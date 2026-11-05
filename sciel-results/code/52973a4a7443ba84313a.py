import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 75.0  # safety margin before 90s kill

    n = 128
    N = n + 1  # indices 0..128
    # target sums 128..256 inclusive
    lo, hi = 128, 256
    T = hi - lo + 1  # 129 target sums

    # convolution: c[s] = sum_{i+j=s} x[i]*x[j]  (this counts ordered pairs)
    # unordered with equal summands allowed: for s even, the i=j term counted once in ordered,
    # but "unordered permit equal" means {a,b} with a<=b. The multiplicity of unordered pairs
    # equals: (ordered_count + diagonal_count)/2 where diagonal = 1 if s even and x[s/2]=1.
    # We'll compute unordered directly.
    # ordered = conv(x,x)[s]; diag = 1 if s%2==0 and x[s//2] else 0
    # unordered = (ordered + diag) / 2

    # Precompute FFT of x for fast conv
    def compute_unordered(x):
        # x: 0/1 array length N
        # ordered convolution via FFT
        L = 1
        need = 2 * N - 1
        while L < need:
            L *= 2
        fx = np.fft.rfft(x, L)
        conv = np.fft.irfft(fx * fx, L)[:need]
        conv = np.rint(conv)  # integer
        # target slice
        ordered_t = conv[lo:hi+1]  # length T
        # diagonal: for s in [lo,hi], diag=1 if s even and x[s//2]==1 (and s//2 in [0,n])
        s_arr = np.arange(lo, hi+1)
        even = (s_arr % 2 == 0)
        idx = s_arr // 2
        valid = (idx >= 0) & (idx <= n)
        diag = np.where(even & valid, x[idx], 0)
        unordered = (ordered_t + diag) / 2.0
        return unordered

    # objective helpers
    def peak(c):
        return float(np.max(c)) if c.size else 0.0

    # Initialize: start from a covering set. Use random dense set then repair.
    # We want coverage first. Use a simple construction: A = [0..128] gives coverage but huge peak.
    # Better start: random subset of moderate density, then repair to cover.
    rng = np.random.default_rng(0)

    def random_init(density):
        x = (rng.random(N) < density).astype(np.float64)
        # ensure nonempty
        if not x.any():
            x[rng.integers(N)] = 1
        return x

    def repair(x):
        # ensure every target sum has at least one representation
        # iterate: find uncovered sums, add elements to cover them
        for _ in range(200):
            c = compute_unordered(x)
            uncov = np.where(c < 1)[0]
            if uncov.size == 0:
                break
            # pick an uncovered sum s, try adding an element a such that s-a in A (or add both)
            s = int(uncov[0])
            added = False
            # try existing a in A with s-a in A? then it'd be covered already. So need to add.
            # add a and s-a (if in range)
            # choose a to maximize new coverage: try a in 0..n
            best_a, best_gain = None, 0
            # candidate a values: those where s-a in [0,n]
            a_min = max(0, s - n)
            a_max = min(n, s)
            # try a few random candidates + symmetric
            cands = list(range(a_min, a_max+1))
            rng.shuffle(cands)
            for a in cands[:200]:
                b = s - a
                if b < 0 or b > n:
                    continue
                # gain: number of currently-uncovered sums that become covered by adding {a,b}
                # new sums from a: a + A ; from b: b + A
                # approximate gain by checking target sums
                gain = 0
                # sums involving a: a + x_j for j in A
                # and b + x_j
                # and a+b
                # we'll just compute after adding both and compare; but that's expensive per cand.
                # Instead approximate: count uncovered s' in [a, a+n] and [b, b+n]
                # quick: check if a in A already
                if x[a] == 1 and x[b] == 1:
                    continue
                # estimate gain
                gain = 0
                # target sums that a can form with existing A
                for s2 in range(max(lo, a), min(hi, a+n)+1):
                    j = s2 - a
                    if 0 <= j <= n and x[j] == 1 and c[s2-lo] < 1:
                        gain += 1
                for s2 in range(max(lo, b), min(hi, b+n)+1):
                    j = s2 - b
                    if 0 <= j <= n and x[j] == 1 and c[s2-lo] < 1:
                        gain += 1
                # a+b itself
                if a+b >= lo and a+b <= hi and c[a+b-lo] < 1:
                    gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_a = a
            if best_a is not None:
                b = s - best_a
                x[best_a] = 1
                x[b] = 1
            else:
                # fallback: add s//2
                x[s//2] = 1
        return x

    # multiplicative weights / gradient step
    def step(x, lr, lam, it):
        c = compute_unordered(x)
        # gradient of sum c^2 w.r.t x_k:
        # c[s] = 0.5*(ordered[s]+diag[s]); ordered[s]=sum_i x[i]x[s-i]
        # d c[s]/d x[k] = 0.5*( (x[s-k] + x[k] if s-k==k else x[s-k]) + diag_derived )
        # simpler: use subgradient via finite-ish: for each k, contribution to c[s] is x[s-k]
        # (for s-k != k) plus 0.5 for the diagonal when 2k==s.
        # We'll compute gradient g[k] = sum_s 2*c[s] * d c[s]/d x[k]
        # d c[s]/d x[k] = 0.5 * (x[s-k] + (1 if s-k==k else 0))   [since ordered has x[k]x[s-k] twice when k!=s-k]
        # Let's compute numerically-safe:
        # g[k] = sum_s c[s] * (x[s-k] + (1 if 2k==s else 0))   (absorbing factor)
        # coverage penalty: for s with c[s]<1, push up sums that k participates in
        # We'll do: for each k, delta = -lr * g[k] + lam * cover_grad[k]
        # cover_grad[k] = sum_{s: c[s]<1} d c[s]/d x[k]
        # To keep it fast, vectorize over k using convolution-like ops.
        # g[k] = sum_s c[s] x[s-k]  (over s in [lo,hi], s-k in [0,n])  +  c[2k] (if 2k in range)
        # The first term is a cross-correlation of c (padded) with x.
        # Let's build c_full length (2n) and compute for each k: sum_s c[s]*x[s-k]
        # = (c * x_reversed) style. Use FFT.
        c_full = np.zeros(2*n+1)
        c_full[lo:hi+1] = c
        # term1[k] = sum_s c_full[s] * x[s-k] for s-k in [0,n], s in [lo,hi]
        # = sum_j x[j] * c_full[j+k]
        # = cross-correlation
        L = 1
        need = 2*N
        while L < need:
            L *= 2
        fx = np.fft.rfft(x, L)
        fc = np.fft.rfft(c_full, L)
        # correlation: sum_j x[j] c[j+k] = ifft( fx * conj(fc) )[k]
        corr = np.fft.irfft(fx * np.conj(fc), L)[:N]
        term1 = np.rint(corr)
        # term2: c[2k] if 2k in [lo,hi]
        term2 = np.zeros(N)
        ks = np.arange(N)
        two_k = 2*ks
        inr = (two_k >= lo) & (two_k <= hi)
        term2[inr] = c_full[two_k[inr]]
        g = term1 + term2  # this is sum_s c[s] * (x[s-k] + diag)  ~ 2 * c * dc/dx

        # coverage gradient
        uncov_mask = (c < 1)
        # cover_grad[k] = sum_{s:uncov} (x[s-k] + diag_s_k)
        c_uncov = np.zeros(2*n+1)
        c_uncov[lo:hi+1] = uncov_mask.astype(np.float64)
        fx2 = np.fft.rfft(x, L)
        fc2 = np.fft.rfft(c_uncov, L)
        corr2 = np.fft.irfft(fx2 * np.conj(fc2), L)[:N]
        term1c = np.rint(corr2)
        term2c = np.zeros(N)
        term2c[inr] = c_uncov[two_k[inr]]
        cg = term1c + term2c

        # update: x += -lr*g + lam*cg, then project to [0,1], then maybe binarize
        x_new = x - lr * g + lam * cg
        x_new = np.clip(x_new, 0, 1)
        return x_new, x

    best_x = None
    best_score = None

    def evaluate(x):
        c = compute_unordered(x)
        cov_ok = np.all(c >= 1)
        pk = peak(c)
        if not cov_ok:
            return None, pk, c
        return pk, pk, c

    # Phase 1: get a covering solution via repair from a few random inits, keep best peak
    for density in [0.5, 0.6, 0.7]:
        if time.time() > DEADLINE:
            break
        x = random_init(density)
        x = repair(x)
        sc, pk, c = evaluate(x)
        if sc is not None:
            if best_score is None or sc < best_score:
                best_score = sc
                best_x = x.copy()

    if best_x is None:
        # fallback: full set
        x = np.ones(N)
        x = x.astype(np.float64)
        best_x = x
        best_score, _, _ = evaluate(x)

    # Phase 2: multiplicative / gradient refinement to reduce peak
    # Work in continuous x, periodically binarize and repair.
    x = best_x.copy()
    lr = 0.05
    lam = 0.5
    it = 0
    no_improve = 0
    while time.time() < DEADLINE:
        x_new, _ = step(x, lr, lam, it)
        # binarize with threshold, keep density similar
        thr = 0.5
        xb = (x_new >= thr).astype(np.float64)
        # ensure nonempty
        if not xb.any():
            xb[np.argmax(x_new)] = 1
        # repair coverage
        xb = repair(xb)
        sc, pk, c = evaluate(xb)
        if sc is not None and (best_score is None or sc < best_score):
            best_score = sc
            best_x = xb.copy()
            no_improve = 0
            # decay lr slightly on improvement
            lr = max(0.01, lr * 0.95)
        else:
            no_improve += 1
            # adapt: if stuck, try higher lambda or different lr
            if no_improve % 10 == 0:
                lr = max(0.01, min(0.2, lr * 1.2))
                lam = min(2.0, lam * 1.1)
        x = x_new
        it += 1

    # Phase 3: local search on best_x: try removing elements (if coverage stays) to lower peak,
    # and swapping.
    def local_search(x, budget):
        x = x.copy()
        sc, pk, c = evaluate(x)
        if sc is None:
            return x
        improved = True
        while improved and time.time() < budget:
            improved = False
            # try removing each element
            idxs = np.where(x == 1)[0]
            rng.shuffle(idxs)
            for k in idxs:
                if time.time() > budget:
                    break
                x2 = x.copy()
                x2[k] = 0
                sc2, pk2, c2 = evaluate(x2)
                if sc2 is not None and sc2 < sc:
                    x = x2
                    sc = sc2
                    improved = True
            # try swapping: remove k, add j
            idxs = np.where(x == 1)[0]
            zeros = np.where(x == 0)[0]
            rng.shuffle(idxs)
            rng.shuffle(zeros)
            for k in idxs[:50]:
                if time.time() > budget:
                    break
                for j in zeros[:50]:
                    if time.time() > budget:
                        break
                    x2 = x.copy()
                    x2[k] = 0
                    x2[j] = 1
                    sc2, pk2, c2 = evaluate(x2)
                    if sc2 is not None and sc2 < sc:
                        x = x2
                        sc = sc2
                        improved = True
        return x

    best_x = local_search(best_x, DEADLINE - 2)
    sc, pk, c = evaluate(best_x)
    if sc is not None:
        best_score = sc

    # return as list of ints
    res = [int(i) for i in np.where(best_x == 1)[0]]
    if not res:
        res = [0]
    return res
# EVOLVE-BLOCK-END
