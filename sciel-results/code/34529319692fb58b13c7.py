"""Optimized second autocorrelation inequality: weighted Sidon supports + smooth-max gradient polish."""
import numpy as np
import time
import random

N = 50
MAX_TIME = 130.0  # safety margin before 150s kill
START_TIME = time.time()

def _remaining():
    return MAX_TIME - (time.time() - START_TIME)

def _ratio(h):
    """||h||_2^2 / (||h||_1 * ||h*h||_inf). Scale-invariant."""
    h = np.asarray(h, dtype=float)
    l1 = h.sum()
    if l1 <= 1e-15:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    linf = conv.max()
    if linf <= 1e-15:
        return 0.0
    return (h @ h) / (l1 * linf)

def _max_paircount(sidon_set):
    """Max number of ordered pairs (a,b) in S with a-b = d, over all d."""
    s = list(sidon_set)
    n = len(s)
    if n == 0:
        return 0
    counts = {}
    for a in s:
        for b in s:
            d = a - b
            counts[d] = counts.get(d, 0) + 1
    return max(counts.values())

def _is_sidon(S):
    """Check S is a Sidon set: all pairwise sums a+b (a<=b) distinct."""
    s = list(S)
    seen = set()
    for i in range(len(s)):
        for j in range(i, len(s)):
            sm = s[i] + s[j]
            if sm in seen:
                return False
            seen.add(sm)
    return True

def _find_sidon_set(n, target_size, seed=0):
    """Greedy Sidon set of given size within [0, n)."""
    rng = random.Random(seed)
    # Try multiple random orderings
    for attempt in range(20):
        order = list(range(n))
        if attempt == 0:
            order.sort()
        else:
            rng.shuffle(order)
        sidon = []
        sums = set()
        for i in order:
            ok = True
            for a in sidon:
                if (a + i) in sums:
                    ok = False
                    break
            if ok:
                # also check i+i
                if (i + i) in sums:
                    ok = False
            if ok:
                for a in sidon:
                    sums.add(a + i)
                sums.add(i + i)
                sidon.append(i)
                if len(sidon) >= target_size:
                    return sidon
    return sidon

def _local_search_support(sidon_set, n, iters=3000):
    """Improve support by swapping elements to minimize max_paircount."""
    current = list(sidon_set)
    cur_mpc = _max_paircount(current)
    cur_set = set(current)
    for _ in range(iters):
        if _remaining() < 5:
            break
        improved = False
        for idx, elem in enumerate(current):
            for new_elem in range(n):
                if new_elem in cur_set:
                    continue
                trial = list(current)
                trial[idx] = new_elem
                trial_set = set(trial)
                mpc = _max_paircount(trial_set)
                if mpc < cur_mpc:
                    current = trial
                    cur_set = trial_set
                    cur_mpc = mpc
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return current, cur_mpc

def _optimize_weights(support, max_iter=4000, lr=0.05):
    """Maximize ratio via projected gradient ascent on weights (log-sum-exp smooth max).
    Fix ||h||_1 = 1 by normalizing; optimize shape."""
    k = len(support)
    if k == 0:
        return None
    support = np.array(support, dtype=int)
    h = np.zeros(N)
    h[support] = 1.0 / k  # start uniform, ||h||_1 = 1

    # Precompute: for each lag d, the set of index pairs (i,j) with support[i]-support[j]=d
    # conv[c] = sum_{i-j = c-(N-1)} h[i]*h[j]  (full mode index)
    # We'll work in full-convolution index space.
    best_h = h.copy()
    best_r = _ratio(h)

    # Build pair list for each convolution bin
    # full conv index: c ranges 0..2N-2, c = i + j where i,j are 0..N-1
    # We only have h nonzero on support.
    # For gradient of log-sum-exp over conv, we need d(conv)/d(h)
    # conv[c] = sum_{i+j=c} h[i]*h[j]
    # d(conv[c])/d(h[m]) = sum_{j: m+j=c} h[j] + sum_{i: i+m=c} h[i]  (but careful with i=j)
    # Actually conv[c] = sum_i h[i] h[c-i], so d/dc: d(conv[c])/d(h[m]) = h[c-m] + h[c-m] if m != c-m, else h[c-m]
    # Simpler: use the fact that conv is quadratic. Use finite-difference-free analytic grad.

    def smooth_max(conv, tau):
        m = conv.max()
        return m + tau * np.log(np.sum(np.exp((conv - m) / tau)))

    def grad_ratio_logsumexp(h, tau):
        """Gradient of ratio w.r.t. h, using log-sum-exp for ||conv||_inf."""
        l1 = h.sum()
        conv = np.convolve(h, h, mode='full')
        # log-sum-exp of conv
        m = conv.max()
        z = np.exp((conv - m) / tau)
        zsum = z.sum()
        # d/d(conv[c]) of (m + tau*log(zsum)) = tau * z[c]/zsum
        w = tau * z / zsum  # weights per conv bin
        # d(conv[c])/d(h[m]) : conv[c] = sum_i h[i] h[c-i]
        # grad_conv[m] = sum_c w[c] * d(conv[c])/d(h[m])
        # d(conv[c])/d(h[m]) = h[c-m] + h[c-m] (counting both i=m and j=m) but if c-m==m then just h[m] once? 
        # No: conv[c] = sum_{i} h[i]*h[c-i]. Term with i=m: h[m]*h[c-m]. Term with j=m (i=c-m): h[c-m]*h[m].
        # If m != c-m, these are two distinct terms each contributing h[c-m] and h[m] respectively to derivative.
        # d/d(h[m]) [h[m]*h[c-m]] = h[c-m] (when m != c-m); if m==c-m, d/d(h[m])[h[m]^2] = 2 h[m].
        # So d(conv[c])/d(h[m]) = 2*h[c-m] if m != c-m, else 2*h[m]. Wait: if m==c-m, term is h[m]*h[m]=h[m]^2, deriv = 2 h[m].
        # If m != c-m: two terms h[m]h[c-m] and h[c-m]h[m], each deriv w.r.t h[m] is h[c-m], total 2 h[c-m].
        # So in all cases: d(conv[c])/d(h[m]) = 2 * h[c-m] (where h[c-m]=0 if out of range).
        # Let's verify: grad_conv[m] = sum_c w[c] * 2 * h[c-m]
        # = 2 * sum_c w[c] h[c-m]  (with h out of range = 0)
        # This is 2 * (w * h)[m] where * is convolution? sum_c w[c] h[c-m] = (w ⊛ h)[m] (cross-correlation).
        # Actually sum_c w[c] h[c-m] = sum_k w[k+m] h[k] = (w flipped convolve h)... let's just compute directly.
        grad_conv = np.zeros(N)
        # For each m in 0..N-1, grad_conv[m] = 2 * sum_{c} w[c] * h[c-m] where c-m in [0,N-1]
        # c ranges 0..2N-2, c-m in [0,N-1] => c in [m, m+N-1]
        for m in range(N):
            c_lo = m
            c_hi = m + N - 1
            if c_hi >= len(conv):
                c_hi = len(conv) - 1
            if c_lo < 0:
                c_lo = 0
            # sum over c in [c_lo, c_hi] of w[c] * h[c-m]
            idx = np.arange(c_lo, c_hi + 1)
            h_idx = idx - m  # 0..N-1
            grad_conv[m] = 2.0 * np.sum(w[idx] * h[h_idx])

        # ratio = (h@h) / (l1 * S) where S = smooth_max(conv)
        # d(ratio)/d(h) = [2 h * l1 * S - (h@h)(grad_l1 * S + l1 * grad_S)] / (l1^2 S^2)
        # grad_l1 = 1 vector, grad_S = dS/d(h) = grad_conv (since S = smooth max of conv, dS/d(h[m]) = grad_conv[m])
        # Wait: S depends on h through conv. dS/d(h[m]) = sum_c (dS/d(conv[c])) (d(conv[c])/d(h[m])) = sum_c (tau z[c]/zsum) * d(conv[c])/d(h[m]) = grad_conv[m]. Yes.
        num = h @ h
        dnum = 2.0 * h
        dl1 = np.ones(N)
        dS = grad_conv
        # ratio = num / (l1 * S)
        # d(ratio) = [dnum*(l1*S) - num*(dl1*S + l1*dS)] / (l1*S)^2
        denom = (l1 * S) ** 2
        grad = (dnum * (l1 * S) - num * (dl1 * S + l1 * dS)) / denom
        return grad, _ratio(h)

    # Adaptive gradient ascent with line search
    h_work = h.copy()
    for it in range(max_iter):
        if _remaining() < 5:
            break
        tau = 0.01 + 0.05 * (it / max_iter)  # anneal tau
        grad, r = grad_ratio_logsumexp(h_work, tau)
        # Project: keep nonnegative, normalize to l1=1
        # Simple step
        h_new = h_work + lr * grad
        h_new = np.maximum(h_new, 0)
        if h_new.sum() <= 1e-15:
            h_new = h_work.copy()
        h_new = h_new / h_new.sum()
        r_new = _ratio(h_new)
        if r_new > r:
            h_work = h_new
            if r_new > best_r:
                best_r = r_new
                best_h = h_work.copy()
        else:
            lr = max(lr * 0.8, 0.001)

    return best_h, best_r

def construct():
    random.seed(42)
    best_h = np.ones(N) / N
    best_r = _ratio(best_h)

    # Try various Sidon set sizes
    for target_size in [7, 8, 9, 10, 11, 12, 13, 14, 15]:
        if _remaining() < 15:
            break
        for seed in range(8):
            if _remaining() < 10:
                break
            sidon = _find_sidon_set(N, target_size, seed=seed)
            if len(sidon) < 5:
                continue
            # Local search on support
            improved_sidon, mpc = _local_search_support(sidon, N, iters=1500)
            # Optimize weights
            res = _optimize_weights(improved_sidon, max_iter=2500)
            if res is not None:
                h_opt, r_opt = res
                if r_opt > best_r:
                    best_r = r_opt
                    best_h = h_opt.copy()

    # Also try random supports of various sizes with weight optimization
    for size in [6, 8, 10, 12, 14]:
        if _remaining() < 10:
            break
        for trial in range(10):
            if _remaining() < 8:
                break
            support = random.sample(range(N), size)
            res = _optimize_weights(support, max_iter=1500)
            if res is not None:
                h_opt, r_opt = res
                if r_opt > best_r:
                    best_r = r_opt
                    best_h = h_opt.copy()

    # Final polish: refine best_h with more iterations
    if _remaining() > 10:
        support = np.where(best_h > 1e-8)[0]
        if len(support) > 0:
            res = _optimize_weights(support, max_iter=3000, lr=0.02)
            if res is not None:
                h_opt, r_opt = res
                if r_opt > best_r:
                    best_r = r_opt
                    best_h = h_opt.copy()

    # Ensure valid
    best_h = np.maximum(best_h, 0)
    if best_h.sum() < 1e-12:
        return [1.0] * N
    # Normalize to sum=1 (scale-invariant, but keep reasonable magnitude)
    best_h = best_h / best_h.sum()
    return list(best_h)

# EVOLVE-BLOCK-END
