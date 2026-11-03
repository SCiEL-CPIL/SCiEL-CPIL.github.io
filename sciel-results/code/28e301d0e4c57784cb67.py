import numpy as np
import time
from scipy.optimize import minimize

N = 400
SIZE = 2 * N - 1
FSIZE = 1
while FSIZE < SIZE:
    FSIZE *= 2
INV_N = 1.0 / FSIZE

# ---- global time budget ----
_T0 = time.time()
_DEADLINE = _T0 + 135.0  # safety margin before the 150 s kill


def _remaining():
    return _DEADLINE - time.time()


# ---------------- core FFT helpers (incremental) ----------------
def _rfft_of_h(h):
    return np.fft.rfft(h, FSIZE)


def _ac_from_H(H):
    """autoconvolution (first SIZE taps) from H = rfft(h)."""
    ac = np.fft.irfft(H * H, FSIZE)
    return ac[:SIZE]


def _full_ac(h):
    return _ac_from_H(_rfft_of_h(h))


def compute_raw_value(h):
    """max(|autoconvolution|) / (sum(h))^2."""
    h = np.asarray(h, dtype=float)
    ac = _full_ac(h)
    m = float(np.max(np.abs(ac)))
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return float('inf')
    return m / (s * s)


# ---------------- incremental flip (4 FFTs per trial) ----------------
def _try_flip(h, H, i, best_val):
    """Flip index i; return (improved, new_H, new_val)."""
    dh = -2.0 * h[i]
    dH = np.zeros(FSIZE // 2 + 1, dtype=np.complex128)
    dH[i] = dh
    Hn = H + np.fft.rfft(dh * np.ones(N), FSIZE)  # cheap: rfft of constant*one-hot
    acn = _ac_from_H(Hn)
    val = float(np.max(np.abs(acn)))  # s unchanged by a sign flip
    if val < best_val:
        return True, Hn, val
    return False, None, None


def _local_search_flip(h, max_passes=8):
    h = np.sign(h).astype(float)
    h[h == 0] = 1.0
    H = _rfft_of_h(h)
    ac = _ac_from_H(H)
    best_val = float(np.max(np.abs(ac)))
    for _ in range(max_passes):
        if _remaining() < 1.0:
            break
        improved = False
        for i in range(N):
            if _remaining() < 0.3:
                break
            ok, Hn, val = _try_flip(h, H, i, best_val)
            if ok:
                h[i] = -h[i]
                H = Hn
                best_val = val
                improved = True
    return h, best_val


def _local_search_block(h, block_sizes=(3, 5, 7), step=7):
    h = np.sign(h).astype(float)
    h[h == 0] = 1.0
    H = _rfft_of_h(h)
    ac = _ac_from_H(H)
    best_val = float(np.max(np.abs(ac)))
    improved = True
    while improved and _remaining() > 1.0:
        improved = False
        for bs in block_sizes:
            for start in range(0, N - bs + 1, step):
                if _remaining() < 0.5:
                    break
                dh = -2.0 * h[start:start + bs]
                dH = np.fft.rfft(np.pad(dh, (0, N - bs)), FSIZE)
                Hn = H + dH
                val = float(np.max(np.abs(_ac_from_H(Hn))))
                if val < best_val:
                    h[start:start + bs] = -h[start:start + bs]
                    H = Hn
                    best_val = val
                    improved = True
    return h, best_val


# ---------------- smooth surrogate with ANALYTICAL gradient ----------------
def _soft_obj_and_grad(x, beta):
    """f = soft_max(|ac(x)|) / s^2  with exact gradient.

    soft_max(|ac|) = m + (1/beta)*log(sum(exp(beta*(|ac|-m))) ), m = max|ac|.
    """
    H = _rfft_of_h(x)
    ac = _ac_from_H(H)
    abs_ac = np.abs(ac)
    m = float(np.max(abs_ac))
    if m < 1e-14:
        m = 1e-14
    scaled = np.clip(beta * (abs_ac - m), -50.0, 50.0)
    e = np.exp(scaled)
    sm = m + float(np.sum(e)) / beta
    s = float(np.sum(x))
    if abs(s) < 1e-9:
        s = 1e-9
    val = sm / (s * s)

    # d softmax / d ac
    w = e / float(np.sum(e))  # softmax weights over |ac|
    # d|ac|/dac = sign(ac) (subgradient at 0 is fine)
    sign_ac = np.sign(ac)
    # dac/dx = 2 * circular-conv(x, x) restricted to first SIZE taps
    # circular conv of length FSIZE; we only need first SIZE taps
    cc = np.fft.irfft(H * H, FSIZE)  # == ac here (linear since zero-padded)
    # grad of softmax w.r.t. x: sum_k w_k sign_k * (2 * (x * x)_k)
    # (x*x)_k = sum_j x_j x_{(k-j) mod FSIZE}; with zero padding it equals ac_k
    # but the full circular gradient is needed; compute via 2 * ifft(H * X_k)
    # Efficient: grad_k = 2 * ifft( H * X_k ) where X_k = rfft(x).
    # We need vector g = sum_k (2 w_k sign_k) * ifft(H * X_k)
    # = ifft( H * rfft( 2 * (w*sign)_padded ) )  using linearity
    coef = 2.0 * w * sign_ac
    coef_full = np.zeros(FSIZE, dtype=float)
    coef_full[:SIZE] = coef
    g = np.fft.irfft(H * np.fft.rfft(coef_full, FSIZE), FSIZE)
    g = g[:N]
    # chain rule: f = sm / s^2 ; df/dx = (g*1 - 2*sm/s) / s^2
    grad = (g - 2.0 * sm / s * np.ones(N)) / (s * s)
    return val, grad


def _smooth_optimize(h0, betas, maxiter=400):
    """Run L-BFGS-B with analytical gradient over an increasing-beta schedule."""
    s0 = float(np.sum(h0))
    if abs(s0) < 1e-6:
        return None
    h = h0 / s0  # normalize to sum=1 (removes scale; objective scale-invariant)
    best = None
    for beta in betas:
        if _remaining() < 2.0:
            break
        try:
            def obj(x):
                v, g = _soft_obj_and_grad(x, beta)
                return v, g
            res = minimize(obj, h, jac=True, method='L-BFGS-B',
                           options={'maxiter': maxiter, 'disp': False, 'gtol': 1e-8})
            h = res.x
            val = compute_raw_value(h)
            if best is None or val < best[1]:
                best = (h.copy(), val)
            # also keep the binarized version as a candidate
            hb = np.sign(h)
            hb[hb == 0] = 1.0
            vb = compute_raw_value(hb)
            if vb < best[1]:
                best = (hb.copy(), vb)
        except Exception:
            pass
    return best


# ---------------- epigraph / tangent (DC) refinement on the exact max ----------------
def _tangent_refine(h, max_iter=200, tol=1e-9):
    """Minimize max|ac(x)|/s^2 by DC: at each step minimize a linearization
    (tangent) of max|ac| at the current peak, then re-normalize. This drives the
    dominant peak down while keeping the rest controlled."""
    h = np.asarray(h, dtype=float).copy()
    s = float(np.sum(h))
    if abs(s) < 1e-6:
        return h
    h = h / s
    best = h.copy()
    best_val = compute_raw_value(best)
    for _ in range(max_iter):
        if _remaining() < 0.5:
            break
        H = _rfft_of_h(h)
        ac = _ac_from_H(H)
        abs_ac = np.abs(ac)
        k = int(np.argmax(abs_ac))
        target = abs_ac[k]
        sign_k = 1.0 if ac[k] >= 0 else -1.0
        # linear surrogate: L(x) = ac_k(x) * sign_k  (we want to push it down)
        # minimize L(x) subject to sum(x)=1  -> use L-BFGS-B with gradient
        # dL/dx = 2 * (x * x)_k = 2 * ifft(H * X_k)
        def lin_obj(x):
            Hx = _rfft_of_h(x)
            acx = _ac_from_H(Hx)
            val = sign_k * acx[k]
            g = np.fft.irfft(Hx * np.fft.rfft(np.eye(1)[0] * 0 + 0, FSIZE), FSIZE)  # placeholder
            # proper gradient:
            g = 2.0 * np.fft.irfft(Hx * np.fft.rfft(x, FSIZE), FSIZE) * 0  # not used
            return val
        # Use closed-form gradient of ac_k(x) = sum_j x_j x_{k-j}
        # d ac_k / d x_i = 2 * x_{(k - i) mod N}  (within linear conv, zero outside)
        def ac_k_and_grad(x):
            Hx = _rfft_of_h(x)
            acx = _ac_from_H(Hx)
            val = sign_k * acx[k]
            # shift: x_{(k-i) mod N} for i in [0,N)
            idx = (k - np.arange(N)) % N
            g = 2.0 * sign_k * x[idx]
            return val, g
        try:
            res = minimize(ac_k_and_grad, h, jac=True, method='L-BFGS-B',
                           options={'maxiter': 60, 'disp': False, 'gtol': 1e-9})
            h = res.x
            # re-normalize
            ss = float(np.sum(h))
            if abs(ss) < 1e-6:
                break
            h = h / ss
            val = compute_raw_value(h)
            if val < best_val - tol:
                best_val = val
                best = h.copy()
            else:
                # try binarized
                hb = np.sign(h)
                hb[hb == 0] = 1.0
                vb = compute_raw_value(hb)
                if vb < best_val:
                    best_val = vb
                    best = hb.copy()
        except Exception:
            pass
    return best


# ---------------- candidate generation ----------------
def _make_candidates(rng):
    cands = []
    t = np.arange(N)
    # all ones
    cands.append(np.ones(N))
    # random
    for _ in range(24):
        cands.append(rng.choice([1.0, -1.0], N))
    # quadratic chirps
    for freq in [1.0, 2.0, 0.5, 3.0, 4.0]:
        ph = 2 * np.pi * freq * (t ** 2) / N
        x = np.sign(np.cos(ph)); x[x == 0] = 1
        cands.append(x)
    # linear chirps
    for freq in [1.0, 2.0, 0.5, 3.0]:
        ph = 2 * np.pi * freq * t / N
        x = np.sign(np.cos(ph)); x[x == 0] = 1
        cands.append(x)
    # golden ratio
    phi = (1 + np.sqrt(5)) / 2
    x = np.sign(np.cos(2 * np.pi * phi * t)); x[x == 0] = 1
    cands.append(x)
    # m-sequence style (pseudo-random from LFSR)
    def lfsr(n, seed=1):
        a = [0] * n
        a[0] = seed
        for i in range(1, n):
            a[i] = (a[i - 1] >> 1) | (((a[i - 1] & 1) ^ ((a[i - 1] >> 3) & 1)) << (7))
        return a
    # simple xorshift-based
    state = 123456789
    seq = np.zeros(N)
    for i in range(N):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= (state >> 17)
        state ^= (state << 5) & 0xFFFFFFFF
        seq[i] = 1.0 if (state & 1) else -1.0
    cands.append(seq)
    return cands


def construct():
    global _T0, _DEADLINE
    _T0 = time.time()
    _DEADLINE = _T0 + 135.0

    rng = np.random.default_rng(42)
    cands = _make_candidates(rng)

    # evaluate all, keep the best
    best_x = None
    best_val = float('inf')
    for x in cands:
        v = compute_raw_value(x)
        if v < best_val:
            best_val = v
            best_x = x.copy()

    # ---- Phase A: multi-start smooth surrogate optimization (analytical grad) ----
    # sort candidates by value, take diverse top starts
    scored = sorted([(compute_raw_value(x), x) for x in cands], key=lambda t: t[0])
    top = [x for _, x in scored[:12]]
    # also include a few random starts
    top += [rng.choice([1.0, -1.0], N) for _ in range(6)]

    for h0 in top:
        if _remaining() < 20.0:
            break
        res = _smooth_optimize(h0, betas=[40, 80, 160, 320], maxiter=250)
        if res is not None:
            hx, vx = res
            if vx < best_val:
                best_val = vx
                best_x = hx.copy()

    # ---- Phase B: incremental hill-climbing from multiple starts ----
    starts = [best_x] + [x for _, x in scored[:6]]
    for h0 in starts:
        if _remaining() < 15.0:
            break
        h, v = _local_search_flip(h0, max_passes=10)
        if v < best_val:
            best_val = v
            best_x = h.copy()
        if _remaining() < 8.0:
            break
        h, v = _local_search_block(h, block_sizes=(3, 5, 7), step=5)
        if v < best_val:
            best_val = v
            best_x = h.copy()

    # ---- Phase C: epigraph/tangent refinement on the exact max ----
    if _remaining() > 15.0:
        h = _tangent_refine(best_x, max_iter=150)
        v = compute_raw_value(h)
        if v < best_val:
            best_val = v
            best_x = h.copy()

    # ---- Phase D: final polish (smooth + flip) on the best ----
    if _remaining() > 12.0:
        res = _smooth_optimize(best_x, betas=[320, 640, 1280], maxiter=300)
        if res is not None:
            hx, vx = res
            if vx < best_val:
                best_val = vx
                best_x = hx.copy()
    if _remaining() > 6.0:
        h, v = _local_search_flip(best_x, max_passes=12)
        if v < best_val:
            best_val = v
            best_x = h.copy()
    if _remaining() > 3.0:
        h, v = _local_search_block(best_x, block_sizes=(3, 5), step=3)
        if v < best_val:
            best_val = v
            best_x = h.copy()

    # ensure valid
    best_x = np.asarray(best_x, dtype=float)
    return best_x


if __name__ == "__main__":
    x = construct()
    print("Best raw value:", compute_raw_value(x))
    print("Sum:", np.sum(x))
