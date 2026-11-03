"""Combined program: smooth surrogate + peak suppression + fast local search."""
import time
import numpy as np
from scipy.optimize import minimize

N = 400
_FFT_SIZE = 1 << ((2 * N - 1) - 1).bit_length()  # next power of two >= 2N-1


def _autoconv_full(h):
    """Full linear autoconvolution via FFT (fixed size)."""
    H = np.fft.rfft(h, _FFT_SIZE)
    ac = np.fft.irfft(H * H, _FFT_SIZE)
    return ac[:2 * N - 1]


def _score(x):
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    conv = _autoconv_full(x)
    return np.max(np.abs(conv)) / (s * s)


def _soft_max_abs(ac, beta):
    """Smooth max(|ac|) via log-sum-exp."""
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    scaled = np.clip(beta * (abs_ac - m), -50, 50)
    return m + np.log(np.sum(np.exp(scaled))) / beta


def _smooth_obj(h, beta):
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _soft_max_abs(_autoconv_full(h), beta) / (s * s)


def _local_search_flip(h, max_passes=6, deadline=None):
    """Fast single-bit sign-flip local search with O(1) score update."""
    h = np.sign(h).astype(np.float64)
    h[h == 0] = 1
    s = float(np.sum(h))
    ac = _autoconv_full(h)
    abs_ac = np.abs(ac)
    best_val = float(np.max(abs_ac)) / (s * s)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            # flip element i
            delta = -2.0 * h[i]
            s_new = s + delta
            if abs(s_new) < 1e-9:
                h[i] = -h[i]
                continue
            ac_new = ac + delta * h  # autoconv(x + delta*e_i) = ac + delta*(x + delta*e_i)
            # but x + delta*e_i = h with i flipped = h_new, so ac_new = ac + delta*h_new
            # h_new = h with h[i] flipped; delta*h_new = delta*h - delta*h[i] - delta*h[i] = delta*h - 2*delta*h[i]
            # Simpler: recompute via the identity ac_new[k] = ac[k] + delta*h_new[k-i]
            # Use direct: ac_new = ac + delta * h_new where h_new[i] = -h[i]
            h[i] = -h[i]
            h_new = h
            ac_new = ac + delta * h_new
            s_new = float(np.sum(h_new))
            abs_new = np.max(np.abs(ac_new))
            val_new = abs_new / (s_new * s_new)
            if val_new < best_val:
                ac = ac_new
                s = s_new
                best_val = val_new
                improved = True
            else:
                h[i] = -h[i]  # revert
        if not improved:
            break
    return h, best_val


def _local_search_block(h, block_sizes=(3, 5, 7), deadline=None):
    """Block sign-flip local search."""
    h = np.sign(h).astype(np.float64)
    h[h == 0] = 1
    best_val = _score(h)
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        for bs in block_sizes:
            step = max(1, (N - bs) // 40)
            for start in range(0, N - bs + 1, step):
                if deadline is not None and time.time() > deadline:
                    break
                h[start:start + bs] = -h[start:start + bs]
                val = _score(h)
                if val < best_val:
                    best_val = val
                    improved = True
                else:
                    h[start:start + bs] = -h[start:start + bs]
    return h, best_val


def _continuous_optimize(h_init, deadline):
    """Multi-stage smooth optimization + peak suppression, returns best continuous x."""
    best_x = h_init.copy()
    best_val = _score(best_x)

    s0 = np.sum(h_init)
    if abs(s0) < 1e-6:
        return best_x, best_val
    h_norm = h_init / s0

    # Stage 1: increasing-beta smooth optimization
    for beta in [20.0, 50.0, 120.0]:
        if time.time() > deadline - 15:
            break
        try:
            res = minimize(lambda x: _smooth_obj(x, beta), h_norm,
                           method='L-BFGS-B',
                           options={'maxiter': 150, 'disp': False, 'gtol': 1e-6})
            h_opt = res.x
            val = _score(h_opt)
            if val < best_val:
                best_val = val
                best_x = h_opt.copy()
            h_norm = h_opt
        except Exception:
            pass

    # Stage 2: peak suppression — directly penalize the argmax of |autoconv|
    if time.time() < deadline - 10:
        for _ in range(3):
            ac = _autoconv_full(best_x)
            peak_idx = int(np.argmax(np.abs(ac)))
            s = np.sum(best_x)

            def peak_obj(x):
                a = _autoconv_full(x)
                ss = np.sum(x)
                if abs(ss) < 1e-9:
                    return 1e10
                base = np.max(np.abs(a)) / (ss * ss)
                # extra penalty on the current peak location
                return base + 0.5 * abs(a[peak_idx]) / (ss * ss)

            try:
                res = minimize(peak_obj, best_x, method='L-BFGS-B',
                               options={'maxiter': 100, 'disp': False, 'gtol': 1e-6})
                val = _score(res.x)
                if val < best_val:
                    best_val = val
                    best_x = res.x.copy()
                else:
                    break
            except Exception:
                break

    return best_x, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    # ---- 1. Baseline + structured templates (current program's strength) ----
    best_x = np.ones(N, dtype=np.float64)
    best_val = _score(best_x)
    candidates = [best_x.copy()]

    # Two/three/four block patterns
    for k in range(1, N):
        x = np.concatenate([np.ones(k), -np.ones(N - k)])
        candidates.append(x)
    for k1 in range(1, N // 2 + 1, 8):
        for k2 in range(1, N - k1 + 1, 8):
            k3 = N - k1 - k2
            if k3 < 0:
                continue
            x = np.concatenate([np.ones(k1), -np.ones(k2), np.ones(k3)])
            candidates.append(x)
            if time.time() > deadline - 100:
                break

    # Sinusoidal / chirp / golden-ratio sign patterns
    t = np.arange(N)
    for freq in [1, 2, 3, 5, 7, 10]:
        x = np.sign(np.sin(2 * np.pi * freq * t / N))
        x[x == 0] = 1
        candidates.append(x.astype(np.float64))
    for freq in [1.0, 2.0, 0.5]:
        phase = 2 * np.pi * freq * (t ** 2) / N
        x = np.sign(np.cos(phase))
        x[x == 0] = 1
        candidates.append(x.astype(np.float64))
    phi = (1 + np.sqrt(5)) / 2
    xg = np.sign(np.cos(2 * np.pi * phi * t))
    xg[xg == 0] = 1
    candidates.append(xg.astype(np.float64))

    # Random starts
    rng = np.random.default_rng(42)
    for _ in range(8):
        candidates.append(rng.choice([1.0, -1.0], N))

    # Evaluate all candidates
    scored = []
    for x in candidates:
        v = _score(x)
        scored.append((v, x))
    scored.sort(key=lambda z: z[0])
    if scored[0][0] < best_val:
        best_val = scored[0][0]
        best_x = scored[0][1].copy()

    # ---- 2. Continuous optimization from top diverse starts (inspiration) ----
    top = scored[:5]
    for v0, h0 in top:
        if time.time() > deadline - 40:
            break
        x_opt, v_opt = _continuous_optimize(h0, deadline - 20)
        if v_opt < best_val:
            best_val = v_opt
            best_x = x_opt.copy()
        # also try binarized
        hb = np.sign(x_opt)
        hb[hb == 0] = 1
        vb = _score(hb)
        if vb < best_val:
            best_val = vb
            best_x = hb.copy()

    # ---- 3. Fast local search on the best (refinement machinery) ----
    if time.time() < deadline - 15:
        h_work = np.sign(best_x)
        h_work[h_work == 0] = 1
        h_work, v_work = _local_search_flip(h_work, max_passes=5, deadline=deadline - 10)
        if v_work < best_val:
            best_val = v_work
            best_x = h_work.copy()

    if time.time() < deadline - 8:
        h_work, v_work = _local_search_block(h_work, block_sizes=(3, 5, 7),
                                             deadline=deadline - 4)
        if v_work < best_val:
            best_val = v_work
            best_x = h_work.copy()

    # ---- 4. A couple of random restarts with local search if time remains ----
    if time.time() < deadline - 20:
        for _ in range(4):
            if time.time() > deadline - 12:
                break
            xr = rng.choice([1.0, -1.0], N)
            hr, vr = _local_search_flip(xr, max_passes=2, deadline=deadline - 8)
            if vr < best_val:
                best_val = vr
                best_x = hr.copy()

    return best_x
