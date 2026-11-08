import numpy as np
import time
import scipy.optimize as opt
from scipy.fft import dct, idct

N = 50
M = 2 * N - 1  # 99
W = 1.0 / (M + 1)  # 1/100
TIME_LIMIT = 150.0
RNG = np.random.RandomState(50)
_DEADLINE = 0.0


def compute_value(h):
    h = np.asarray(h, dtype=float).ravel()
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    if not np.all(np.isfinite(h)) or np.any(h < -1e-12):
        return 0.0
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(g) / (m + 1)
    norm_inf = np.max(g)
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)


def _value_neg(h):
    return -compute_value(h)


def _smooth_max(g, tau=0.01):
    g_max = np.max(g)
    if g_max <= 0:
        return 0.0
    shifted = g / g_max
    log_sum = np.log(np.sum(np.exp(shifted / tau - 1.0))) + np.log(g_max)
    return np.exp(log_sum)


def _dct_basis(K):
    """Generate DCT-II basis vectors for K components."""
    n = N
    x = np.arange(n, dtype=float)
    basis = np.zeros((K, n))
    for k in range(K):
        basis[k] = np.cos(np.pi * (2 * x + 1) * k / (2 * n))
    return basis


def _dct_to_h(coeffs, basis, enforce_nonneg=True):
    """Convert DCT coefficients to step heights."""
    h = coeffs @ basis
    if enforce_nonneg:
        h = np.maximum(0.0, h)
    return h


def _h_to_dct_softplus(h, basis):
    """Invert softplus to get coefficients: c_i = log(expm1(h_i)) for h_i > 0, clipped."""
    h = np.maximum(1e-10, h)
    coeffs = np.log(np.expm1(h))
    return coeffs


def _value_neg_smooth(h, tau=0.01, peak_penalty=0.0):
    """Smooth objective with soft-max for norm_inf."""
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(g) / (m + 1)
    norm_inf_smooth = _smooth_max(g, tau)
    if norm_1 <= 0 or norm_inf_smooth <= 0:
        return 0.0
    val = l2 / (norm_1 * norm_inf_smooth)
    if peak_penalty > 0:
        g_sorted = np.sort(g)[::-1]
        g_sum = np.sum(g)
        if g_sum > 0:
            top1 = g_sorted[0] / g_sum
            val = val - peak_penalty * (top1 - 1.0 / len(g)) ** 2
    return -val


def _value_neg_dct(coeffs, basis, tau=0.01, peak_penalty=0.0):
    """Smooth objective in DCT coefficient space using log-transform for non-negativity."""
    # Use softplus-like transform: h_i = softplus(c_i) to ensure h_i > 0
    # softplus(x) = log(1 + exp(x)), derivative = sigmoid(x)
    h = np.log1p(np.exp(coeffs))
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(g) / (m + 1)
    norm_inf_smooth = _smooth_max(g, tau)
    if norm_1 <= 0 or norm_inf_smooth <= 0:
        return 0.0
    val = l2 / (norm_1 * norm_inf_smooth)
    if peak_penalty > 0:
        g_sorted = np.sort(g)[::-1]
        g_sum = np.sum(g)
        if g_sum > 0:
            top1 = g_sorted[0] / g_sum
            val = val - peak_penalty * (top1 - 1.0 / len(g)) ** 2
    return -val


def _value_neg_dct_exact(coeffs, basis):
    """Exact objective in DCT coefficient space."""
    h = _dct_to_h(coeffs, basis)
    return _value_neg(h)


def _h_to_dct(h, basis):
    """Project h onto DCT basis to get coefficients."""
    # Use least squares for robustness
    try:
        coeffs, _, _, _ = np.linalg.lstsq(basis.T, h, rcond=None)
    except:
        coeffs = np.zeros(basis.shape[0])
    return coeffs


def _generate_structured_starts():
    """Generate diverse structured starting points."""
    x_grid = np.arange(N, dtype=float)
    starts = []

    # Uniform
    starts.append(np.ones(N))

    # Hann window
    hann = 0.5 * (1 + np.cos(2 * np.pi * x_grid / (N - 1)))
    starts.append(hann)

    # Cosine windows
    for width in [0.5, 0.7, 0.8, 0.9, 1.0]:
        arg = np.pi * width * (x_grid - (N - 1) / 2) / ((N - 1) / 2)
        h = np.cos(arg)
        starts.append(np.maximum(0.0, h))

    # Gaussian bumps
    for center in [10, 15, 20, 25, 30, 35, 40]:
        for width in [3, 5, 8, 12, 15, 20]:
            h = np.exp(-0.5 * ((x_grid - center) / width) ** 2)
            starts.append(h)

    # Two Gaussians
    for c1 in [8, 12, 15, 18]:
        for c2 in [32, 35, 38, 42]:
            for w in [3, 5, 8, 12]:
                h = np.exp(-0.5 * ((x_grid - c1) / w) ** 2) + \
                    np.exp(-0.5 * ((x_grid - c2) / w) ** 2)
                starts.append(h)

    # Power-law bumps
    for center in [15, 25, 35]:
        for r in [8, 12, 16, 20]:
            for p in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
                base = np.maximum(0.0, 1.0 - np.abs(x_grid - center) / r)
                h = base ** p
                starts.append(h)

    # Semicircle
    for center in [15, 25, 35]:
        for radius in [8, 12, 16, 20]:
            d = (x_grid - center) / radius
            h = np.sqrt(np.maximum(0.0, 1.0 - d ** 2))
            starts.append(h)

    # Triangle
    for center in [15, 25, 35]:
        for hw in [5, 10, 15, 20]:
            h = np.maximum(0.0, 1.0 - np.abs(x_grid - center) / hw)
            starts.append(h)

    # Sparse spikes
    for k in [2, 3, 5]:
        for _ in range(3):
            idx = RNG.choice(N, size=k, replace=False)
            h = np.zeros(N)
            h[idx] = 1.0
            starts.append(h)

    # Comb / periodic
    for period in [4, 5, 7, 10]:
        h = (np.arange(N) % period < period // 2).astype(float)
        starts.append(h)

    return starts


def _coordinate_descent(h0, max_iter=2000):
    h = np.maximum(0.0, h0.copy())
    for _ in range(max_iter):
        if time.time() > _DEADLINE:
            break
        improved = False
        for i in range(N):
            current_val = h[i]
            candidates = [
                current_val * 0.5,
                current_val * 0.75,
                current_val * 1.25,
                current_val * 1.5,
                current_val + 0.05,
                current_val - 0.05,
                0.0,
                1.0,
            ]
            if current_val > 0:
                candidates.extend([current_val * (1 + RNG.normal(0, 0.1)) for _ in range(3)])
            candidates = [max(0.0, c) for c in candidates]

            best_cand = current_val
            best_cand_val = compute_value(h)

            for c in candidates:
                h[i] = c
                v = compute_value(h)
                if v > best_cand_val:
                    best_cand_val = v
                    best_cand = c

            h[i] = best_cand
            if best_cand != current_val:
                improved = True

        if not improved:
            break
    return h


def _simulated_annealing(h0, n_iter=5000):
    h = np.maximum(0.0, h0.copy())
    current_val = compute_value(h)
    best_local_h = h.copy()
    best_local_val = current_val
    T = 0.5
    cooling = 0.9995

    for _ in range(n_iter):
        if time.time() > _DEADLINE:
            break
        i = RNG.randint(N)
        delta = RNG.normal(0, 0.1)
        old_val = h[i]
        new_val = max(0.0, old_val + delta)
        h[i] = new_val

        new_val_score = compute_value(h)
        delta_val = new_val_score - current_val

        if delta_val > 0 or RNG.random() < np.exp(delta_val / max(T, 1e-10)):
            current_val = new_val_score
            if current_val > best_local_val:
                best_local_val = current_val
                best_local_h = h.copy()
        else:
            h[i] = old_val

        T *= cooling

    return best_local_h


def _dct_optimize(h0, K=8, max_iter=300):
    """Optimize in DCT coefficient space with low-rank constraint using softplus transform."""
    basis = _dct_basis(K)
    # Project h0 onto DCT basis
    coeffs_proj = _h_to_dct(h0, basis)
    # Convert to softplus parameterization
    h_init = np.maximum(1e-10, coeffs_proj @ basis)
    c_init = _h_to_dct_softplus(h_init, basis)

    def obj(c):
        return _value_neg_dct(c, basis, tau=0.01)

    try:
        res = opt.minimize(obj, c_init, method='L-BFGS-B',
                           options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10})
        if res.x is not None:
            h_opt = np.log1p(np.exp(res.x))  # softplus inverse
            return h_opt
    except Exception:
        pass

    return h0


def construct():
    global _DEADLINE
    start_time = time.time()
    _DEADLINE = start_time + (TIME_LIMIT - 25.0)  # 25s safety margin

    best_val = 0.0
    best_h = np.ones(N)

    def track(h):
        nonlocal best_val, best_h
        h = np.maximum(0.0, h)
        v = compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()

    # ===== Phase 1: Generate and evaluate diverse structured starts =====
    starts = _generate_structured_starts()
    for s in starts:
        s = np.maximum(0.0, s)
        if np.sum(s) > 0:
            track(s)

    # ===== Phase 2: DCT-basis optimization from top starts =====
    scored = [(compute_value(np.maximum(0.0, s)), s) for s in starts]
    scored.sort(key=lambda t: -t[0])
    top_starts = [s for _, s in scored[:15]]

    for K in [6, 8, 10, 12]:
        if time.time() > _DEADLINE:
            break
        for s in top_starts:
            if time.time() > _DEADLINE:
                break
            s = np.maximum(0.0, s)
            h_dct = _dct_optimize(s, K=K, max_iter=200)
            track(h_dct)

    # ===== Phase 3: Smooth gradient-based optimization from top starts =====
    for s in top_starts[:10]:
        if time.time() > _DEADLINE:
            break
        s = np.maximum(0.0, s)
        for tau in [0.005, 0.01, 0.02, 0.05]:
            if time.time() > _DEADLINE:
                break
            for pp in [0.0, 0.5, 1.0, 2.0]:
                if time.time() > _DEADLINE:
                    break
                try:
                    res = opt.minimize(
                        lambda h: _value_neg_smooth(h, tau, pp),
                        s, method='L-BFGS-B', bounds=[(0, None)] * N,
                        options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-10}
                    )
                    if res.x is not None:
                        track(res.x)
                except Exception:
                    pass

    # ===== Phase 4: Coordinate descent with peak-aware updates =====
    for s in top_starts[:8]:
        if time.time() > _DEADLINE:
            break
        s = np.maximum(0.0, s)
        h_cd = _coordinate_descent(s, max_iter=800)
        track(h_cd)

    for _ in range(6):
        if time.time() > _DEADLINE:
            break
        pert = best_h + RNG.normal(0, 0.1, N)
        pert = np.maximum(0.0, pert)
        h_cd = _coordinate_descent(pert, max_iter=400)
        track(h_cd)

    # ===== Phase 5: Simulated annealing for global exploration =====
    for s in top_starts[:5]:
        if time.time() > _DEADLINE:
            break
        h_sa = _simulated_annealing(s, n_iter=4000)
        track(h_sa)

    # ===== Phase 6: Nelder-Mead polish on exact objective =====
    def _nm_optimize(h0):
        if time.time() > _DEADLINE:
            return
        h0 = np.maximum(0.0, h0)

        def obj(h):
            h = np.maximum(0.0, h)
            return _value_neg(h)

        try:
            res = opt.minimize(obj, h0, method='Nelder-Mead',
                               options={'maxiter': 600, 'xatol': 1e-8,
                                        'fatol': 1e-10, 'adaptive': True})
            if res.x is not None:
                track(res.x)
        except Exception:
            pass

    x_grid = np.arange(N, dtype=float)
    nm_starts = [
        np.ones(N),
        np.exp(-0.5 * ((x_grid - 25) / 8) ** 2),
        np.exp(-0.5 * ((x_grid - 15) / 5) ** 2) + np.exp(-0.5 * ((x_grid - 35) / 5) ** 2),
        best_h.copy(),
    ]
    for s in nm_starts:
        if time.time() > _DEADLINE:
            break
        _nm_optimize(s)

    # ===== Phase 7: Fine-tune with very small perturbations =====
    for _ in range(30):
        if time.time() > _DEADLINE:
            break
        for scale in [0.01, 0.05, 0.1]:
            if time.time() > _DEADLINE:
                break
            pert = best_h + RNG.normal(0, scale, N)
            pert = np.maximum(0.0, pert)
            v = compute_value(pert)
            if v > best_val:
                best_val = v
                best_h = pert.copy()
                # Local optimize from this improvement
                try:
                    res = opt.minimize(
                        _value_neg, best_h, method='L-BFGS-B', bounds=[(0, None)] * N,
                        options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
                    )
                    if res.x is not None:
                        track(res.x)
                except Exception:
                    pass

    # ===== Phase 8: Final DCT polish =====
    if time.time() < _DEADLINE:
        for K in [10, 12, 15]:
            if time.time() > _DEADLINE:
                break
            h_dct = _dct_optimize(best_h, K=K, max_iter=150)
            track(h_dct)

    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    return best_h.tolist()


if __name__ == "__main__":
    h = construct()
    print("Score:", compute_value(h))
