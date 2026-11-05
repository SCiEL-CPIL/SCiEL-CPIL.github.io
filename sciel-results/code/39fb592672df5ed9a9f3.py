"""Combined approach: low-dimensional Fourier-basis smooth optimization + exact active-set refinement."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 148.0  # Safety margin before 150s kill

# ---------------------------------------------------------------------------
# Precompute Fourier basis
# ---------------------------------------------------------------------------
_n = np.arange(N)
_K = 30  # number of positive-frequency Fourier modes
_dim = 2 * _K + 1  # DC + K cos + K sin

_F_cos = np.cos(2.0 * np.pi * np.outer(np.arange(1, _K + 1), _n) / N)  # (K, N)
_F_sin = np.sin(2.0 * np.pi * np.outer(np.arange(1, _K + 1), _n) / N)  # (K, N)


def _reconstruct(x):
    """Reconstruct h from Fourier coefficients x = [a0, a1..aK, b1..bK]."""
    h = np.full(N, x[0])
    h += _F_cos.T @ x[1:1 + _K]
    h += _F_sin.T @ x[1 + _K:]
    return h


def _eval_value(h):
    """Compute the objective value (lower is better)."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(~np.isfinite(h)):
        return np.inf
    sh = float(np.sum(h))
    if abs(sh) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    max_conv = float(np.max(conv))
    val = abs(2.0 * N * max_conv / (sh * sh))
    return val


# ---------------------------------------------------------------------------
# Smooth surrogate objective in Fourier-coefficient space
# ---------------------------------------------------------------------------
def _smooth_objective(x, beta=60.0):
    """Smooth surrogate for |2N * max(conv) / sum(h)^2| via log-sum-exp."""
    h = _reconstruct(x)
    s = float(np.sum(h))
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h)
    max_c = float(np.max(conv))
    shifted = beta * (conv - max_c)
    # Numerically stable log-sum-exp
    m = np.max(shifted)
    log_sum_exp = max_c + (m + np.log(np.sum(np.exp(shifted - m)))) / beta
    val = abs(2.0 * N * log_sum_exp / (s * s))
    if not np.isfinite(val):
        return 1e10
    return val


def _smooth_grad(x, beta=60.0):
    """Gradient of the smooth surrogate w.r.t. Fourier coefficients."""
    h = _reconstruct(x)
    s = float(np.sum(h))
    if abs(s) < 1e-10:
        return np.zeros_like(x)

    conv = np.convolve(h, h)
    max_c = float(np.max(conv))
    shifted = beta * (conv - max_c)
    m = np.max(shifted)
    exp_shifted = np.exp(shifted - m)
    wsum = float(np.sum(exp_shifted))
    if wsum < 1e-300:
        return np.zeros_like(x)
    weights = exp_shifted / wsum

    # Gradient of log-sum-exp (soft max) w.r.t. h:
    # grad_soft_max[k] = 2 * sum_i weights[i] * h[i-k]  (for valid i-k)
    grad_soft_max = np.zeros(N)
    for k in range(N):
        grad_soft_max[k] = 2.0 * float(np.dot(weights[k:k + N], h))

    lse = max_c + (m + np.log(wsum)) / beta
    f = 2.0 * N * lse / (s * s)

    # Gradient of f w.r.t. h
    df_dh = (2.0 * N / (s * s)) * grad_soft_max - (4.0 * N * lse / (s ** 3)) * np.ones(N)

    # Chain rule to Fourier coefficients
    if f >= 0:
        grad_h = df_dh
    else:
        grad_h = -df_dh

    grad_x = np.zeros_like(x)
    grad_x[0] = float(np.sum(grad_h))
    grad_x[1:1 + _K] = _F_cos @ grad_h
    grad_x[1 + _K:] = _F_sin @ grad_h

    return grad_x


# ---------------------------------------------------------------------------
# Exact active-set refinement in full h-space
# ---------------------------------------------------------------------------
def _direct_refine(h0, max_iters=200, tol=1e-10):
    """Direct refinement using active-set method on the max autocorrelation lag."""
    h = h0.copy().astype(np.float64)
    s = float(np.sum(h))
    if abs(s) < 1e-10:
        return h

    for it in range(max_iters):
        conv = np.convolve(h, h)
        k_star = int(np.argmax(conv))

        # Build matrix for this lag: C[i,j] = 1 iff i+j == k_star
        C = np.zeros((N, N))
        for i in range(N):
            j = k_star - i
            if 0 <= j < N:
                C[i, j] = 1.0

        qh = float(h @ C @ h)
        s = float(np.sum(h))
        if abs(s) < 1e-10:
            break

        f = 2.0 * N * qh / (s * s)
        grad = 2.0 * N * (2.0 * C @ h) / (s * s) - 4.0 * N * qh / (s * s * s) * np.ones(N)

        if f < 0:
            grad = -grad

        gnorm = float(np.linalg.norm(grad))
        if gnorm < tol:
            break

        # Backtracking line search
        alpha = 1.0
        f_current = abs(f)
        improved = False
        for ls_iter in range(30):
            h_new = h - alpha * grad
            s_new = float(np.sum(h_new))
            if abs(s_new) < 1e-10:
                alpha *= 0.5
                continue
            conv_new = np.convolve(h_new, h_new)
            k_new = int(np.argmax(conv_new))
            C_new = np.zeros((N, N))
            for i in range(N):
                j = k_new - i
                if 0 <= j < N:
                    C_new[i, j] = 1.0
            qh_new = float(h_new @ C_new @ h_new)
            f_new = abs(2.0 * N * qh_new / (s_new * s_new))

            if f_new < f_current - 1e-12 * alpha * gnorm * gnorm:
                h = h_new
                improved = True
                break
            alpha *= 0.5
        if not improved:
            break

    return h


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def construct():
    t0 = time.time()
    best_h = np.ones(N)
    best_val = _eval_value(best_h)  # 2.0

    def elapsed():
        return time.time() - t0

    rng = np.random.RandomState(42)

    # --- Generate diverse initial points in Fourier coefficient space ---
    initial_points = []

    # Seed: all ones => a0 = 1, rest = 0
    x_seed = np.zeros(_dim)
    x_seed[0] = 1.0
    initial_points.append(x_seed)

    # Random Fourier coefficients (low-frequency dominated)
    for _ in range(20):
        x_rand = np.zeros(_dim)
        x_rand[0] = rng.uniform(0.5, 2.0)  # positive DC bias
        for k in range(1, _K + 1):
            decay = 1.0 / k ** 0.7  # low-frequency dominance
            x_rand[k] = rng.uniform(-1, 1) * decay
            x_rand[_K + k] = rng.uniform(-1, 1) * decay
        initial_points.append(x_rand)

    # Pure single-frequency starts
    for freq in [1, 2, 3, 4, 5, 7, 10, 15, 20, 25]:
        if freq <= _K:
            for phase in [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]:
                x_single = np.zeros(_dim)
                x_single[0] = 1.0  # DC offset to keep sum positive
                amp = 0.5
                x_single[freq] = amp * np.cos(phase)
                x_single[_K + freq] = amp * np.sin(phase)
                initial_points.append(x_single)

    # Multi-frequency random synthesis
    for n_comp in [2, 3, 5, 8]:
        for _ in range(8):
            freqs = rng.choice(np.arange(1, _K + 1), size=n_comp, replace=False)
            x_multi = np.zeros(_dim)
            x_multi[0] = rng.uniform(0.5, 1.5)
            for f in freqs:
                amp = rng.uniform(0.3, 1.0) / f ** 0.5
                phase = rng.uniform(0, 2 * np.pi)
                x_multi[f] += amp * np.cos(phase)
                x_multi[_K + f] += amp * np.sin(phase)
            initial_points.append(x_multi)

    # --- Evaluate and sort initial points ---
    init_values = []
    for x_init in initial_points:
        h_init = _reconstruct(x_init)
        val_init = _eval_value(h_init)
        init_values.append((val_init, x_init))
    init_values.sort(key=lambda x: x[0])

    # Take top candidates for optimization
    top_inits = [x for _, x in init_values[:25]]

    # --- Phase 1: Smooth surrogate optimization in Fourier space ---
    for x_init in top_inits:
        if elapsed() > DEADLINE - 45:
            break

        try:
            result = minimize(
                lambda x, b=60.0: _smooth_objective(x, b),
                x_init,
                jac=lambda x, b=60.0: _smooth_grad(x, b),
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 400, 'gtol': 1e-6}
            )

            if elapsed() > DEADLINE - 40:
                break

            h_candidate = _reconstruct(result.x)
            v = _eval_value(h_candidate)
            if v < best_val and np.all(np.isfinite(h_candidate)) and abs(np.sum(h_candidate)) > 1e-10:
                best_val = v
                best_h = h_candidate.copy()
        except Exception:
            pass

    # --- Phase 2: Direct active-set refinement from best and top candidates ---
    if elapsed() < DEADLINE - 25:
        try:
            h_refined = _direct_refine(best_h.copy(), max_iters=300, tol=1e-12)
            v = _eval_value(h_refined)
            if v < best_val and np.all(np.isfinite(h_refined)) and abs(np.sum(h_refined)) > 1e-10:
                best_val = v
                best_h = h_refined.copy()
        except Exception:
            pass

    # Refine from top initial points (reconstruct h first)
    for x_init in top_inits[:12]:
        if elapsed() > DEADLINE - 20:
            break
        try:
            h_init = _reconstruct(x_init)
            h_refined = _direct_refine(h_init.copy(), max_iters=200, tol=1e-11)
            v = _eval_value(h_refined)
            if v < best_val and np.all(np.isfinite(h_refined)) and abs(np.sum(h_refined)) > 1e-10:
                best_val = v
                best_h = h_refined.copy()
        except Exception:
            pass

    # --- Phase 3: Final high-precision polish ---
    if elapsed() < DEADLINE - 8:
        try:
            h_polished = _direct_refine(best_h.copy(), max_iters=500, tol=1e-14)
            v = _eval_value(h_polished)
            if v < best_val and np.all(np.isfinite(h_polished)) and abs(np.sum(h_polished)) > 1e-10:
                best_val = v
                best_h = h_polished.copy()
        except Exception:
            pass

    # --- Phase 4: Local perturbation search around best ---
    if elapsed() < DEADLINE - 3:
        try:
            for _ in range(40):
                if elapsed() > DEADLINE - 2:
                    break
                noise = rng.normal(0, 0.01, N)
                h_perturbed = best_h + noise
                s_p = float(np.sum(h_perturbed))
                if abs(s_p) < 1e-10:
                    continue
                h_refined = _direct_refine(h_perturbed.copy(), max_iters=100, tol=1e-12)
                v = _eval_value(h_refined)
                if v < best_val and np.all(np.isfinite(h_refined)) and abs(np.sum(h_refined)) > 1e-10:
                    best_val = v
                    best_h = h_refined.copy()
        except Exception:
            pass

    # Final safety: ensure finite and non-zero sum
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        best_h = np.ones(N)

    return best_h
