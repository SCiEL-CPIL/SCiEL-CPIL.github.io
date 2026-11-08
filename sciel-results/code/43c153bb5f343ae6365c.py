import numpy as np
import time
import scipy.optimize as opt

N = 50
M = 2 * N - 1  # 99
W = 1.0 / (M + 1)  # 1/100
TIME_LIMIT = 150.0


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


def _smooth_max(g, tau=0.01):
    """Smooth approximation of max(g) using log-sum-exp."""
    g_max = np.max(g)
    if g_max <= 0:
        return 0.0
    shifted = g / g_max
    log_sum = np.log(np.sum(np.exp(shifted / tau - 1.0))) + np.log(g_max)
    return np.exp(log_sum)


def _value_neg_smooth(h, tau=0.01):
    """Smooth version of negative value for gradient optimization."""
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
    return -l2 / (norm_1 * norm_inf_smooth)


def construct():
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - 20.0)  # 20s safety margin
    rng = np.random.RandomState(42)

    best_val = 0.0
    best_h = np.ones(N)

    def track(h):
        nonlocal best_val, best_h
        h = np.maximum(0.0, h)
        v = compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()

    x_grid = np.arange(N, dtype=float)

    # ===== Phase 1: Generate diverse structured starts =====
    starts = []

    # Uniform
    starts.append(np.ones(N))

    # Hann window (raised cosine)
    hann = 0.5 * (1 + np.cos(2 * np.pi * x_grid / (N - 1)))
    starts.append(hann)

    # Various cosine windows
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
            for p in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
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

    # Evaluate all starts
    for s in starts:
        s = np.maximum(0.0, s)
        if np.sum(s) > 0:
            track(s)

    # ===== Phase 2: Smooth gradient-based optimization from top starts =====
    scored = [(compute_value(np.maximum(0.0, s)), s) for s in starts]
    scored.sort(key=lambda t: -t[0])
    top_starts = [s for _, s in scored[:20]]

    bounds = [(0, None)] * N

    for s in top_starts:
        if time.time() > deadline:
            break
        s = np.maximum(0.0, s)
        # Try multiple tau values for smooth max
        for tau in [0.005, 0.01, 0.02, 0.05]:
            if time.time() > deadline:
                break
            try:
                res = opt.minimize(
                    lambda h: _value_neg_smooth(h, tau),
                    s, method='L-BFGS-B', bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if res.x is not None:
                    track(res.x)
            except Exception:
                pass

    # ===== Phase 3: Coordinate descent with peak-aware updates =====
    def _coordinate_descent(h0, max_iter=2000):
        h = np.maximum(0.0, h0.copy())
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            for i in range(N):
                # Try a small range of values for h[i]
                current_val = h[i]
                # Sample new values
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
                    candidates.extend([current_val * (1 + rng.normal(0, 0.1)) for _ in range(3)])
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
        track(h)
        return h

    # Coordinate descent from top starts
    for s in top_starts[:10]:
        if time.time() > deadline:
            break
        s = np.maximum(0.0, s)
        _coordinate_descent(s, max_iter=1000)

    # Also from current best with perturbations
    for _ in range(10):
        if time.time() > deadline:
            break
        pert = best_h + rng.normal(0, 0.1, N)
        pert = np.maximum(0.0, pert)
        _coordinate_descent(pert, max_iter=500)

    # ===== Phase 4: Simulated annealing for global exploration =====
    def _simulated_annealing(h0, n_iter=5000):
        h = np.maximum(0.0, h0.copy())
        current_val = compute_value(h)
        best_local_h = h.copy()
        best_local_val = current_val
        T = 0.5
        cooling = 0.9995

        for _ in range(n_iter):
            if time.time() > deadline:
                break
            # Random perturbation
            i = rng.randint(N)
            delta = rng.normal(0, 0.1)
            old_val = h[i]
            new_val = max(0.0, old_val + delta)
            h[i] = new_val

            new_val_score = compute_value(h)
            delta_val = new_val_score - current_val

            if delta_val > 0 or rng.random() < np.exp(delta_val / max(T, 1e-10)):
                current_val = new_val_score
                if current_val > best_local_val:
                    best_local_val = current_val
                    best_local_h = h.copy()
            else:
                h[i] = old_val

            T *= cooling

        track(best_local_h)
        return best_local_h

    # Simulated annealing from multiple starts
    for s in top_starts[:8]:
        if time.time() > deadline:
            break
        s = np.maximum(0.0, s)
        _simulated_annealing(s, n_iter=3000)

    # From current best
    for _ in range(5):
        if time.time() > deadline:
            break
        _simulated_annealing(best_h.copy(), n_iter=3000)

    # ===== Phase 5: Fine-tune with very small perturbations =====
    for _ in range(100):
        if time.time() > deadline:
            break
        for scale in [0.005, 0.01, 0.02, 0.05]:
            if time.time() > deadline:
                break
            pert = best_h + rng.normal(0, scale, N)
            pert = np.maximum(0.0, pert)
            v = compute_value(pert)
            if v > best_val:
                best_val = v
                best_h = pert.copy()

    # Final smooth optimization from best
    for tau in [0.001, 0.005, 0.01]:
        if time.time() > deadline:
            break
        try:
            res = opt.minimize(
                lambda h: _value_neg_smooth(h, tau),
                best_h, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.x is not None:
                track(res.x)
        except Exception:
            pass

    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    return best_h.tolist()


if __name__ == "__main__":
    h = construct()
    print("Score:", compute_value(h))
