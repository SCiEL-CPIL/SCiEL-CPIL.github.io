"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
TARGET_SUM = N / 2.0

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp

def _get_linear_overlaps(h):
    """
    Linear cross-correlation of h with (1-h) for all non-zero shifts.
    C_k = sum_i h_i * (1 - h_{i+k})
    """
    g = 1.0 - h
    c = np.correlate(h, g, mode='full')
    c_neg = c[:N-1]
    c_pos = c[N:]
    return np.concatenate([c_neg, c_pos])

def _max_overlap(h):
    """Return the maximum linear overlap value."""
    overlaps = _get_linear_overlaps(h)
    if len(overlaps) == 0:
        return 0.0
    return float(np.max(overlaps))

def _project_to_feasible(h):
    """
    Project h onto the feasible set: 0 <= h_i <= 1 and sum(h) = N/2.
    Uses bisection on a uniform shift for exact sum.
    """
    h = np.clip(h, 0.0, 1.0).copy()
    target = TARGET_SUM

    def _sum_at_delta(delta):
        return np.sum(np.clip(h + delta, 0.0, 1.0))

    s0 = np.sum(h)
    if abs(s0 - target) < 1e-13:
        return h

    lo, hi = -1.5, 1.5
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        s = _sum_at_delta(mid)
        if s < target:
            lo = mid
        else:
            hi = mid

    delta = 0.5 * (lo + hi)
    h = np.clip(h + delta, 0.0, 1.0)

    # Distribute any tiny residual to interior elements.
    diff = target - np.sum(h)
    if abs(diff) > 1e-14:
        mask = (h > 1e-9) & (h < 1.0 - 1e-9)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
    return h

def _objective_epigraph(x, beta, lam_sum):
    """
    Epigraph objective. x = [h_0..h_{N-1}, t].
    Minimize t + (softmax(overlaps) - t)_+^2 + sum_penalty.
    """
    h = np.clip(x[:N], 0.0, 1.0)
    t = x[N]

    overlaps = _get_linear_overlaps(h)
    if len(overlaps) == 0:
        lse = 0.0
    else:
        max_ov = np.max(overlaps)
        shifted = overlaps - max_ov
        shifted = np.clip(shifted, -50.0 / beta, 0.0)
        lse = max_ov + logsumexp(beta * shifted) / beta

    if lse > t:
        penalty = (lse - t) ** 2
    else:
        penalty = 0.0

    sum_err = np.sum(h) - TARGET_SUM
    sum_penalty = lam_sum * (sum_err ** 2)
    return t + penalty + sum_penalty

def _coordinate_search(h, deadline, max_rounds=300):
    """
    Local search: adjust one coordinate, re-project, keep if max overlap drops.
    Uses adaptive step sizes.
    """
    current_h = h.copy()
    current_score = _max_overlap(current_h)
    
    # Start with larger steps, shrink over time
    base_steps = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    for round_num in range(max_rounds):
        if time.time() > deadline - 3.0:
            break
        improved = False
        # Shrink steps as we progress
        scale = max(0.1, 1.0 - round_num / max_rounds)
        steps = [s * scale for s in base_steps]
        
        for i in range(N):
            for delta in steps:
                for sign in (1, -1):
                    new_h = current_h.copy()
                    new_h[i] += sign * delta
                    new_h = _project_to_feasible(new_h)
                    new_score = _max_overlap(new_h)
                    if new_score < current_score - 1e-13:
                        current_h = new_h
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return current_h, current_score

def _generate_candidates(rng):
    """Generate diverse initial candidates."""
    candidates = []
    t = np.arange(N)
    
    # Uniform
    candidates.append(np.full(N, 0.5))
    
    # Alternating patterns
    for hi, lo in [(0.6, 0.4), (0.8, 0.2), (0.9, 0.1), (0.7, 0.3), (0.95, 0.05)]:
        h_alt = np.full(N, 0.5)
        h_alt[::2] = hi
        h_alt[1::2] = lo
        candidates.append(h_alt)
    
    # Block patterns
    for block_size in [2, 3, 4, 5, 7, 10]:
        for hi, lo in [(0.7, 0.3), (0.8, 0.2), (0.6, 0.4)]:
            h_block = np.zeros(N)
            for i in range(0, N, block_size * 2):
                h_block[i:i+block_size] = hi
                h_block[i+block_size:i+2*block_size] = lo
            candidates.append(h_block)
    
    # Sinusoidal
    for freq in range(1, 15):
        for amp in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
            for phase in [0.0, np.pi / 4, np.pi / 2, 3*np.pi/4]:
                candidates.append(0.5 + amp * np.sin(2 * np.pi * freq * t / N + phase))
    
    # Random
    for _ in range(20):
        candidates.append(rng.uniform(0.0, 1.0, size=N))
    
    return candidates

def construct():
    """
    Construct step heights h in [0,1], sum = N/2, minimizing the maximum
    overlap of h with shifted (1-h). Uses an epigraph LSE surrogate optimized
    by L-BFGS-B with an increasing-beta schedule, followed by coordinate
    local search and an exact-sum correction.
    """
    start_time = time.time()
    deadline = start_time + 118.0

    rng = np.random.default_rng(42)
    
    # --- Generate and evaluate diverse candidates ---
    candidates = _generate_candidates(rng)
    
    best_h = None
    best_score = np.inf
    for h_cand in candidates:
        h_proj = _project_to_feasible(h_cand)
        score = _max_overlap(h_proj)
        if score < best_score:
            best_score = score
            best_h = h_proj
    
    # --- Epigraph optimization with increasing beta ---
    t_init = best_score + 0.1
    bounds = [(0.0, 1.0)] * N + [(0.0, 10.0)]
    
    # More granular beta schedule, especially at high values
    betas = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    
    for beta in betas:
        if time.time() > deadline - 20.0:
            break
        lam_sum = 100.0 + 10.0 * beta
        x_init = np.concatenate([best_h, [t_init]])

        def obj_func(x):
            return _objective_epigraph(x, beta, lam_sum)

        try:
            result = minimize(
                obj_func, x_init, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-13, 'gtol': 1e-11, 'maxls': 40}
            )
            h_new = _project_to_feasible(result.x[:N])
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
                t_init = max(0.0, result.x[N])
        except Exception:
            pass
    
    # --- High-precision polish with very high beta ---
    if time.time() < deadline - 10.0:
        x_init = np.concatenate([best_h, [t_init]])

        def obj_polish(x):
            return _objective_epigraph(x, 20000.0, 10000.0)

        try:
            result = minimize(
                obj_polish, x_init, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 50}
            )
            h_polished = _project_to_feasible(result.x[:N])
            score_polished = _max_overlap(h_polished)
            if score_polished < best_score:
                best_score = score_polished
                best_h = h_polished
        except Exception:
            pass
    
    # --- Coordinate local search on the true (nonsmooth) max objective ---
    if time.time() < deadline - 5.0:
        best_h, best_score = _coordinate_search(best_h, deadline)
    
    # --- Final exact-sum correction ---
    best_h = _project_to_feasible(best_h)
    sum_h = np.sum(best_h)
    diff = TARGET_SUM - sum_h
    if abs(diff) > 1e-12:
        mask = (best_h > 1e-9) & (best_h < 1.0 - 1e-9)
        if np.any(mask):
            best_h[mask] += diff / np.sum(mask)
            best_h = np.clip(best_h, 0.0, 1.0)
        else:
            best_h = best_h * (TARGET_SUM / sum_h)
            best_h = _project_to_feasible(best_h)

    return best_h

# EVOLVE-BLOCK-END
