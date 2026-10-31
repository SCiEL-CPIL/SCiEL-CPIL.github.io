"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

_start_time = time.time()
_DEADLINE = 115  # seconds, safety margin before 150s kill


def _compute_max_overlap(h):
    """Compute max overlap of h with shifted 1-h over all cyclic shifts.
    overlap_k = sum_i h_i (1 - h_{i+k}) = sum(h) - sum_i h_i h_{i+k} = N/2 - circ_corr(h,h)[k]
    So minimizing max overlap = maximizing min circ_corr.
    """
    n = len(h)
    # circular autocorrelation via FFT
    F = np.fft.rfft(h, n=2 * n)
    corr = np.fft.irfft(F * np.conj(F), n=2 * n)[:n]
    # corr[k] = sum_i h_i h_{i+k mod n}
    max_overlap = n / 2.0 - np.min(corr)
    return max_overlap


def _all_overlaps(h):
    n = len(h)
    F = np.fft.rfft(h, n=2 * n)
    corr = np.fft.irfft(F * np.conj(F), n=2 * n)[:n]
    return n / 2.0 - corr


def _objective(params):
    """Soft-max of overlaps (lower is better)."""
    h = np.clip(params, 0.0, 1.0)
    ov = _all_overlaps(h)
    beta = 25.0
    # smooth max
    m = np.max(ov)
    return m + np.log(np.sum(np.exp(beta * (ov - m)))) / beta


def _project_to_constraint(h):
    """Project h to satisfy 0<=h<=1 and sum(h)=N/2 exactly."""
    h = np.clip(h, 0.0, 1.0)
    diff = N / 2.0 - np.sum(h)
    if abs(diff) > 1e-14:
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
        diff2 = N / 2.0 - np.sum(h)
        if abs(diff2) > 1e-14:
            if diff2 > 0:
                idx = np.argmax(1.0 - h)
                add = min(diff2, 1.0 - h[idx])
                h[idx] += add
            else:
                idx = np.argmin(h)
                sub = min(-diff2, h[idx])
                h[idx] -= sub
    return h


def _try_optimize(h_init, max_iter=3000):
    """Use scipy SLSQP with linear equality constraint."""
    A_eq = np.ones((1, N))
    b_eq = np.array([N / 2.0])
    constraints = [LinearConstraint(A_eq, N / 2.0, N / 2.0)]
    bounds = [(0.0, 1.0)] * N
    try:
        result = minimize(
            _objective,
            h_init,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iter, 'ftol': 1e-10}
        )
        h = _project_to_constraint(result.x)
        return h, _compute_max_overlap(h)
    except Exception:
        h = _project_to_constraint(h_init)
        return h, _compute_max_overlap(h)


def construct():
    global _start_time
    best_h = np.full(N, 0.5)
    best_score = _compute_max_overlap(best_h)

    np.random.seed(42)

    # Build a diverse set of initializations
    inits = []
    # Uniform
    inits.append(np.full(N, 0.5))
    # Alternating
    alt = np.where(np.arange(N) % 2 == 0, 1.0, 0.0)
    inits.append(alt)
    # Block
    blk = np.zeros(N)
    blk[:N // 2] = 1.0
    inits.append(blk)
    # Sine patterns
    for freq in [1, 2, 3, 5, 10]:
        s = 0.5 + 0.5 * np.sin(2 * np.pi * freq * np.arange(N) / N)
        inits.append(s)
    # Random inits
    for _ in range(15):
        r = np.random.uniform(0, 1, N)
        inits.append(r)

    # Sort by initial score, try best first
    scored = []
    for h0 in inits:
        h0 = _project_to_constraint(h0)
        s = _compute_max_overlap(h0)
        scored.append((s, h0))
    scored.sort(key=lambda x: x[0])

    # Also keep a pool of current best solutions for local refinement
    solution_pool = [best_h]

    n_iter = 0
    while time.time() - _start_time < _DEADLINE:
        n_iter += 1
        # Pick an init: cycle through sorted list, plus perturbations of best
        if n_iter <= len(scored):
            h0 = scored[n_iter - 1][1].copy()
        else:
            # Perturb a random solution from the pool
            base = solution_pool[np.random.randint(len(solution_pool))]
            h0 = base + np.random.normal(0, 0.1, N)
            h0 = np.clip(h0, 0, 1)

        h0 = _project_to_constraint(h0)

        h, score = _try_optimize(h0, max_iter=2000)

        if score < best_score - 1e-12:
            best_score = score
            best_h = h.copy()
            solution_pool.append(h.copy())

        # Periodic local refinement of the best
        if n_iter % 5 == 0 and time.time() - _start_time < _DEADLINE:
            h_ref = best_h + np.random.normal(0, 0.05, N)
            h_ref = _project_to_constraint(h_ref)
            h2, s2 = _try_optimize(h_ref, max_iter=1500)
            if s2 < best_score - 1e-12:
                best_score = s2
                best_h = h2.copy()
                solution_pool.append(h2.copy())

        # Early stop if we've converged well
        if best_score < 0.25:
            break

    # Final projection to ensure exact sum
    diff = N / 2.0 - np.sum(best_h)
    if abs(diff) > 1e-12:
        best_h = best_h + diff / N
        best_h = np.clip(best_h, 0.0, 1.0)
        diff2 = N / 2.0 - np.sum(best_h)
        if abs(diff2) > 1e-12:
            idx = np.argmin(best_h) if diff2 > 0 else np.argmax(best_h)
            best_h[idx] += diff2

    return best_h
# EVOLVE-BLOCK-END
