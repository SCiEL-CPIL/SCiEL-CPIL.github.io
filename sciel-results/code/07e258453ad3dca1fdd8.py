"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _overlap_score(h):
    """
    Compute the maximum overlap of h with shifted 1-h.
    Overlap for shift k is sum_i h[i] * (1 - h[(i+k) % N]).
    We want to minimize the maximum over all shifts k.
    """
    N = len(h)
    one_minus_h = 1.0 - h
    max_overlap = 0.0
    for k in range(N):
        # Circular shift
        shifted = np.roll(one_minus_h, k)
        overlap = np.dot(h, shifted)
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Initial guess: sinusoidal wave clipped and normalized
    # h[i] = min(ceiling, base + amplitude * sin(frequency * i + phase))
    # Then normalize so sum is N/2 and values are in [0, 1]
    
    best_h = np.full(N, 0.5)
    best_score = _overlap_score(best_h)
    
    # Parameterize: frequency (cycles over N), phase, amplitude, ceiling
    # We'll use a smooth representation:
    # raw[i] = base + amplitude * sin(2*pi*frequency*i/N + phase)
    # h[i] = min(ceiling, max(0, raw[i]))
    # Then scale to ensure sum = N/2 and values in [0,1]
    
    def params_to_h(params):
        freq, phase, amplitude, ceiling = params
        i = np.arange(N)
        raw = 0.5 + amplitude * np.sin(2 * np.pi * freq * i / N + phase)
        # Clip to [0, ceiling]
        clipped = np.clip(raw, 0.0, ceiling)
        # Normalize so sum is N/2
        s = np.sum(clipped)
        if s < 1e-10:
            return np.full(N, 0.5)
        h = clipped * (N / 2.0) / s
        # Ensure values are in [0, 1]
        h = np.clip(h, 0.0, 1.0)
        # Re-normalize to ensure sum is exactly N/2
        s2 = np.sum(h)
        if s2 < 1e-10:
            return np.full(N, 0.5)
        h = h * (N / 2.0) / s2
        h = np.clip(h, 0.0, 1.0)
        return h
    
    def objective(params):
        h = params_to_h(params)
        return _overlap_score(h)
    
    # Bounds for parameters
    # freq: 0.1 to 10.0 cycles
    # phase: 0 to 2*pi
    # amplitude: 0.0 to 0.5
    # ceiling: 0.1 to 1.0
    bounds = [(0.1, 10.0), (0.0, 2 * np.pi), (0.0, 0.5), (0.1, 1.0)]
    
    # Try differential evolution for global search
    try:
        if time.time() < deadline:
            result = differential_evolution(
                objective, bounds, seed=42, maxiter=100, tol=1e-4,
                polish=False, updating='deferred'
            )
            if result.success or time.time() < deadline:
                h_candidate = params_to_h(result.x)
                score_candidate = _overlap_score(h_candidate)
                if score_candidate < best_score:
                    best_score = score_candidate
                    best_h = h_candidate
    except Exception:
        pass
    
    # Local refinement with Nelder-Mead
    try:
        if time.time() < deadline:
            # Start from the best found so far
            # Reverse-map: find parameters that produce best_h approximately
            # Or just do local search in parameter space from several starts
            for init_params in [
                [1.0, 0.0, 0.3, 0.8],
                [2.0, np.pi/4, 0.4, 0.9],
                [3.0, np.pi/2, 0.5, 1.0],
                [0.5, 0.0, 0.2, 0.7],
            ]:
                if time.time() > deadline - 5.0:
                    break
                res = minimize(
                    objective, init_params, method='Nelder-Mead',
                    bounds=bounds, options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res.success:
                    h_candidate = params_to_h(res.x)
                    score_candidate = _overlap_score(h_candidate)
                    if score_candidate < best_score:
                        best_score = score_candidate
                        best_h = h_candidate
    except Exception:
        pass
    
    # Final check: ensure sum is exactly N/2
    s = np.sum(best_h)
    if abs(s - N/2.0) > 1e-10:
        best_h = best_h * (N/2.0) / s
        best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h
# EVOLVE-BLOCK-END
