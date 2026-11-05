"""
Strategy: Genetic Algorithm over structured sinusoidal+chirp representation.

Instead of optimizing 400 free parameters with gradient methods, we search
a ~20-dimensional structured space:
  h[n] = offset + sum_k amp_k * cos(2*pi*freq_k*n/N + phase_k) + chirp * n^2/N^2

The GA uses tournament selection, uniform crossover, and Gaussian mutation
directly on this parameter vector. This allows effective global search over
the structured family without getting stuck in local optima of the 400-dim
landscape. After GA, a short local polish on the full 400-dim h is applied.
"""

import numpy as np
import time

N = 400
_DEADLINE = 140.0  # wall-clock safety margin

# Parameter bounds for the structured representation
# params = [offset, (amp, freq, phase) x K, chirp]
K_MODES = 4  # number of sinusoidal modes
PARAM_DIM = 1 + 3 * K_MODES + 1  # offset + K*(amp,freq,phase) + chirp = 14

# Bounds: offset in [0.5, 3.0], amp in [-2, 2], freq in [0.5, 20], phase in [0, 2*pi], chirp in [-5, 5]
P_LOW = np.array([0.5] + [-2.0, 0.5, 0.0] * K_MODES + [-5.0])
P_HIGH = np.array([3.0] + [2.0, 20.0, 2.0 * np.pi] * K_MODES + [5.0])


def _params_to_h(params):
    """Convert parameter vector to step heights."""
    n = np.arange(N, dtype=np.float64)
    offset = params[0]
    h = np.full(N, offset, dtype=np.float64)
    idx = 1
    for k in range(K_MODES):
        amp = params[idx]
        freq = params[idx + 1]
        phase = params[idx + 2]
        h += amp * np.cos(2.0 * np.pi * freq * n / N + phase)
        idx += 3
    chirp = params[idx]
    h += chirp * (n / N) ** 2
    return h


def _evaluate(h):
    """Compute the task value."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _evaluate_params(params):
    """Evaluate a parameter vector."""
    h = _params_to_h(params)
    if not np.all(np.isfinite(h)):
        return np.inf
    return _evaluate(h)


def _ga_search(deadline, pop_size=200, n_gen=500, seed=42):
    """
    Genetic algorithm over the structured parameter space.
    Returns best params and best value.
    """
    rng = np.random.default_rng(seed)
    
    # Initialize population
    pop = np.empty((pop_size, PARAM_DIM))
    for i in range(pop_size):
        pop[i] = rng.uniform(P_LOW, P_HIGH)
    
    # Evaluate initial population
    vals = np.array([_evaluate_params(pop[i]) for i in range(pop_size)])
    
    best_idx = np.argmin(vals)
    best_val = vals[best_idx]
    best_params = pop[best_idx].copy()
    
    elite_size = 10
    tournament_k = 5
    crossover_rate = 0.8
    mutation_rate = 0.3
    mutation_sigma = 0.15
    
    gen = 0
    while time.time() < deadline:
        gen += 1
        
        # Sort by fitness
        order = np.argsort(vals)
        pop = pop[order]
        vals = vals[order]
        
        # Track best
        if vals[0] < best_val:
            best_val = vals[0]
            best_params = pop[0].copy()
        
        # Create next generation
        new_pop = np.empty((pop_size, PARAM_DIM))
        
        # Elitism: keep top individuals
        new_pop[:elite_size] = pop[:elite_size]
        
        # Fill rest with offspring
        for i in range(elite_size, pop_size):
            if time.time() > deadline:
                break
            
            # Tournament selection for two parents
            t1 = rng.integers(0, tournament_k, size=tournament_k)
            t2 = rng.integers(0, tournament_k, size=tournament_k)
            p1_idx = t1[np.argmin(vals[t1])]
            p2_idx = t2[np.argmin(vals[t2])]
            p1 = pop[p1_idx]
            p2 = pop[p2_idx]
            
            # Crossover
            if rng.random() < crossover_rate:
                mask = rng.random(PARAM_DIM) < 0.5
                child = np.where(mask, p1, p2)
            else:
                child = p1.copy()
            
            # Mutation
            if rng.random() < mutation_rate:
                mut_mask = rng.random(PARAM_DIM) < 0.5
                noise = rng.normal(0, mutation_sigma, PARAM_DIM) * mut_mask
                child = child + noise
                # Clamp to bounds
                child = np.clip(child, P_LOW, P_HIGH)
            
            new_pop[i] = child
        
        # Evaluate new population (only non-elite)
        for i in range(elite_size, pop_size):
            if time.time() > deadline:
                break
            new_pop[i] = np.clip(new_pop[i], P_LOW, P_HIGH)
            new_vals = _evaluate_params(new_pop[i])
            vals[i] = new_vals
        
        pop = new_pop
        # Re-sort for next generation
        order = np.argsort(vals)
        pop = pop[order]
        vals = vals[order]
    
    return best_params, best_val


def _local_polish(h, deadline):
    """
    Short local refinement on the full 400-dim h using coordinate descent
    with random restarts.
    """
    best_h = h.copy()
    best_val = _evaluate(h)
    rng = np.random.default_rng(7)
    
    # Coordinate descent: try adjusting each element
    n_iters = 0
    while time.time() < deadline and n_iters < 30000:
        n_iters += 1
        # Pick a random subset of coordinates to try
        n_try = rng.integers(5, 20)
        idx = rng.choice(N, size=n_try, replace=False)
        
        improved = False
        for i in idx:
            # Try a few step sizes
            for ds in [0.01, 0.05, 0.1, -0.01, -0.05, -0.1]:
                h_new = best_h.copy()
                h_new[i] += ds
                v_new = _evaluate(h_new)
                if np.isfinite(v_new) and v_new < best_val - 1e-14:
                    best_val = v_new
                    best_h = h_new
                    improved = True
                    break
        if not improved and n_iters % 500 == 0:
            # Random restart from best
            h_restart = best_h + rng.normal(0, 0.02, N)
            v_restart = _evaluate(h_restart)
            if np.isfinite(v_restart) and v_restart < best_val:
                best_val = v_restart
                best_h = h_restart
    
    return best_h, best_val


def construct():
    """
    Construct step heights to minimize the autocorrelation inequality value.
    """
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    # Phase 1: GA search over structured parameter space
    ga_deadline = deadline - 20  # reserve 20s for local polish
    best_params, best_val = _ga_search(ga_deadline, pop_size=150, n_gen=1000, seed=42)
    
    # Convert best params to h
    best_h = _params_to_h(best_params)
    
    # Also try a second GA run with different seed if time permits
    if time.time() < deadline - 25:
        bp2, bv2 = _ga_search(deadline - 25, pop_size=100, n_gen=500, seed=123)
        if bv2 < best_val:
            best_val = bv2
            best_params = bp2
            best_h = _params_to_h(best_params)
    
    # Phase 2: Local polish on full 400-dim h
    if time.time() < deadline - 5:
        best_h, best_val = _local_polish(best_h, deadline - 5)
    
    # Final validation
    final_val = _evaluate(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        # Fallback: try a few structured candidates directly
        n = np.arange(N, dtype=np.float64)
        candidates = []
        # Simple cosine with offset
        for freq in [1.0, 2.0, 3.0, 5.0, 10.0]:
            for offset in [1.0, 1.5, 2.0]:
                h_c = offset + np.cos(2.0 * np.pi * freq * n / N)
                candidates.append(h_c)
        # Chirp
        for c in [1.0, 2.0, -1.0, -2.0]:
            h_c = 1.0 + c * (n / N) ** 2
            candidates.append(h_c)
        
        best_fallback = np.ones(N)
        best_fv = 2.0
        for h_c in candidates:
            v = _evaluate(h_c)
            if np.isfinite(v) and v < best_fv:
                best_fv = v
                best_fallback = h_c
        best_h = best_fallback
    
    return best_h
