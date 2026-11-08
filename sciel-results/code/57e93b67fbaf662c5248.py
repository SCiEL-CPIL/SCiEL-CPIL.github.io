"""
Structural change: Replace stochastic population-based Differential Evolution
with a deterministic Multi-Start Cyclic Coordinate Descent (CCD) strategy.
CCD optimizes one dimension at a time in a cyclic manner. By using large
initial step sizes and many diverse restarts, it explores the landscape
thoroughly without the overhead of population management and mutation
crossover. This approach is significantly faster per iteration, allowing
for a much higher number of independent optimization runs within the time
budget.
"""
import numpy as np
import time

N = 50
M = 2 * N - 1          # len(g) = 99
W = 1.0 / (M + 1)      # 1/100
TIME_LIMIT = 150.0
RNG = np.random.RandomState(50)

def _build_y(g):
    y = np.zeros(M + 2)
    y[1:-1] = g
    return y

def compute_value(h):
    """Exact evaluator metric for given heights (defensive)."""
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
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _cyclic_coordinate_descent(h, max_iter=200, initial_step=1.0):
    """
    Perform Cyclic Coordinate Descent.
    Optimizes each coordinate in turn, using a golden-section-like or
    simple line search to find the best value for that coordinate.
    """
    h = np.maximum(0.0, h.copy())
    current_val = compute_value(h)
    
    # Adaptive step size
    step = initial_step
    
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            # Simple line search around h[i]
            # Try a few candidate values
            best_h_i = h[i]
            best_val_i = current_val
            
            # We will try values: h[i] - step, h[i], h[i] + step, and maybe 0
            candidates = [max(0.0, h[i] - step), h[i], min(10.0, h[i] + step), 0.0]
            
            # Add some random perturbations if step is small
            if step < 0.1:
                candidates.extend([max(0.0, h[i] + RNG.uniform(-step, step)) for _ in range(5)])
            
            for cand in candidates:
                if cand < 0:
                    cand = 0.0
                old_h_i = h[i]
                h[i] = cand
                val = compute_value(h)
                if val > best_val_i:
                    best_val_i = val
                    best_h_i = cand
            
            if best_h_i != h[i]:
                h[i] = best_h_i
                improved = True
                current_val = best_val_i
            else:
                h[i] = best_h_i # Ensure consistency even if no improvement found in candidates (might be same)
        
        if not improved:
            # If no improvement in a full cycle, reduce step size
            step *= 0.7
            if step < 1e-4:
                break
        else:
            # If improvement found, we can keep step size or reduce slightly
            # to ensure convergence
            step *= 0.95

    return h, compute_value(h)

def _generate_starts():
    x = np.linspace(0, N - 1, N)
    starts = []
    # Uniform
    starts.append(np.ones(N))
    # Gaussian
    starts.append(np.exp(-0.5 * ((x - N / 2) / (N / 4)) ** 2))
    # Two peaks
    p1 = np.exp(-0.5 * ((x - N * 0.25) / (N / 8)) ** 2)
    p2 = np.exp(-0.5 * ((x - N * 0.75) / (N / 8)) ** 2)
    starts.append(p1 + p2)
    # Semicircle
    starts.append(np.sqrt(np.maximum(0.0, 1 - ((2 * x / (N - 1) - 1)) ** 2)))
    # Cosine bump
    starts.append(0.5 * (1 + np.cos(2 * np.pi * x / N)))
    # Triangle
    starts.append(1 - 2 * np.abs(x - (N - 1) / 2) / (N - 1))
    # Random smooth profiles
    for _ in range(100):
        freqs = RNG.randint(1, 8, size=6)
        phases = RNG.uniform(0, 2 * np.pi, size=6)
        amps = RNG.uniform(0.5, 1.0, size=6)
        prof = np.zeros(N)
        for f_, p_, a_ in zip(freqs, phases, amps):
            prof += a_ * np.cos(2 * np.pi * f_ * x / N + p_)
        starts.append(np.maximum(0.1, prof))
        
    # Add some random sparse spikes
    for _ in range(50):
        h = np.zeros(N)
        idx = RNG.randint(0, N, size=RNG.randint(1, 5))
        h[idx] = RNG.uniform(0.5, 2.0)
        starts.append(h)

    return starts

def construct():
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - 10.0)  # 10s safety margin

    best_val = 0.0
    best_h = np.ones(N)

    starts = _generate_starts()
    
    # Evaluate starts
    for s in starts:
        v = compute_value(s)
        if v > best_val:
            best_val = v
            best_h = s.copy()

    # Multi-Start Coordinate Descent
    # We will run CCD from the best start and from many random starts
    # until time runs out
    
    # Sort starts by initial value to prioritize promising regions?
    # Or just iterate through them.
    
    # To maximize coverage, we will run CCD on a subset of the best starts
    # and then on random perturbations of the global best.
    
    # First, run CCD on all initial starts (they are cheap enough)
    # But 155 starts * 50 dims * ~10 evals per dim = 77500 evals.
    # Each eval is ~1ms? 77s. Too slow.
    # We need to be smarter.
    
    # Strategy:
    # 1. Run CCD on the top 20 best initial starts.
    # 2. Run CCD on 100 random perturbations of the current global best.
    
    # Evaluate all starts to get top 20
    scored_starts = [(compute_value(s), s) for s in starts]
    scored_starts.sort(key=lambda x: x[0], reverse=True)
    
    top_starts = [s for v, s in scored_starts[:20]]
    
    for s in top_starts:
        if time.time() > deadline:
            break
        h_opt, v_opt = _cyclic_coordinate_descent(s, max_iter=100, initial_step=2.0)
        if v_opt > best_val:
            best_val = v_opt
            best_h = h_opt.copy()

    # Now, refine the global best with many random restarts
    # This helps escape local minima near the best found so far
    for _ in range(500):
        if time.time() > deadline:
            break
        
        # Create a perturbation of the best solution
        # Mix of small and large perturbations
        if RNG.rand() < 0.5:
            pert = best_h + RNG.normal(0, 0.1, N)
        else:
            pert = best_h * RNG.uniform(0.5, 1.5, N)
            if RNG.rand() < 0.2:
                # Inject a random spike
                idx = RNG.randint(0, N)
                pert[idx] += RNG.uniform(0.5, 2.0)
        
        pert = np.maximum(0.0, pert)
        
        h_opt, v_opt = _cyclic_coordinate_descent(pert, max_iter=50, initial_step=1.0)
        if v_opt > best_val:
            best_val = v_opt
            best_h = h_opt.copy()

    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    return best_h.tolist()

if __name__ == "__main__":
    h = construct()
    print("Score:", compute_value(h))
