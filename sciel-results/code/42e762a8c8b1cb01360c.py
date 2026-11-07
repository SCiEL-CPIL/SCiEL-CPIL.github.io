"""
Strategy: Structural Simulated Annealing with Symmetry Constraints.

Unlike the DCT approach which forces smoothness, this strategy explores the
full 50-dimensional space of non-negative heights using Simulated Annealing.
Key differences from the current program:
1. No DCT parametrization: Direct optimization of h, allowing for sharp peaks
   or non-smooth structures that might yield higher l2/(norm_1*norm_inf).
2. Symmetry enforcement: The problem is invariant under h -> reverse(h).
   We exploit this by optimizing only the first 25 heights and mirroring them,
   effectively halving the dimensionality and search space.
3. Simulated Annealing: Uses a cooling schedule to escape local minima,
   which are prevalent in non-convex objectives involving ratios of norms.
4. Adaptive Steps: Step size adapts based on acceptance rate to balance
   exploration and exploitation.
"""
import time
import numpy as np

N = 50
HALF_N = 25

def _eval_full(h):
    """Exact evaluator for a full vector of length N."""
    h = np.asarray(h, dtype=float)
    h = np.maximum(h, 0.0)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    l2 = (w / 3.0) * np.sum(y[:-1] * y[:-1] + y[:-1] * y[1:] + y[1:] * y[1:])
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _eval_half(h_half):
    """
    Evaluate a half-vector (length 25) by mirroring it to form the full vector.
    This enforces symmetry h[i] == h[N-1-i].
    """
    h_full = np.concatenate([h_half, h_half[::-1]])
    return _eval_full(h_full)

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # 5s safety margin
    
    # Initial best: Seed is 50 equal heights.
    # In half-space, this is 25 equal heights.
    best_h_half = np.ones(HALF_N) / HALF_N  # Scale doesn't matter for ratio, but keep finite
    best_val = _eval_half(best_h_half)
    
    # We will also try a few diverse initial states
    # 1. Seed (uniform)
    # 2. Gaussian-like
    # 3. Triangular
    # 4. Random
    
    rng = np.random.default_rng(42)
    
    candidates = []
    candidates.append(np.ones(HALF_N) / HALF_N)
    
    # Gaussian centered
    x = np.linspace(-1, 1, HALF_N)
    candidates.append(np.exp(-x**2 * 2))
    candidates.append(np.exp(-x**2 * 5))
    
    # Triangular
    candidates.append(1 - np.abs(x))
    
    # Bimodal (two peaks)
    candidates.append(np.exp(-((x-0.5)**2)*10) + np.exp(-((x+0.5)**2)*10))
    
    # Random smooth
    for _ in range(5):
        r = rng.uniform(0.1, 1.0, HALF_N)
        # Smooth it
        r = np.convolve(r, np.ones(5)/5, mode='same')
        candidates.append(r)
    
    # Initialize best from candidates
    for c in candidates:
        c = np.maximum(c, 0)
        v = _eval_half(c)
        if v > best_val:
            best_val = v
            best_h_half = c.copy()
    
    # Simulated Annealing
    current_h = best_h_half.copy()
    current_val = best_val
    
    # Temperature schedule
    T_start = 1e-3
    T_end = 1e-8
    n_steps = 50000  # Will be adjusted based on time
    
    # Adaptive step size
    step_size = 0.05
    accept_count = 0
    total_count = 0
    
    # Pre-compute indices for random selection
    indices = np.arange(HALF_N)
    
    for i in range(n_steps):
        if time.time() > deadline - 5.0:
            break
        
        # Cooling
        progress = i / n_steps
        T = T_start * (T_end / T_start) ** progress
        
        # Generate perturbation
        # Randomly select a few indices to perturb (sparse update for speed)
        n_perturb = rng.integers(1, 4)
        idxs = rng.choice(HALF_N, size=n_perturb, replace=False)
        
        old_h = current_h.copy()
        
        # Perturb
        for idx in idxs:
            # Multiplicative perturbation to maintain non-negativity and scale invariance
            # Log-normal perturbation
            factor = np.exp(rng.normal(0, step_size))
            current_h[idx] *= factor
            # Clamp to prevent overflow/underflow issues, though ratio is scale invariant
            # We don't need to normalize because _eval_half handles it (ratio is scale invariant)
            # Actually, _eval_full uses g=conv(h,h). If h scales by k, g scales by k^2.
            # l2 scales by k^2, norm_1 by k^2, norm_inf by k^2.
            # So value is scale invariant.
        
        new_val = _eval_half(current_h)
        
        # Acceptance criterion
        if new_val > current_val:
            current_val = new_val
            accept_count += 1
            total_count += 1
            # Update best
            if new_val > best_val:
                best_val = new_val
                best_h_half = current_h.copy()
        else:
            total_count += 1
            # Metropolis acceptance
            delta = current_val - new_val
            if delta < 1e-12:
                # Equivalent
                if rng.random() < 0.5:
                    current_val = new_val
                    accept_count += 1
            else:
                # P = exp(-delta / T)
                # Avoid overflow/underflow in exp
                log_p = -delta / T
                if log_p > 0: # Should not happen if new_val < current_val
                    accept_count += 1
                else:
                    p = np.exp(log_p)
                    if rng.random() < p:
                        accept_count += 1
                    else:
                        # Revert
                        current_h = old_h
        
        # Adaptive step size adjustment every 1000 steps
        if i % 1000 == 999 and total_count > 0:
            acc_rate = accept_count / total_count
            if acc_rate > 0.8:
                step_size *= 1.5
            elif acc_rate < 0.2:
                step_size *= 0.7
            # Reset counters
            accept_count = 0
            total_count = 0
    
    # Final local polish with coordinate ascent on the full 50-dim vector
    # to break the symmetry constraint if it's suboptimal, or just refine.
    # Let's try to break symmetry: take best symmetric, then optimize full 50.
    
    best_h_full = np.concatenate([best_h_half, best_h_half[::-1]])
    
    if time.time() < deadline - 10.0:
        # Simple gradient-free local search on full 50 dims
        # Using small random perturbations
        current_h_full = best_h_full.copy()
        current_val_full = best_val
        step_sz = 0.01
        
        for _ in range(5000):
            if time.time() > deadline - 2.0:
                break
            
            idx = rng.integers(0, N)
            old_val = current_h_full[idx]
            
            # Try a few multipliers
            improved = False
            for mult in [1.1, 0.9, 1.2, 0.8, 1.05, 0.95]:
                cand = current_h_full.copy()
                cand[idx] = old_val * mult
                if cand[idx] < 0:
                    continue
                v = _eval_full(cand)
                if v > current_val_full:
                    current_h_full = cand
                    current_val_full = v
                    best_val = v
                    best_h_full = cand
                    improved = True
                    break
            
            if not improved:
                # Try a random jump if stuck
                if rng.random() < 0.1:
                    idx2 = rng.integers(0, N)
                    current_h_full[idx2] = max(0, current_h_full[idx2] + rng.normal(0, 0.1))
                    v = _eval_full(current_h_full)
                    if v > current_val_full:
                        current_val_full = v
                        best_val = v
                        best_h_full = current_h_full
    
    return list(best_h_full)
