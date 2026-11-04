import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp

N = 100
TARGET_SUM = N / 2.0

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s kill

    def get_correlations_linear(h):
        """
        Compute linear cross-correlation of h with (1-h) for all non-zero shifts.
        Returns array of C_k for k != 0.
        """
        g = 1.0 - h
        c = np.correlate(h, g, mode='full')
        # c[N-1] is zero-shift. We exclude it.
        # c[0:N-1] are negative shifts, c[N:2N-1] are positive shifts.
        return np.concatenate([c[:N-1], c[N:]])

    def max_overlap(h):
        """Return the maximum overlap value."""
        c = get_correlations_linear(h)
        if len(c) == 0:
            return 0.0
        return np.max(c)

    def project(h):
        """
        Project h onto feasible set: 0 <= h_i <= 1 and sum(h) = N/2.
        Uses bisection on a Lagrange multiplier for exact projection.
        """
        h = np.clip(h, 0.0, 1.0).copy()
        current = np.sum(h)
        
        if abs(current - TARGET_SUM) < 1e-12:
            return h
        
        # Bisection on lambda: find lambda such that sum(clip(h - lambda, 0, 1)) = TARGET_SUM
        # When lambda < 0, we're adding; when lambda > 0, we're subtracting
        lo, hi = -max(h) - 1, 1 - min(h) + 1
        
        def sum_at_lambda(lam):
            return np.sum(np.clip(h - lam, 0, 1))
        
        for _ in range(60):
            mid = (lo + hi) / 2
            s = sum_at_lambda(mid)
            if s > TARGET_SUM:
                lo = mid
            else:
                hi = mid
        
        lam = (lo + hi) / 2
        h_proj = np.clip(h - lam, 0.0, 1.0)
        
        # Final correction for floating point
        diff = TARGET_SUM - np.sum(h_proj)
        if abs(diff) > 1e-12:
            mask = (h_proj > 1e-8) & (h_proj < 1 - 1e-8)
            if np.any(mask):
                adj = diff / np.sum(mask)
                h_proj[mask] += adj
                h_proj = np.clip(h_proj, 0.0, 1.0)
        
        return h_proj

    def objective_lse(x, beta, lam_sum):
        """
        Objective function using LogSumExp approximation of the max overlap.
        x is the vector of heights.
        """
        h = np.clip(x, 0.0, 1.0)
        overlaps = get_correlations_linear(h)
        
        if len(overlaps) == 0:
            max_ov = 0.0
            lse = 0.0
        else:
            max_ov = np.max(overlaps)
            shifted = overlaps - max_ov
            lse = max_ov + logsumexp(beta * shifted) / beta
        
        # Penalty for sum constraint
        sum_err = np.sum(h) - TARGET_SUM
        sum_penalty = lam_sum * (sum_err ** 2)
        
        return lse + sum_penalty

    def run_optimization(x0, beta, lam_sum, maxiter=200):
        """
        Run L-BFGS-B optimization from x0 with given parameters.
        """
        bounds = [(0.0, 1.0) for _ in range(N)]
        
        def obj_func(x):
            return objective_lse(x, beta, lam_sum)
        
        try:
            result = minimize(
                obj_func, x0, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            return result.x
        except Exception:
            return x0

    def local_search_swap(h, max_iters=2000):
        """
        Perform swap-based local search. Try swapping pairs of values to reduce max overlap.
        Maintains sum constraint by only swapping pairs.
        """
        current_h = h.copy()
        current_score = max_overlap(current_h)
        indices = np.arange(N)
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5.0:
                break
                
            # Try random pair swaps
            improved = False
            for _ in range(50):
                i, j = rng.choice(indices, size=2, replace=False)
                if i == j:
                    continue
                    
                # Swap values at i and j
                new_h = current_h.copy()
                new_h[i], new_h[j] = current_h[j], current_h[i]
                
                # Check if swap is feasible (values still in [0,1])
                if not (0.0 <= new_h[i] <= 1.0 and 0.0 <= new_h[j] <= 1.0):
                    continue
                    
                new_score = max_overlap(new_h)
                if new_score < current_score - 1e-12:
                    current_h = new_h
                    current_score = new_score
                    improved = True
                    break
                    
            if not improved:
                # Try small perturbations instead of swaps
                perturbed = False
                for _ in range(50):
                    idx = rng.choice(indices)
                    delta = rng.uniform(-0.05, 0.05)
                    new_h = current_h.copy()
                    new_h[idx] = np.clip(new_h[idx] + delta, 0.0, 1.0)
                    
                    # Adjust another index to maintain sum
                    sum_diff = TARGET_SUM - np.sum(new_h)
                    if abs(sum_diff) > 1e-10:
                        # Find another index to adjust
                        other_indices = indices[indices != idx]
                        if len(other_indices) > 0:
                            other_idx = rng.choice(other_indices)
                            new_h[other_idx] = np.clip(new_h[other_idx] + sum_diff, 0.0, 1.0)
                            # Re-check sum
                            if abs(np.sum(new_h) - TARGET_SUM) > 1e-8:
                                continue
                    
                    new_score = max_overlap(new_h)
                    if new_score < current_score - 1e-12:
                        current_h = new_h
                        current_score = new_score
                        perturbed = True
                        break
                        
                if not perturbed:
                    break
                    
        return current_h, current_score

    # --- Initialization ---
    rng = np.random.default_rng(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Uniform
    candidates.append(np.full(N, 0.5))
    
    # 2. Alternating binary-like (smoothed)
    h_alt = np.zeros(N)
    h_alt[::2] = 0.6
    h_alt[1::2] = 0.4
    candidates.append(h_alt)
    
    # 3. Stronger alternating patterns
    h_alt2 = np.zeros(N)
    h_alt2[::2] = 0.8
    h_alt2[1::2] = 0.2
    candidates.append(h_alt2)
    
    h_alt3 = np.zeros(N)
    h_alt3[::2] = 0.9
    h_alt3[1::2] = 0.1
    candidates.append(h_alt3)
    
    # 4. Block patterns
    for block_size in [2, 3, 4, 5]:
        h_block = np.zeros(N)
        for i in range(0, N, block_size * 2):
            h_block[i:i+block_size] = 0.7
            h_block[i+block_size:i+2*block_size] = 0.3
        candidates.append(h_block)
    
    # 5. Sinusoidal patterns
    t = np.arange(N)
    for freq in range(1, 10):
        for amp in [0.1, 0.2, 0.3, 0.4]:
            for phase in [0, np.pi/2]:
                trial = 0.5 + amp * np.sin(2*np.pi*freq*t/N + phase)
                candidates.append(trial.copy())
                
    # 6. Random
    for _ in range(20):
        h_rand = rng.uniform(0.0, 1.0, size=N)
        candidates.append(h_rand)
        
    # Evaluate and select best initial candidate
    best_h = None
    best_score = np.inf
    
    for h_cand in candidates:
        h_proj = project(h_cand)
        score = max_overlap(h_proj)
        if score < best_score:
            best_score = score
            best_h = h_proj
            
    # --- Optimization Phases ---
    # We increase beta to sharpen the max approximation.
    # We also increase the sum penalty lambda to enforce constraint more strictly as we converge.
    
    beta_schedule = [5, 10, 20, 50, 100, 200]
    
    current_h = best_h.copy()
    
    for beta in beta_schedule:
        if time.time() > deadline - 15.0:
            break
            
        # Increase lambda as beta increases to keep sum constraint tight
        # Early on, smooth objective is more important. Late on, exact constraint is critical.
        lam_sum = 10.0 * (beta / 5.0)
        
        # Run optimization from current best
        new_h = run_optimization(current_h, beta, lam_sum, maxiter=300)
        
        # Project to ensure strict feasibility for evaluation
        new_h_proj = project(new_h)
        score_new = max_overlap(new_h_proj)
        
        if score_new < best_score:
            best_score = score_new
            best_h = new_h_proj
            current_h = new_h_proj
        else:
            # If it didn't improve, keep current_h but maybe try a random restart?
            # Given time constraints, we stick to the main trajectory to be safe.
            # However, if we are stuck, a small perturbation might help.
            # Let's just keep the previous best_h as the starting point for next beta.
            pass

    # --- Local Search Phase ---
    # Use swap-based local search to polish the gradient-optimized solution
    if time.time() < deadline - 10.0:
        ls_h, ls_score = local_search_swap(best_h, max_iters=3000)
        if ls_score < best_score:
            best_score = ls_score
            best_h = ls_h

    # --- Final Polish ---
    # Use very high beta and high lambda for final precision
    if time.time() < deadline - 5.0:
        beta_final = 500
        lam_final = 1000.0
        new_h = run_optimization(best_h, beta_final, lam_final, maxiter=500)
        new_h_proj = project(new_h)
        score_new = max_overlap(new_h_proj)
        
        if score_new < best_score:
            best_score = score_new
            best_h = new_h_proj

    # Final Validation
    best_h = project(best_h)
    
    # Ensure strict constraints
    # The projection should handle this, but let's double check
    assert np.all(best_h >= 0) and np.all(best_h <= 1), "h out of bounds"
    # Sum constraint might be slightly off due to floating point in projection loop
    # Re-project one more time with tighter tolerance if needed
    diff = TARGET_SUM - np.sum(best_h)
    if abs(diff) > 1e-9:
        # Simple adjustment
        # Find indices not at 0 or 1 to adjust
        mask = (best_h > 1e-6) & (best_h < 1 - 1e-6)
        if np.any(mask):
            adj = diff / np.sum(mask)
            best_h[mask] += adj
            best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h
