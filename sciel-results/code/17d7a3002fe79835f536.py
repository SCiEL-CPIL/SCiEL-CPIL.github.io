# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Maximizes the ratio D/M where:
    D = H(x - y)
    M = max(H(x), H(x + y), H(x + 2y))
    
    Strategy:
    1. Precompute projection indices for x, y, x-y, x+y, x+2y.
    2. Use Simulated Annealing (SA) with geometric steps to explore the 100-dimensional 
       probability simplex. SA is effective for high-dimensional non-convex optimization 
       (maximizing a ratio of concave functions) and can escape local minima better 
       than deterministic local search, often finding better basins than Differential 
       Evolution within a fixed time budget due to lower overhead per step.
    3. Polish the best result found by SA using L-BFGS-B for high precision.
    """
    n = 10
    deadline = time.time() + 590.0  # Safety margin (limit is 630s)
    
    # Grid coordinates
    # P[y, x] -> y is row, x is col
    ys, xs = np.mgrid[0:n, 0:n]
    
    # Flatten for binning
    d_val = (xs - ys).flatten()
    m0_val = xs.flatten()
    m1_val = (xs + ys).flatten()
    m2_val = (xs + 2 * ys).flatten()
    
    # Unique values and inverse mapping
    d_vals, d_inv = np.unique(d_val, return_inverse=True)
    m0_vals, m0_inv = np.unique(m0_val, return_inverse=True)
    m1_vals, m1_inv = np.unique(m1_val, return_inverse=True)
    m2_vals, m2_inv = np.unique(m2_val, return_inverse=True)
    
    num_d = len(d_vals)
    num_m0 = len(m0_vals)
    num_m1 = len(m1_vals)
    num_m2 = len(m2_vals)

    def compute_entropies_from_p(p_flat):
        """
        p_flat: shape (100,) normalized probabilities
        Returns: D, M
        """
        p_d = np.bincount(d_inv, weights=p_flat, minlength=num_d)
        p_m0 = np.bincount(m0_inv, weights=p_flat, minlength=num_m0)
        p_m1 = np.bincount(m1_inv, weights=p_flat, minlength=num_m1)
        p_m2 = np.bincount(m2_inv, weights=p_flat, minlength=num_m2)
        
        def shannon(p_dist):
            p_nonzero = p_dist[p_dist > 1e-15]
            if len(p_nonzero) == 0:
                return 0.0
            return -np.sum(p_nonzero * np.log2(p_nonzero))
        
        H_d = shannon(p_d)
        H_m0 = shannon(p_m0)
        H_m1 = shannon(p_m1)
        H_m2 = shannon(p_m2)
        
        M = max(H_m0, H_m1, H_m2)
        return H_d, M

    def get_score(p_flat):
        """
        Returns the ratio D/M. Higher is better.
        """
        H_d, M = compute_entropies_from_p(p_flat)
        if M < 1e-9:
            return -1e9
        return H_d / M

    # Initial best: Uniform distribution
    p_best = np.ones(100) / 100.0
    best_score = get_score(p_best)
    
    # Simulated Annealing
    rng = np.random.default_rng(42)
    
    # Temperature schedule: Exponential cooling
    # Start high enough to allow large moves, end low enough for convergence
    T_start = 0.5
    T_end = 0.001
    steps = 150000  # Total number of SA steps
    
    # We will run SA for a fixed number of steps, checking time periodically
    # Each step involves generating a candidate and evaluating it.
    # Evaluation is fast (bincount + entropy), so 150k steps should take < 10s.
    
    current_p = p_best.copy()
    current_score = best_score
    
    # Precompute log steps for efficiency? No, just compute on fly.
    # We use a "move" strategy: pick two random indices, move mass from one to the other.
    # This preserves the simplex constraint (sum=1) and non-negativity (if we clip).
    
    for i in range(steps):
        if time.time() > deadline - 60:  # Leave 60s for polishing
            break
            
        # Temperature
        # Linear interpolation between T_start and T_end
        T = T_start + (T_end - T_start) * (i / steps)
        
        # Generate candidate
        # Pick two random indices
        idx1 = rng.integers(0, 100)
        idx2 = rng.integers(0, 100)
        
        if idx1 == idx2:
            continue
            
        # Determine direction of mass transfer
        # Move mass from idx1 to idx2 or vice versa
        # To ensure we stay in the simplex, we move a fraction of the mass.
        # Let's move a random fraction of the mass at idx1 to idx2 (or vice versa).
        
        # To be more effective, we can scale the step size with temperature.
        # Max mass we can move is min(current_p[idx1], current_p[idx2] + something)?
        # Actually, we can move up to current_p[idx1] from idx1 to idx2.
        # Let's pick a random amount to move, scaled by T.
        
        # Heuristic: Step size proportional to T * max(current_p)
        max_p = np.max(current_p)
        step_size = T * max_p * 0.1 # Scale factor
        
        # Randomly decide direction
        if rng.random() > 0.5:
            # Move from idx1 to idx2
            max_move = current_p[idx1]
            move_amt = min(step_size, max_move)
            if move_amt < 1e-15:
                continue
            candidate_p = current_p.copy()
            candidate_p[idx1] -= move_amt
            candidate_p[idx2] += move_amt
        else:
            # Move from idx2 to idx1
            max_move = current_p[idx2]
            move_amt = min(step_size, max_move)
            if move_amt < 1e-15:
                continue
            candidate_p = current_p.copy()
            candidate_p[idx2] -= move_amt
            candidate_p[idx1] += move_amt
        
        # Evaluate candidate
        cand_score = get_score(candidate_p)
        
        # Acceptance criterion
        delta = cand_score - current_score
        if delta > 0 or rng.random() < np.exp(delta / T):
            current_p = candidate_p
            current_score = cand_score
            
            if current_score > best_score:
                best_score = current_score
                p_best = current_p.copy()
        
        # Optional: Occasionally do a larger "jump" to escape deep local minima
        if i % 1000 == 0 and i > 0:
            # Jump: pick a random point in the simplex and blend
            # This helps if we are stuck in a bad region.
            # However, blending with a random point is very destructive.
            # Instead, let's just rely on the temperature schedule.
            pass

    # Phase 2: Local Refinement with L-BFGS-B
    # We need to optimize in log-space or use a constrained optimizer.
    # L-BFGS-B works on unconstrained parameters. We map log-parameters to probabilities via softmax.
    
    def objective(log_p_params):
        """
        log_p_params: shape (100,) log-probabilities (unnormalized)
        Returns: negative ratio -D/M (to be minimized)
        """
        if time.time() > deadline:
            return 1e9
            
        log_p_flat = log_p_params - np.max(log_p_params)
        p_flat = np.exp(log_p_flat)
        sum_p = np.sum(p_flat)
        
        if sum_p < 1e-15:
            return 1e9
            
        p_flat /= sum_p
        
        H_d, M = compute_entropies_from_p(p_flat)
        
        if M < 1e-9:
            return 1e9
            
        return -H_d / M

    def run_polishing(start_params, maxiter=2000):
        try:
            res = minimize(
                objective, 
                start_params, 
                method='L-BFGS-B', 
                options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            return res
        except Exception:
            return None

    # Convert best probability vector to log-probabilities for polishing
    # Avoid log(0) by clipping
    p_safe = np.clip(p_best, 1e-15, 1.0)
    log_p_start = np.log(p_safe)
    
    # Polish the SA result
    if time.time() < deadline - 10:
        res_polish = run_polishing(log_p_start)
        if res_polish is not None:
            # Convert back to check score
            log_p_flat = res_polish.x - np.max(res_polish.x)
            p_flat = np.exp(log_p_flat)
            p_flat /= np.sum(p_flat)
            polish_score = get_score(p_flat)
            
            if polish_score > best_score:
                best_score = polish_score
                p_best = p_flat

    # Additional random restarts with polishing if time permits
    # We try a few different starting points for L-BFGS-B to see if we can find a better local max.
    
    starts = [
        log_p_start, # Best found so far
        np.zeros(100), # Uniform
    ]
    
    # Add some random starts
    for _ in range(3):
        if time.time() > deadline - 10:
            break
        # Random log-probabilities
        rand_log_p = rng.normal(0, 1.0, 100)
        starts.append(rand_log_p)
        
    for start in starts:
        if time.time() > deadline - 10:
            break
            
        res = run_polishing(start, maxiter=1000)
        if res is not None:
            log_p_flat = res.x - np.max(res.x)
            p_flat = np.exp(log_p_flat)
            p_flat /= np.sum(p_flat)
            current_score = get_score(p_flat)
            
            if current_score > best_score:
                best_score = current_score
                p_best = p_flat

    # Final conversion and validation
    P = p_best.reshape((n, n))
    P = np.clip(P, 0.0, None)
    s = np.sum(P)
    if s > 1e-9:
        P = P / s
    else:
        P = np.ones((n, n)) / (n * n)
        
    return P

# EVOLVE-BLOCK-END
