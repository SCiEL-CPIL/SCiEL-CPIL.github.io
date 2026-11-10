import numpy as np
import time
from scipy.optimize import minimize

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # Safety margin

    def compute_score(support, weights):
        n = len(support)
        w_sum = np.sum(weights)
        if w_sum <= 1e-9:
            return 0.0
        p = weights / w_sum
        
        # Pairwise sums and weights for Z = X1 + X2
        s_sums = support[:, None] + support[None, :]
        p_prods = p[:, None] * p[None, :]
        
        # Flatten and sort
        s_sums_flat = s_sums.flatten()
        p_prods_flat = p_prods.flatten()
        
        sort_idx = np.argsort(s_sums_flat)
        s_sorted = s_sums_flat[sort_idx]
        p_sorted = p_prods_flat[sort_idx]
        
        # Prefix sums of probabilities for Z = X1 + X2
        p_prefix = np.cumsum(p_sorted)
        
        # Thresholds T_{kl} = 2*s_l - s_k
        T = 2.0 * support[None, :] - support[:, None]
        
        # Count of elements in s_sorted < T[k, l]
        idx = np.searchsorted(s_sorted, T, side='left')
        
        # Zero-padded prefix sum
        p_prefix_pad = np.zeros(n * n + 1)
        p_prefix_pad[1:] = p_prefix
        
        # Prob(Z < T[k,l])
        prob_z_less = p_prefix_pad[idx]
        
        # Total prob = sum_{k, l} p_k * p_l * prob_z_less[k, l]
        total_prob = np.sum(p[:, None] * p[None, :] * prob_z_less)
        
        return total_prob

    def objective_weights(weights, support):
        # Minimize negative score
        # Ensure weights are non-negative and sum to 1
        # scipy Powell doesn't handle constraints well, so we use a transformation or penalty?
        # Better: use SLSQP with constraints.
        # Or, since n is small, we can just use Powell on the simplex directly?
        # Powell is derivative free.
        # Let's use SLSQP for better convergence on the simplex.
        w_sum = np.sum(weights)
        if w_sum <= 1e-9:
            return 0.0
        p = weights / w_sum
        # SLSQP will handle the constraint sum(w)=1, w>=0
        # But we need to pass the raw weights to the objective?
        # No, we define the objective to take weights, and the optimizer handles constraints.
        # However, if the optimizer steps outside, compute_score might fail or be invalid.
        # We clip inside compute_score? No, compute_score assumes valid input.
        # We will use a wrapper that clips and normalizes, but this breaks gradients.
        # Since we are using Powell (derivative free), we can just evaluate on the raw vector
        # and rely on the optimizer to stay near the simplex?
        # Actually, let's use SLSQP with constraints.
        
        # To be safe and fast, let's just use a custom hill-climbing for weights if SLSQP is too slow?
        # n=16, SLSQP should be fast.
        
        # For Powell, we can transform variables to ensure positivity?
        # Let's stick to SLSQP.
        
        # But wait, I'm calling this from within an optimizer loop.
        # Let's just compute the score.
        return -compute_score(support, weights)

    def optimize_weights(support, initial_weights, time_limit):
        n = len(support)
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'ineq', 'fun': lambda w: w}  # w >= 0
        ]
        
        # Bounds
        bounds = [(0.0, 1.0) for _ in range(n)]
        
        # Use SLSQP
        try:
            res = minimize(
                objective_weights, 
                initial_weights, 
                args=(support,), 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-9}
            )
            return res.x
        except Exception:
            return initial_weights

    def refine_support(support, weights, rng, time_limit):
        n = len(support)
        current_support = support.copy()
        current_weights = weights.copy()
        current_score = compute_score(current_support, current_weights)
        
        # SA parameters
        T = 1.0
        T_min = 1e-5
        cooling_rate = 0.99
        
        step_size = 0.1
        
        while time.time() < time_limit:
            if T < T_min:
                break
                
            # Perturb support
            new_support = current_support.copy()
            idx = rng.integers(n)
            new_support[idx] += rng.normal(0, step_size)
            new_support = np.maximum(new_support, 0)
            
            # Optimize weights for the new support
            # Use current weights as init for speed
            new_weights = optimize_weights(new_support, current_weights, time_limit)
            
            new_score = compute_score(new_support, new_weights)
            
            # Acceptance criterion
            if new_score >= current_score:
                current_support = new_support
                current_weights = new_weights
                current_score = new_score
            else:
                delta = new_score - current_score
                # Accept worse with probability exp(delta / T)
                if rng.random() < np.exp(delta / T):
                    current_support = new_support
                    current_weights = new_weights
                    current_score = new_score
            
            T *= cooling_rate
            if rng.random() < 0.05:
                step_size *= 0.95
            else:
                step_size *= 1.01
                
        return current_support, current_weights, current_score

    best_score = -1.0
    best_support = None
    best_weights = None
    
    rng = np.random.default_rng(42)
    
    # Focus on n=16 or 20
    n = 16
    
    # Initial random supports
    num_restarts = 5
    time_per_restart = (deadline - 10.0) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Random initialization
        support_init = np.sort(rng.uniform(0, 2.0, n))
        weights_init = np.ones(n) / n
        
        # First, optimize weights for the initial support
        weights_opt = optimize_weights(support_init, weights_init, time.time() + 10.0)
        
        # Then, refine support using SA
        s, w, score = refine_support(support_init, weights_opt, rng, time.time() + time_per_restart)
        
        if score > best_score:
            best_score = score
            best_support = s.copy()
            best_weights = w.copy()
            
        # Print progress
        # print(f"Restart {r+1}: Score {score:.6f}, Time {time.time()-start_time:.1f}s")

    if best_support is None:
        n = 8
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n) / n
        best_score = compute_score(best_support, best_weights)

    # Final polish: Optimize weights one last time with higher precision
    if time.time() < deadline - 10.0:
        final_weights = optimize_weights(best_support, best_weights, deadline - 5.0)
        final_score = compute_score(best_support, final_weights)
        if final_score > best_score:
            best_score = final_score
            best_weights = final_weights

    # Final cleanup
    best_support = np.maximum(best_support, 0)
    best_weights = np.maximum(best_weights, 0)
    
    # Normalize weights to ensure sum is exactly 1?
    # The evaluator normalizes, but let's be clean.
    w_sum = np.sum(best_weights)
    if w_sum > 1e-9:
        best_weights = best_weights / w_sum

    best_support = best_support.flatten()
    best_weights = best_weights.flatten()
    
    return best_support, best_weights
