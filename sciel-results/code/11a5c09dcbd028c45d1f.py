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
    1. Precompute projection indices.
    2. Use a smooth surrogate for the max in M (log-sum-exp) to enable
       efficient gradient-based optimization with L-BFGS-B.
    3. Anneal the temperature of the soft-max to converge to the true max.
    4. Multiple restarts from structured and random initializations.
    """
    n = 10
    deadline = time.time() + 590.0  # Safety margin (limit is 630s)
    
    # Grid coordinates
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
        Returns: D, M, H_m0, H_m1, H_m2
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
        return H_d, M, H_m0, H_m1, H_m2

    def get_p_from_logits(logits):
        """
        Convert logits to probabilities using softmax.
        """
        logits = logits - np.max(logits)
        p = np.exp(logits)
        s = np.sum(p)
        if s < 1e-15:
            return None
        return p / s

    def objective_smooth(logits, temperature):
        """
        Minimize -D / M_smooth where M_smooth is a log-sum-exp of entropies.
        This makes the objective differentiable.
        """
        if time.time() > deadline:
            return 1e9, np.zeros(100)
            
        p_flat = get_p_from_logits(logits)
        if p_flat is None:
            return 1e9, np.zeros(100)
            
        H_d, M, H_m0, H_m1, H_m2 = compute_entropies_from_p(p_flat)
        
        if M < 1e-9:
            return 1e9, np.zeros(100)
            
        # Soft max of H_m0, H_m1, H_m2
        # We want to approximate max(H_m0, H_m1, H_m2)
        # log-sum-exp is typically for sum. For max, we use:
        # max(a, b, c) approx (1/t) * log(exp(t*a) + exp(t*b) + exp(t*c))
        # Here t = 1/temperature. As temperature -> 0, this -> max.
        
        h_vals = np.array([H_m0, H_m1, H_m2])
        # Scale for numerical stability
        max_h = np.max(h_vals)
        if max_h < 1e-15:
            M_smooth = 0.0
        else:
            # t = 1/temperature
            # To avoid overflow, we subtract max_h from h_vals before exp
            # M_smooth = (1/t) * log(sum(exp(t * (h - max_h)))) + max_h
            # But we can just use the standard formula if we are careful.
            # Let's use a large t if temperature is small.
            t = 1.0 / temperature
            # If t is huge, exp(t*h) overflows.
            # M_smooth = max_h + (1/t) * log(sum(exp(t * (h - max_h))))
            shifted = h_vals - max_h
            # shifted <= 0
            exp_shifted = np.exp(t * shifted)
            sum_exp = np.sum(exp_shifted)
            if sum_exp < 1e-300:
                M_smooth = max_h
            else:
                M_smooth = max_h + (1.0 / t) * np.log(sum_exp)
                
        if M_smooth < 1e-9:
            return 1e9, np.zeros(100)
            
        val = -H_d / M_smooth
        return val, p_flat

    def run_annealing(start_params, max_iter=1000, t_start=1.0, t_end=1e-4, decay=0.99):
        """
        Run L-BFGS-B with annealing temperature.
        """
        current_t = t_start
        best_val = 1e9
        best_params = start_params.copy()
        
        for i in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            # Define objective for current temperature
            def obj(logits):
                val, _ = objective_smooth(logits, current_t)
                return val
            
            try:
                res = minimize(
                    obj, 
                    best_params, 
                    method='L-BFGS-B', 
                    options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-6}
                )
                
                if res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x.copy()
                    
            except Exception:
                pass
            
            current_t *= decay
            if current_t < t_end:
                current_t = t_end
                
        return best_val, best_params

    # Initial best
    init_params = np.zeros(100)
    val, _ = objective_smooth(init_params, 1.0)
    best_score = -val
    best_params = init_params.copy()
    
    # Define starts
    rng = np.random.default_rng(42)
    
    starts = [
        np.zeros(100),
    ]
    
    # Random starts
    for _ in range(10):
        if time.time() > deadline - 20:
            break
        starts.append(rng.normal(0, 1.0, 100))
        
    # Structured start: favor high entropy D (x-y)
    # Maximize spread of x-y.
    # x-y ranges from -9 to 9.
    # Uniform is already good for D.
    # Maybe concentrate mass on cells where x-y is large?
    # But M also matters.
    
    # Let's try a few more structured starts
    # Start with high weight on corners
    corner_start = np.zeros(100)
    corner_start[0] += 2.0    # (0,0)
    corner_start[9] += 2.0    # (0,9)
    corner_start[90] += 2.0   # (9,0)
    corner_start[99] += 2.0   # (9,9)
    starts.append(corner_start)
    
    # Run annealing for each start
    for start in starts:
        if time.time() > deadline - 20:
            break
            
        val, params = run_annealing(start)
        if -val > best_score:
            best_score = -val
            best_params = params

    # Final polish with very low temperature (approximating true max)
    if time.time() < deadline - 10:
        def obj_final(logits):
            val, _ = objective_smooth(logits, 1e-6)
            return val
        
        try:
            res = minimize(
                obj_final, 
                best_params, 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun < -best_score:
                best_score = -res.fun
                best_params = res.x
        except Exception:
            pass

    # Convert best parameters back to P
    p_flat = get_p_from_logits(best_params)
    if p_flat is None:
        p_flat = np.ones(100) / 100.0
    else:
        p_flat /= np.sum(p_flat)
    
    P = p_flat.reshape((n, n))
    
    # Final validation
    P = np.clip(P, 0.0, None)
    s = np.sum(P)
    if s > 1e-9:
        P = P / s
    else:
        P = np.ones((n, n)) / (n * n)
        
    return P

# EVOLVE-BLOCK-END
