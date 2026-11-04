import numpy as np
import scipy.optimize
import time

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin
    
    N = 100
    target_sum = N / 2.0
    
    def get_correlations_linear(h):
        g = 1.0 - h
        c = np.correlate(h, g, mode='full')
        # Exclude lag 0 (index N-1)
        c_nonzero = np.concatenate([c[:N-1], c[N:]])
        return c_nonzero
    
    def score_h(h):
        c_all = get_correlations_linear(h)
        return np.max(c_all)
    
    def find_optimal_x(A, B):
        """
        Minimize max_k (A_k * x + B_k) for x in [0, 1].
        A and B are arrays of coefficients.
        """
        def f(x):
            return np.max(A * x + B)
        
        # The function is convex.
        # We can use ternary search or find intersections.
        # Ternary search is simple and robust for small K.
        lo, hi = 0.0, 1.0
        for _ in range(50):
            m1 = lo + (hi - lo) / 3.0
            m2 = hi - (hi - lo) / 3.0
            if f(m1) < f(m2):
                hi = m2
            else:
                lo = m1
        return (lo + hi) / 2.0

    def coordinate_descent(h, max_iters=500):
        h = h.copy()
        # Pre-compute indices for linear correlation?
        # np.correlate(h, g, 'full')
        # To compute A and B, we need to know which terms involve h_i.
        # Instead of deriving analytically, we can use the difference method.
        # C_k(h) is linear in h_i.
        # C_k(h) = C_k(h with h_i=0) + h_i * (C_k(h with h_i=1) - C_k(h with h_i=0))
        # So B_k = C_k(h_i=0)
        # A_k = C_k(h_i=1) - C_k(h_i=0)
        
        for it in range(max_iters):
            if time.time() - start_time > time_limit - 5:
                break
            
            for i in range(N):
                # Save original value
                h_orig = h[i]
                
                # Set h_i = 0
                h[i] = 0.0
                C0 = get_correlations_linear(h)
                
                # Set h_i = 1
                h[i] = 1.0
                C1 = get_correlations_linear(h)
                
                # Restore h_i temporarily to calculate A and B
                # A = C1 - C0
                # B = C0
                A = C1 - C0
                B = C0
                
                # Find optimal x
                x_opt = find_optimal_x(A, B)
                
                # Update h_i
                h[i] = x_opt
            
            # Enforce sum constraint
            current_sum = np.sum(h)
            diff = target_sum - current_sum
            if abs(diff) > 1e-10:
                h += diff / N
                h = np.clip(h, 0, 1)
                
                # Clip might have changed the sum, do one more correction if needed?
                # The coordinate descent will fix it in the next iteration.
                
        return h

    def local_search_swap(h, max_swaps=200):
        """
        Simple swap local search to break symmetry or fix small issues.
        """
        h = h.copy()
        current_score = score_h(h)
        for _ in range(max_swaps):
            if time.time() - start_time > time_limit - 2:
                break
            i, j = np.random.choice(N, 2, replace=False)
            h_new = h.copy()
            h_new[i], h_new[j] = h_new[j], h_new[i]
            # Check if sum is preserved (yes) and bounds (yes)
            new_score = score_h(h_new)
            if new_score < current_score - 1e-10:
                h = h_new
                current_score = new_score
        return h

    # Initialization
    # Try a few different starts
    best_h = None
    best_score = np.inf
    
    rng = np.random.default_rng(42)
    
    # Start 1: Uniform 0.5
    h0 = np.full(N, 0.5)
    
    # Start 2: Random
    h1 = rng.uniform(0.2, 0.8, N)
    h1 = h1 / np.sum(h1) * target_sum
    h1 = np.clip(h1, 0, 1)
    
    # Start 3: Alternating 0.1, 0.9
    h2 = np.zeros(N)
    h2[::2] = 0.1
    h2[1::2] = 0.9
    h2 = h2 / np.sum(h2) * target_sum
    h2 = np.clip(h2, 0, 1)
    
    starts = [h0, h1, h2]
    
    for idx, h_init in enumerate(starts):
        if time.time() - start_time > time_limit - 20:
            break
            
        # Run coordinate descent
        h_opt = coordinate_descent(h_init, max_iters=200)
        
        # Polish with SLSQP
        # Define objective for SLSQP
        def obj_slsqp(h):
            c_all = get_correlations_linear(h)
            # Use LSE for smoothness in SLSQP
            max_c = np.max(c_all)
            if max_c > 100: max_c = 100
            return max_c + np.log(np.sum(np.exp(10 * (c_all - max_c)))) / 10
        
        def cons_slsqp(h):
            return np.sum(h) - target_sum
        
        constraints = [{'type': 'eq', 'fun': cons_slsqp}]
        bounds = [(0, 1) for _ in range(N)]
        
        try:
            res = scipy.optimize.minimize(
                obj_slsqp, 
                h_opt, 
                method='SLSQP', 
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            h_slsqp = res.x
            h_slsqp = np.clip(h_slsqp, 0, 1)
            
            # Correct sum exactly
            for _ in range(50):
                diff = target_sum - np.sum(h_slsqp)
                if abs(diff) < 1e-14: break
                h_slsqp += diff / N
                h_slsqp = np.clip(h_slsqp, 0, 1)
                
            score = score_h(h_slsqp)
            
            if score < best_score:
                best_score = score
                best_h = h_slsqp.copy()
                
        except Exception:
            # If SLSQP fails, use the coordinate descent result
            score = score_h(h_opt)
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()

    if best_h is None:
        best_h = np.full(N, 0.5)
        
    # Final local search
    best_h = local_search_swap(best_h, max_swaps=500)
    
    # Final sum correction
    for _ in range(50):
        diff = target_sum - np.sum(best_h)
        if abs(diff) < 1e-14: break
        best_h += diff / N
        best_h = np.clip(best_h, 0, 1)
        
    return best_h
