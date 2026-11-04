import numpy as np
import scipy.optimize
import time

N = 100
TARGET_SUM = N / 2.0

def construct():
    """
    Construct step heights h to minimize the maximum overlap of h with shifted 1-h.
    
    The overlap C_s = sum h_i (1 - h_{i+s}).
    We use a structural change: reformulate as a convex optimization problem
    using the epigraph variable t and LogSumExp smoothing, then solve with
    L-BFGS-B and diverse restarts, followed by a high-precision polish pass.
    """
    start_time = time.time()
    # Evaluator kills at 150s. We use 135s as our internal deadline to be safe.
    deadline = 135.0
    safety_margin = 5.0
    effective_deadline = deadline - safety_margin
    
    # 1. Define the objective functions and projection
    
    def get_correlations(h):
        """
        Compute linear correlations of h and 1-h for all non-zero shifts.
        Returns an array of correlation values for shifts k = -(N-1) ... -1, 1 ... N-1.
        """
        g = 1.0 - h
        c = np.correlate(h, g, mode='full')
        # c has length 2N-1. Index N-1 is shift 0.
        # We want all shifts except 0.
        c_nonzero = np.concatenate([c[:N-1], c[N:]])
        return c_nonzero

    def project_to_feasible(h):
        """
        Project h onto the feasible set: 0 <= h_i <= 1, sum(h) = N/2.
        Uses iterative water-filling with clipping.
        """
        h = np.clip(h, 0.0, 1.0)
        for _ in range(50):
            current_sum = np.sum(h)
            diff = TARGET_SUM - current_sum
            if abs(diff) < 1e-14:
                break
            h += diff / N
            h = np.clip(h, 0.0, 1.0)
        # Final exact correction: distribute residual among non-boundary elements
        residual = TARGET_SUM - np.sum(h)
        if abs(residual) > 1e-15:
            interior = (h > 1e-10) & (h < 1.0 - 1e-10)
            if np.sum(interior) > 0:
                h[interior] += residual / np.sum(interior)
                h = np.clip(h, 0.0, 1.0)
        return h

    def lse_objective(h, beta):
        """
        LogSumExp of correlations for smooth approximation of max.
        This is differentiable and convex (since max of linear functions is convex).
        """
        c_all = get_correlations(h)
        max_c = np.max(c_all)
        if max_c > 500:
            max_c = 500.0
        lse = max_c + np.log(np.sum(np.exp(beta * (c_all - max_c)))) / beta
        return lse

    def score_h(h):
        """Calculate the true max overlap score."""
        c_all = get_correlations(h)
        return np.max(c_all)

    # 2. Optimization routine
    
    def optimize_instance(h0, beta, max_iter, ftol=1e-10, gtol=1e-10):
        """
        Optimize using L-BFGS-B with projection-based constraint handling.
        """
        bounds = [(0, 1) for _ in range(N)]
        
        def obj(h):
            h_proj = project_to_feasible(h)
            return lse_objective(h_proj, beta)

        try:
            res = scipy.optimize.minimize(
                obj, 
                h0, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
            )
            h_opt = res.x
        except Exception:
            h_opt = h0
        
        h_opt = project_to_feasible(h_opt)
        return h_opt

    # 3. Search Strategy
    
    best_h = np.full(N, 0.5)
    best_score = score_h(best_h)  # Baseline: 25.0
    
    rng = np.random.default_rng(42)
    
    # Multi-restart strategy with varying beta and initializations
    restarts = []
    
    # 1. Uniform start (baseline)
    restarts.append((np.full(N, 0.5), 5.0, 500))
    
    # 2-7. Random perturbations
    for i in range(6):
        h_rand = rng.uniform(0.1, 0.9, size=N)
        h_rand = project_to_feasible(h_rand)
        beta = 5.0 + i * 5.0
        restarts.append((h_rand, beta, 500))
        
    # 8-13. Bimodal initializations (encouraging structure)
    for i in range(6):
        h_rand = rng.uniform(0.0, 1.0, size=N)
        # Thresholding to create blocks
        h_rand = np.where(h_rand > 0.5, 0.8, 0.2)
        h_rand = project_to_feasible(h_rand)
        beta = 20.0 + i * 5.0
        restarts.append((h_rand, beta, 500))

    # Run all restarts
    for idx, (h0, beta, max_iter) in enumerate(restarts):
        if time.time() > effective_deadline:
            break
            
        h_opt = optimize_instance(h0, beta, max_iter)
        current_score = score_h(h_opt)
        
        if current_score < best_score:
            best_score = current_score
            best_h = h_opt.copy()
            
    # 4. High-precision polish pass on the best solution
    if time.time() < effective_deadline - 10.0:
        # Use very high beta to sharpen the approximation of max
        polish_beta = 100.0
        h_polished = optimize_instance(best_h, polish_beta, 1000, ftol=1e-12, gtol=1e-12)
        polished_score = score_h(h_polished)
        
        if polished_score < best_score:
            best_score = polished_score
            best_h = h_polished.copy()
            
    # Final constraint verification
    best_h = project_to_feasible(best_h)
    
    return best_h
