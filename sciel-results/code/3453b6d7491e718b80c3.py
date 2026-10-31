"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def get_dists(pts):
        # Returns array of squared distances for i < j
        # pts: (N, 2)
        d_sq = pdist(pts, 'sqeuclidean')
        return d_sq

    def score_raw(pts):
        d_sq = get_dists(pts)
        d_min = np.sqrt(d_sq.min())
        d_max = np.sqrt(d_sq.max())
        if d_min < 1e-12:
            return 1e10
        return (d_max / d_min) ** 2

    def normalize(pts):
        pts = pts - np.mean(pts, axis=0)
        scale = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
        if scale > 1e-9:
            pts = pts / scale
        return pts

    # Strategy:
    # The current program uses a stochastic hill-climbing approach on the raw ratio,
    # which is non-smooth and prone to getting stuck in local minima.
    #
    # New Strategy:
    # 1. Use a smooth, differentiable objective function: f = log(max_dist) - log(min_dist).
    #    Minimizing this is equivalent to minimizing the ratio.
    #    To handle the non-differentiability of min and max, we use a "soft" approximation
    #    or simply rely on the fact that L-BFGS-B can handle the kinks reasonably well if
    #    initialized close to the optimum, OR we can use a log-sum-exp trick for max and
    #    a negative log-sum-exp trick for min (min = -max(-x)).
    #    
    #    However, a simpler and very effective approach for packing problems is to maximize
    #    the minimum distance (Tammes problem variant) while constraining the max distance,
    #    or to minimize the variance of distances.
    #    
    #    Let's try a two-stage approach:
    #    Stage 1: Use Differential Evolution (global, derivative-free) to find a good basin.
    #             DE is robust to non-smoothness and high dimensionality (32 dims here).
    #    Stage 2: Use L-BFGS-B with a smooth objective (sum of squared deviations from
    #             the mean distance, or a soft max/min) to polish the solution.
    #
    #    Objective for DE: (max_d / min_d)^2 is fine, but it's non-smooth.
    #    Let's use a slightly different objective for the global search that is smoother:
    #    f = (max_d^2 / min_d^2) + lambda * (variance of distances)
    #    Actually, just minimizing the ratio directly with DE is often sufficient.
    
    def objective_ratio(x):
        pts = x.reshape(-1, 2)
        # Check for coincident points
        d_sq = get_dists(pts)
        d_min_sq = np.min(d_sq)
        if d_min_sq < 1e-12:
            return 1e10
        d_max_sq = np.max(d_sq)
        ratio = d_max_sq / d_min_sq
        return ratio

    # For L-BFGS-B, we need a smooth objective.
    # Let's use: f = log(max_d) - log(min_d)
    # To make it differentiable, we approximate max and min using log-sum-exp.
    # max(x) ≈ (1/beta) * log(sum(exp(beta*x)))
    # min(x) ≈ -(1/beta) * log(sum(exp(-beta*x)))
    # So f ≈ (1/beta) * (log(sum(exp(beta*d))) + log(sum(exp(-beta*d))))
    # This is smooth for any beta > 0. As beta -> infinity, it approaches the true max-min.
    # We can anneal beta during optimization or fix it at a high value.
    
    def objective_smooth(x, beta):
        pts = x.reshape(-1, 2)
        d_sq = get_dists(pts)
        # Work with distances, not squared, to keep units consistent with ratio?
        # The ratio is (max_d/min_d)^2 = max_d_sq / min_d_sq.
        # Let's stick to squared distances to avoid sqrt in the inner loop for speed,
        # but the log-sum-exp works on the values.
        # If we use d_sq, max_d_sq/min_d_sq = exp(log(max_d_sq) - log(min_d_sq)).
        # So minimizing log(max_d_sq) - log(min_d_sq) is equivalent.
        
        # Approximation of log(max(d_sq))
        # max(d_sq) = (1/beta) * log(sum(exp(beta * d_sq)))
        # log(max(d_sq)) = log(sum(exp(beta * d_sq))) - log(beta)
        # Approximation of log(min(d_sq))
        # min(d_sq) = -(1/beta) * log(sum(exp(-beta * d_sq)))
        # log(min(d_sq)) = log(sum(exp(-beta * d_sq))) - log(beta)
        
        # So log(max) - log(min) = [log(sum(exp(beta*d))) - log(beta)] - [log(sum(exp(-beta*d))) - log(beta)]
        #                         = log(sum(exp(beta*d))) - log(sum(exp(-beta*d)))
        
        # To prevent overflow in exp, we use the standard log-sum-exp trick:
        # log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
        
        max_d_sq = np.max(d_sq)
        min_d_sq = np.min(d_sq)
        
        # If points are too close, return a large penalty
        if min_d_sq < 1e-12:
            return 1e10
            
        # We want to minimize (max/min)^2.
        # log(ratio^2) = 2 * (log(max) - log(min)).
        # Minimizing log(ratio) is equivalent.
        
        # Let's just use the raw ratio for DE, and for L-BFGS-B use a soft version.
        # Actually, L-BFGS-B can struggle with the sharp kinks of min/max.
        # A better smooth proxy for "minimize spread" is the variance of the distances.
        # But we specifically want to minimize the ratio.
        
        # Let's try minimizing: sum( (d_i - mean_d)^2 ) / (min_d * max_d) ? No, that's complicated.
        
        # Let's stick to the log-sum-exp approximation for the ratio.
        # f = log(max_d_sq) - log(min_d_sq)
        
        # For numerical stability, we shift d_sq by min_d_sq before exp?
        # log(sum(exp(beta * d_sq))) = beta * max_d_sq + log(sum(exp(beta * (d_sq - max_d_sq))))
        # log(sum(exp(-beta * d_sq))) = -beta * min_d_sq + log(sum(exp(-beta * (d_sq - min_d_sq))))
        
        # This is getting complex. Let's use a simpler smooth objective:
        # f = (max_d_sq - min_d_sq) / min_d_sq  => max_d_sq/min_d_sq - 1.
        # This is still non-smooth.
        
        # Alternative: Use the "Tammes" objective: maximize min_d_sq, subject to max_d_sq <= 1.
        # This is a standard way to solve packing problems.
        # We can use L-BFGS-B to maximize min_d_sq (i.e., minimize -min_d_sq) with a constraint.
        # But constraints are hard to define for min.
        
        # Let's try a different smooth objective:
        # f = sum_{i<j} (d_ij^2 - d_mean^2)^2 / (d_min^2 * d_max^2)
        # This penalizes variance relative to the range.
        
        # Actually, the most robust method for this specific "ratio" task in literature is often
        # just to use a global optimizer like DE or CMA-ES on the raw ratio, as the landscape
        # is not too rugged for 32 dimensions if the initial population is good.
        # The current program uses a very basic hill climb. DE is much more powerful.
        
        return None # Placeholder

    # Let's implement the DE approach.
    # Bounds for points: [-2, 2] for x and y.
    bounds = [(-2, 2)] * (2 * N)
    
    # Initial best
    best_pts = None
    best_score = 1e10
    
    # 1. Differential Evolution
    # DE is global and handles non-smooth functions well.
    # We will run DE for a fixed number of iterations or time.
    
    def de_objective(x):
        pts = x.reshape(-1, 2)
        d_sq = get_dists(pts)
        d_min_sq = np.min(d_sq)
        if d_min_sq < 1e-12:
            return 1e10
        d_max_sq = np.max(d_sq)
        return d_max_sq / d_min_sq

    # Run DE
    try:
        # maxiter=1000 might be too slow? Let's monitor time.
        # popsize=15 is standard.
        # tol=1e-8
        # We will wrap it in a time check if possible, but DE doesn't easily support time-based stopping.
        # We'll estimate iterations. 32 dims, popsize 15 -> 480 evals/iter.
        # 1000 iters -> 480,000 evals. Might be too slow.
        # Let's try maxiter=200 first. 96,000 evals.
        
        # To be safe, we'll run DE with a moderate number of iterations.
        # If it's too slow, we'll reduce it.
        
        # Let's use a smaller population and fewer iterations to fit in time.
        # popsize=10, maxiter=100 -> 3200 evals/iter * 100 = 320,000 evals.
        # Each eval is O(N^2) = 256 ops. 320k * 256 ~ 80M ops. Should be < 10s in numpy.
        
        result_de = differential_evolution(
            de_objective, 
            bounds, 
            maxiter=100, 
            popsize=10, 
            tol=1e-8, 
            seed=42,
            init='sobol' # Better initial sampling
        )
        
        if result_de.success or result_de.fun < best_score:
            best_pts = result_de.x.reshape(-1, 2)
            best_score = result_de.fun
            # Note: result_de.fun is the ratio (max/min)^2? No, de_objective returns d_max_sq/d_min_sq which is (max/min)^2.
            # So best_score is the raw score.
            
    except Exception as e:
        pass

    # 2. Local Refinement with L-BFGS-B
    # Now that we have a good starting point from DE, we can try to polish it.
    # We'll use a smooth approximation of the objective.
    # f = log(max_d_sq) - log(min_d_sq)
    # We'll use the log-sum-exp trick with a large beta.
    
    def smooth_objective(x, beta=50.0):
        pts = x.reshape(-1, 2)
        d_sq = get_dists(pts)
        if np.min(d_sq) < 1e-12:
            return 1e10
        
        # log(max_d_sq) - log(min_d_sq)
        # Using log-sum-exp:
        # LSE(x) = max(x) + log(sum(exp(x - max(x))))
        # log(max_d_sq) is not exactly LSE, but LSE approximates log(sum(exp(beta*x))) / beta + max(x).
        # For large beta, LSE(x)/beta + max(x) -> max(x).
        # Actually, (1/beta) * log(sum(exp(beta * x))) approximates max(x).
        
        # Let's use:
        # approx_log_max = (1/beta) * np.log(np.sum(np.exp(beta * d_sq)))
        # approx_log_min = (1/beta) * np.log(np.sum(np.exp(-beta * d_sq))) * (-1)  # Wait, min is negative max of negative.
        # min(d) = -max(-d).
        # log(min(d)) = log(-max(-d)). This is tricky if min(d) is negative, but d_sq is positive.
        # So min(d_sq) > 0.
        
        # Let's just use the raw ratio for L-BFGS-B as well, but with a finite difference gradient?
        # No, L-BFGS-B can use analytical gradients or finite differences.
        # Finite differences are slow.
        
        # Let's try to implement the gradient of the log-sum-exp approximation.
        # f = (1/beta) * (log(sum(exp(beta*d))) - log(sum(exp(-beta*d))))
        # df/dd_i = (1/beta) * (beta * exp(beta*d_i)/sum(exp(beta*d)) - (-beta) * exp(-beta*d_i)/sum(exp(-beta*d)))
        #          = exp(beta*d_i)/sum(exp(beta*d)) + exp(-beta*d_i)/sum(exp(-beta*d))
        
        # This is smooth!
        
        # We need the gradient with respect to the point coordinates.
        # d_i = ||p_i - p_j||^2
        # d/dp_k (d_i) = 2 * (p_k - p_l) if k is in the pair, else 0.
        
        # This is getting complicated to implement the full Jacobian.
        # Let's just use L-BFGS-B with finite differences on the raw ratio.
        # It might be slow, but with a good start from DE, it might converge in few steps.
        
        # Alternatively, use COBYLA which doesn't need gradients.
        
        return None

    # Let's use COBYLA for the final polish, as it handles non-smooth functions and doesn't need gradients.
    if best_pts is not None:
        def cobyla_objective(x):
            pts = x.reshape(-1, 2)
            d_sq = get_dists(pts)
            d_min_sq = np.min(d_sq)
            if d_min_sq < 1e-12:
                return 1e10
            d_max_sq = np.max(d_sq)
            return d_max_sq / d_min_sq

        try:
            result_cobyla = minimize(
                cobyla_objective, 
                best_pts.flatten(), 
                method='COBYLA', 
                options={'maxiter': 1000, 'rhobeg': 1e-3}
            )
            
            if result_cobyla.fun < best_score:
                best_pts = result_cobyla.x.reshape(-1, 2)
                best_score = result_cobyla.fun
                
        except Exception as e:
            pass

    # 3. If we still have time, try a few random restarts with DE
    if best_pts is None or time.time() < deadline - 20:
        rng = np.random.default_rng(123)
        for i in range(3):
            if time.time() > deadline - 10:
                break
            
            # Random initial population
            # We can't easily change DE's init, so we'll just run DE again with a different seed
            # and maybe tighter bounds if we have a good solution.
            
            try:
                result_de2 = differential_evolution(
                    de_objective, 
                    bounds, 
                    maxiter=50, 
                    popsize=10, 
                    tol=1e-8, 
                    seed=123 + i,
                    init='sobol'
                )
                
                if result_de2.fun < best_score:
                    best_pts = result_de2.x.reshape(-1, 2)
                    best_score = result_de2.fun
                    
            except Exception as e:
                pass

    # Final Fallback
    if best_pts is None:
        # Generate a simple grid
        side = int(np.ceil(np.sqrt(N)))
        pts = []
        for i in range(side):
            for j in range(side):
                pts.append([i, j])
                if len(pts) == N:
                    break
            if len(pts) == N:
                break
        best_pts = np.array(pts[:N])
        best_score = score_raw(best_pts)

    # Normalize the final result to ensure it's centered and scaled, although the score is scale-invariant.
    # The problem asks for points in 2D Euclidean space. Normalization doesn't change the score.
    # But it's good practice.
    best_pts = normalize(best_pts)

    return best_pts

# EVOLVE-BLOCK-END
