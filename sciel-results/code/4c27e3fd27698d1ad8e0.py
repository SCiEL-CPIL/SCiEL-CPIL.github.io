import numpy as np
import scipy.optimize
import time

N = 100

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    def project_to_feasible(h):
        """
        Project h onto the feasible set: 0 <= h_i <= 1, sum(h) = N/2.
        """
        h = np.clip(h, 0.0, 1.0)
        for _ in range(50):
            current_sum = np.sum(h)
            diff = N / 2.0 - current_sum
            if abs(diff) < 1e-12:
                break
            h += diff / N
            h = np.clip(h, 0.0, 1.0)
        
        # Final exact correction
        residual = N / 2.0 - np.sum(h)
        if abs(residual) > 1e-14:
            # Distribute residual among elements not at 0 or 1
            mask = (h > 1e-8) & (h < 1.0 - 1e-8)
            if np.sum(mask) > 0:
                h[mask] += residual / np.sum(mask)
                h = np.clip(h, 0.0, 1.0)
            else:
                # If all are at bounds, we can't satisfy the sum exactly with this projection
                # But for N=100, sum=50, it's always possible to have interior points
                # unless we are very unlucky.
                pass
        return h

    def get_correlations(h):
        """
        Compute linear correlations of h and 1-h for all non-zero shifts.
        """
        g = 1.0 - h
        c = np.correlate(h, g, mode='full')
        # c has length 2N-1. Index N-1 is shift 0.
        # We want shifts k = -(N-1) ... -1 and 1 ... N-1
        c_nonzero = np.concatenate([c[:N-1], c[N:]])
        return c_nonzero

    def score_h(h):
        """Calculate the true max overlap score."""
        c_all = get_correlations(h)
        return np.max(c_all)

    def lse_objective(h, beta):
        """
        LogSumExp of correlations for smooth approximation of max.
        """
        c_all = get_correlations(h)
        max_c = np.max(c_all)
        if max_c > 500:
            max_c = 500.0
        lse = max_c + np.log(np.sum(np.exp(beta * (c_all - max_c)))) / beta
        return lse

    def objective_de(h):
        """
        Objective for Differential Evolution.
        Penalizes deviation from sum constraint and maximizes overlap.
        """
        h_proj = project_to_feasible(h)
        return score_h(h_proj)

    # Initial best guess: Uniform
    best_h = np.full(N, 0.5)
    best_score = score_h(best_h)
    
    # Run Differential Evolution
    # Bounds are [0, 1] for each element
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    try:
        # DE is stochastic, use a seed for reproducibility in testing, but random in prod?
        # The task allows random.
        de_result = scipy.optimize.differential_evolution(
            objective_de, 
            bounds, 
            maxiter=50, 
            popsize=15,
            tol=1e-6,
            seed=42,
            polish=False # We will polish manually
        )
        h_de = de_result.x
        h_de = project_to_feasible(h_de)
        score_de = score_h(h_de)
        
        if score_de < best_score:
            best_score = score_de
            best_h = h_de.copy()
            
    except Exception:
        pass

    # Polish the best result using L-BFGS-B with LSE objective
    if time.time() - start_time < time_limit:
        beta = 50.0
        h0 = best_h
        
        def obj_polish(h):
            h_proj = project_to_feasible(h)
            return lse_objective(h_proj, beta)

        try:
            res = scipy.optimize.minimize(
                obj_polish, 
                h0, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            h_polished = res.x
            h_polished = project_to_feasible(h_polished)
            score_polished = score_h(h_polished)
            
            if score_polished < best_score:
                best_score = score_polished
                best_h = h_polished.copy()
                
        except Exception:
            pass

    return best_h
