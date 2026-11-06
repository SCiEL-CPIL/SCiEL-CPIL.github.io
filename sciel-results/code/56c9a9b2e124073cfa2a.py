import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 24
    k = np.arange(n)
    
    # Grid for evaluation - higher resolution for better local search
    m = 4096
    thetas = np.linspace(0, 2*np.pi, m, endpoint=False)
    powers_grid = np.exp(1j * np.outer(k, thetas))
    
    def get_max_norm_from_vals(vals):
        return np.max(np.abs(vals))

    def get_max_norm(c):
        V = np.dot(c, powers_grid)
        return get_max_norm_from_vals(V)

    # Smooth surrogate: log-sum-exp approximation of max norm
    def smooth_max_norm(vals, beta=200.0):
        """Soft max using log-sum-exp for smoothness."""
        abs_vals = np.abs(vals)
        max_val = np.max(abs_vals)
        if max_val < 1e-30:
            return 0.0
        shifted = abs_vals - max_val
        log_sum_exp = max_val + np.log(np.sum(np.exp(beta * shifted))) / beta
        return log_sum_exp

    def smooth_flip_scores(c, V, beta=200.0):
        """Compute smooth surrogate scores for all single-bit flips."""
        V_flips = V[np.newaxis, :] - 2.0 * c[:, np.newaxis] * powers_grid
        return np.array([smooth_max_norm(V_flips[i], beta) for i in range(n)])

    best_c = np.random.choice([-1, 1], size=n)
    best_V = np.dot(best_c, powers_grid)
    best_score = get_max_norm_from_vals(best_V)
    
    Z_all = powers_grid
    
    def try_double_flip(c, V, current_score, Z_all):
        """Try all double-bit flips to escape local minima."""
        n_local = len(c)
        for i in range(n_local):
            if time.time() > deadline - 15.0:
                return None, None, None
            for j in range(i+1, n_local):
                V_new = V - 2.0 * c[i] * Z_all[i] - 2.0 * c[j] * Z_all[j]
                score_new = np.max(np.abs(V_new))
                if score_new < current_score - 1e-12:
                    c[i] *= -1
                    c[j] *= -1
                    return c, V_new, score_new
        return None, None, None

    def local_search(c, V, current_score):
        """Steepest descent with double flips to escape local minima."""
        improved = True
        double_flip_count = 0
        beta = 200.0
        while improved and time.time() < deadline - 20.0:
            improved = False
            
            # Compute max norm for all single-bit flips
            V_flips = V[np.newaxis, :] - 2.0 * c[:, np.newaxis] * Z_all
            abs_flips = np.abs(V_flips)
            max_norms = np.max(abs_flips, axis=1)
            
            min_idx = np.argmin(max_norms)
            min_norm = max_norms[min_idx]
            
            if min_norm < current_score - 1e-12:
                c[min_idx] *= -1
                V = V_flips[min_idx]
                current_score = min_norm
                improved = True
                double_flip_count = 0
            else:
                # No single-bit flip improves - try double-bit flips
                if double_flip_count < 5:
                    result = try_double_flip(c, V, current_score, Z_all)
                    if result[0] is not None:
                        c, V, current_score = result
                        improved = True
                        double_flip_count += 1
                    else:
                        # Try smooth-guided flip: pick flip that best reduces smooth surrogate
                        smooth_scores = smooth_flip_scores(c, V, beta)
                        best_smooth_idx = np.argmin(smooth_scores)
                        # Verify with exact max norm
                        V_test = V_flips[best_smooth_idx]
                        test_score = np.max(np.abs(V_test))
                        if test_score < current_score - 1e-12:
                            c[best_smooth_idx] *= -1
                            V = V_test
                            current_score = test_score
                            improved = True
                            double_flip_count = 0
                        else:
                            break
                else:
                    break
        
        return c, V, current_score

    # Main search loop with simulated annealing flavor
    temperature = 0.1
    best_score_ever = best_score
    patience = 0
    max_patience = 50
    
    while time.time() < deadline - 30.0:
        # Determine starting point
        r = random.random()
        if r < 0.3 or best_score > 1.27:
            # Full random restart
            c = np.random.choice([-1, 1], size=n)
        else:
            # Perturb best solution
            c = best_c.copy()
            num_flips = random.randint(2, 5)
            for _ in range(num_flips):
                idx = random.randint(0, n-1)
                c[idx] *= -1
        
        V = np.dot(c, powers_grid)
        current_score = get_max_norm_from_vals(V)
        
        # Local search
        c, V, current_score = local_search(c, V, current_score)
        
        if current_score < best_score - 1e-12:
            best_score = current_score
            best_c = c.copy()
            best_score_ever = min(best_score_ever, best_score)
            patience = 0
            # Decrease temperature on improvement
            temperature *= 0.8
        else:
            patience += 1
            # Increase temperature to explore more
            temperature = min(temperature * 1.05, 0.5)
        
        # If stuck for too long, do a more aggressive restart
        if patience > max_patience:
            c = np.random.choice([-1, 1], size=n)
            V = np.dot(c, powers_grid)
            current_score = get_max_norm_from_vals(V)
            c, V, current_score = local_search(c, V, current_score)
            if current_score < best_score - 1e-12:
                best_score = current_score
                best_c = c.copy()
                patience = 0
            else:
                patience = 0

    # Final high-precision polish on the best candidate
    def refine_peak(c, theta_center, delta, max_iter=80):
        """Golden section search for the peak near theta_center."""
        def eval_p(theta):
            z = np.exp(1j * theta)
            result = c[-1]
            for coeff in reversed(c[:-1]):
                result = result * z + coeff
            return np.abs(result)
        
        gr = 0.6180339887498949
        a = theta_center - delta
        b = theta_center + delta
        c1 = b - gr * (b - a)
        c2 = a + gr * (b - a)
        f1 = eval_p(c1)
        f2 = eval_p(c2)
        
        for _ in range(max_iter):
            if f1 < f2:
                b = c2
                c2 = c1
                f2 = f1
                c1 = b - gr * (b - a)
                f1 = eval_p(c1)
            else:
                a = c1
                c1 = c2
                f1 = f2
                c2 = a + gr * (b - a)
                f2 = eval_p(c2)
                
        return max(f1, f2, eval_p(theta_center))

    # Find peaks in best_c on a finer grid
    m_fine = 16384
    thetas_fine = np.linspace(0, 2*np.pi, m_fine, endpoint=False)
    powers_fine = np.exp(1j * np.outer(k, thetas_fine))
    V_best = np.dot(best_c, powers_fine)
    abs_best = np.abs(V_best)
    
    # Get top 20 indices
    top_indices = np.argsort(abs_best)[-20:]
    
    delta = 2 * np.pi / m_fine
    max_refined = 0
    for idx in top_indices:
        theta_peak = thetas_fine[idx]
        val = refine_peak(best_c, theta_peak, 8*delta)
        if val > max_refined:
            max_refined = val
            
    return best_c.tolist()
