import numpy as np
from scipy.optimize import minimize

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    
    def get_polygon_from_angles(angles):
        base_angles = 2 * np.pi * np.arange(n) / n
        total_angles = base_angles + angles
        
        x = np.cos(total_angles)
        y = np.sin(total_angles)
        poly = np.column_stack((x, y))
        
        # Normalize
        centroid = np.mean(poly, axis=0)
        poly = poly - centroid
        dists = np.linalg.norm(poly, axis=1)
        mean_d = np.mean(dists)
        if mean_d < 0.01:
            return None
        poly = poly / mean_d
        
        # Check strict convexity
        edges = np.roll(poly, -1, axis=0) - poly
        cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
        if np.any(cross <= 0):
            return None
            
        return poly

    def compute_scores(poly):
        scores = np.zeros(n)
        for i in range(n):
            v = poly[i]
            dists = np.linalg.norm(poly - v, axis=1)
            dists[i] = np.inf
            sorted_dists = np.sort(dists)[:19]
            
            if sorted_dists[0] < 0.05:
                scores[i] = 0.0
                continue
                
            gaps = np.diff(sorted_dists)
            k = np.argmin(gaps)
            
            if k == 0:
                D = (sorted_dists[0] + sorted_dists[1]) / 2.0
                four = sorted_dists[0:4]
            elif k == 17:
                D = (sorted_dists[17] + sorted_dists[18]) / 2.0
                four = sorted_dists[15:19]
            else:
                D = (sorted_dists[k] + sorted_dists[k+1]) / 2.0
                four = sorted_dists[k-1:k+3]
                
            if D < 0.1:
                scores[i] = 0.0
                continue
                
            ratios = np.maximum(four / D, D / four)
            raw = np.mean(ratios)
            scores[i] = min(1.0, 1.0 / raw)
            
        return scores

    def smooth_objective(angles):
        poly = get_polygon_from_angles(angles)
        if poly is None:
            return 100.0
            
        scores = compute_scores(poly)
        
        # Log-sum-exp approximation of the minimum score
        # We want to maximize min(scores), so minimize -min(scores).
        # approx min(s) = -1/beta * log(sum(exp(-beta * s)))
        # So objective = 1/beta * log(sum(exp(-beta * s)))
        
        beta = 5.0
        # Clip scores to avoid underflow/overflow issues in exp, though scores are [0,1]
        # exp(-5 * 1) = exp(-5) ~ 0.006, fine.
        # If score is 0, exp(0) = 1.
        
        # Handle case where all scores are 0? Sum is 20. log(20)/beta.
        # If one score is high, others 0. Sum ~ 1 + 19*exp(-beta).
        
        # To be safe with numerical stability:
        # max_exp = max(-beta * scores)
        # sum_exp = sum(exp(-beta * scores - max_exp))
        # log_sum_exp = max_exp + log(sum_exp)
        # approx_min = -1/beta * log_sum_exp
        # objective = -approx_min = 1/beta * log_sum_exp
        
        neg_beta_scores = -beta * scores
        max_val = np.max(neg_beta_scores)
        sum_exp = np.sum(np.exp(neg_beta_scores - max_val))
        
        if sum_exp == 0:
            return 100.0
            
        log_sum_exp = max_val + np.log(sum_exp)
        obj_val = (1.0 / beta) * log_sum_exp
        
        return obj_val

    # Multi-start optimization
    best_score = -np.inf
    best_angles = np.zeros(n)
    
    # Initial points: regular polygon (zeros) and some random perturbations
    starts = [np.zeros(n)]
    rng = np.random.default_rng(42)
    for _ in range(10):
        starts.append(rng.uniform(-np.pi/25, np.pi/25, size=n))
    
    for start in starts:
        res = minimize(smooth_objective, start, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
        
        if res.success or res.fun < best_score:
            # Evaluate the actual hard minimum score for this result
            poly = get_polygon_from_angles(res.x)
            if poly is not None:
                scores = compute_scores(poly)
                actual_score = np.min(scores)
                if actual_score > best_score:
                    best_score = actual_score
                    best_angles = res.x
    
    best_poly = get_polygon_from_angles(best_angles)
    if best_poly is None:
        # Fallback
        phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_poly = np.column_stack((np.cos(phi), np.sin(phi)))
        
    return best_poly
