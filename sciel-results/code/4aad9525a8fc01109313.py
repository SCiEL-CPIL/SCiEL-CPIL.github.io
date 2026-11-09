import numpy as np
from scipy.optimize import differential_evolution

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    
    def get_polygon_from_angles(angles):
        base_angles = 2 * np.pi * np.arange(n) / n
        total_angles = base_angles + angles
        
        x = np.cos(total_angles)
        y = np.sin(total_angles)
        poly = np.column_stack((x, y))
        
        centroid = np.mean(poly, axis=0)
        poly = poly - centroid
        dists = np.linalg.norm(poly, axis=1)
        mean_d = np.mean(dists)
        if mean_d < 0.01:
            return None
        poly = poly / mean_d
        
        edges = np.roll(poly, -1, axis=0) - poly
        cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
        if np.any(cross <= 0):
            return None
            
        return poly

    def compute_vertex_scores(poly):
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

    def smooth_min_score(angles, temperature=0.05):
        poly = get_polygon_from_angles(angles)
        if poly is None:
            return 0.0
            
        scores = compute_vertex_scores(poly)
        
        # Use log-sum-exp to create a smooth approximation of the minimum
        # min(s_i) ≈ -log(sum(exp(-s_i/T))) * T
        neg_scores = -scores / temperature
        max_neg = np.max(neg_scores)
        log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_scores - max_neg)))
        smooth_min = -log_sum_exp * temperature
        
        return smooth_min

    def objective(angles):
        # We want to maximize the smooth minimum score
        # So we minimize the negative of it
        s = smooth_min_score(angles)
        return -s

    # Use a smaller bound to ensure convexity and avoid issues
    bound = np.pi / 30
    bounds = [(-bound, bound) for _ in range(n)]
    
    # Run differential evolution with the smooth objective
    res = differential_evolution(
        objective, 
        bounds, 
        seed=42, 
        maxiter=300, 
        tol=1e-6, 
        polish=True,
        mutation=(0.5, 1.5),
        recombination=0.8
    )
    
    best_poly = get_polygon_from_angles(res.x)
    
    if best_poly is None:
        phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_poly = np.column_stack((np.cos(phi), np.sin(phi)))
        
    return best_poly
