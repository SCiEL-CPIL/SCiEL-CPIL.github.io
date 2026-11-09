import numpy as np
from scipy.optimize import minimize

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    
    def get_polygon(params):
        # params: coefficients for cos(2*k*phi)
        # r_i = 1 + sum c_k cos(2*k*phi_i)
        phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
        r = np.ones(n)
        for i, c in enumerate(params):
            r += c * np.cos(2 * (i + 1) * phi)
        
        # Ensure r is positive and reasonable
        if np.any(r < 0.5):
            return None
            
        x = r * np.cos(phi)
        y = r * np.sin(phi)
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
        # Cross products of consecutive edges
        edges = np.roll(poly, -1, axis=0) - poly
        cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
        # For CCW, all cross should be > 0
        if np.any(cross <= 0):
            return None
            
        return poly

    def compute_scores(poly):
        scores = []
        for i in range(n):
            v = poly[i]
            dists = np.linalg.norm(poly - v, axis=1)
            dists[i] = np.inf
            sorted_dists = np.sort(dists)[:19]
            
            if sorted_dists[0] < 0.05:
                scores.append(0.0)
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
                scores.append(0.0)
                continue
                
            ratios = np.maximum(four / D, D / four)
            raw = np.mean(ratios)
            scores.append(min(1.0, 1.0 / raw))
            
        return np.min(scores)

    def objective(params):
        poly = get_polygon(params)
        if poly is None:
            return 10.0
        score = compute_scores(poly)
        return -score

    # Optimize
    # Start with regular polygon (zeros)
    # Try a few initial points?
    # Nelder-Mead is deterministic.
    x0 = [0.0, 0.0, 0.0, 0.0]
    
    # To ensure we find a good spot, we might run a small grid or multiple starts?
    # Given 600s, we can run 10 starts.
    best_score = -1
    best_poly = None
    
    for _ in range(10):
        # Small random perturbation
        x0_try = np.random.normal(0, 0.1, size=4)
        res = minimize(objective, x0_try, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4})
        
        poly = get_polygon(res.x)
        if poly is not None:
            sc = compute_scores(poly)
            if sc > best_score:
                best_score = sc
                best_poly = poly

    if best_poly is None:
        # Fallback to regular polygon
        phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_poly = np.column_stack((np.cos(phi), np.sin(phi)))
        
    return best_poly
