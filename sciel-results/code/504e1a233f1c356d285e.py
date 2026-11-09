# EVOLVE-BLOCK-START
import numpy as np


def _vertex_score(polygon, i):
    """Calculate the score for a single vertex i in the polygon."""
    n = len(polygon)
    # Calculate distances from vertex i to all other vertices
    diffs = polygon - polygon[i]
    dists = np.linalg.norm(diffs, axis=1)
    
    # Exclude self-distance
    dists = np.delete(dists, i)
    
    # Sort distances
    d = np.sort(dists)
    
    # Check for repeated points (d_0 < 0.05)
    if d[0] < 0.05:
        return 0.0
    
    # Find k: first index minimizing d_{k+1} - d_k
    gaps = np.diff(d)
    k = np.argmin(gaps)
    
    D = (d[k] + d[k+1]) / 2.0
    if D < 0.1:
        return 0.0
        
    # Determine the four distances to use
    if k == 0:
        idxs = [0, 1, 2, 3]
    elif k == 17: # n=20, so max index for d is 18, gap index max is 17
        idxs = [15, 16, 17, 18]
    else:
        idxs = [k-1, k, k+1, k+2]
        
    four_d = d[idxs]
    
    # Calculate raw score
    ratios = np.maximum(four_d / D, D / four_d)
    raw = np.mean(ratios)
    
    score = min(1.0, 1.0 / raw)
    return score

def _evaluate_polygon(polygon):
    """Evaluate the entire polygon based on the task metric."""
    # Check convexity and validity first
    n = len(polygon)
    if n != 20:
        return 0.0
        
    # Check for repeated points
    if np.any(np.linalg.norm(np.diff(polygon, axis=0, append=polygon[:1]), axis=1) < 1e-6):
         # This check is a bit loose, let's rely on the vertex score's d[0] < 0.05 mostly
         pass
         
    # Check strict convexity: all turns same sign, total turn +-2pi
    # Calculate cross products of consecutive edge vectors
    edges = np.diff(polygon, axis=0, append=polygon[:1])
    cross = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    
    # All cross products must have the same sign (non-zero)
    if np.any(cross == 0):
        return 0.0
    if np.any(cross > 0) and np.any(cross < 0):
        return 0.0
        
    # Total turning angle should be +-2*pi. For a simple convex polygon, this is guaranteed if it's strictly convex and closed.
    # We can verify by summing angles, but the cross product sign check is usually sufficient for "strictly convex" in this context 
    # assuming simple polygon.
    
    # Calculate minimum vertex score
    min_score = 1.0
    for i in range(n):
        s = _vertex_score(polygon, i)
        if s < min_score:
            min_score = s
            if min_score == 0.0:
                return 0.0
                
    return min_score

def equidistant_points_polygon20() -> np.ndarray:
    """
    Design a convex polygon using Sine-Modulated Radius (Harmonic Perturbation).
    r(theta) = 1 + A * cos(k * theta + phi)
    
    We perform a small grid search over A, k, and phi to maximize the minimum vertex score.
    """
    n = 20
    best_score = -1.0
    best_poly = None
    
    # Parameters to search
    # A: Amplitude of modulation. Must be small to maintain convexity.
    # For r(θ) = 1 + A cos(kθ), convexity requires r + r'' > 0.
    # r' = -A k sin(kθ)
    # r'' = -A k^2 cos(kθ)
    # Condition: 1 + A cos(kθ) - A k^2 cos(kθ) > 0 => 1 + A(1-k^2)cos(kθ) > 0.
    # Worst case cos(kθ) = -1 (if k>1): 1 - A(1-k^2) > 0 => A < 1/(1-k^2) if k<1, or A < 1/(k^2-1) if k>1?
    # Actually, if k=2, 1 + A(1-4)cos(2θ) = 1 - 3A cos(2θ). Min is 1-3A. So A < 1/3.
    # Let's keep A small, e.g., 0.05 to 0.3.
    
    # k: Frequency. Integer values ensure closure.
    # phi: Phase.
    
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    # Search space
    # We want to break symmetry of the regular polygon.
    # k=2, 3, 4, 5 are good candidates.
    # A in [0.05, 0.1, 0.15, 0.2, 0.25]
    # phi in [0, pi/4, pi/2, 3pi/4, pi]
    
    A_vals = [0.05, 0.10, 0.15, 0.20, 0.25]
    k_vals = [2, 3, 4, 5]
    phi_vals = np.linspace(0, np.pi, 5, endpoint=False)
    
    for A in A_vals:
        for k in k_vals:
            for phi in phi_vals:
                r = 1.0 + A * np.cos(k * angles + phi)
                x = r * np.cos(angles)
                y = r * np.sin(angles)
                poly = np.stack([x, y], axis=1)
                
                # Normalize
                centroid = np.mean(poly, axis=0)
                poly = poly - centroid
                dists_to_centroid = np.linalg.norm(poly, axis=1)
                mean_dist = np.mean(dists_to_centroid)
                
                if mean_dist <= 0.01:
                    continue
                    
                poly = poly / mean_dist
                
                score = _evaluate_polygon(poly)
                
                if score > best_score:
                    best_score = score
                    best_poly = poly.copy()
                    
    if best_poly is None:
        # Fallback to regular polygon if something went wrong
        x = np.cos(angles)
        y = np.sin(angles)
        best_poly = np.stack([x, y], axis=1)
        
    return best_poly

# EVOLVE-BLOCK-END
