# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    Uses a high-symmetry combinatorial template (4 vertices + 10 edge midpoints of a tetrahedron)
    and optimizes 5 radial scaling factors via coordinate descent and local search.
    """
    
    deadline = time.time() + 600  # 600s limit, keep safety margin
    
    # Define the 5 symmetry orbits for 14 points in 3D based on a regular tetrahedron
    # Orbit 0: 4 vertices of the regular tetrahedron
    # Orbit 1-4: 10 edge midpoints (each orbit corresponds to edges sharing a vertex? No, edges are not transitive under Td acting on vertices only? 
    # Wait, the tetrahedral group Td acts transitively on edges. So all 10 edge midpoints are in ONE orbit under the full symmetry group of the tetrahedron.
    # However, if we fix the vertices, the edge midpoints are all equivalent.
    # So we have:
    # Orbit A: 4 vertices
    # Orbit B: 10 edge midpoints
    # This gives only 2 degrees of freedom (radial scaling of A and B).
    # But the problem asks for 14 points. A regular tetrahedron has 4 vertices and 6 edges. 4+6=10 points. We need 4 more.
    # Perhaps face centers? There are 4 faces. 4+6+4 = 14 points.
    # Let's check the symmetry.
    # Vertices: 4
    # Edge midpoints: 6
    # Face centers: 4
    # Total: 14.
    # The tetrahedral group Td acts transitively on vertices, edges, and faces.
    # So we have 3 orbits:
    # Orbit 0: 4 vertices
    # Orbit 1: 6 edge midpoints
    # Orbit 2: 4 face centers
    # This gives 3 radial scaling factors to optimize.
    
    # Let's define the base positions for a regular tetrahedron centered at the origin.
    # Standard coordinates for regular tetrahedron vertices:
    # (1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)
    # These are at distance sqrt(3) from origin.
    
    v1 = np.array([1, 1, 1])
    v2 = np.array([1, -1, -1])
    v3 = np.array([-1, 1, -1])
    v4 = np.array([-1, -1, 1])
    
    vertices = np.array([v1, v2, v3, v4])
    
    # Edge midpoints
    e12 = (v1 + v2) / 2.0
    e13 = (v1 + v3) / 2.0
    e14 = (v1 + v4) / 2.0
    e23 = (v2 + v3) / 2.0
    e24 = (v2 + v4) / 2.0
    e34 = (v3 + v4) / 2.0
    
    edges = np.array([e12, e13, e14, e23, e24, e34])
    
    # Face centers
    # Face opposite v1: v2, v3, v4 -> center = (v2+v3+v4)/3
    f1 = (v2 + v3 + v4) / 3.0
    f2 = (v1 + v3 + v4) / 3.0
    f3 = (v1 + v2 + v4) / 3.0
    f4 = (v1 + v2 + v3) / 3.0
    
    faces = np.array([f1, f2, f3, f4])
    
    # Base points (unit scaled? No, just the geometry)
    # We will scale each orbit by a factor.
    # Let the scaling factors be s0, s1, s2.
    # Points = concat(s0 * vertices, s1 * edges, s2 * faces)
    
    # We want to maximize (d_min / d_max)^2.
    # This is equivalent to maximizing d_min / d_max.
    # We can fix one scale, say s0 = 1, and optimize s1, s2.
    # Or optimize all 3 and normalize.
    # Since the ratio is scale-invariant, we can fix s0 = 1.
    # Then we have 2 variables: s1, s2.
    
    def get_points(s1, s2):
        pts = np.vstack([
            vertices,          # s0 = 1
            s1 * edges,
            s2 * faces
        ])
        return pts
    
    def objective(s):
        s1, s2 = s
        pts = get_points(s1, s2)
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2
    
    # Initial guess
    # The base points are already somewhat optimized?
    # Let's start with s1=1, s2=1.
    s_init = np.array([1.0, 1.0])
    
    # Coordinate Descent + Local Search
    # Since the space is only 2D, we can do a fine grid search or use scipy.optimize.
    # Let's use scipy.optimize.minimize with Nelder-Mead for robustness.
    
    from scipy.optimize import minimize
    
    def neg_objective(s):
        return -objective(s)
    
    best_score = 0.0
    best_s = s_init
    
    # Try multiple starting points to avoid local minima
    starts = [
        [1.0, 1.0],
        [0.5, 0.5],
        [1.5, 1.5],
        [0.8, 1.2],
        [1.2, 0.8],
        [0.0, 1.0],
        [1.0, 0.0],
        [2.0, 0.5],
        [0.5, 2.0]
    ]
    
    for start in starts:
        if time.time() > deadline - 10:
            break
        res = minimize(neg_objective, start, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
        score = -res.fun
        if score > best_score:
            best_score = score
            best_s = res.x
    
    # Refine with a finer local search around the best point found
    if time.time() < deadline - 5:
        # Simple gradient-free refinement
        s_curr = best_s.copy()
        step = 0.01
        for _ in range(200):
            if time.time() > deadline - 2:
                break
            improved = False
            for i in range(2):
                for sign in [-1, 1]:
                    s_try = s_curr.copy()
                    s_try[i] += sign * step
                    score_try = objective(s_try)
                    if score_try > best_score:
                        best_score = score_try
                        best_s = s_try
                        s_curr = s_try
                        improved = True
            if not improved:
                step *= 0.5
                if step < 1e-6:
                    break
    
    final_points = get_points(best_s[0], best_s[1])
    
    # Ensure finite
    if not np.all(np.isfinite(final_points)):
        # Fallback to initial
        final_points = get_points(1.0, 1.0)
    
    return final_points

# EVOLVE-BLOCK-END
