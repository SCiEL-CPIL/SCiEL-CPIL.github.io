# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def _score(points):
    """Compute (d_min / d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    # Normalize to unit sphere to remove scale dependency and keep numbers stable
    # This effectively fixes d_max to be <= 2, but we compute the ratio anyway
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    points_norm = points / norms
    
    dists = pdist(points_norm, metric='euclidean')
    d_min = np.min(dists)
    d_max = np.max(dists)
    
    if d_max == 0 or np.isnan(d_min) or np.isnan(d_max):
        return 0.0
    
    return (d_min / d_max) ** 2

def _objective(x, n=14, d=3):
    """Objective function for minimization (negative of score)."""
    points = x.reshape(n, d)
    return -_score(points)

def _generate_cuboctahedron_14():
    """Generate 12 cuboctahedron vertices + 2 points."""
    # Cuboctahedron vertices: permutations of (±1, ±1, 0)
    verts = []
    for x in [1, -1]:
        for y in [1, -1]:
            verts.append([x, y, 0])
            verts.append([x, 0, y])
            verts.append([0, x, y])
    # This generates 12 unique vertices?
    # (1,1,0), (1,0,1), (0,1,1)
    # (-1,1,0), (-1,0,1), (0,-1,1)
    # (1,-1,0), (1,0,-1), (0,1,-1)
    # (-1,-1,0), (-1,0,-1), (0,-1,-1)
    # Yes, 12 unique.
    points = np.array(verts)
    
    # Add 2 points. Let's start them at (±2, 0, 0) to be far out, 
    # normalization will pull them in.
    p1 = np.array([2.0, 0.0, 0.0])
    p2 = np.array([-2.0, 0.0, 0.0])
    
    points = np.vstack([points, p1, p2])
    return points

def _generate_truncated_octahedron_14():
    """Generate 14 points from Truncated Octahedron vertices."""
    # Vertices are permutations of (0, ±1, ±2)
    verts = []
    for x in [0, 1, -1, 2, -2]:
        for y in [0, 1, -1, 2, -2]:
            for z in [0, 1, -1, 2, -2]:
                if sorted([abs(x), abs(y), abs(z)]) == [0, 1, 2]:
                    verts.append([x, y, z])
    
    # There are 24 vertices. We need 14.
    # Let's pick the first 14? Or a symmetric subset?
    # A symmetric subset might be better.
    # Let's take all 12 with a 0? 
    # No, let's just take 14 randomly from the 24 to break symmetry slightly 
    # or take a specific symmetric set.
    # For now, take first 14. The optimizer will adjust.
    points = np.array(verts[:14])
    return points

def _generate_random(n=14, d=3, seed=None):
    if seed is not None:
        np.random.seed(seed)
    points = np.random.randn(n, d)
    return points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the coordinates of the 14 points.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 600  # 600s limit, keep margin
    
    best_score = -1.0
    best_points = None
    
    # Initial guesses
    guesses = []
    
    # 1. Cuboctahedron based
    guesses.append(_generate_cuboctahedron_14())
    
    # 2. Truncated Octahedron based
    guesses.append(_generate_truncated_octahedron_14())
    
    # 3. Random guesses with different seeds
    for i in range(10):
        guesses.append(_generate_random(n, d, seed=100 + i))
        
    # 4. Icosahedron based (12 verts) + 2
    # Icosahedron vertices: (0, ±1, ±φ), (±1, ±φ, 0), (±φ, 0, ±1)
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = []
    for x in [0, 1, -1]:
        for y in [0, 1, -1]:
            for z in [0, 1, -1]:
                # This is too many. 
                # Standard ico: 
                # (0, ±1, ±φ), (±1, ±φ, 0), (±φ, 0, ±1)
                pass
    # Let's just generate them properly
    ico_verts = []
    for s1 in [1, -1]:
        for s2 in [1, -1]:
            ico_verts.append([0, s1, s2*phi])
            ico_verts.append([s1, s2*phi, 0])
            ico_verts.append([s2*phi, 0, s1])
    # This gives 12 vertices.
    points_ico = np.array(ico_verts)
    p1 = np.array([3.0, 0.0, 0.0])
    p2 = np.array([-3.0, 0.0, 0.0])
    guesses.append(np.vstack([points_ico, p1, p2]))
    
    # Optimize each guess
    for i, guess in enumerate(guesses):
        if time.time() > deadline - 30:
            break
            
        x0 = guess.flatten()
        
        # Use Powell method for non-smooth objective
        try:
            res = minimize(_objective, x0, method='Powell', options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if res.success or res.fun < -best_score:
                current_score = -res.fun
                if current_score > best_score:
                    best_score = current_score
                    best_points = res.x.reshape(n, d)
        except Exception as e:
            pass
            
    # If optimization failed or didn't improve, use the best initial guess
    if best_points is None:
        # Fallback to the first guess normalized
        best_points = guesses[0]
        best_score = _score(best_points)
        
    # Final refinement with more iterations if time permits
    if time.time() < deadline - 60 and best_points is not None:
        x0 = best_points.flatten()
        try:
            res = minimize(_objective, x0, method='Powell', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if -res.fun > best_score:
                best_score = -res.fun
                best_points = res.x.reshape(n, d)
        except Exception as e:
            pass
            
    # Ensure output is finite and correct shape
    if best_points is None:
        # Absolute fallback
        best_points = np.random.randn(n, d)
        
    # Normalize output to unit sphere for consistency, though not strictly required
    # as the metric is scale invariant. But it keeps numbers small.
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    best_points = best_points / norms
    
    return best_points

# EVOLVE-BLOCK-END
