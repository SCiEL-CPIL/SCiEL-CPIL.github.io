import numpy as np
import time
from scipy.optimize import differential_evolution
from scipy.spatial.distance import pdist

N = 14
D = 3

def get_pairwise_sq_dists(points):
    """Calculate squared pairwise distances for N points in D dimensions."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    """Calculate the actual score: (max_dist / min_dist)^2."""
    sq_dists = get_pairwise_sq_dists(points)
    min_sq = np.min(sq_dists)
    if min_sq < 1e-12:
        return np.inf
    max_sq = np.max(sq_dists)
    return (max_sq / min_sq)

def generate_24_cell_vertices():
    """Generate the 24 vertices of the 24-cell."""
    verts = []
    # Permutations of (1, 1, 0, 0) with all sign combinations
    for i in range(4):
        for j in range(i+1, 4):
            # Positions i and j are non-zero
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    v = np.zeros(4)
                    v[i] = si
                    v[j] = sj
                    verts.append(v)
    return np.array(verts)

def objective_projection(p_flat, vertices):
    """
    Objective function for differential evolution.
    p_flat: 12 floats representing a 3x4 matrix P.
    vertices: 14x4 array of selected 24-cell vertices.
    """
    P = p_flat.reshape(3, 4)
    # Project vertices
    points = vertices @ P.T
    
    # Check for degeneracy (points too close or identical)
    sq_dists = get_pairwise_sq_dists(points)
    min_sq = np.min(sq_dists)
    if min_sq < 1e-8:
        return 1e10
        
    max_sq = np.max(sq_dists)
    
    # We want to minimize the ratio. 
    # To avoid issues with scale, we can normalize the points.
    # However, the ratio is scale invariant, so we don't strictly need to, 
    # but it helps numerical stability in the optimizer if the values are O(1).
    norm = np.linalg.norm(points)
    if norm < 1e-8:
        return 1e10
    
    return max_sq / min_sq

def construct():
    t0 = time.time()
    deadline = 145.0  # Safety margin before 150s
    
    # Generate 24-cell vertices
    all_verts = generate_24_cell_vertices()
    
    best_score = np.inf
    best_points = None
    
    # We will try a few different subsets of 14 vertices from the 24.
    # Strategy: 
    # 1. First 14
    # 2. Vertices with v[0] >= 0 (12 verts) + 2 with v[0] < 0
    # 3. Random subsets (fixed seed)
    
    subsets = []
    
    # Subset 1: First 14
    subsets.append(all_verts[:14])
    
    # Subset 2: Half with v[0]>0, half with v[0]<0
    pos = all_verts[all_verts[:, 0] > 0]
    neg = all_verts[all_verts[:, 0] < 0]
    if len(pos) >= 7 and len(neg) >= 7:
        subsets.append(np.vstack([pos[:7], neg[:7]]))
        
    # Subset 3: Random selection
    rng = np.random.default_rng(42)
    for _ in range(3):
        idx = rng.choice(24, size=14, replace=False)
        subsets.append(all_verts[idx])
        
    # Define bounds for the 12 parameters of P
    # P entries can be roughly between -2 and 2
    bounds = [(-2.0, 2.0)] * 12
    
    for sub_idx, vertices in enumerate(subsets):
        if time.time() - t0 > deadline - 10:
            break
            
        try:
            # Use differential evolution for global optimization in low dimension
            # maxiter=100, popsize=20 should be fast enough for 12 dimensions
            result = differential_evolution(
                objective_projection, 
                bounds, 
                args=(vertices,),
                maxiter=100, 
                popsize=20,
                tol=1e-6,
                seed=42+sub_idx,
                init='sobol'
            )
            
            if result.success and result.fun < best_score:
                # Reconstruct points
                P = result.x.reshape(3, 4)
                points = vertices @ P.T
                score = raw_score(points)
                if score < best_score:
                    best_score = score
                    best_points = points
        except Exception as e:
            pass

    # If DE didn't find anything good, fallback to a local polish of the best found or a simple heuristic
    if best_points is None:
        # Fallback: Use the first subset and a simple random projection
        vertices = subsets[0]
        P = np.random.default_rng(1).normal(size=(3, 4))
        points = vertices @ P.T
        best_points = points
        best_score = raw_score(points)

    # Final polish with L-BFGS-B on the original coordinates (42 dims) starting from the projected solution
    # This allows the points to deviate from the linear projection manifold if it helps.
    if time.time() - t0 < deadline - 5:
        from scipy.optimize import minimize
        
        def obj_full(x):
            pts = x.reshape(N, D)
            # Center
            pts = pts - np.mean(pts, axis=0, keepdims=True)
            sq_d = get_pairwise_sq_dists(pts)
            min_sq = np.min(sq_d)
            if min_sq < 1e-10:
                return 1e10
            max_sq = np.max(sq_d)
            return max_sq / min_sq
            
        x0 = best_points.flatten()
        try:
            res = minimize(obj_full, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-10})
            if res.fun < best_score:
                best_points = res.x.reshape(N, D)
                best_score = res.fun
        except Exception:
            pass

    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {1.0/raw_score(pts)}") # Print the "search score" equivalent for visibility
    print(f"Raw Ratio Squared: {raw_score(pts)}")
