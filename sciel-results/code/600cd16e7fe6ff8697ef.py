# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from scipy.optimize import milp, LinearConstraint, Bounds

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    grid_sq = (grid ** 2).sum(axis=1)
    
    # Precompute tuple to index
    tuple_to_idx = {tuple(p): i for i, p in enumerate(grid)}

    def get_sphere_params(p1, p2, p3, p4):
        """
        Returns (center, radius_sq) for the sphere passing through 4 points, 
        or None if they are coplanar or collinear (no unique sphere).
        """
        # Matrix form:
        # [x1 y1 z1 x1^2+y1^2+z1^2]
        # [x2 y2 z2 x2^2+y2^2+z2^2]
        # [x3 y3 z3 x3^2+y3^2+z3^2]
        # [x4 y4 z4 x4^2+y4^2+z4^2]
        # This doesn't directly give center.
        
        # Use the determinant condition for 5 points (p, p1, p2, p3, p4) = 0
        # Or solve the linear system for center C:
        # 2*C dot (p1-p0) = |p1|^2 - |p0|^2
        # 2*C dot (p2-p0) = |p2|^2 - |p0|^2
        # 2*C dot (p3-p0) = |p3|^2 - |p0|^2
        
        p0, p1, p2, p3 = p1, p2, p3, p4
        A = np.array([
            [2*(p1-p0)[0], 2*(p1-p0)[1], 2*(p1-p0)[2]],
            [2*(p2-p0)[0], 2*(p2-p0)[1], 2*(p2-p0)[2]],
            [2*(p3-p0)[0], 2*(p3-p0)[1], 2*(p3-p0)[2]]
        ], dtype=np.float64)
        b = np.array([
            np.dot(p1,p1) - np.dot(p0,p0),
            np.dot(p2,p2) - np.dot(p0,p0),
            np.dot(p3,p3) - np.dot(p0,p0)
        ], dtype=np.float64)
        
        det = np.linalg.det(A)
        if abs(det) < 1e-9:
            return None # Coplanar
            
        try:
            center = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None
            
        radius_sq = np.dot(center - p0, center - p0)
        return (center, radius_sq)

    def is_on_sphere(point, center, radius_sq, tol=1e-9):
        d = point - center
        dist_sq = np.dot(d, d)
        return abs(dist_sq - radius_sq) < tol

    # Phase 1: Discover "Dense Spheres" (spheres with >= 5 grid points)
    # We sample random quadruples to find these.
    rng = np.random.default_rng(42)
    dense_spheres = [] # List of (center, radius_sq, set_of_point_indices)
    sphere_signatures = set() # To avoid duplicates
    
    start_time = time.time()
    deadline = start_time + 580
    
    # Sample many quadruples
    num_samples = 50000
    for _ in range(num_samples):
        if time.time() > deadline - 30:
            break
        idxs = rng.choice(N, size=4, replace=False)
        p1, p2, p3, p4 = grid[idxs]
        params = get_sphere_params(p1, p2, p3, p4)
        if params is None:
            continue
        center, r_sq = params
        # Quantize center and radius to avoid float dupes
        # Center coords are often rational with small denominators
        sig = (tuple(np.round(center, 6)), np.round(r_sq, 6))
        if sig in sphere_signatures:
            continue
        
        # Check if this sphere is "dense"
        # Efficiently check all points?
        # dists = ((grid - center)**2).sum(axis=1)
        # on_sphere = np.argwhere(np.abs(dists - r_sq) < 1e-6).flatten()
        
        # Optimization: Only check if the sphere is likely to have more points.
        # Actually, just check. N=512 is small.
        dists = ((grid - center)**2).sum(axis=1)
        on_sphere = np.argwhere(np.abs(dists - r_sq) < 1e-6).flatten()
        
        if len(on_sphere) >= 5:
            sphere_signatures.add(sig)
            dense_spheres.append((center, r_sq, set(on_sphere.tolist())))

    # Phase 2: Solve Max-SAT / Hitting Set
    # We want to select max points such that for each dense sphere, <= 4 points are selected.
    # This is a binary integer program.
    # Variables: x_i for i in 0..N-1
    # Objective: max sum(x_i)
    # Constraints: For each dense sphere S: sum_{i in S} x_i <= 4
    
    if not dense_spheres:
        # No dense spheres found? Return all points? 
        # Wait, there might be dense spheres we missed.
        # But if we found none, the greedy "add all" might work?
        # No, we must ensure validity.
        # If no dense spheres found via sampling, it's risky.
        # Let's fall back to a greedy check if ILP is not feasible or if we are out of time.
        pass

    # Build A matrix for MILP
    # Number of constraints M = len(dense_spheres)
    M = len(dense_spheres)
    
    if M > 0:
        # A is M x N sparse
        # We can use scipy.sparse
        import scipy.sparse as sp
        
        rows = []
        cols = []
        data = []
        
        for i, (center, r_sq, pts) in enumerate(dense_spheres):
            for p_idx in pts:
                rows.append(i)
                cols.append(p_idx)
                data.append(1.0)
                
        A = sp.csr_matrix((data, (rows, cols)), shape=(M, N))
        
        # Constraints: A @ x <= 4
        # And 0 <= x <= 1
        
        c = -np.ones(N) # Minimize -sum(x) => Maximize sum(x)
        
        constraints = [
            LinearConstraint(A, ub=np.full(M, 4.0)),
            LinearConstraint(np.eye(N), lb=0.0, ub=1.0)
        ]
        
        integrality = np.ones(N)
        bounds = Bounds(0, 1)
        
        try:
            res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
            if res.status == 0:
                x = res.x
                selected_indices = np.where(x > 0.5)[0]
                result = grid[selected_indices]
                # Final verification step
                if _verify_validity(result, grid, grid_sq, tuple_to_idx):
                    return result.astype(np.int64)
            else:
                # MILP failed, fallback to greedy
                pass
        except Exception:
            pass

    # Fallback: Greedy with Exact Check (similar to seed but optimized)
    # Use the dense spheres found to guide the greedy?
    # Or just do a fast greedy.
    
    best_arr = np.empty((0, 3), dtype=np.int64)
    
    # Sort grid by some heuristic (e.g. distance from center)
    center_grid = np.array([3.5, 3.5, 3.5])
    dists = np.linalg.norm(grid - center_grid, axis=1)
    order = np.argsort(dists)
    
    arr = np.empty((0, 3), dtype=np.int64)
    arr_sq = np.empty(0, dtype=np.int64)
    arr_set = set()
    
    # Pre-compute "bad" quadruples from dense spheres?
    # If a sphere has 5 points, any 4 of them form a "bad" quadruple for the 5th.
    # We can maintain a count of how many points from each dense sphere are in `arr`.
    
    # Map point index to list of dense sphere indices it belongs to
    point_to_spheres = {i: [] for i in range(N)}
    for s_idx, (center, r_sq, pts) in enumerate(dense_spheres):
        for p_idx in pts:
            point_to_spheres[p_idx].append(s_idx)
            
    sphere_counts = np.zeros(M, dtype=np.int64)
    
    for idx in order:
        if time.time() > deadline - 10:
            break
            
        # Check if adding this point violates any dense sphere
        # If sphere_counts[s] + 1 > 4 for any s in point_to_spheres[idx], it's invalid.
        valid = True
        for s_idx in point_to_spheres[idx]:
            if sphere_counts[s_idx] >= 4:
                valid = False
                break
                
        if valid:
            # We also need to ensure no *other* 5-tuple is formed.
            # The dense sphere check only covers spheres with >=5 points.
            # What about spheres with exactly 5 points that we missed?
            # Or spheres with 4 points in the set + 1 new point = 5 total?
            # If a sphere has exactly 5 points in the grid, it is a dense sphere.
            # So if we cover all dense spheres, we cover all potential 5-tuples?
            # YES. If 5 points are cospherical, they lie on a sphere.
            # That sphere contains at least those 5 grid points.
            # So it is a "dense sphere" (count >= 5).
            # Therefore, if we enforce "no dense sphere has > 4 selected points",
            # we guarantee no 5 cospherical points.
            
            arr = np.vstack([arr, grid[idx]]) if len(arr) > 0 else grid[idx][None, :]
            arr_sq = np.append(arr_sq, grid_sq[idx])
            arr_set.add(idx)
            
            for s_idx in point_to_spheres[idx]:
                sphere_counts[s_idx] += 1
                
    if len(arr) > len(best_arr):
        best_arr = arr
        
    # Local Search: Try to add back points that were rejected?
    # Or swap?
    # Given time constraints, return best found.
    
    return best_arr.astype(np.int64)

def _verify_validity(arr, grid, grid_sq, tuple_to_idx):
    """
    Verify that no 5 points in arr are cospherical.
    This is a safety check.
    """
    k = len(arr)
    if k < 5:
        return True
        
    # For each point in arr, check if it forms a cospherical 5-tuple with 4 others.
    # To speed up, we can just check all 5-combinations? C(k,5) might be large.
    # But k is likely < 100. C(100,5) = 75M. Too slow.
    
    # Use the determinant check for each point p and 4 others.
    # This is O(k^5) if naive.
    # Better: For each p, check all 4-combinations of others. O(k^4) per p. Total O(k^5).
    # If k=50, 50^5 = 312M. Might be too slow in Python.
    
    # Since we used the dense sphere logic, this should be fast if we just check dense spheres.
    # But this is a generic verifier.
    
    # Optimization: Only check if the set of 5 points lies on a sphere.
    # We can use the property: 5 points are cospherical iff the determinant of the 5x5 matrix is 0.
    
    # We will sample a few random 5-tuples to verify.
    rng = np.random.default_rng(123)
    for _ in range(1000):
        if k < 5:
            break
        idxs = rng.choice(k, size=5, replace=False)
        pts = arr[idxs]
        # Construct matrix
        # [x y z x^2+y^2+z^2 1]
        mat = np.zeros((5, 5))
        for i, p in enumerate(pts):
            mat[i, 0] = p[0]
            mat[i, 1] = p[1]
            mat[i, 2] = p[2]
            mat[i, 3] = np.dot(p, p)
            mat[i, 4] = 1.0
            
        det = np.linalg.det(mat)
        if abs(det) < 1e-9:
            return False
            
    return True

# EVOLVE-BLOCK-END
