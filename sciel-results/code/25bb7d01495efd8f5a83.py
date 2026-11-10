"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def _covering_radius_fast(points):
    """
    Compute the covering radius in degrees for points on the unit sphere.
    Uses vectorized circumradius computation for triangular facets.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        
        max_circumradius = 0.0
        
        # Handle triangular facets (most common in 3D convex hull)
        tri_facets = [f for f in facets if len(f) == 3]
        
        if len(tri_facets) > 0:
            tri_facets = np.array(tri_facets)
            a = points[tri_facets[:, 0]]
            b = points[tri_facets[:, 1]]
            c = points[tri_facets[:, 2]]
            
            # Compute cross products
            ab = np.cross(a, b)
            bc = np.cross(b, c)
            ca = np.cross(c, a)
            
            # Triple product: a . (b x c)
            triple = np.einsum('ij,ij->i', a, np.cross(b, c))
            denom = 4.0 * np.abs(triple)
            
            # Avoid division by zero
            valid = denom > 1e-15
            if np.any(valid):
                norms_ab = np.linalg.norm(ab[valid], axis=1)
                norms_bc = np.linalg.norm(bc[valid], axis=1)
                norms_ca = np.linalg.norm(ca[valid], axis=1)
                
                sin_R = (norms_ab * norms_bc * norms_ca) / denom[valid]
                sin_R = np.clip(sin_R, 0.0, 1.0)
                R = np.arcsin(sin_R)
                max_circumradius = max(max_circumradius, np.max(R))
        
        # Handle non-triangular facets (rare, but possible)
        non_tri_facets = [f for f in facets if len(f) != 3]
        for simplex in non_tri_facets:
            pts = points[simplex]
            n_pts = len(pts)
            if n_pts < 3:
                continue
            if n_pts == 4:
                a, b, c, d = pts[0], pts[1], pts[2], pts[3]
                v1 = a - b
                v2 = a - c
                v3 = a - d
                x = np.cross(v1, np.cross(v2, v3))
                nx = np.linalg.norm(x)
                if nx < 1e-15:
                    continue
                x = x / nx
                cos_R = np.clip(np.dot(x, a), -1.0, 1.0)
                R = np.arccos(cos_R)
                max_circumradius = max(max_circumradius, R)
            else:
                centroid = np.mean(pts, axis=0)
                nc = np.linalg.norm(centroid)
                if nc < 1e-15:
                    continue
                centroid = centroid / nc
                for p in pts:
                    cos_R = np.clip(np.dot(centroid, p), -1.0, 1.0)
                    R = np.arccos(cos_R)
                    max_circumradius = max(max_circumradius, R)
        
        return np.degrees(max_circumradius)
    except Exception:
        return np.inf

def _spectral_optimization(points, max_time):
    """
    Optimize points using a spectral approach.
    We view the problem as finding a set of vectors that are well-distributed.
    A common spectral method for sphere packing/covering involves maximizing
    the minimum eigenvalue of a Gram matrix or related structure, or using
    eigenvalue decomposition of a matrix constructed from the points to refine them.
    
    Here, we use an iterative scheme where we construct a matrix M from the points
    and update the points based on the eigenvectors of M, aiming to spread them out.
    Specifically, we can use the idea of "eigenvalue optimization" where we
    maximize the sum of the smallest eigenvalues of the Gram matrix, which
    encourages isotropy.
    """
    start_time = time.time()
    current_points = points.copy()
    current_radius = _covering_radius_fast(current_points)
    best_points = current_points.copy()
    best_radius = current_radius
    
    # We will iterate by updating points based on a spectral flow.
    # Let G be the Gram matrix. We want to maximize the minimum eigenvalue of G?
    # Or perhaps use the Laplacian of the graph formed by nearest neighbors?
    # A simpler spectral approach: 
    # 1. Compute the covariance matrix of the points.
    # 2. If it's not isotropic, rotate the points to make it isotropic.
    # 3. This is a global move that can escape local minima.
    
    # Let's try a method based on the "Spectral Flow" or similar.
    # Actually, a known technique is to use the eigenvectors of the adjacency matrix
    # of the nearest neighbor graph to perturb the points.
    
    # Let's try a different spectral idea:
    # Maximize the minimum distance is hard.
    # But we can maximize the trace of the Gram matrix subject to constraints? No, trace is fixed for unit vectors.
    
    # Let's try to use the eigen-decomposition of the matrix K = sum_i v_i v_i^T.
    # If K is not proportional to identity, the distribution is anisotropic.
    # We can apply a transformation to make K more isotropic.
    # This is similar to "whitening" the distribution.
    
    step = 0
    while time.time() - start_time < max_time:
        # Compute the second moment matrix
        M = current_points.T @ current_points
        
        # Check if M is close to identity * (N/3)
        target_M = (N / 3.0) * np.eye(3)
        
        # If M is already isotropic, the spectral move is zero.
        # We need a way to move towards isotropy.
        # One way: diagonalize M = U D U^T.
        # We want to transform points by U D^{-1/2} U^T to make the covariance identity.
        # But this is a linear transformation that doesn't preserve unit norm.
        # So we apply the transformation and then renormalize.
        
        try:
            eigvals, eigvecs = np.linalg.eigh(M)
            # Avoid division by zero
            eigvals = np.clip(eigvals, 1e-10, None)
            # Compute the whitening matrix
            # W = U D^{-1/2} U^T
            W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            # Apply W to all points
            new_points = current_points @ W.T
            
            # Renormalize
            norms = np.linalg.norm(new_points, axis=1)
            norms[norms < 1e-10] = 1.0
            new_points = new_points / norms[:, np.newaxis]
            
            # Evaluate
            new_radius = _covering_radius_fast(new_points)
            
            if new_radius < best_radius:
                best_radius = new_radius
                best_points = new_points.copy()
                current_points = new_points.copy()
                current_radius = new_radius
            else:
                # If whitening didn't help, try a small perturbation based on the eigenvectors
                # to escape the isotropic state if it's a local minimum.
                # Or simply stop if we've reached isotropy?
                # Let's try a small rotation based on the eigenvectors.
                # Actually, if M is isotropic, the points are well-distributed in terms of moments.
                # We might need to break the symmetry.
                
                # Let's try moving points along the directions of the eigenvectors with small weights.
                # This is a heuristic.
                break
        except Exception:
            break
            
        step += 1
        
    return best_points, best_radius

def _local_hill_climb(points, max_time):
    """
    Fast local hill climbing to polish the result.
    """
    start_time = time.time()
    current_points = points.copy()
    current_radius = _covering_radius_fast(current_points)
    best_points = current_points.copy()
    best_radius = current_radius
    step_size = 0.01
    
    while time.time() - start_time < max_time:
        improved = False
        # Try moving each point
        for i in range(N):
            if time.time() - start_time > max_time:
                break
            # Try a few random directions
            for _ in range(3):
                perturb = np.random.randn(3) * step_size
                new_point = current_points[i] + perturb
                new_point /= np.linalg.norm(new_point)
                
                old_point = current_points[i].copy()
                current_points[i] = new_point
                
                new_radius = _covering_radius_fast(current_points)
                
                if new_radius < current_radius:
                    current_radius = new_radius
                    if new_radius < best_radius:
                        best_radius = new_radius
                        best_points = current_points.copy()
                    improved = True
                else:
                    current_points[i] = old_point
        
        if not improved:
            step_size *= 0.8
            if step_size < 1e-6:
                break
                
    return best_points, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # Start with a high-quality Fibonacci lattice
    initial_points = _fibonacci_sphere(N)
    
    best_points = initial_points
    best_radius = _covering_radius_fast(best_points)
    
    # Phase 1: Spectral Optimization
    # This attempts to make the distribution isotropic, which is a necessary condition
    # for optimal covering.
    sp_points, sp_radius = _spectral_optimization(best_points, max_time=30.0)
    if sp_radius < best_radius:
        best_points = sp_points
        best_radius = sp_radius
        
    # Phase 2: Local Hill Climbing to polish
    if time.time() < deadline - 10.0:
        lc_points, lc_radius = _local_hill_climb(best_points, max_time=30.0)
        if lc_radius < best_radius:
            best_points = lc_points
            best_radius = lc_radius
            
    # Phase 3: Another round of Spectral Optimization
    if time.time() < deadline - 5.0:
        sp2_points, sp2_radius = _spectral_optimization(best_points, max_time=20.0)
        if sp2_radius < best_radius:
            best_points = sp2_points
            best_radius = sp2_radius
            
        # Final polish
        if time.time() < deadline - 2.0:
            lc2_points, lc2_radius = _local_hill_climb(best_points, max_time=5.0)
            if lc2_radius < best_radius:
                best_points = lc2_points
                best_radius = lc2_radius
                
    return best_points

def _fibonacci_sphere(n):
    """Generate points on a sphere using the Fibonacci lattice."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
# EVOLVE-BLOCK-END
