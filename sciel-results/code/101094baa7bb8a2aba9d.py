"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time
import math

N = 150
DEADLINE = 115.0  # safety margin before 120s kill

def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for N points on unit sphere.
    
    Energy = sum_{i<j} log(1/|xi - xj|) = -sum_{i<j} log(|xi - xj|)
    Gradient w.r.t. xi: sum_{j!=i} (xi - xj) / |xi - xj|^2
    """
    n = len(pts)
    # Compute pairwise distances
    # dist[i,j] = |xi - xj|
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist, 1.0)
    
    # Energy: -sum_{i<j} log(dist[i,j])
    # Use upper triangle only
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: grad_i = sum_{j!=i} (xi - xj) / |xi - xj|^2
    # Note: d/dxi [-log|xi-xj|] = -(xi-xj)/|xi-xj|^2 ... wait
    # d/dxi log|xi-xj| = (xi-xj)/|xi-xj|^2
    # So d/dxi [-log|xi-xj|] = -(xi-xj)/|xi-xj|^2
    # But we want to MINIMIZE energy, so gradient of energy is what we need.
    # energy = -sum log(dist), so grad_i = -sum_{j!=i} (xi-xj)/dist[i,j]^2
    # Wait, let me re-derive:
    # E = sum_{i<j} log(1/dist) = -sum_{i<j} log(dist)
    # dE/dxi = -sum_{j!=i, j>i or j<i} d(log(dist_ij))/dxi
    # For each pair (i,j) with i<j: dE/dxi = -d(log(dist))/dxi = -(xi-xj)/dist^2
    #                            dE/dxj = -d(log(dist))/dxj = -(xj-xi)/dist^2 = (xi-xj)/dist^2
    # So for point i: grad_i = -sum_{j!=i} (xi - xj) / dist[i,j]^2
    #                    = sum_{j!=i} (xj - xi) / dist[i,j]^2
    
    # Compute gradient efficiently
    # grad[i] = sum_{j!=i} (xj - xi) / dist[i,j]^2
    # = sum_{j!=i} xj/dist[i,j]^2 - xi * sum_{j!=i} 1/dist[i,j]^2
    
    inv_dist_sq = np.zeros_like(dist)
    np.fill_diagonal(inv_dist_sq, 0.0)
    np.divide(1.0, dist * dist, out=inv_dist_sq, where=dist > 1e-15)
    
    # sum_xj_over_dist_sq[i] = sum_j xj * inv_dist_sq[i,j]
    sum_xj = pts.T @ inv_dist_sq  # shape (3, n)
    sum_xj = sum_xj.T  # shape (n, 3)
    
    sum_inv = np.sum(inv_dist_sq, axis=1)  # shape (n,)
    
    grad = sum_xj - pts * sum_inv[:, np.newaxis]
    
    return energy, grad

def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _fibonacci_sphere(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _geodesic_dome(n_target):
    """Generate points via subdivided icosahedron, then adjust to n_target."""
    # Start with icosahedron vertices
    t = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Faces of icosahedron
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    # Subdivide until we have enough points
    # Each subdivision multiplies faces by 4, points roughly by 4
    # Start: 12 points, 20 faces
    # After 1 sub: ~42 points, 80 faces
    # After 2 sub: ~162 points, 320 faces
    # After 3 sub: ~642 points, 1280 faces
    
    # We want ~150 points, so 2 subdivisions gives ~162, which is close
    for level in range(2):
        new_verts = [list(v) for v in verts]
        new_faces = []
        mid_cache = {}
        
        def midpoint(v1, v2):
            key = (tuple(v1), tuple(v2))
            if key in mid_cache:
                return mid_cache[key]
            mid = (np.array(v1) + np.array(v2)) / 2.0
            mid = mid / np.linalg.norm(mid)
            idx = len(new_verts)
            new_verts.append(list(mid))
            mid_cache[key] = idx
            return idx
        
        for face in faces:
            a, b, c = face
            ab = midpoint(verts[a], verts[b])
            bc = midpoint(verts[b], verts[c])
            ca = midpoint(verts[c], verts[a])
            new_faces.append([a, ab, ca])
            new_faces.append([b, bc, ab])
            new_faces.append([c, ca, bc])
            new_faces.append([ab, bc, ca])
        
        verts = np.array(new_verts)
        faces = new_faces
    
    # Now we have ~162 points. If we have more than n_target, randomly select n_target
    if len(verts) >= n_target:
        # Select n_target points (use first n_target for determinism, or random)
        rng = np.random.default_rng(42)
        idx = rng.choice(len(verts), size=n_target, replace=False)
        return verts[idx]
    else:
        # If too few, pad with fibonacci points
        fib = _fibonacci_sphere(n_target)
        return fib

def construct():
    start_time = time.time()
    
    # Generate multiple diverse starting points
    starts = []
    
    # 1. Fibonacci spiral
    starts.append(_fibonacci_sphere(N))
    
    # 2. Geodesic dome
    try:
        starts.append(_geodesic_dome(N))
    except:
        pass
    
    # 3-11. Perturbed Fibonacci with different random seeds
    for seed in range(9):
        rng = np.random.default_rng(seed + 100)
        pts = _fibonacci_sphere(N)
        # Add small random perturbation
        noise = rng.normal(0, 0.1, size=pts.shape)
        pts = _project_to_sphere(pts + noise)
        starts.append(pts)
    
    # Also add a few more diverse starts
    for seed in range(5):
        rng = np.random.default_rng(seed + 200)
        pts = rng.normal(0, 1, size=(N, 3))
        pts = _project_to_sphere(pts)
        starts.append(pts)
    
    best_energy = np.inf
    best_pts = starts[0].copy()
    
    # Phase 1: L-BFGS-B from each start
    for start_idx, start_pts in enumerate(starts):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            break
        
        try:
            result = minimize(
                _energy_and_grad,
                start_pts,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < best_energy:
                # Project back to sphere
                result_pts = _project_to_sphere(result.x)
                e, _ = _energy_and_grad(result_pts)
                if e < best_energy:
                    best_energy = e
                    best_pts = result_pts.copy()
        except:
            pass
    
    # Phase 2: Simulated annealing on the best result
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 10:
        pts = best_pts.copy()
        current_energy = best_energy
        
        # SA parameters
        T_start = 0.1
        T_end = 0.001
        n_iter = min(5000, int((DEADLINE - 10 - elapsed) * 100))
        n_iter = max(n_iter, 1000)
        
        rng = np.random.default_rng(12345)
        
        for it in range(n_iter):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            T = T_start * (T_end / T_start) ** (it / n_iter)
            
            # Pick a random point to move
            i = rng.integers(N)
            old_pt = pts[i].copy()
            
            # Random direction on tangent space
            rand_dir = rng.normal(0, 1, size=3)
            # Project to tangent space: remove component along pts[i]
            rand_dir = rand_dir - np.dot(rand_dir, pts[i]) * pts[i]
            norm_dir = np.linalg.norm(rand_dir)
            if norm_dir < 1e-10:
                continue
            rand_dir = rand_dir / norm_dir
            
            # Step size proportional to temperature
            step = T * 0.5
            new_pt = _project_to_sphere(old_pt + step * rand_dir)
            
            # Compute energy difference (approximate: only affected pairs)
            # Full recompute for accuracy
            pts[i] = new_pt
            new_energy, _ = _energy_and_grad(pts)
            
            delta_e = new_energy - current_energy
            if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 1e-10)):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pts = pts.copy()
            else:
                pts[i] = old_pt
    
    # Phase 3: Final high-precision polish
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 2:
        try:
            result = minimize(
                _energy_and_grad,
                best_pts,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            if result.fun < best_energy:
                result_pts = _project_to_sphere(result.x)
                e, _ = _energy_and_grad(result_pts)
                if e < best_energy:
                    best_energy = e
                    best_pts = result_pts.copy()
        except:
            pass
    
    # Ensure final points are on unit sphere and valid
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Nudge apart
                best_pts[j] = _project_to_sphere(best_pts[j] + 1e-6 * (best_pts[j] - best_pts[i]))
    
    return best_pts
