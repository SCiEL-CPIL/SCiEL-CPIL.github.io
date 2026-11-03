"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
import time
import sys

def _normalize(P):
    """Normalize rows of P to unit length."""
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return P / norms

def _fibonacci_points(N):
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    P = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(P)

def _random_points(N, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    P = rng.randn(N, 3)
    return _normalize(P)

def _icosahedron_based_points(N):
    """Start from icosahedron vertices and subdivide to get N points."""
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    verts = _normalize(verts)
    
    # If N == 12, return icosahedron
    if N == 12:
        return verts
    
    # Subdivide: add midpoints of edges, then more points
    # For N > 12, add random points near existing ones
    P = verts.copy()
    while len(P) < N:
        # Pick a random existing point and add a perturbed version
        idx = np.random.randint(len(P))
        new_pt = P[idx] + 0.1 * np.random.randn(3)
        new_pt = new_pt / np.linalg.norm(new_pt)
        P = np.vstack([P, new_pt.reshape(1, 3)])
    
    return P[:N]

def _covering_radius(P):
    """Compute covering radius using Delaunay circumcenters."""
    try:
        tri = Delaunay(P)
        simplices = tri.simplices
        if len(simplices) == 0:
            return np.inf
        
        p_i = P[simplices[:, 0]]
        p_j = P[simplices[:, 1]]
        p_k = P[simplices[:, 2]]
        
        v1 = p_i - p_j
        v2 = p_i - p_k
        normals = np.cross(v1, v2)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        circumcenters = normals / norms
        
        # Ensure circumcenter is on the correct side
        centroids = (p_i + p_j + p_k) / 3.0
        dots = np.sum(circumcenters * centroids, axis=1)
        circumcenters[dots < 0] *= -1
        
        # Distance to nearest of the 3 defining points
        dists = np.arccos(np.clip(np.sum(circumcenters * p_i, axis=1), -1.0, 1.0))
        return np.max(dists) * 180.0 / np.pi
    except Exception:
        return np.inf

def _local_optimize(P, time_limit=20.0):
    """Local optimization via gradient descent on covering radius."""
    start = time.time()
    best_P = P.copy()
    best_r = _covering_radius(P)
    
    # Use a simple iterative approach: move each point toward the centroid of its
    # Voronoi cell (spherical Lloyd) but with momentum and adaptive step size
    P = P.copy()
    velocity = np.zeros_like(P)
    lr = 0.1
    momentum = 0.9
    
    for it in range(2000):
        if time.time() - start > time_limit:
            break
        
        try:
            tri = Delaunay(P)
            simplices = tri.simplices
        except Exception:
            break
        
        if len(simplices) == 0:
            break
        
        p_i = P[simplices[:, 0]]
        p_j = P[simplices[:, 1]]
        p_k = P[simplices[:, 2]]
        
        v1 = p_i - p_j
        v2 = p_i - p_k
        normals = np.cross(v1, v2)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        circumcenters = normals / norms
        
        centroids = (p_i + p_j + p_k) / 3.0
        dots = np.sum(circumcenters * centroids, axis=1)
        circumcenters[dots < 0] *= -1
        
        # Accumulate centroids for each point
        new_P = np.zeros_like(P)
        counts = np.zeros(N)
        idx_i = simplices[:, 0]
        idx_j = simplices[:, 1]
        idx_k = simplices[:, 2]
        
        np.add.at(new_P, idx_i, circumcenters)
        np.add.at(new_P, idx_j, circumcenters)
        np.add.at(new_P, idx_k, circumcenters)
        
        np.add.at(counts, idx_i, 1)
        np.add.at(counts, idx_j, 1)
        np.add.at(counts, idx_k, 1)
        
        counts[counts == 0] = 1.0
        centroid = new_P / counts[:, np.newaxis]
        
        # Spherical gradient step: move toward centroid
        # The gradient of the covering radius w.r.t. each point points away from
        # the worst circumcenter. Here we use the Lloyd update as a proxy.
        diff = centroid - P
        # Project onto tangent space
        P_norm = np.linalg.norm(P, axis=1, keepdims=True)
        P_unit = P / P_norm
        diff_tangent = diff - np.sum(diff * P_unit, axis=1, keepdims=True) * P_unit
        
        velocity = momentum * velocity + lr * diff_tangent
        P = P + velocity
        P = _normalize(P)
        
        r = _covering_radius(P)
        if r < best_r:
            best_r = r
            best_P = P.copy()
        else:
            # Revert and reduce learning rate
            P = best_P.copy()
            lr *= 0.9
            if lr < 1e-6:
                break
    
    return best_P, best_r

def _kick_and_optimize(P, kick_size, time_limit):
    """Apply a random kick and then locally optimize."""
    P_kicked = P + kick_size * np.random.randn(N, 3)
    P_kicked = _normalize(P_kicked)
    return _local_optimize(P_kicked, time_limit)

def construct():
    """
    Constructs a configuration of 50 points on the unit sphere minimizing the covering radius.
    Uses multi-start local optimization with diverse initializations.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_P = None
    best_r = np.inf
    
    # Strategy: multiple restarts from different initializations
    initializations = []
    
    # 1. Fibonacci spiral
    initializations.append(('fib', _fibonacci_points(N)))
    
    # 2. Icosahedron-based
    initializations.append(('ico', _icosahedron_based_points(N)))
    
    # 3. Random starts
    for seed in [42, 123, 456, 789, 1000, 2000]:
        initializations.append((f'rand_{seed}', _random_points(N, seed=seed)))
    
    # 4. Perturbed Fibonacci
    fib = _fibonacci_points(N)
    for kick in [0.05, 0.1, 0.2]:
        P_pert = fib + kick * np.random.randn(N, 3)
        initializations.append((f'fib_kick_{kick}', _normalize(P_pert)))
    
    for name, P_init in initializations:
        if time.time() > deadline - 15.0:
            break
        
        # First, try a quick local optimization
        P_opt, r_opt = _local_optimize(P_init, time_limit=10.0)
        
        if r_opt < best_r:
            best_r = r_opt
            best_P = P_opt.copy()
        
        # If we have time, try kicks from the best so far
        if best_P is not None and time.time() < deadline - 20.0:
            for kick_size in [0.05, 0.1, 0.15, 0.2]:
                if time.time() > deadline - 10.0:
                    break
                P_kicked, r_kicked = _kick_and_optimize(best_P, kick_size, time_limit=8.0)
                if r_kicked < best_r:
                    best_r = r_kicked
                    best_P = P_kicked.copy()
    
    # Final polish: one more local optimization on the best
    if best_P is not None and time.time() < deadline - 5.0:
        P_final, r_final = _local_optimize(best_P, time_limit=min(10.0, deadline - time.time() - 2.0))
        if r_final < best_r:
            best_r = r_final
            best_P = P_final.copy()
    
    if best_P is None:
        best_P = _fibonacci_points(N)
    
    return best_P
# EVOLVE-BLOCK-END
