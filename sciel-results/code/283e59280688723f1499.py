import numpy as np
from scipy.spatial import Delaunay
import time

N = 50

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return points / norms

def _covering_radius(points):
    """
    Computes the covering radius of a set of points on the unit sphere.
    The covering radius is the maximum distance from any point on the sphere to the nearest site.
    This is equal to the maximum circumradius of the Delaunay triangles (Voronoi vertices).
    """
    try:
        tri = Delaunay(points)
        simplices = tri.simplices
        
        p_i = points[simplices[:, 0]]
        p_j = points[simplices[:, 1]]
        p_k = points[simplices[:, 2]]
        
        # Compute circumcenters (Voronoi vertices)
        # The circumcenter of a spherical triangle is the normalized cross product of two edge vectors.
        # We need to ensure the direction is correct (pointing towards the triangle).
        
        v1 = p_i - p_j
        v2 = p_i - p_k
        normals = np.cross(v1, v2)
        
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        # Handle degenerate triangles
        norms[norms == 0] = 1.0
        circumcenters = normals / norms
        
        # Ensure circumcenters point towards the triangle centroid
        centroids = (p_i + p_j + p_k) / 3.0
        dots = np.sum(circumcenters * centroids, axis=1)
        circumcenters[dots < 0] *= -1
        
        # The distance from the circumcenter to the nearest site is the angular distance to any of the 3 vertices.
        # Since the circumcenter is equidistant to all 3, we just compute the dot product with one.
        # dist = arccos(circumcenter . p_i)
        dots_cc = np.sum(circumcenters * p_i, axis=1)
        dots_cc = np.clip(dots_cc, -1.0, 1.0)
        dists = np.arccos(dots_cc)
        
        return np.max(dists)
    except Exception:
        return np.inf

def _lloyd_step(points, num_steps=10):
    """
    Performs Lloyd's algorithm steps to converge to a local minimum of the covering radius.
    """
    P = points.copy()
    for _ in range(num_steps):
        try:
            tri = Delaunay(P)
            simplices = tri.simplices
            
            new_P = np.zeros_like(P)
            counts = np.zeros(N)
            
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
            new_P = new_P / counts[:, np.newaxis]
            new_P = _normalize(new_P)
            
            # Check convergence
            if np.linalg.norm(new_P - P) < 1e-8:
                P = new_P
                break
            P = new_P
        except Exception:
            break
    return P

def _fibonacci_points(N):
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    P = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(P)

def _random_points(N, seed=None):
    if seed is not None:
        np.random.seed(seed)
    P = np.random.randn(N, 3)
    return _normalize(P)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_P = None
    best_radius = np.inf
    
    # List of initial configurations
    inits = []
    inits.append(_fibonacci_points(N))
    inits.append(_random_points(N, seed=42))
    inits.append(_random_points(N, seed=123))
    
    # Add a perturbed Fibonacci
    fib = _fibonacci_points(N)
    fib_pert = fib + 0.1 * np.random.randn(N, 3)
    inits.append(_normalize(fib_pert))
    
    for init_P in inits:
        if time.time() > deadline:
            break
            
        # 1. Initial Lloyd's relaxation
        P = _lloyd_step(init_P, num_steps=50)
        current_radius = _covering_radius(P)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_P = P.copy()
            
        # 2. Simulated Annealing
        # We will run SA for a fixed number of iterations or until time runs out
        T = 0.05  # Initial temperature (radians)
        T_min = 0.001
        cooling_rate = 0.995
        max_sa_iters = 2000
        
        for sa_iter in range(max_sa_iters):
            if time.time() > deadline:
                break
                
            # Pick a random point to perturb
            idx = np.random.randint(N)
            original_P = P[idx].copy()
            
            # Perturb the point
            # Move it in a random direction on the sphere
            random_vec = np.random.randn(3)
            # Ensure it's not parallel to P[idx]
            if np.dot(random_vec, P[idx]) > 0.99:
                random_vec = np.random.randn(3)
            
            # New point is normalized sum of original and small random step
            step_size = 0.05
            new_vec = P[idx] + step_size * random_vec
            new_P = _normalize(new_vec.reshape(1,3))[0]
            
            P[idx] = new_P
            
            new_radius = _covering_radius(P)
            
            # Acceptance criterion
            if new_radius < current_radius:
                current_radius = new_radius
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_P = P.copy()
            else:
                # Reject with probability
                delta = new_radius - current_radius
                if delta < 1e-10:
                    # Accept if equal (or very close) to allow exploration
                    pass
                else:
                    prob = np.exp(-delta / T)
                    if np.random.rand() > prob:
                        # Reject
                        P[idx] = original_P
            
            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min
                
            # Periodically apply Lloyd's step to re-converge
            if sa_iter % 50 == 49:
                P = _lloyd_step(P, num_steps=5)
                current_radius = _covering_radius(P)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_P = P.copy()
                
                # Reheat if stuck?
                if current_radius > best_radius * 1.01:
                    T = 0.02

    # Final Lloyd's refinement on the best found
    if best_P is not None and time.time() < deadline - 10:
        best_P = _lloyd_step(best_P, num_steps=100)
        final_radius = _covering_radius(best_P)
        if final_radius < best_radius:
            best_radius = final_radius
    
    return best_P

