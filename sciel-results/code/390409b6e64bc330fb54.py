"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import math
import random

def _covering_radius(points):
    """Compute the covering radius (in radians) as the max circumradius of convex hull facets."""
    try:
        hull = ConvexHull(points)
        max_r = 0.0
        for simplex in hull.simplices:
            p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            v = p2 - p1
            w = p3 - p1
            vv = np.dot(v, v)
            ww = np.dot(w, w)
            vw = np.dot(v, w)
            det = vv * ww - vw * vw
            if abs(det) < 1e-12:
                continue
            t1 = (0.5 * vv * ww - 0.5 * ww * vw) / det
            t2 = (0.5 * ww * vv - 0.5 * vv * vw) / det
            C_plane = p1 + t1 * v + t2 * w
            norm_C = np.linalg.norm(C_plane)
            if norm_C < 1e-12:
                continue
            C = C_plane / norm_C
            cos_R = np.clip(np.dot(C, p1), -1.0, 1.0)
            R = np.arccos(cos_R)
            if R > max_r:
                max_r = R
        return max_r
    except Exception:
        return np.pi

def _compute_facet_circumcenters(points, hull):
    """Compute circumcenters and radii for all hull facets."""
    facet_radii = []
    facet_centers = []
    for simplex in hull.simplices:
        p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        v = p2 - p1
        w = p3 - p1
        vv = np.dot(v, v)
        ww = np.dot(w, w)
        vw = np.dot(v, w)
        det = vv * ww - vw * vw
        if abs(det) < 1e-12:
            facet_radii.append(0.0)
            facet_centers.append(np.zeros(3))
            continue
        t1 = (0.5 * vv * ww - 0.5 * ww * vw) / det
        t2 = (0.5 * ww * vv - 0.5 * vv * vw) / det
        C_plane = p1 + t1 * v + t2 * w
        norm_C = np.linalg.norm(C_plane)
        if norm_C < 1e-12:
            facet_radii.append(0.0)
            facet_centers.append(np.zeros(3))
            continue
        C = C_plane / norm_C
        cos_R = np.clip(np.dot(C, p1), -1.0, 1.0)
        R = np.arccos(cos_R)
        facet_radii.append(R)
        facet_centers.append(C)
    return np.array(facet_radii), np.array(facet_centers)

def _local_polish(points, iterations=300):
    """Local optimization using circumcenter moves with adaptive step size."""
    points = points.copy()
    best_points = points.copy()
    best_energy = _covering_radius(points)
    
    for it in range(iterations):
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        facet_radii, facet_centers = _compute_facet_circumcenters(points, hull)
        max_idx = np.argmax(facet_radii)
        max_r = facet_radii[max_idx]
        
        if max_r < 0.197:
            break
        
        simplex = hull.simplices[max_idx]
        idx1, idx2, idx3 = simplex
        
        C = facet_centers[max_idx]
        
        # Adaptive step size: larger when radius is large
        step = 0.002 + 0.008 * max_r
        
        # Move all three vertices toward the circumcenter
        for idx in [idx1, idx2, idx3]:
            p = points[idx]
            new_p = p + step * (C - p)
            new_p = new_p / np.linalg.norm(new_p)
            points[idx] = new_p
        
        energy = _covering_radius(points)
        if energy < best_energy:
            best_energy = energy
            best_points = points.copy()
    
    return best_points

def _repulsion_init(rng, n_points=130, iterations=2000, alpha=1.0):
    """Initialize points using repulsion-based method."""
    points = rng.normal(size=(n_points, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    for it in range(iterations):
        # Compute pairwise distances
        dists = np.zeros((n_points, n_points))
        for i in range(n_points):
            for j in range(i+1, n_points):
                d = np.linalg.norm(points[i] - points[j])
                dists[i, j] = d
                dists[j, i] = d
        
        # Compute repulsive forces
        forces = np.zeros((n_points, 3))
        for i in range(n_points):
            for j in range(n_points):
                if i == j:
                    continue
                d = dists[i, j]
                if d < 1e-10:
                    continue
                # Repulsion proportional to 1/d^alpha
                force_mag = alpha / (d ** 2)
                force_dir = (points[i] - points[j]) / d
                forces[i] += force_mag * force_dir
        
        # Move points
        step = 0.01 * (1.0 - it / iterations) + 0.001
        for i in range(n_points):
            new_p = points[i] + step * forces[i]
            new_p = new_p / np.linalg.norm(new_p)
            points[i] = new_p
    
    return points

def construct():
    rng = np.random.default_rng(42)
    
    # Try multiple initializations and pick the best
    candidates = []
    
    # Initialization 1: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    candidates.append(fib_points)
    
    # Initialization 2: Repulsion-based
    rep_points = _repulsion_init(rng, N, iterations=1000, alpha=0.5)
    candidates.append(rep_points)
    
    # Initialization 3: Random with local polish
    rand_points = rng.normal(size=(N, 3))
    rand_points /= np.linalg.norm(rand_points, axis=1, keepdims=True)
    rand_points = _local_polish(rand_points, iterations=50)
    candidates.append(rand_points)
    
    # Pick best candidate
    best_init = None
    best_init_energy = np.pi
    for cand in candidates:
        e = _covering_radius(cand)
        if e < best_init_energy:
            best_init_energy = e
            best_init = cand.copy()
    
    # Multi-scale simulated annealing
    current_points = best_init.copy()
    current_energy = best_init_energy
    best_points = current_points.copy()
    best_energy = current_energy
    
    # Phase 1: Large steps
    T_start = 0.05
    T_end = 0.005
    n_iter1 = 3000
    for i_iter in range(n_iter1):
        T = T_start * (T_end / T_start) ** (i_iter / n_iter1)
        idx = rng.integers(0, N)
        move_step = 0.05 * (T / T_start + 0.1)
        noise = rng.normal(size=3)
        noise /= np.linalg.norm(noise)
        new_points = current_points.copy()
        new_points[idx] = new_points[idx] + move_step * noise
        new_points[idx] /= np.linalg.norm(new_points[idx])
        new_energy = _covering_radius(new_points)
        delta_E = new_energy - current_energy
        if delta_E < 0 or rng.random() < math.exp(-delta_E / T):
            current_points = new_points
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = current_points.copy()
    
    # Phase 2: Medium steps
    T_start = 0.005
    T_end = 0.001
    n_iter2 = 3000
    for i_iter in range(n_iter2):
        T = T_start * (T_end / T_start) ** (i_iter / n_iter2)
        idx = rng.integers(0, N)
        move_step = 0.02 * (T / T_start + 0.1)
        noise = rng.normal(size=3)
        noise /= np.linalg.norm(noise)
        new_points = current_points.copy()
        new_points[idx] = new_points[idx] + move_step * noise
        new_points[idx] /= np.linalg.norm(new_points[idx])
        new_energy = _covering_radius(new_points)
        delta_E = new_energy - current_energy
        if delta_E < 0 or rng.random() < math.exp(-delta_E / T):
            current_points = new_points
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = current_points.copy()
    
    # Phase 3: Small steps
    T_start = 0.001
    T_end = 0.0001
    n_iter3 = 2000
    for i_iter in range(n_iter3):
        T = T_start * (T_end / T_start) ** (i_iter / n_iter3)
        idx = rng.integers(0, N)
        move_step = 0.008 * (T / T_start + 0.1)
        noise = rng.normal(size=3)
        noise /= np.linalg.norm(noise)
        new_points = current_points.copy()
        new_points[idx] = new_points[idx] + move_step * noise
        new_points[idx] /= np.linalg.norm(new_points[idx])
        new_energy = _covering_radius(new_points)
        delta_E = new_energy - current_energy
        if delta_E < 0 or rng.random() < math.exp(-delta_E / T):
            current_points = new_points
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = current_points.copy()
    
    # Final local polish
    polished = _local_polish(best_points, iterations=200)
    polished_energy = _covering_radius(polished)
    
    if polished_energy < best_energy:
        best_points = polished
        best_energy = polished_energy
    
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    return best_points
# EVOLVE-BLOCK-END
