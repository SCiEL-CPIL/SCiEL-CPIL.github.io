"""Rewritten program for Heilbronn convex regions n=14 using Powell local optimization from diverse starts."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools

N = 14

# EVOLVE-BLOCK-START

def _triangle_areas(points):
    """
    Calculate areas of all triangles formed by triples of points.
    Returns an array of areas.
    """
    n = points.shape[0]
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    
    indices = np.array(indices)
    pi = points[indices[:, 0]]
    pj = points[indices[:, 1]]
    pk = points[indices[:, 2]]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return areas

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    try:
        if np.any(np.isnan(points)):
            return 0.0
        
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
        
        min_area = np.min(_triangle_areas(points))
        return min_area / hull_area
    except Exception:
        return 0.0

def _score_from_angles(angles):
    """
    Calculate score given angles on the unit circle.
    angles: array of shape (N,)
    """
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    return _score(points)

def _negative_score_from_angles(angles):
    """
    Objective function for minimization on the manifold of angles.
    """
    return -_score_from_angles(angles)

def _powell_from_angles(start_angles, maxiter=5000, tol=1e-8):
    """
    Run Powell optimization from a given set of angles.
    Returns (score, best_angles).
    """
    def objective(x):
        return _negative_score_from_angles(x)
    
    try:
        res = minimize(objective, start_angles, method='Powell',
                       options={'maxiter': maxiter, 'ftol': tol, 'xtol': tol})
        score = -res.fun
        if score > 0 and not np.isnan(score):
            return score, res.x
        else:
            return 0.0, start_angles
    except Exception:
        return 0.0, start_angles

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (limit 150s)
    
    best_score = -np.inf
    best_angles = None
    
    # Regular N-gon angles
    regular_angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    # Strategy: Multiple Powell local optimizations from diverse initializations
    # Powell is effective for smooth objectives and converges well
    
    # Generate diverse starting points
    rng = np.random.default_rng(42)
    
    # Start 1: Regular polygon
    starts = [regular_angles.copy()]
    
    # Start 2-10: Jittered regular polygon with varying jitter magnitudes
    for jitter_mag in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        jittered = regular_angles + rng.normal(0, jitter_mag, N)
        # Normalize to stay in [0, 2*pi)
        jittered = jittered % (2 * np.pi)
        starts.append(jittered)
    
    # Start 11-20: Random angles (uniform distribution)
    for _ in range(10):
        random_angles = rng.uniform(0, 2*np.pi, N)
        starts.append(random_angles)
    
    # Start 21-30: Clustered angles (some points close together)
    for _ in range(10):
        base = rng.uniform(0, 2*np.pi, N)
        # Add some clustering by taking differences and making some small
        diffs = np.diff(np.sort(base))
        # Make some diffs small
        for i in range(rng.integers(2, 5)):
            diffs[rng.integers(0, len(diffs))] *= 0.1
        # Reconstruct
        sorted_angles = np.sort(base)
        reconstructed = np.concatenate(([sorted_angles[0]], sorted_angles[0] + np.cumsum(diffs)))
        reconstructed = reconstructed % (2 * np.pi)
        starts.append(reconstructed)
    
    # Run Powell optimization from each start
    for i, start in enumerate(starts):
        if time.time() > deadline:
            break
        
        # Time per optimization: allocate remaining time / remaining starts
        remaining_starts = len(starts) - i
        time_per_opt = (deadline - time.time()) / max(remaining_starts, 1)
        # Cap maxiter based on time available (rough estimate: ~0.01s per iteration)
        maxiter = int(min(10000, max(100, time_per_opt / 0.01)))
        
        score, angles = _powell_from_angles(start, maxiter=maxiter)
        
        if score > best_score:
            best_score = score
            best_angles = angles
    
    # If optimization failed or score is bad, fallback to regular polygon
    if best_angles is None or best_score < 0.1:
        best_angles = regular_angles
    
    # Final refinement: Run Powell again from the best result with tighter tolerance
    if time.time() < deadline - 10:
        final_score, final_angles = _powell_from_angles(best_angles, maxiter=20000, tol=1e-10)
        if final_score > best_score:
            best_score = final_score
            best_angles = final_angles
    
    # Convert best angles to points
    final_points = np.column_stack([np.cos(best_angles), np.sin(best_angles)])
    
    return final_points
# EVOLVE-BLOCK-END
