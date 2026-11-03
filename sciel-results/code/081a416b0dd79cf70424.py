"""Improved program: vectorized gradient ascent for circle packing."""
import numpy as np
import time

N = 300
TIME_LIMIT = 110.0  # seconds, safety margin below 120s


def _score(points):
    """Compute the maximum common radius for given centres."""
    # Distance between all pairs
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    
    # Distance from centre to boundary
    radii = np.sqrt(np.sum(points * points, axis=1))
    min_boundary = 1.0 - np.max(radii)
    
    r = min(min_pair_dist / 2.0, min_boundary)
    return r


def _gradient(points, r):
    """Compute gradient of the score with respect to points.
    
    We want to maximize r = min(min_pair_dist/2, min_boundary).
    Use a soft approach: gradient of each constraint, weighted by how binding it is.
    """
    n = len(points)
    grad = np.zeros_like(points)
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2) + 1e-30)
    np.fill_diagonal(dist, np.inf)
    
    # Find the minimum pair distance
    min_pair_dist = np.min(dist)
    min_boundary = 1.0 - np.max(np.sqrt(np.sum(points * points, axis=1)))
    
    r_val = min(min_pair_dist / 2.0, min_boundary)
    
    # Gradient from pairwise constraints
    # For each pair (i,j) where dist_ij is close to 2*r, push apart
    if min_pair_dist / 2.0 <= min_boundary:
        # Pairwise constraint is binding
        # Soft weighting: use exp(-(dist - 2r)^2 / sigma^2) to focus on near-binding pairs
        sigma = 0.01
        weights = np.exp(-((dist - 2.0 * r_val) ** 2) / (2.0 * sigma ** 2))
        np.fill_diagonal(weights, 0.0)
        
        # Gradient contribution: for pair (i,j), direction is (points[i] - points[j]) / dist
        # We want to increase dist, so gradient for point i is +(points[i]-points[j])/dist
        # and for point j is -(points[i]-points[j])/dist
        # Normalized direction
        dir_norm = diff / (dist + 1e-30)[:, :, np.newaxis]
        
        # Sum weighted contributions
        grad += np.sum(weights[:, :, np.newaxis] * dir_norm, axis=1)
        # Each pair counted twice (i,j) and (j,i), so divide by 2
        grad /= 2.0
        # Scale by 0.5 because r = dist/2
        grad *= 0.5
    else:
        # Boundary constraint is binding
        # Push all points inward
        radii = np.sqrt(np.sum(points * points, axis=1)) + 1e-30
        grad -= points / radii[:, np.newaxis]
    
    return grad, r_val


def _initial_hex():
    """Generate initial hexagonal lattice configuration."""
    spacing = 0.106
    points = []
    for j in range(-30, 31):
        for i in range(-30, 31):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x*x + y*y <= (1.0 - spacing/2.0)**2:
                points.append([x, y])
    
    points = np.array(points)
    # Sort by radius and take first N
    radii = np.sqrt(np.sum(points*points, axis=1))
    points = points[np.argsort(radii, kind='stable')[:N]]
    return points


def construct():
    start_time = time.time()
    
    # Initial configuration
    points = _initial_hex()
    best_points = points.copy()
    best_r = _score(points)
    
    # Adaptive step size
    step = 0.001
    prev_r = best_r
    
    # Gradient ascent with adaptive step size
    iteration = 0
    while time.time() - start_time < TIME_LIMIT:
        grad, r_val = _gradient(points, best_r)
        
        # Adaptive step: increase if improving, decrease if not
        new_r = _score(points + step * grad)
        if new_r > r_val:
            points = points + step * grad
            best_points = points.copy()
            best_r = new_r
            step *= 1.05
        else:
            step *= 0.7
            if step < 1e-8:
                break
        
        iteration += 1
        if iteration % 100 == 0:
            if time.time() - start_time > TIME_LIMIT - 10:
                break
    
    # Final polish: try smaller steps
    step = 1e-5
    for _ in range(50):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        grad, r_val = _gradient(points, best_r)
        new_points = points + step * grad
        new_r = _score(new_points)
        if new_r > best_r:
            points = new_points
            best_points = points.copy()
            best_r = new_r
        else:
            step *= 0.5
            if step < 1e-10:
                break
    
    # Verify validity
    radii = np.sqrt(np.sum(best_points * best_points, axis=1))
    if np.any(radii >= 1.0):
        # Scale down to be valid
        scale = 0.999 / np.max(radii)
        best_points *= scale
    
    return best_points
