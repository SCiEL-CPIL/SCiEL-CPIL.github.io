"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random

def _sphere_radius(centers):
    """Compute the maximum radius for given centers in [-0.5, 0.5]^3."""
    # Min distance between pairs
    min_pair_dist = float('inf')
    n = len(centers)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_pair_dist:
                min_pair_dist = d
    
    # Min distance to boundary
    min_boundary = float('inf')
    for c in centers:
        for k in range(3):
            dist = 0.5 - abs(c[k])
            if dist < min_boundary:
                min_boundary = dist
    
    return min(min_pair_dist / 2.0, min_boundary)

def construct():
    # Use a body-centered cubic (BCC) inspired lattice with perturbation
    # and local optimization to improve packing density.
    
    # Start with a well-known good packing structure: 
    # We'll use a combination of grid-based initialization with BCC-like offsets
    # and then refine with local search.
    
    np.random.seed(42)
    
    # Initialize with a 4x4x4 grid (64 positions) and pick 50, 
    # but use BCC-like offsets for better spacing
    m = 4
    base_ticks = (np.arange(m) + 0.5) / m - 0.5
    
    # Generate all grid points
    grid_points = []
    for z in base_ticks:
        for y in base_ticks:
            for x in base_ticks:
                grid_points.append([x, y, z])
    
    # Add BCC-like offset points (center of each small cube)
    offset = 0.5 / m
    bcc_points = []
    for z in base_ticks:
        for y in base_ticks:
            for x in base_ticks:
                bcc_points.append([x + offset, y + offset, z + offset])
    
    # Combine and select 50 points that maximize minimum distance
    all_points = grid_points + bcc_points
    n_total = len(all_points)
    
    # Greedy selection: pick points that maximize minimum distance
    selected = []
    remaining = list(range(n_total))
    
    # Start from center-ish
    center_idx = n_total // 2
    selected.append(center_idx)
    remaining.remove(center_idx)
    
    for _ in range(N - 1):
        best_idx = -1
        best_min_dist = -1
        for idx in remaining:
            pt = np.array(all_points[idx])
            min_d = float('inf')
            for sel_idx in selected:
                sel_pt = np.array(all_points[sel_idx])
                d = np.linalg.norm(pt - sel_pt)
                if d < min_d:
                    min_d = d
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)
    
    centers = np.array([all_points[i] for i in selected])
    
    # Local optimization using simulated annealing-like approach
    current_radius = _sphere_radius(centers)
    best_centers = centers.copy()
    best_radius = current_radius
    
    # Simulated annealing
    T = 0.01
    T_min = 1e-6
    cooling = 0.995
    max_iter = 5000
    
    for it in range(max_iter):
        if T < T_min:
            break
        
        # Perturb a random point
        i = np.random.randint(N)
        delta = np.random.normal(0, T, 3)
        new_centers = centers.copy()
        new_centers[i] = centers[i] + delta
        
        # Check bounds
        if np.any(np.abs(new_centers[i]) > 0.5):
            continue
        
        new_radius = _sphere_radius(new_centers)
        
        # Accept if better or with probability exp((new - old)/T)
        if new_radius > current_radius:
            centers = new_centers
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = centers.copy()
        else:
            prob = np.exp((new_radius - current_radius) / T)
            if np.random.random() < prob:
                centers = new_centers
                current_radius = new_radius
        
        T *= cooling
    
    return best_centers

# EVOLVE-BLOCK-END
