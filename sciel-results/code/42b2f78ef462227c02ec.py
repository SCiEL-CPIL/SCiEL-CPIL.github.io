"""Rewritten program for packing 300 equal circles in a unit circle."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import math

def _hex_grid(n):
    """Generate a hexagonal grid of points roughly covering the unit disk."""
    # Start with a hexagonal lattice that has enough points
    # We'll generate points on a hexagonal lattice and filter to the unit disk
    # The spacing needs to be small enough to get >= n points
    
    # For 300 points in unit disk, hexagonal packing density ~ 0.9069
    # Area of unit disk = pi, so area per point ~ pi/300
    # Hexagonal lattice: area per point = (sqrt(3)/2) * s^2 where s is spacing
    # So s ~ sqrt(2*pi/(300*sqrt(3))) ~ sqrt(2*pi/(300*1.732)) ~ sqrt(0.01209) ~ 0.11
    
    best_points = None
    best_spacing = None
    
    for trial in range(20):
        # Try different spacings
        s = 0.10 + trial * 0.005
        # Generate hex grid points
        points = []
        max_r = 1.0
        # Number of rings needed
        max_ring = int(max_r / s) + 3
        
        for ring in range(max_ring + 1):
            if ring == 0:
                points.append([0.0, 0.0])
            else:
                # Hexagonal ring: 6*ring points
                for i in range(6 * ring):
                    # Angle for this point
                    angle = (i / (6.0 * ring)) * 2.0 * np.pi
                    # But hexagonal rings are not circular, they follow hex shape
                    # Let's use a simpler approach: generate on a square grid with hex offset
                    
        # Better: generate on a rectangular grid with hexagonal offset
        points = []
        for j in range(-max_ring, max_ring + 1):
            for i in range(-max_ring, max_ring + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3.0) / 2.0
                if x*x + y*y <= max_r * max_r:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) >= n:
            # Select the n closest to origin
            dists = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
            idx = np.argsort(dists, kind='stable')[:n]
            pts = points[idx]
            
            if best_points is None:
                best_points = pts
                best_spacing = s
            else:
                # Check which gives better min distance
                def min_dist(p):
                    # Compute pairwise distances
                    d = np.sqrt(((p[:, np.newaxis, :] - p[np.newaxis, :, :])**2).sum(-1))
                    np.fill_diagonal(d, np.inf)
                    return d.min()
                
                if min_dist(pts) > min_dist(best_points):
                    best_points = pts
                    best_spacing = s
    
    return best_points

def _initial_hexagonal():
    """Create initial hexagonal lattice configuration."""
    # Start with a reasonable spacing
    # For 300 points, spacing around 0.11-0.12
    s = 0.115
    
    # Generate hex grid
    points = []
    max_ring = int(1.0 / s) + 3
    
    for j in range(-max_ring, max_ring + 1):
        for i in range(-max_ring, max_ring + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 1.0:
                points.append([x, y])
    
    points = np.array(points)
    
    if len(points) < N:
        # Need smaller spacing
        s *= 0.95
        points = []
        max_ring = int(1.0 / s) + 3
        for j in range(-max_ring, max_ring + 1):
            for i in range(-max_ring, max_ring + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 1.0:
                    points.append([x, y])
        points = np.array(points)
    
    # Select N closest to origin
    dists = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    idx = np.argsort(dists, kind='stable')[:N]
    return points[idx]

def _score(points):
    """Compute the packing score: min(min pairwise dist/2, min (1 - |c_i|))."""
    n = len(points)
    # Pairwise distances
    d = np.sqrt(((points[:, np.newaxis, :] - points[np.newaxis, :, :])**2).sum(-1))
    np.fill_diagonal(d, np.inf)
    min_pair = d.min()
    
    # Distance to boundary
    radii = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    min_boundary = (1.0 - radii).min()
    
    return min(min_pair / 2.0, min_boundary)

def _optimize_from_initial(points, time_budget):
    """Run gradient-based optimization to improve the packing."""
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Use a simple iterative improvement approach
    # For each point, try to move it in the direction that increases the minimum constraint
    
    current_score = _score(points)
    best_points = points.copy()
    best_score = current_score
    
    # Iterative local search
    for iteration in range(500):
        if time.time() > deadline - 5:
            break
        
        improved = False
        for i in range(N):
            if time.time() > deadline - 5:
                break
            
            # Find the most constrained neighbor
            d = np.sqrt(((points[i] - points)**2).sum(-1))
            d[i] = np.inf
            min_d = d.min()
            j = d.argmin()
            
            # Also check boundary constraint
            r_i = np.sqrt(points[i, 0]**2 + points[i, 1]**2)
            boundary_margin = 1.0 - r_i
            
            # Determine which constraint is binding
            if min_d / 2.0 < boundary_margin:
                # Distance constraint binding
                # Move away from j
                diff = points[i] - points[j]
                norm = np.sqrt(diff[0]**2 + diff[1]**2)
                if norm > 1e-10:
                    direction = diff / norm
                else:
                    direction = np.array([1.0, 0.0])
            else:
                # Boundary constraint binding
                # Move toward origin
                if r_i > 1e-10:
                    direction = -points[i] / r_i
                else:
                    direction = np.array([1.0, 0.0])
            
            # Try small moves
            for step in [0.001, 0.005, 0.01, 0.02]:
                new_points = points.copy()
                new_points[i] = points[i] + direction * step
                
                # Check if still inside unit circle
                new_r = np.sqrt(new_points[i, 0]**2 + new_points[i, 1]**2)
                if new_r >= 1.0:
                    continue
                
                new_score = _score(new_points)
                if new_score > current_score + 1e-12:
                    points = new_points
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_points = points.copy()
                    break
        
        if not improved:
            # Try random perturbations
            for _ in range(10):
                if time.time() > deadline - 5:
                    break
                i = np.random.randint(N)
                angle = np.random.uniform(0, 2*np.pi)
                step = np.random.uniform(0.001, 0.01)
                new_points = points.copy()
                new_points[i] += step * np.array([np.cos(angle), np.sin(angle)])
                
                new_r = np.sqrt(new_points[i, 0]**2 + new_points[i, 1]**2)
                if new_r >= 1.0:
                    continue
                
                new_score = _score(new_points)
                if new_score > current_score + 1e-12:
                    points = new_points
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_points = points.copy()
                    break
    
    return best_points

def construct():
    """Construct the packing of 300 equal circles in the unit circle."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start with hexagonal lattice
    points = _initial_hexagonal()
    
    # Ensure we have exactly N points
    if len(points) < N:
        # Add more points by using smaller spacing
        s = 0.10
        while len(points) < N:
            points = []
            max_ring = int(1.0 / s) + 3
            for j in range(-max_ring, max_ring + 1):
                for i in range(-max_ring, max_ring + 1):
                    x = (i + 0.5 * (j % 2)) * s
                    y = j * s * np.sqrt(3.0) / 2.0
                    if x*x + y*y <= 1.0:
                        points.append([x, y])
            points = np.array(points)
            s *= 0.95
        
        dists = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        idx = np.argsort(dists, kind='stable')[:N]
        points = points[idx]
    elif len(points) > N:
        dists = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        idx = np.argsort(dists, kind='stable')[:N]
        points = points[idx]
    
    # Run optimization
    time_budget = min(120.0, deadline - time.time())
    if time_budget > 10:
        points = _optimize_from_initial(points, time_budget)
    
    # Final check and return
    if time.time() > deadline - 2:
        # Return current best
        pass
    
    return points.astype(np.float64)
# EVOLVE-BLOCK-END
