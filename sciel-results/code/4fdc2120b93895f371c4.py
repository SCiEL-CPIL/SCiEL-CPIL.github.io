"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree

N = 300

# EVOLVE-BLOCK-START

def _compute_radius(points):
    """
    Compute the feasible common radius for a set of points.
    Radius is limited by:
    1. Distance to boundary: min(1 - |p_i|)
    2. Distance to other circles: min(|p_i - p_j|) / 2
    """
    if points is None or len(points) == 0:
        return 0.0
    
    # 1. Boundary constraint
    dists_to_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = float(np.min(dists_to_boundary))
    
    # 2. Pairwise distance constraint
    d = pdist(points)
    if len(d) == 0:
        min_pair = np.inf
    else:
        min_pair = float(np.min(d)) / 2.0
        
    return min(min_boundary, min_pair)

def _hexagonal_lattice(spacing):
    """
    Generate hexagonal lattice points inside the unit disk.
    Returns None if not enough points are generated.
    """
    k = int(1.0 / spacing) + 3
    pts = []
    sqrt3 = np.sqrt(3.0)
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * sqrt3 / 2.0
            if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                pts.append((x, y))
    
    if len(pts) < N:
        return None
        
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists, kind="stable")[:N]
    return pts[idx]

def _square_grid(spacing):
    """
    Generate square grid points inside the unit disk.
    """
    k = int(1.0 / spacing) + 3
    pts = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = i * spacing
            y = j * spacing
            if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                pts.append((x, y))
                
    if len(pts) < N:
        return None
        
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists, kind="stable")[:N]
    return pts[idx]

def _concentric_rings(n_rings):
    """
    Generate points on concentric rings.
    """
    if n_rings <= 0:
        return None
        
    pts = []
    pts.append([0.0, 0.0])
    
    remaining = N - 1
    ring_counts = [1]
    if n_rings > 1:
        radii = np.linspace(0.3, 0.95, n_rings - 1)
        weights = radii
        weights /= np.sum(weights)
        counts = np.round(weights * remaining).astype(int)
        
        diff = remaining - np.sum(counts)
        if diff != 0:
            idx_max = np.argmax(counts)
            counts[idx_max] += diff
            
        ring_counts.extend(counts.tolist())
        
    for ring_idx in range(1, n_rings):
        count = ring_counts[ring_idx]
        if count <= 0:
            continue
        ring_radius = 0.95 * ring_idx / n_rings
        for i in range(count):
            angle = 2 * np.pi * i / count
            x = ring_radius * np.cos(angle)
            y = ring_radius * np.sin(angle)
            pts.append([x, y])
            
    if len(pts) != N:
        return None
        
    return np.array(pts)

def _fast_radius(points, tree, point_idx, new_pos):
    """
    Fast radius computation after moving one point.
    Uses KDTree for neighbor queries instead of full pdist.
    """
    # Boundary constraint for the moved point
    new_dist = np.linalg.norm(new_pos)
    new_boundary = 1.0 - new_dist
    
    # Query neighbors of the moved point
    # We need to check distances to all other points
    # But we can use the tree to find close neighbors efficiently
    # However, for exact min distance, we might still need to be careful.
    # Let's just use pdist for the full set but cache the min pair distance
    # if only one point moves, we can update it incrementally?
    # Actually, for N=300, pdist is ~0.1ms. 10000 iters = 1s.
    # The bottleneck is the number of pdist calls.
    # Let's stick to pdist but optimize the loop structure.
    
    # For now, let's just compute it directly to ensure correctness.
    # The "fast" part is avoiding recomputing the boundary min for all points
    # if we only move one.
    
    # We need min of:
    # 1. min_j (1 - |p_j|) for j != idx, and (1 - |new_pos|)
    # 2. min_{i<j} |p_i - p_j| / 2
    
    # This is still O(N^2) due to the pairwise term.
    # Let's try a different approach: only check the moved point's neighbors
    # and keep track of the global min pair distance.
    
    # This requires maintaining the global min pair distance.
    # If the moved point was part of the min pair, we need to recompute.
    
    # For simplicity and robustness, let's just use pdist.
    # The speedup from KDTree might not be worth the complexity for N=300.
    
    points[idx] = new_pos
    r = _compute_radius(points)
    return r

def _refine_local(points, deadline, max_iters=5000, step_size=0.05):
    """
    Hill-climbing local refinement.
    Tries to move individual points to increase the minimum radius.
    """
    points = points.copy()
    best_r = _compute_radius(points)
    best_points = points.copy()
    
    rng = np.random.default_rng(42)
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
            
        # Pick a random point
        idx = rng.integers(0, N)
        orig_x, orig_y = points[idx]
        
        # Try a few random perturbations
        best_local_r = best_r
        best_local_pt = (orig_x, orig_y)
        
        for _ in range(10):
            if time.time() > deadline:
                break
            angle = rng.uniform(0, 2 * np.pi)
            step = step_size * rng.uniform(0.1, 1.0)
            dx = step * np.cos(angle)
            dy = step * np.sin(angle)
            
            new_x = orig_x + dx
            new_y = orig_y + dy
            
            if new_x*new_x + new_y*new_y > 1.0:
                continue
                
            points[idx] = [new_x, new_y]
            r = _compute_radius(points)
            
            if r > best_local_r:
                best_local_r = r
                best_local_pt = (new_x, new_y)
                
        points[idx] = best_local_pt
        
        if best_local_r > best_r:
            best_r = best_local_r
            best_points = points.copy()
            
    return best_points, best_r

def _simulated_annealing(points, deadline, start_temp=0.01, end_temp=0.001, max_iters=5000):
    """
    Simulated annealing to escape local minima.
    """
    points = points.copy()
    current_r = _compute_radius(points)
    best_r = current_r
    best_points = points.copy()
    
    rng = np.random.default_rng(42)
    n_iters = max_iters
    
    for i in range(n_iters):
        if time.time() > deadline:
            break
            
        # Temperature schedule
        t = start_temp * (end_temp / start_temp) ** (i / n_iters)
        
        # Pick a random point
        idx = rng.integers(0, N)
        orig_x, orig_y = points[idx]
        
        # Random perturbation
        angle = rng.uniform(0, 2 * np.pi)
        step = 0.02 * rng.uniform(0.1, 1.0)
        dx = step * np.cos(angle)
        dy = step * np.sin(angle)
        
        new_x = orig_x + dx
        new_y = orig_y + dy
        
        if new_x*new_x + new_y*new_y > 1.0:
            continue
            
        points[idx] = [new_x, new_y]
        new_r = _compute_radius(points)
        
        # Accept if better or with probability exp(-delta / T)
        delta = new_r - current_r
        if delta > 0 or (t > 0 and rng.random() < np.exp(delta / t)):
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_points = points.copy()
        else:
            points[idx] = [orig_x, orig_y]
            
    return best_points, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    candidates = []
    
    # 1. Hexagonal Lattice Candidates
    spacings = np.linspace(0.10, 0.13, 20)
    for s in spacings:
        pts = _hexagonal_lattice(s)
        if pts is not None:
            r = _compute_radius(pts)
            candidates.append((pts, r))
            
    # 2. Square Grid Candidates
    spacings_sq = np.linspace(0.10, 0.13, 10)
    for s in spacings_sq:
        pts = _square_grid(s)
        if pts is not None:
            r = _compute_radius(pts)
            candidates.append((pts, r))
            
    # 3. Concentric Rings Candidates
    for n_rings in range(5, 20):
        pts = _concentric_rings(n_rings)
        if pts is not None:
            r = _compute_radius(pts)
            candidates.append((pts, r))
            
    if not candidates:
        pts = np.random.randn(N, 2) * 0.5
        return pts

    # Sort by radius, best first
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    best_points = candidates[0][0]
    best_r = candidates[0][1]
    
    # 4. Refinement: First do hill climbing, then simulated annealing
    num_to_refine = min(3, len(candidates))
    for i in range(num_to_refine):
        if time.time() > deadline - 30.0:
            break
            
        cand_pts, cand_r = candidates[i]
        
        # Phase 1: Hill climbing
        refined_pts, refined_r = _refine_local(cand_pts, deadline - 15.0, max_iters=5000, step_size=0.05)
        
        # Phase 2: Simulated annealing on the result
        if time.time() < deadline - 5.0:
            sa_pts, sa_r = _simulated_annealing(refined_pts, deadline - 5.0, start_temp=0.01, end_temp=0.0001, max_iters=5000)
            if sa_r > refined_r:
                refined_pts, refined_r = sa_pts, sa_r
        
        if refined_r > best_r:
            best_r = refined_r
            best_points = refined_pts
            
    return best_points

# EVOLVE-BLOCK-END
