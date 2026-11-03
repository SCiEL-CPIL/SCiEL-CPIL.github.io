"""Improved packing of 200 equal circles in a unit square."""
import numpy as np
import time
from scipy.spatial.distance import pdist, cdist
from scipy.spatial import Delaunay

N = 200


def _compute_radius(points):
    """Compute the max common radius for points in [-0.5, 0.5]^2."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    if not np.all((points >= -0.5) & (points <= 0.5)):
        return 0.0
    # Min distance to boundary
    dist_boundary = 0.5 - np.max(np.abs(points), axis=1)
    min_boundary = np.min(dist_boundary)
    if min_boundary < 0:
        return 0.0
    # Min pairwise distance
    if N < 2:
        min_pair = float('inf')
    else:
        d = pdist(points)
        min_pair = np.min(d)
    return min(min_pair / 2.0, min_boundary)


def _soft_min_radius(points, beta=50.0):
    """
    Smooth surrogate for the min-radius objective using log-sum-exp.
    Returns a smooth approximation of min(min_pair/2, min_boundary).
    Lower beta = smoother (more averaged), higher beta = sharper (closer to true min).
    """
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    if not np.all((points >= -0.5) & (points <= 0.5)):
        return 0.0
    
    dist_boundary = 0.5 - np.max(np.abs(points), axis=1)
    # Soft min of boundary distances
    # log-sum-exp trick for stability
    max_b = np.max(dist_boundary)
    soft_min_boundary = max_b - np.log(np.sum(np.exp(dist_boundary - max_b))) / len(dist_boundary)
    
    if N < 2:
        return soft_min_boundary
    
    d = pdist(points)
    max_d = np.max(d)
    soft_min_pair = max_d - np.log(np.sum(np.exp(d - max_d))) / len(d)
    
    # Combine: soft min of (soft_min_pair/2, soft_min_boundary)
    vals = np.array([soft_min_pair / 2.0, soft_min_boundary])
    max_v = np.max(vals)
    soft_min_combined = max_v - np.log(np.sum(np.exp(vals - max_v))) / len(vals)
    
    return soft_min_combined


def _greedy_lattice_select(num_points, time_limit=30.0):
    """
    Greedy selection of points from a fine hexagonal lattice.
    At each step, pick the lattice site that maximizes the minimum distance
    to already-selected points and to the boundary.
    """
    start_time = time.time()
    
    # Create a fine hexagonal lattice of candidate sites
    # The lattice spacing should be small enough to allow fine positioning
    # but large enough to keep the number of candidates manageable
    # For 200 points in a unit square, we want maybe 50-100 candidates per row
    
    # Let's create a hex lattice with spacing ~0.02
    # That gives roughly 50x50 = 2500 candidates, which is manageable
    
    spacing = 0.015  # Fine spacing
    # Generate hex lattice points
    candidates = []
    dy = spacing * np.sqrt(3) / 2
    # Range to cover [-0.5, 0.5] with some margin
    x_min, x_max = -0.5, 0.5
    y_min, y_max = -0.5, 0.5
    
    row = 0
    y = y_min
    while y <= y_max:
        offset = (row % 2) * spacing / 2
        x = x_min + offset
        while x <= x_max:
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                candidates.append([x, y])
            x += spacing
        y += dy
        row += 1
    
    candidates = np.array(candidates)
    n_cand = len(candidates)
    
    if n_cand < num_points:
        # Fallback to square grid if not enough candidates
        return None
    
    # Greedy selection
    selected = []
    # For efficiency, we'll track distances incrementally
    # But for simplicity, let's use a straightforward approach with early termination
    
    for step in range(num_points):
        if time.time() - start_time > time_limit:
            break
        
        if len(selected) == 0:
            # First point: maximize distance to boundary (i.e., center)
            # The point closest to center (0,0)
            dists_center = np.sqrt(candidates[:, 0]**2 + candidates[:, 1]**2)
            best_idx = np.argmin(dists_center)
        else:
            selected_arr = np.array(selected)
            # Distance from each candidate to each selected point
            dists = cdist(candidates, selected_arr)
            min_dists_to_selected = np.min(dists, axis=1)
            
            # Distance to boundary for each candidate
            dist_boundary = 0.5 - np.max(np.abs(candidates), axis=1)
            
            # We want to maximize the minimum of (min_dists_to_selected, dist_boundary)
            # This is a bottleneck objective
            min_val = np.minimum(min_dists_to_selected, dist_boundary)
            best_idx = np.argmax(min_val)
        
        selected.append(candidates[best_idx])
        
        # Remove selected point from candidates to avoid re-selection
        # (optional, but prevents duplicates)
    
    if len(selected) < num_points:
        return None
    
    return np.array(selected[:num_points])


def _topological_swap(points, time_budget=20.0):
    """
    Perform topological swaps on a proximity graph to equalize distances.
    Build Delaunay triangulation, identify longest edges (bottlenecks),
    and swap endpoints to reduce the longest edge while maintaining validity.
    """
    start_time = time.time()
    points = points.copy()
    current_r = _compute_radius(points)
    
    if current_r <= 0:
        return points, current_r
    
    no_improve = 0
    
    while time.time() - start_time < time_budget and no_improve < 5:
        # Build Delaunay triangulation
        try:
            tri = Delaunay(points)
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    a, b = simplex[i], simplex[(i + 1) % 3]
                    edges.add((min(a, b), max(a, b)))
        except Exception:
            break
        
        if not edges:
            break
        
        # Compute edge lengths
        edge_list = list(edges)
        edge_lengths = np.array([np.linalg.norm(points[e[0]] - points[e[1]]) for e in edge_list])
        
        # Find the longest edge (bottleneck in the graph)
        max_edge_idx = np.argmax(edge_lengths)
        max_edge_len = edge_lengths[max_edge_idx]
        i, j = edge_list[max_edge_idx]
        
        # Try to swap point i with another point k that is closer to j
        # Or swap j with another point k that is closer to i
        # Strategy: find a point k such that swapping i and k reduces the longest edge
        improved = False
        
        # Try swapping point i with a nearby point
        # Find points within some radius of i
        dists_to_i = np.linalg.norm(points - points[i], axis=1)
        # Consider points that are close to i but not too close
        candidates = np.where((dists_to_i > 0.5 * max_edge_len) & (dists_to_i < 2.0 * max_edge_len))[0]
        candidates = [c for c in candidates if c != i and c != j]
        
        # Sort by distance to i
        if candidates:
            candidates = sorted(candidates, key=lambda c: dists_to_i[c])
        
        for k in candidates[:10]:  # Try up to 10 candidates
            if time.time() - start_time > time_budget:
                break
            
            # Try swapping i and k
            new_points = points.copy()
            new_points[i], new_points[k] = new_points[k].copy(), new_points[i].copy()
            
            if np.all((new_points >= -0.5) & (new_points <= 0.5)):
                new_r = _compute_radius(new_points)
                if new_r > current_r + 1e-12:
                    current_r = new_r
                    points = new_points
                    improved = True
                    no_improve = 0
                    break
        
        # Also try swapping j with a nearby point
        if not improved:
            dists_to_j = np.linalg.norm(points - points[j], axis=1)
            candidates = np.where((dists_to_j > 0.5 * max_edge_len) & (dists_to_j < 2.0 * max_edge_len))[0]
            candidates = [c for c in candidates if c != i and c != j]
            
            if candidates:
                candidates = sorted(candidates, key=lambda c: dists_to_j[c])
            
            for k in candidates[:10]:
                if time.time() - start_time > time_budget:
                    break
                
                # Try swapping j and k
                new_points = points.copy()
                new_points[j], new_points[k] = new_points[k].copy(), new_points[j].copy()
                
                if np.all((new_points >= -0.5) & (new_points <= 0.5)):
                    new_r = _compute_radius(new_points)
                    if new_r > current_r + 1e-12:
                        current_r = new_r
                        points = new_points
                        improved = True
                        no_improve = 0
                        break
        
        if not improved:
            no_improve += 1
    
    return points, current_r


def _local_refine(points, time_budget=10.0):
    """
    Local refinement using coordinate ascent with adaptive step size.
    For each point, try to move it in a direction that increases the min distance.
    """
    start_time = time.time()
    points = points.copy()
    current_r = _compute_radius(points)
    
    if current_r <= 0:
        return points, current_r
    
    # Determine the bottleneck: is it a pair or the boundary?
    def _get_bottleneck(pts):
        dist_boundary = 0.5 - np.max(np.abs(pts), axis=1)
        min_boundary = np.min(dist_boundary)
        min_boundary_idx = np.argmin(dist_boundary)
        
        if N >= 2:
            d = pdist(pts)
            min_pair = np.min(d)
            # Find which pair
            min_pair_idx = np.argmin(d)
            # Convert from condensed index to (i, j)
            n = N
            # Invert triangular index
            k = min_pair_idx
            # i < j, index = i*n - i*(i+1)/2 + (j - i - 1)
            # Solve for i, j
            for i in range(n):
                max_j = n - i - 1
                if k < max_j:
                    j = i + 1 + k
                    break
                k -= max_j
            pair = (i, j)
        else:
            min_pair = float('inf')
            pair = None
        
        if min_pair / 2.0 < min_boundary:
            return 'pair', min_pair / 2.0, pair
        else:
            return 'boundary', min_boundary, min_boundary_idx
    
    step = 0.005
    no_improve_count = 0
    
    while time.time() - start_time < time_budget:
        improved = False
        for i in range(N):
            if time.time() - start_time > time_budget:
                break
            
            # Try moving point i in 8 directions + diagonal
            for dx in [-step, 0, step]:
                for dy in [-step, 0, step]:
                    if dx == 0 and dy == 0:
                        continue
                    new_points = points.copy()
                    new_points[i] += np.array([dx, dy])
                    if np.all((new_points >= -0.5) & (new_points <= 0.5)):
                        new_r = _compute_radius(new_points)
                        if new_r > current_r + 1e-12:
                            current_r = new_r
                            points = new_points
                            improved = True
                            no_improve_count = 0
                            break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= 3:
                # Reduce step size
                step *= 0.5
                if step < 1e-6:
                    break
                no_improve_count = 0
        else:
            # Keep step size or slightly increase
            pass
    
    return points, current_r


def construct():
    deadline = time.time() + 145.0  # Safety margin (evaluator kills at 150s)
    
    best_points = None
    best_r = 0.0
    
    # Strategy 1: Greedy lattice selection
    greedy_points = _greedy_lattice_select(N, time_limit=40.0)
    if greedy_points is not None:
        r = _compute_radius(greedy_points)
        if r > best_r:
            best_r = r
            best_points = greedy_points.copy()
    
    # Strategy 2: Hex grid with optimization
    # Try a hexagonal grid arrangement
    for rows in range(12, 22):
        if time.time() > deadline - 20:
            break
        cols = N // rows
        rem = N % rows
        
        # Hex grid: row j has offset (j%2)*s/2
        # x = -0.5 + s/2 + i*s + (j%2)*s/2
        # y = -0.5 + s*sqrt(3)/4 + j*s*sqrt(3)/2
        
        # We need to fit in the square
        # Width: (cols-1)*s + s/2 (for offset) + s (diameter margin) ≈ cols*s
        # Actually, the max x is -0.5 + s/2 + (cols-1)*s + s/2 = -0.5 + cols*s
        # So we need cols*s <= 1 => s <= 1/cols
        # Height: (rows-1)*s*sqrt(3)/2 + s*sqrt(3)/2 = rows*s*sqrt(3)/2
        # Wait, y ranges from -0.5 + s*sqrt(3)/4 to -0.5 + s*sqrt(3)/4 + (rows-1)*s*sqrt(3)/2
        # Max y - min y = (rows-1)*s*sqrt(3)/2
        # We need the full range to be within [-0.5, 0.5]
        # So (rows-1)*s*sqrt(3)/2 + s*sqrt(3)/2 <= 1  (adding margin for radius)
        # Actually, the circle radius is s/2, so we need:
        # min y - s/2 >= -0.5 and max y + s/2 <= 0.5
        # min y = -0.5 + s*sqrt(3)/4
        # max y = -0.5 + s*sqrt(3)/4 + (rows-1)*s*sqrt(3)/2
        # Constraint 1: -0.5 + s*sqrt(3)/4 - s/2 >= -0.5 => s*(sqrt(3)/4 - 1/2) >= 0 => always true for s>0? No.
        # sqrt(3)/4 ≈ 0.433, 1/2 = 0.5, so sqrt(3)/4 - 1/2 < 0. So this constraint is violated!
        # We need to shift the grid. Let's center it.
        
        # Let's just generate and check
        s_x = 1.0 / cols if cols > 0 else 1.0
        s_y = 1.0 / (max(1, (rows - 1) * np.sqrt(3) / 2)) if rows > 1 else 1.0
        s = min(s_x, s_y)
        
        points = []
        count = 0
        for j in range(rows):
            offset = (j % 2) * s / 2
            n_in_row = cols + (1 if j < rem else 0)
            for i in range(n_in_row):
                x = -0.5 + s / 2 + i * s + offset
                y = -0.5 + s * np.sqrt(3) / 4 + j * s * np.sqrt(3) / 2
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
                    count += 1
                    if count >= N:
                        break
            if count >= N:
                break
        
        if len(points) == N:
            points = np.array(points)
            r = _compute_radius(points)
            if r > best_r:
                best_r = r
                best_points = points.copy()
    
    # Strategy 3: Square grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    sq_points = grid[:N]
    r = _compute_radius(sq_points)
    if r > best_r:
        best_r = r
        best_points = sq_points.copy()
    
    if best_points is None:
        # Fallback
        best_points = np.zeros((N, 2))
        return best_points
    
    # Topological swap refinement
    if time.time() < deadline - 15:
        remaining = deadline - time.time() - 12
        if remaining > 5:
            swapped_points, swapped_r = _topological_swap(best_points, time_budget=remaining)
            if swapped_r > best_r:
                best_r = swapped_r
                best_points = swapped_points.copy()
    
    # Local refinement
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 3
        if remaining > 2:
            best_points, best_r = _local_refine(best_points, time_budget=remaining)
    
    return best_points
