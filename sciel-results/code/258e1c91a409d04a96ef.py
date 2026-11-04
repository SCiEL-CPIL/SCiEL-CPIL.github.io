"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time

N = 21

# EVOLVE-BLOCK-START

def _get_contact_graph():
    """
    Define the contact graph for a hexagonal-like packing of 21 circles.
    Rows: 5, 4, 5, 4, 3
    Indices are assigned row by row.
    """
    rows = [5, 4, 5, 4, 3]
    pairs = []
    idx = 0
    for i, count in enumerate(rows):
        for j in range(count):
            # Touch right neighbor in same row
            if j < count - 1:
                pairs.append((idx, idx + 1))
            
            # Touch neighbors in the row below
            if i + 1 < len(rows):
                next_count = rows[i + 1]
                next_start = idx + count
                # In a hexagonal lattice, circle (i, j) touches (i+1, j) and (i+1, j+1)
                # assuming standard offset. 
                # With rows 5,4,5..., the offsets are:
                # Row 0 (5): centers at 0.5, 1.5, 2.5, 3.5, 4.5 (units of cell width)
                # Row 1 (4): centers at 1.0, 2.0, 3.0, 4.0
                # Row 2 (5): centers at 0.5, 1.5, 2.5, 3.5, 4.5
                # So (0,0) touches (1,0). (0,1) touches (1,0) and (1,1).
                
                # General rule for this specific 5-4-5-4-3 pattern:
                # If current row has C circles, next row has C-1 or C+1.
                # Here: 5->4 (next is smaller), 4->5 (next is larger), 5->4, 4->3.
                
                if next_count < count:
                    # Next row is indented (smaller)
                    # Circle j touches next row j
                    if j < next_count:
                        pairs.append((idx, next_start + j))
                    # Circle j touches next row j+1 ? 
                    # In hex packing, if row below is smaller, 
                    # circle j touches (below, j) and (below, j+1) is not standard.
                    # Actually, let's just add potential contacts based on geometric proximity 
                    # in the initialization and let the solver refine.
                    # For the graph, we assume standard hex connectivity.
                    if j < next_count:
                        pairs.append((idx, next_start + j))
                    if j + 1 < next_count:
                        pairs.append((idx, next_start + j + 1))
                else:
                    # Next row is larger (or equal)
                    # Circle j touches (below, j-1) and (below, j)
                    if j - 1 >= 0:
                        pairs.append((idx, next_start + j - 1))
                    if j < next_count:
                        pairs.append((idx, next_start + j))
        idx += count
    
    # Remove duplicates and sort
    unique_pairs = set(map(tuple, map(sorted, pairs)))
    return sorted(list(unique_pairs))

def _initialize_positions(w, h):
    """
    Create initial positions based on a hexagonal grid scaled to fit w, h.
    Returns (positions, radii) where radii are all 1.0 initially (scale is determined later).
    """
    rows = [5, 4, 5, 4, 3]
    total_rows = len(rows)
    positions = []
    
    # We define the "unit" packing in a virtual space, then scale it.
    # Let's define the packing in a square of side 1 for simplicity, then scale x by w, y by h?
    # No, better to define positions relative to w and h directly.
    
    # Hexagonal spacing logic:
    # Assume all circles have radius R.
    # Horizontal distance between centers in same row: 2R
    # Vertical distance between rows: R * sqrt(3)
    # Width of row with k circles: (k-1)*2R + 2R = k*2R? No, span is (k-1)*2R + 2R = 2kR? 
    # Actually, width occupied by k circles of radius R in a row is 2R * k if they touch boundaries?
    # Let's just place them in a grid that looks hexagonal and let the solver fix it.
    
    for i, count in enumerate(rows):
        y = (i + 0.5) * h / total_rows
        for j in range(count):
            # Offset for hexagonal look
            if i % 2 == 1:
                # Indent the row
                x = (j + 0.5) * w / 5 + 0.5 * (w / 5) * 0.5 # Approximate offset
            else:
                x = (j + 0.5) * w / 5
            
            # Clamp x to be within [0, w]
            x = np.clip(x, 0.05*w, 0.95*w)
            positions.append([x, y])
            
    return np.array(positions, dtype=float)

def _projected_gauss_seidel(positions, radii, w, h, contact_pairs, max_iters=2000, tol=1e-10):
    """
    Iterative solver that strictly enforces non-overlap and boundary constraints.
    It adjusts positions to satisfy distance constraints (dist >= r_i + r_j) and boundaries.
    """
    n = len(positions)
    pos = positions.copy()
    rad = radii.copy()
    
    # Precompute pair indices for faster access
    pairs_i = np.array([p[0] for p in contact_pairs])
    pairs_j = np.array([p[1] for p in contact_pairs])
    
    for it in range(max_iters):
        max_move = 0.0
        
        # 1. Boundary Projection
        # For each circle, clamp position to [r, w-r] and [r, h-r]
        # This is a hard constraint.
        for i in range(n):
            r = rad[i]
            # X bounds
            if pos[i, 0] < r:
                pos[i, 0] = r
            elif pos[i, 0] > w - r:
                pos[i, 0] = w - r
            
            # Y bounds
            if pos[i, 1] < r:
                pos[i, 1] = r
            elif pos[i, 1] > h - r:
                pos[i, 1] = h - r
        
        # 2. Overlap Projection (Gauss-Seidel style)
        # We iterate through the contact pairs and push circles apart if they overlap.
        # To be robust, we should check ALL pairs, not just contacts, in case the graph is incomplete.
        # However, for speed, we first check contacts, then do a global check if needed.
        # Given N=21, checking all 210 pairs is cheap. Let's check all pairs for robustness.
        
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist_sq = dx*dx + dy*dy
                min_dist = rad[i] + rad[j]
                
                if dist_sq < min_dist * min_dist:
                    dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
                    overlap = min_dist - dist
                    
                    if dist < 1e-9:
                        # If overlapping centers, push in random direction
                        angle = np.random.uniform(0, 2 * np.pi)
                        push_x = np.cos(angle) * overlap * 0.5
                        push_y = np.sin(angle) * overlap * 0.5
                    else:
                        push_x = (dx / dist) * overlap * 0.5
                        push_y = (dy / dist) * overlap * 0.5
                    
                    pos[i, 0] -= push_x
                    pos[i, 1] -= push_y
                    pos[j, 0] += push_x
                    pos[j, 1] += push_y
                    
                    max_move = max(max_move, overlap)
                    
        # Check convergence
        if max_move < tol:
            break
            
    return pos

def _check_validity(pos, rad, w, h, tol=1e-7):
    """
    Check if the configuration is valid (no overlaps, inside bounds).
    """
    n = len(pos)
    # Bounds
    for i in range(n):
        if pos[i, 0] - rad[i] < -tol or pos[i, 0] + rad[i] > w + tol:
            return False
        if pos[i, 1] - rad[i] < -tol or pos[i, 1] + rad[i] > h + tol:
            return False
            
    # Overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < rad[i] + rad[j] - tol:
                return False
    return True

def _optimize_for_aspect(aspect, deadline):
    """
    For a given aspect ratio, find the maximum scale (radius multiplier) such that
    the hexagonal packing fits in the rectangle.
    """
    w = 2.0 * aspect / (1.0 + aspect)
    h = 2.0 / (1.0 + aspect)
    
    # Get contact graph
    pairs = _get_contact_graph()
    
    # Initial positions (unit radii)
    pos0 = _initialize_positions(w, h)
    
    # We will binary search for the maximum scale factor S such that
    # radii = S * base_radii fits.
    # Base radii: let's assume all circles have radius 1.0 in the "unit" space?
    # No, the positions are already in w,h space.
    # Let's define base radii as some fraction of the local grid spacing.
    # Simpler: Let's define the packing in a normalized space where min(w,h)=1?
    # Actually, let's just scale the *positions* and *radii* together?
    
    # Better approach:
    # 1. Define a "template" packing with unit radii in a canonical box (e.g., 1x1 or specific aspect).
    # 2. Scale it to fit the target aspect ratio?
    # This is complex because aspect ratio changes the optimal packing.
    
    # Alternative:
    # Fix the aspect ratio.
    # Initialize positions in w,h box.
    # Set initial radii to a small value.
    # Use Gauss-Seidel to resolve overlaps.
    # Then, iteratively increase radii?
    
    # Most robust method for this problem:
    # 1. Initialize positions in w,h box.
    # 2. Set radii to 1.0.
    # 3. Run GS solver to find a valid configuration with radii=1.0?
    #    If w,h are too small, it will fail (circles pushed out or stuck).
    #    If it succeeds, we can scale up.
    
    # Let's try this:
    # 1. Start with small radii (e.g., 0.01).
    # 2. Solve for positions.
    # 3. Increase radii by 10% and re-solve.
    # 4. Repeat until failure or time limit.
    
    rad = np.ones(N) * 0.01
    pos = pos0.copy()
    current_scale = 0.01
    
    max_scale = 0.0
    best_pos = None
    best_rad = None
    
    # Heuristic max radius: min(w,h) / (2 * sqrt(21)) ?
    # Let's just iterate.
    
    for step in range(50): # 50 steps of scaling
        if time.time() > deadline:
            break
            
        # Solve for current radii
        pos_new = _projected_gauss_seidel(pos, rad, w, h, pairs, max_iters=500)
        
        # Check validity
        if _check_validity(pos_new, rad, w, h):
            max_scale = current_scale
            best_pos = pos_new
            best_rad = rad.copy()
            pos = pos_new # Use converged position as start for next scale
            
            # Increase scale
            # If we are close to the limit, increase slowly.
            current_scale *= 1.1
            rad = np.ones(N) * current_scale
        else:
            # Failed to pack at this scale
            # Try to recover by reducing scale slightly and solving again?
            # Or just break.
            break
            
    if best_pos is None:
        return None, None, 0.0
        
    total_radius = np.sum(best_rad)
    return best_pos, best_rad, total_radius

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    best_score = 0.0
    best_circles = None
    
    # Try a range of aspect ratios
    # Perimeter = 2(w+h) = 4 => w+h=2
    # aspect = w/h
    # w = 2a/(1+a), h = 2/(1+a)
    
    aspects = [1.0, 0.8, 1.25, 0.6, 1.6, 0.5, 2.0]
    
    for aspect in aspects:
        if time.time() > deadline - 5.0:
            break
            
        pos, rad, score = _optimize_for_aspect(aspect, deadline)
        
        if pos is not None and score > best_score:
            best_score = score
            # Combine pos and rad into (x, y, r) format
            best_circles = np.column_stack((pos, rad))
            
    # If no aspect worked well, try a random restart with a square
    if best_circles is None:
        w, h = 1.0, 1.0
        pos0 = _initialize_positions(w, h)
        rad = np.ones(N) * 0.05
        pos = _projected_gauss_seidel(pos0, rad, w, h, [], max_iters=1000)
        if _check_validity(pos, rad, w, h):
            best_circles = np.column_stack((pos, rad))
            best_score = np.sum(rad)
        else:
            # Fallback: tiny circles
            best_circles = np.column_stack((pos0, np.ones(N)*0.01))
            best_score = 0.21

    return best_circles

# EVOLVE-BLOCK-END
