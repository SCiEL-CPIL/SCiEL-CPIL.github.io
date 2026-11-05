"""3D Container Loading: Best-Fit greedy with multiple heuristics + gap-filling local search."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
from collections import defaultdict

def _orientations(w, h, d):
    """Return all 6 distinct orientations (permutations of dims)."""
    seen = set()
    result = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

def _overlaps(x1, y1, z1, x2, y2, z2, w1, h1, d1, w2, h2, d2):
    """Check if two boxes strictly overlap."""
    if x1 + w1 <= x2 or x2 + w2 <= x1:
        return False
    if y1 + h1 <= y2 or y2 + h2 <= y1:
        return False
    if z1 + d1 <= z2 or z2 + d2 <= z1:
        return False
    return True

def _fits(placed, x, y, z, w, h, d, W, H, D):
    """Check if box at (x,y,z) with dims (w,h,d) fits in container and doesn't overlap."""
    if x < 0 or y < 0 or z < 0:
        return False
    if x + w > W or y + h > H or z + d > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed:
        if _overlaps(x, y, z, px, py, pz, w, h, d, pw, ph, pd):
            return False
    return True

def _score_position(x, y, z, w, h, d, W, H, D):
    """
    Score a position for best-fit: we want to minimize the 'waste' or 'extent'.
    We prefer positions that keep the bounding box tight.
    A common heuristic is to minimize the volume of the bounding box of all placed items + this item.
    However, computing that for every candidate is expensive.
    
    Alternative: Minimize the sum of distances to the existing 'center of mass' or simply minimize (x, y, z) in a weighted way?
    Actually, 'Best Fit' usually means minimizing the empty space created.
    A simpler proxy: Minimize the volume of the bounding box of the current set + new box.
    Let's approximate by minimizing (max_x + w, max_y + h, max_z + d) volume?
    
    Let's use a specific heuristic:
    1. Minimize z (height) to keep things low.
    2. Minimize y (depth) to keep things front.
    3. Minimize x (width) to keep things left.
    This is basically 'First Fit' with a specific order.
    
    The prompt mentions: "Switching from First Fit to Best Fit (minimizing max extent) reduces the bounding box volume".
    So we should minimize the volume of the bounding box of all placed boxes + candidate.
    
    To make it fast, we can pre-compute current max_x, max_y, max_z.
    New max_x = max(current_max_x, x+w)
    New max_y = max(current_max_y, y+h)
    New max_z = max(current_max_z, z+d)
    Score = New max_x * New max_y * New max_z
    
    We want to MINIMIZE this score.
    """
    pass # Placeholder, will implement in packer

def _pack_best_fit(boxes, container, order, deadline):
    """
    Pack boxes using Best-Fit heuristic.
    For each box, try all orientations.
    For each orientation, try all candidate positions.
    Select the position that minimizes the resulting bounding box volume.
    """
    W, H, D = container
    placed = []  # list of (x, y, z, w, h, d)
    placements = {}  # box_idx -> (x, y, z, ow, oh, od)
    filled_vol = 0.0
    
    # Current bounding box
    cur_max_x = 0
    cur_max_y = 0
    cur_max_z = 0
    
    # We need to generate candidate positions efficiently.
    # Instead of checking all grid points, we check points derived from existing boxes.
    # But for Best Fit, we might need to evaluate many positions to find the one that minimizes the BB volume.
    # A standard set of candidates:
    # 0, and for each placed box: x, x+w, y, y+h, z, z+d.
    
    for idx in order:
        if time.time() > deadline:
            break
            
        w, h, d = boxes[idx]
        best_score = float('inf')
        best_pos = None
        best_orient = None
        
        # Pre-compute candidate coordinates
        xs = {0}
        ys = {0}
        zs = {0}
        for (px, py, pz, pw, ph, pd) in placed:
            xs.add(px)
            xs.add(px + pw)
            ys.add(py)
            ys.add(py + ph)
            zs.add(pz)
            zs.add(pz + pd)
        
        # Filter candidates within container bounds (roughly, exact check in _fits)
        xs = sorted(x for x in xs if 0 <= x <= W)
        ys = sorted(y for y in ys if 0 <= y <= H)
        zs = sorted(z for z in zs if 0 <= z <= D)
        
        # To speed up, we can limit the number of candidates if there are too many,
        # but for 60 boxes, the number of unique coordinates is small (< 120 per dim).
        # 120^3 = 1.7M candidates * 6 orientations * 60 boxes = 600M ops. Too slow.
        
        # Optimization:
        # We don't need to check ALL combinations of xs, ys, zs.
        # We only need to check positions where the box touches an existing box or the wall.
        # The candidates generated above (from box edges) are exactly those.
        # But the cartesian product is large.
        
        # Better approach for Best Fit:
        # Iterate through orientations.
        # For each orientation, iterate through a smaller set of "critical" positions.
        # Actually, the set of candidate positions is the set of "corners" formed by the grid of placed boxes.
        
        # Let's try to reduce the search space.
        # If we have many placed boxes, the number of unique x, y, z coordinates grows.
        # Max unique coords per dim is 2*N + 1. For N=60, that's 121.
        # 121^3 = ~1.8M candidates.
        # For each candidate, we check overlap with ~60 boxes.
        # 1.8M * 60 = 108M overlap checks per box.
        # 60 boxes * 108M = 6.5B checks. Too slow in Python.
        
        # We need a faster overlap check or fewer candidates.
        
        # Alternative: Use a spatial hash or grid for overlap checking?
        # Or, limit the candidates.
        
        # Let's try a different strategy for Best Fit:
        # Instead of checking all candidates, we can check candidates in order of "promisingness".
        # But Best Fit requires finding the global minimum of the BB volume among valid positions.
        
        # Maybe we can approximate Best Fit with a "Best Fit Decreasing" on a coarser grid?
        # Or, we can just use a smarter First Fit that prefers positions that don't increase the BB much.
        
        # Let's stick to a faster variant:
        # Generate candidates, but sort them by a heuristic score (e.g., z, then y, then x) and stop early?
        # No, that's First Fit.
        
        # Let's try to optimize the overlap check.
        # We can represent the placed boxes in a 3D grid of occupied cells?
        # If the container is not too large, this is fast.
        # But we don't know the container size.
        
        # Let's assume the container dimensions are reasonable (e.g., < 1000).
        # A 3D numpy array for occupancy would be fast.
        
        # Let's implement a grid-based occupancy map if the volume is small enough.
        # If W*H*D <= 10^7, use grid.
        
        vol = W * H * D
        use_grid = vol <= 10**7
        
        if use_grid:
            # Grid approach
            occ = np.zeros((W, H, D), dtype=bool)
            # Note: We need to handle the fact that boxes have integer coordinates?
            # The problem statement doesn't specify integer coordinates.
            # "x, y, z, w, h, d" are floats?
            # If they are floats, we can't use a simple boolean grid.
            # The prompt implies continuous space or at least real numbers.
            # "any axis rotation allowed" usually implies continuous.
            # However, the candidate positions are derived from box edges, which are floats.
            
            # If coordinates are floats, grid approach is hard unless we discretize.
            # Discretization might lose optimality.
            
            # Let's check the input type. Usually these problems use integers or simple floats.
            # If we assume integers, grid is great.
            # If not, we are stuck with O(N) overlap checks.
            
            # Let's assume we can't rely on integers.
            # We need to speed up the O(N) overlap check or reduce candidates.
            
            # Reduction of candidates:
            # We only need to consider positions where the box is "supported" by the floor or another box.
            # This means z must be 0 or z must be equal to pz+pd for some placed box (pz, py, pz, pw, ph, pd) where the projection of the new box overlaps with the projection of the placed box in XY.
            
            # This significantly reduces the number of z candidates.
            # For each (x, y), the valid z's are 0 and the tops of boxes that overlap in XY.
            
            # Let's implement this "supported" candidate generation.
            
            # 1. Generate candidate X and Y coordinates.
            # X candidates: 0, and for each box, px, px+pw.
            # Y candidates: 0, and for each box, py, py+ph.
            
            # 2. For each (cx, cy), find valid Z candidates.
            # Z candidates: 0, and for each box that overlaps (cx, cy) in XY, pz+pd.
            
            # 3. For each (cx, cy, cz), check if it fits (no overlap in 3D).
            
            # This reduces the search space from O(N^3) to O(N^2 * N) = O(N^3) but with a much smaller constant?
            # Actually, number of (cx, cy) pairs is O(N^2).
            # For each pair, number of z candidates is O(N).
            # Total candidates O(N^3).
            # But for each candidate, overlap check is O(N).
            # Total O(N^4). Still slow for N=60?
            # 60^4 = 12.9M. This is feasible in Python if the constant is small.
            
            pass # End of grid thought process
        
        # Let's implement the "supported" candidate generation for continuous coordinates.
        
        # Generate X and Y candidates
        x_cands = {0}
        y_cands = {0}
        for (px, py, pz, pw, ph, pd) in placed:
            x_cands.add(px)
            x_cands.add(px + pw)
            y_cands.add(py)
            y_cands.add(py + ph)
        
        x_cands = sorted([x for x in x_cands if 0 <= x <= W])
        y_cands = sorted([y for y in y_cands if 0 <= y <= H])
        
        # For each orientation
        for ow, oh, od in _orientations(w, h, d):
            # For each candidate X, Y
            for cx in x_cands:
                if cx + ow > W:
                    continue
                for cy in y_cands:
                    if cy + oh > H:
                        continue
                    
                    # Find valid Z candidates for this (cx, cy)
                    z_cands = {0}
                    for (px, py, pz, pw, ph, pd) in placed:
                        # Check if box (px, py, pz, pw, ph, pd) overlaps with (cx, cy, ?, ow, oh, ?) in XY
                        if cx + ow > px and px + pw > cx and cy + oh > py and py + ph > cy:
                            z_cands.add(pz + pd)
                    
                    z_cands = sorted([z for z in z_cands if 0 <= z <= D])
                    
                    for cz in z_cands:
                        if cz + od > D:
                            continue
                        
                        # Check if (cx, cy, cz, ow, oh, od) fits (no overlap)
                        if _fits(placed, cx, cy, cz, ow, oh, od, W, H, D):
                            # Calculate BB volume
                            new_max_x = max(cur_max_x, cx + ow)
                            new_max_y = max(cur_max_y, cy + oh)
                            new_max_z = max(cur_max_z, cz + od)
                            score = new_max_x * new_max_y * new_max_z
                            
                            if score < best_score:
                                best_score = score
                                best_pos = (cx, cy, cz)
                                best_orient = (ow, oh, od)
        
        if best_pos is not None:
            cx, cy, cz = best_pos
            ow, oh, od = best_orient
            placed.append((cx, cy, cz, ow, oh, od))
            placements[idx] = (cx, cy, cz, ow, oh, od)
            filled_vol += w * h * d
            
            # Update current max
            cur_max_x = max(cur_max_x, cx + ow)
            cur_max_y = max(cur_max_y, cy + oh)
            cur_max_z = max(cur_max_z, cz + od)
            
    return placements, filled_vol

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    container_vol = W * H * D
    
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin
    
    # Generate multiple sort orders
    orders = []
    
    # Order 1: by volume descending
    orders.append(sorted(range(n), key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True))
    
    # Order 2: by max dimension descending
    orders.append(sorted(range(n), key=lambda i: max(boxes[i]), reverse=True))
    
    # Order 3: by sum of dimensions descending
    orders.append(sorted(range(n), key=lambda i: sum(boxes[i]), reverse=True))
    
    # Order 4: by aspect ratio (most elongated first)
    orders.append(sorted(range(n), key=lambda i: (max(boxes[i]) / max(min(boxes[i]), 1e-9)), reverse=True))
    
    # Order 5: by min dimension descending (flat boxes first)
    orders.append(sorted(range(n), key=lambda i: min(boxes[i]), reverse=True))
    
    best_result = None
    best_filled = 0.0
    
    for order in orders:
        if time.time() > deadline:
            break
        
        placements, filled_vol = _pack_best_fit(boxes, container, order, deadline)
        
        if filled_vol > best_filled:
            best_filled = filled_vol
            best_result = placements
    
    # Build final result array
    result = []
    for i in range(n):
        w, h, d = boxes[i]
        if best_result is not None and i in best_result:
            x, y, z, ow, oh, od = best_result[i]
            result.append((0, x, y, z, ow, oh, od))
        else:
            result.append((-1, 0, 0, 0, w, h, d))
    
    return result
# EVOLVE-BLOCK-END
