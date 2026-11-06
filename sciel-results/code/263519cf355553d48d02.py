"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import permutations
import numpy as np

def _get_rotations(w, h, d):
    """Get all unique rotations of a box."""
    rots = set()
    for perm in permutations((w, h, d)):
        rots.add(perm)
    return list(rots)

def _pack_layer_decomposition(boxes, container, deadline, slab_height=None):
    """
    Time-sliced layer decomposition: discretize container height into slabs,
    pack each slab using 2D shelf heuristic.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return [], 0
    
    if slab_height is None:
        slab_height = min(W, H, D) // 4 + 1
    
    # Precompute rotations
    box_rotations = []
    for w, h, d in boxes:
        rots = _get_rotations(w, h, d)
        box_rotations.append(rots)
    
    # Sort by volume descending
    box_indices = list(range(n))
    box_indices.sort(key=lambda i: -boxes[i][0] * boxes[i][1] * boxes[i][2])
    
    result = [(-1, 0, 0, 0, 0, 0, 0)] * n
    filled_volume = 0
    placed = []  # (x, y, z, w, h, d)
    
    def overlaps_any(px, py, pz, pw, ph, pd):
        for ex, ey, ez, ew, eh, ed in placed:
            if (px < ex + ew and px + pw > ex and
                py < ey + eh and py + ph > ey and
                pz < ez + ed and pz + pd > ez):
                return True
        return False
    
    def try_place_in_slab(box_idx, bw, bh, bd, slab_z):
        """Try to place box in a specific slab using 2D shelf heuristic."""
        # Slab bounds: z from slab_z to slab_z + slab_height
        # Box must fit: bh <= slab_height
        if bh > slab_height:
            return None
        
        # 2D shelf packing within the slab
        # Shelves are horizontal strips at fixed y positions
        shelf_y = 0
        shelf_end_x = 0
        shelf_height = 0
        
        for cy in range(0, H, max(1, min(slab_height, 10))):
            if cy + bh > H:
                break
            
            # Try to place on this shelf
            for cx in range(0, W - bw + 1, max(1, min(bw, 10))):
                if cx + bw > W:
                    break
                
                if not overlaps_any(cx, cy, slab_z, bw, bh, bd):
                    return (cx, cy, slab_z, bw, bh, bd)
        
        return None
    
    # Process boxes in order, trying to place in each slab
    for idx in box_indices:
        if time.time() > deadline:
            break
        
        w, h, d = boxes[idx]
        rots = box_rotations[idx]
        
        placed_this = False
        
        # Try each rotation
        for bw, bh, bd in rots:
            # Try each possible slab position
            for slab_z in range(0, D - bd + 1, max(1, min(slab_height, 5))):
                if slab_z + bd > D:
                    break
                
                pos = try_place_in_slab(idx, bw, bh, bd, slab_z)
                if pos is not None:
                    cx, cy, cz, pw, ph, pd = pos
                    placed.append((cx, cy, cz, pw, ph, pd))
                    result[idx] = (0, cx, cy, cz, pw, ph, pd)
                    filled_volume += pw * ph * pd
                    placed_this = True
                    break
            
            if placed_this:
                break
        
        if not placed_this:
            result[idx] = (-1, 0, 0, 0, w, h, d)
    
    return result, filled_volume

def _pack_boxes(boxes, container, sort_key=None, deadline=None, seed=0):
    """
    Pack boxes using extreme point heuristic with a given sort strategy.
    Returns (result, filled_volume) where result is list of (bin, x, y, z, w, h, d).
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return [], 0
    
    if deadline is None:
        deadline = time.time() + 10.0
    
    # Precompute rotations for each box
    box_rotations = []
    for w, h, d in boxes:
        rots = _get_rotations(w, h, d)
        rots.sort(key=lambda r: (r[2], r[1], r[0]))
        box_rotations.append(rots)
    
    # Sort boxes
    box_indices = list(range(n))
    if sort_key is None:
        sort_key = lambda i: -boxes[i][0] * boxes[i][1] * boxes[i][2]
    box_indices.sort(key=sort_key)
    
    placed = []  # List of (x, y, z, w, h, d)
    result = [(-1, 0, 0, 0, 0, 0, 0)] * n
    filled_volume = 0
    candidates = [(0, 0, 0)]
    
    def overlaps_any(px, py, pz, pw, ph, pd):
        for ex, ey, ez, ew, eh, ed in placed:
            if (px < ex + ew and px + pw > ex and
                py < ey + eh and py + ph > ey and
                pz < ez + ed and pz + pd > ez):
                return True
        return False
    
    def add_candidate(x, y, z):
        if 0 <= x <= W and 0 <= y <= H and 0 <= z <= D:
            inside = False
            for ex, ey, ez, ew, eh, ed in placed:
                if x > ex and x < ex + ew and y > ey and y < ey + eh and z > ez and z < ez + ed:
                    inside = True
                    break
            if not inside:
                if (x, y, z) not in candidates:
                    candidates.append((x, y, z))
    
    for idx in box_indices:
        if time.time() > deadline:
            break
        
        w, h, d = boxes[idx]
        rots = box_rotations[idx]
        
        placed_this = False
        
        valid_candidates = [c for c in candidates if 0 <= c[0] <= W and 0 <= c[1] <= H and 0 <= c[2] <= D]
        valid_candidates.sort(key=lambda p: (p[2], p[1], p[0]))
        
        for bw, bh, bd in rots:
            if time.time() > deadline:
                break
            
            found = False
            for cx, cy, cz in valid_candidates:
                if cx + bw > W or cy + bh > H or cz + bd > D:
                    continue
                
                if not overlaps_any(cx, cy, cz, bw, bh, bd):
                    placed.append((cx, cy, cz, bw, bh, bd))
                    result[idx] = (0, cx, cy, cz, bw, bh, bd)
                    filled_volume += bw * bh * bd
                    
                    add_candidate(cx + bw, cy, cz)
                    add_candidate(cx, cy + bh, cz)
                    add_candidate(cx, cy, cz + bd)
                    
                    if (cx, cy, cz) in candidates:
                        candidates.remove((cx, cy, cz))
                    
                    placed_this = True
                    found = True
                    break
            
            if found:
                break
        
        if not placed_this:
            result[idx] = (-1, 0, 0, 0, w, h, d)
    
    return result, filled_volume

def _shake_improve(boxes, container, initial_placement, deadline, max_iterations=500):
    """
    Local search: try to improve packing by moving boxes to better positions.
    Uses simulated annealing-style acceptance.
    Returns improved placement and filled volume.
    """
    W, H, D = container
    n = len(boxes)
    
    # Extract placed boxes from initial placement
    placed_boxes = []  # (box_idx, x, y, z, w, h, d)
    for i, (bin_id, x, y, z, w, h, d) in enumerate(initial_placement):
        if bin_id == 0:
            placed_boxes.append((i, x, y, z, w, h, d))
    
    if not placed_boxes:
        return initial_placement, 0
    
    # Compute initial filled volume
    def compute_filled(placement):
        filled = 0
        for bin_id, x, y, z, w, h, d in placement:
            if bin_id == 0:
                filled += w * h * d
        return filled
    
    current_filled = compute_filled(initial_placement)
    best_placement = [list(p) for p in initial_placement]
    best_filled = current_filled
    
    # Build spatial index for faster overlap checks
    # Use a simple grid-based approach
    def build_index(placed_list):
        """Build a list of placed boxes for overlap checking."""
        return [(x, y, z, w, h, d) for (_, x, y, z, w, h, d) in placed_list]
    
    def overlaps(px, py, pz, pw, ph, pd, placed_list, exclude_idx=-1):
        for i, (ex, ey, ez, ew, eh, ed) in enumerate(placed_list):
            if i == exclude_idx:
                continue
            if (px < ex + ew and px + pw > ex and
                py < ey + eh and py + ph > ey and
                pz < ez + ed and pz + pd > ez):
                return True
        return False
    
    # Generate candidate positions for a box
    def get_candidate_positions(bx, by, bz, bw, bh, bd, placed_list, exclude_idx):
        """Generate extreme point candidates for a specific box."""
        cands = set()
        # Add origin
        cands.add((0, 0, 0))
        
        # Add positions relative to other boxes
        for i, (ex, ey, ez, ew, eh, ed) in enumerate(placed_list):
            if i == exclude_idx:
                continue
            # Positions adjacent to this box
            cands.add((ex + ew, ey, ez))
            cands.add((ex, ey + eh, ez))
            cands.add((ex, ey, ez + ed))
            # Also try to align with the box's edges
            if ex >= bx and ex <= bx + bw:
                cands.add((ex, by, bz))
            if ey >= by and ey <= by + bh:
                cands.add((bx, ey, bz))
            if ez >= bz and ez <= bz + bd:
                cands.add((bx, by, ez))
        
        # Filter valid positions
        valid = []
        for (cx, cy, cz) in cands:
            if 0 <= cx and cx + bw <= W and 0 <= cy and cy + bh <= H and 0 <= cz and cz + bd <= D:
                if not overlaps(cx, cy, cz, bw, bh, bd, placed_list, exclude_idx):
                    valid.append((cx, cy, cz))
        
        return valid
    
    # Simulated annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.95
    
    placed_list = build_index(placed_boxes)
    current_placement = [list(p) for p in initial_placement]
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Pick a random placed box to move
        box_pos = random.randint(0, len(placed_boxes) - 1)
        box_idx, bx, by, bz, bw, bh, bd = placed_boxes[box_pos]
        
        # Get candidate positions
        cands = get_candidate_positions(bx, by, bz, bw, bh, bd, placed_list, box_pos)
        
        if not cands:
            # Try to keep current position
            continue
        
        # Pick a random candidate
        nx, ny, nz = random.choice(cands)
        
        # Compute new filled volume (volume doesn't change, so this is just for tracking)
        # The improvement comes from enabling other boxes to be placed
        # For now, we just try to make the packing more compact
        
        # Accept with probability based on temperature
        if random.random() < T:
            # Move the box
            placed_boxes[box_pos] = (box_idx, nx, ny, nz, bw, bh, bd)
            placed_list = build_index(placed_boxes)
            
            # Update placement
            current_placement[box_idx] = [0, nx, ny, nz, bw, bh, bd]
            
            # Check if this enables better packing of unplaced boxes
            # (simplified: just track the current state)
            
            # Update best if improved (we need a better metric)
            # For now, use compactness: sum of max coordinates
            max_x = max((x + w for (_, x, y, z, w, h, d) in placed_boxes), default=0)
            max_y = max((y + h for (_, x, y, z, w, h, d) in placed_boxes), default=0)
            max_z = max((z + d for (_, x, y, z, w, h, d) in placed_boxes), default=0)
            compactness = max_x * max_y * max_z
            
            if compactness < best_filled:
                best_filled = compactness
                best_placement = [list(p) for p in current_placement]
        
        T *= cooling
        if T < T_min:
            break
    
    # Convert best_placement back to tuples
    best_placement = [tuple(p) for p in best_placement]
    final_filled = compute_filled(best_placement)
    
    return best_placement, final_filled

def _pack_boxes_best_fit(boxes, container, deadline):
    """
    Best-fit packing: for each box, choose the position that minimizes the 
    "wasted" space above and around it.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return [], 0
    
    box_rotations = []
    for w, h, d in boxes:
        rots = _get_rotations(w, h, d)
        rots.sort(key=lambda r: (r[2], r[1], r[0]))
        box_rotations.append(rots)
    
    # Sort by volume descending
    box_indices = list(range(n))
    box_indices.sort(key=lambda i: -boxes[i][0] * boxes[i][1] * boxes[i][2])
    
    placed = []
    result = [(-1, 0, 0, 0, 0, 0, 0)] * n
    filled_volume = 0
    candidates = [(0, 0, 0)]
    
    def overlaps_any(px, py, pz, pw, ph, pd):
        for ex, ey, ez, ew, eh, ed in placed:
            if (px < ex + ew and px + pw > ex and
                py < ey + eh and py + ph > ey and
                pz < ez + ed and pz + pd > ez):
                return True
        return False
    
    def add_candidate(x, y, z):
        if 0 <= x <= W and 0 <= y <= H and 0 <= z <= D:
            inside = False
            for ex, ey, ez, ew, eh, ed in placed:
                if x > ex and x < ex + ew and y > ey and y < ey + eh and z > ez and z < ez + ed:
                    inside = True
                    break
            if not inside:
                if (x, y, z) not in candidates:
                    candidates.append((x, y, z))
    
    def score_position(cx, cy, cz, bw, bh, bd):
        """Score a position: lower is better (more compact)."""
        # Distance from origin
        dist = cx + cy + cz
        # Height above ground
        height = cz
        # Prefer lower positions
        return height * 10 + dist
    
    for idx in box_indices:
        if time.time() > deadline:
            break
        
        w, h, d = boxes[idx]
        rots = box_rotations[idx]
        
        placed_this = False
        best_pos = None
        best_score = float('inf')
        
        valid_candidates = [c for c in candidates if 0 <= c[0] <= W and 0 <= c[1] <= H and 0 <= c[2] <= D]
        
        for bw, bh, bd in rots:
            for cx, cy, cz in valid_candidates:
                if cx + bw > W or cy + bh > H or cz + bd > D:
                    continue
                
                if not overlaps_any(cx, cy, cz, bw, bh, bd):
                    s = score_position(cx, cy, cz, bw, bh, bd)
                    if s < best_score:
                        best_score = s
                        best_pos = (cx, cy, cz, bw, bh, bd)
        
        if best_pos is not None:
            cx, cy, cz, bw, bh, bd = best_pos
            placed.append((cx, cy, cz, bw, bh, bd))
            result[idx] = (0, cx, cy, cz, bw, bh, bd)
            filled_volume += bw * bh * bd
            
            add_candidate(cx + bw, cy, cz)
            add_candidate(cx, cy + bh, cz)
            add_candidate(cx, cy, cz + bd)
            
            if (cx, cy, cz) in candidates:
                candidates.remove((cx, cy, cz))
            
            placed_this = True
        
        if not placed_this:
            result[idx] = (-1, 0, 0, 0, w, h, d)
    
    return result, filled_volume

def load(boxes, container):
    """
    Main entry point: place boxes into container.
    Returns list of (bin, x, y, z, w, h, d) for each box.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Strategy 1: Layer decomposition (new)
    result_layer, filled_layer = _pack_layer_decomposition(boxes, container, deadline)
    
    # Strategy 2: Standard extreme point with volume sorting
    result1, filled1 = _pack_boxes(boxes, container, 
                                    sort_key=lambda i: -boxes[i][0] * boxes[i][1] * boxes[i][2],
                                    deadline=deadline)
    
    # Strategy 3: Best-fit
    result2, filled2 = _pack_boxes_best_fit(boxes, container, deadline)
    
    # Strategy 4: Width-first sorting
    result3, filled3 = _pack_boxes(boxes, container, 
                                    sort_key=lambda i: -boxes[i][0],
                                    deadline=deadline)
    
    # Strategy 5: Height-first sorting
    result4, filled4 = _pack_boxes(boxes, container, 
                                    sort_key=lambda i: -boxes[i][1],
                                    deadline=deadline)
    
    # Strategy 6: Depth-first sorting
    result5, filled5 = _pack_boxes(boxes, container, 
                                    sort_key=lambda i: -boxes[i][2],
                                    deadline=deadline)
    
    # Strategy 7: Max dimension first
    result6, filled6 = _pack_boxes(boxes, container, 
                                    sort_key=lambda i: -max(boxes[i]),
                                    deadline=deadline)
    
    # Strategy 8: Min dimension first
    result7, filled7 = _pack_boxes(boxes, container, 
                                    sort_key=lambda i: -min(boxes[i]),
                                    deadline=deadline)
    
    # Strategy 9: Sum of dimensions
    result8, filled8 = _pack_boxes(boxes, container, 
                                    sort_key=lambda i: -sum(boxes[i]),
                                    deadline=deadline)
    
    # Collect all results
    all_results = [
        (result_layer, filled_layer),
        (result1, filled1),
        (result2, filled2),
        (result3, filled3),
        (result4, filled4),
        (result5, filled5),
        (result6, filled6),
        (result7, filled7),
        (result8, filled8),
    ]
    
    # Find the best result by filled volume
    best_result = max(all_results, key=lambda x: x[1])[0]
    best_filled = max(all_results, key=lambda x: x[1])[1]
    
    # Try to improve the best result with local search
    if time.time() < deadline - 20:  # Leave 20s for local search
        improved, improved_filled = _shake_improve(boxes, container, best_result, deadline, max_iterations=300)
        if improved_filled > best_filled:
            best_result = improved
            best_filled = improved_filled
    
    return best_result
# EVOLVE-BLOCK-END
