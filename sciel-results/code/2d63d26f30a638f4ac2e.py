"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def _get_rotations(w, h, d):
    """Return all unique rotations of a box (w,h,d)."""
    rots = set()
    rots.add((w, h, d))
    rots.add((w, d, h))
    rots.add((h, w, d))
    rots.add((h, d, w))
    rots.add((d, w, h))
    rots.add((d, h, w))
    return list(rots)

def _check_overlap(x, y, z, w, h, d, placed):
    """Check if box at (x,y,z) with dims (w,h,d) overlaps with any placed box."""
    for (px, py, pz, pw, ph, pd) in placed:
        if x < px + pw and px < x + w and \
           y < py + ph and py < y + h and \
           z < pz + pd and pz < z + d:
            return True
    return False

def _find_position(x, y, z, w, h, d, placed, W, H, D):
    """Try to place box at given position, return True if successful."""
    if x + w > W or y + h > H or z + d > D:
        return False
    if _check_overlap(x, y, z, w, h, d, placed):
        return False
    placed.append((x, y, z, w, h, d))
    return True

def _pack_single_strategy(boxes, container, seed, deadline):
    """
    Single packing strategy:
    1. Sort boxes by a heuristic
    2. For each box, try all rotations and find best position
    """
    W, H, D = container
    n = len(boxes)
    rng = random.Random(seed)
    
    # Sort boxes by volume descending, with some randomization
    indexed_boxes = list(range(n))
    volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
    
    # Different sorting strategies based on seed
    strategy = seed % 4
    if strategy == 0:
        # Sort by volume descending
        indexed_boxes.sort(key=lambda i: -volumes[i])
    elif strategy == 1:
        # Sort by max dimension descending
        indexed_boxes.sort(key=lambda i: -max(boxes[i]))
    elif strategy == 2:
        # Sort by surface area descending
        indexed_boxes.sort(key=lambda i: -(2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2])))
    else:
        # Sort by volume descending with random tie-breaking
        indexed_boxes.sort(key=lambda i: (-volumes[i], rng.random()))
    
    placed = []  # List of (x, y, z, w, h, d)
    result = [-1] * n  # -1 means not placed
    
    # Track occupied space for faster lookup
    # We'll use a simple approach: for each box, try candidate positions
    
    for idx in indexed_boxes:
        if time.time() > deadline:
            break
        
        w0, h0, d0 = boxes[idx]
        rots = _get_rotations(w0, h0, d0)
        
        best_pos = None
        best_rot = None
        best_score = -1
        
        # Candidate positions: try corners and along edges
        # Generate candidate x, y, z positions
        candidate_xs = [0]
        candidate_ys = [0]
        candidate_zs = [0]
        
        for (px, py, pz, pw, ph, pd) in placed:
            candidate_xs.append(px + pw)
            candidate_ys.append(py + ph)
            candidate_zs.append(pz + pd)
        
        # Also add some intermediate positions
        for x in candidate_xs[:]:
            if x + w0 <= W:
                pass
        for y in candidate_ys[:]:
            if y + h0 <= H:
                pass
        for z in candidate_zs[:]:
            if z + d0 <= D:
                pass
        
        # Deduplicate and sort
        candidate_xs = sorted(set(x for x in candidate_xs if x >= 0 and x + min(w0, h0, d0) <= W))
        candidate_ys = sorted(set(y for y in candidate_ys if y >= 0 and y + min(w0, h0, d0) <= H))
        candidate_zs = sorted(set(z for z in candidate_zs if z >= 0 and z + min(w0, h0, d0) <= D))
        
        # Limit candidates to avoid explosion
        if len(candidate_xs) > 20:
            candidate_xs = candidate_xs[:20]
        if len(candidate_ys) > 20:
            candidate_ys = candidate_ys[:20]
        if len(candidate_zs) > 20:
            candidate_zs = candidate_zs[:20]
        
        # Try each rotation and each candidate position
        for rot in rots:
            rw, rh, rd = rot
            for cx in candidate_xs:
                for cy in candidate_ys:
                    for cz in candidate_zs:
                        if cx + rw > W or cy + rh > H or cz + rd > D:
                            continue
                        if _check_overlap(cx, cy, cz, rw, rh, rd, placed):
                            continue
                        # Score: prefer lower positions (gravity)
                        score = -(cx + cy + cz)
                        if score > best_score:
                            best_score = score
                            best_pos = (cx, cy, cz)
                            best_rot = rot
        
        if best_pos is not None:
            placed.append((best_pos[0], best_pos[1], best_pos[2], best_rot[0], best_rot[1], best_rot[2]))
            result[idx] = 0
            # Store placement info for later
            # We'll rebuild at the end
        else:
            result[idx] = -1
    
    # Build final placements
    placements = []
    placed_map = {}
    # Re-derive which boxes are placed and where
    # This is tricky since we need to match placed entries to box indices
    # Let's redo this more carefully
    
    return result, placed

def _pack_shelf(boxes, container, seed, deadline):
    """Shelf-based packing with rotation optimization."""
    W, H, D = container
    n = len(boxes)
    rng = random.Random(seed)
    
    # Sort by volume descending
    indexed_boxes = list(range(n))
    volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
    indexed_boxes.sort(key=lambda i: -volumes[i])
    
    placed = []  # (box_idx, x, y, z, w, h, d)
    result = [-1] * n
    
    # Shelf packing: maintain shelves in each layer
    x, y, z = 0, 0, 0
    row_h = 0
    layer_d = 0
    
    for idx in indexed_boxes:
        if time.time() > deadline:
            break
        
        w0, h0, d0 = boxes[idx]
        rots = _get_rotations(w0, h0, d0)
        
        best_rot = None
        best_score = -1
        
        for rot in rots:
            rw, rh, rd = rot
            # Check if fits in current position
            if x + rw <= W and y + rh <= H and z + rd <= D:
                # Score: prefer orientations that minimize waste
                score = -(rw * rh * rd)
                # Prefer keeping x small
                score -= rw * 0.1
                if score > best_score:
                    best_score = score
                    best_rot = rot
        
        if best_rot is None:
            # Try to advance shelf
            x, y, row_h = 0, y + row_h, 0
            for rot in rots:
                rw, rh, rd = rot
                if x + rw <= W and y + rh <= H and z + rd <= D:
                    score = -(rw * rh * rd)
                    score -= rw * 0.1
                    if score > best_score:
                        best_score = score
                        best_rot = rot
        
        if best_rot is None:
            # Try to advance layer
            y, z, layer_d, x, row_h = 0, z + layer_d, 0, 0, 0
            for rot in rots:
                rw, rh, rd = rot
                if x + rw <= W and y + rh <= H and z + rd <= D:
                    score = -(rw * rh * rd)
                    score -= rw * 0.1
                    if score > best_score:
                        best_score = score
                        best_rot = rot
        
        if best_rot is not None:
            placed.append((idx, x, y, z, best_rot[0], best_rot[1], best_rot[2]))
            result[idx] = 0
            x += best_rot[0]
            row_h = max(row_h, best_rot[1])
            layer_d = max(layer_d, best_rot[2])
        else:
            result[idx] = -1
    
    return result, placed

def _build_placements(result, boxes, container, placed):
    """Build final placement list from result and placed info."""
    W, H, D = container
    n = len(boxes)
    placements = []
    
    # Create a mapping from box index to placement
    placed_by_idx = {}
    for entry in placed:
        if len(entry) == 7:
            idx, x, y, z, w, h, d = entry
            placed_by_idx[idx] = (x, y, z, w, h, d)
        else:
            # From _pack_single_strategy, placed is (x, y, z, w, h, d)
            pass
    
    for i in range(n):
        if result[i] == 0:
            if i in placed_by_idx:
                x, y, z, w, h, d = placed_by_idx[i]
                placements.append((0, x, y, z, w, h, d))
            else:
                # Should not happen, but fallback
                placements.append((0, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]))
        else:
            placements.append((-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]))
    
    return placements

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin out of 120s
    
    container_volume = W * H * D
    best_fill = -1
    best_placements = None
    
    # Try multiple strategies
    strategies = [
        ("shelf", _pack_shelf),
    ]
    
    # Try shelf packing with different seeds
    for seed in range(20):
        if time.time() > deadline:
            break
        
        result, placed = _pack_shelf(boxes, container, seed, deadline)
        
        # Calculate fill
        filled_volume = sum(boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n) if result[i] == 0)
        fill = filled_volume / container_volume if container_volume > 0 else 0
        
        if fill > best_fill:
            best_fill = fill
            # Build placements
            placed_by_idx = {}
            for entry in placed:
                idx, x, y, z, w, h, d = entry
                placed_by_idx[idx] = (x, y, z, w, h, d)
            
            placements = []
            for i in range(n):
                if result[i] == 0:
                    if i in placed_by_idx:
                        x, y, z, w, h, d = placed_by_idx[i]
                        placements.append((0, x, y, z, w, h, d))
                    else:
                        placements.append((0, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]))
                else:
                    placements.append((-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]))
            
            best_placements = placements
    
    if best_placements is None:
        # Fallback: simple shelf packing
        placements = []
        x, y, z, row_h, layer_d = 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d, x, row_h = 0, z + layer_d, 0, 0, 0
            if z + d > D:
                placements.append((-1, 0, 0, 0, w, h, d))
                continue
            placements.append((0, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        best_placements = placements
    
    return best_placements
# EVOLVE-BLOCK-END
