"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np
from itertools import product

def _get_rotations(w, h, d):
    """Get all unique rotations of a box."""
    rots = set()
    rots.add((w, h, d))
    rots.add((w, d, h))
    rots.add((h, w, d))
    rots.add((h, d, w))
    rots.add((d, w, h))
    rots.add((d, h, w))
    return list(rots)

def _check_overlap(x, y, z, w, h, d, placed, zmin=0, zmax=None):
    """Check if a box at (x,y,z) with dims (w,h,d) overlaps any placed box.
    Uses z-range pre-filter for speed."""
    for px, py, pz, pw, ph, pd in placed:
        # Quick z-range rejection
        if z + d <= pz or pz + pd <= z:
            continue
        if not (x + w <= px or px + pw <= x or
                y + h <= py or py + ph <= y):
            return True
    return False

def _fits_in_container(x, y, z, w, h, d, W, H, D):
    """Check if box fits within container boundaries."""
    return (x >= 0 and y >= 0 and z >= 0 and
            x + w <= W and y + h <= H and z + d <= D)

def _generate_candidates(placed, W, H, D):
    """Generate candidate positions for placing a new box.
    Includes corner-adjacent positions plus top-face positions."""
    candidates = [(0, 0, 0)]
    for px, py, pz, pw, ph, pd in placed:
        # Corner-adjacent positions (7 faces of the placed box)
        candidates.append((px + pw, py, pz))
        candidates.append((px, py + ph, pz))
        candidates.append((px, py, pz + pd))
        candidates.append((px + pw, py + ph, pz))
        candidates.append((px + pw, py, pz + pd))
        candidates.append((px, py + ph, pz + pd))
        candidates.append((px + pw, py + ph, pz + pd))
        # Top-face positions: place on top, aligned to the placed box's x/y
        candidates.append((px, py, pz + pd))
        candidates.append((px + pw, py, pz + pd))
        candidates.append((px, py + ph, pz + pd))
        candidates.append((px + pw, py + ph, pz + pd))
        # Side positions at the same z-level as the placed box's top
        candidates.append((px + pw, py, pz))
        candidates.append((px, py + ph, pz))
    
    # Deduplicate
    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique

def _try_place(boxes, container, seed=0, strategy='best_fit'):
    """
    Attempt to place boxes into the container.
    Returns a list of placements: (bin, x, y, z, w, h, d) for each box.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Preprocess
    box_rotations = []
    volumes = []
    for w, h, d in boxes:
        box_rotations.append(_get_rotations(w, h, d))
        volumes.append(w * h * d)
    
    container_vol = W * H * D
    if container_vol == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    def _check_time():
        return time.time() < deadline
    
    def _score_result(placement_result):
        """Calculate filled volume fraction."""
        filled = sum(w * h * d for b, x, y, z, w, h, d in placement_result if b == 0)
        return filled / container_vol
    
    def _place_with_order(order, rot_strategy='best', pos_strategy='corner'):
        """
        Place boxes in given order.
        rot_strategy: 'original', 'best', 'min_corner'
        pos_strategy: 'corner', 'lowest'
        """
        if not _check_time():
            return None
        
        placed = []
        result = [None] * n
        failed_indices = []
        
        for idx in order:
            if not _check_time():
                return None
            
            ow, oh, od = boxes[idx]
            rots = box_rotations[idx]
            
            if rot_strategy == 'original':
                rots_to_try = [(ow, oh, od)]
            else:
                rots_to_try = rots
            
            best_pos = None
            best_rot = None
            best_score = float('inf')
            
            for rw, rh, rd in rots_to_try:
                candidates = _generate_candidates(placed, W, H, D)
                candidates.sort(key=lambda p: (p[2], p[1], p[0]))
                
                for cx, cy, cz in candidates:
                    if _fits_in_container(cx, cy, cz, rw, rh, rd, W, H, D):
                        if not _check_overlap(cx, cy, cz, rw, rh, rd, placed):
                            score = cx + cy + cz
                            if score < best_score:
                                best_score = score
                                best_pos = (cx, cy, cz)
                                best_rot = (rw, rh, rd)
                            break  # First valid position for this rotation
            
            if best_pos is not None:
                x, y, z = best_pos
                w, h, d = best_rot
                result[idx] = (0, x, y, z, w, h, d)
                placed.append((x, y, z, w, h, d))
            else:
                result[idx] = (-1, 0, 0, 0, ow, oh, od)
                failed_indices.append(idx)
        
        # Fill in any None entries
        for i in range(n):
            if result[i] is None:
                ow, oh, od = boxes[i]
                result[i] = (-1, 0, 0, 0, ow, oh, od)
        
        # Post-processing: try to place failed boxes into gaps
        if failed_indices and _check_time():
            _try_fill_gaps(failed_indices, placed, result, box_rotations, boxes, W, H, D, _check_time)
        
        return result
    
    def _try_fill_gaps(failed_indices, placed, result, box_rotations, boxes, W, H, D, _check_time):
        """Try to place failed boxes into gaps by searching more candidate positions."""
        # Build a list of placed boxes for reference
        placed_list = list(placed)
        
        for idx in failed_indices:
            if not _check_time():
                break
            
            ow, oh, od = boxes[idx]
            rots = box_rotations[idx]
            placed_vol = sum(pw*ph*pd for _,_,_,pw,ph,pd in placed_list)
            
            # Generate more diverse candidate positions
            extra_candidates = set()
            # Add positions based on all placed box edges
            for px, py, pz, pw, ph, pd in placed_list:
                # Positions along x edges
                for cx in [px, px + pw]:
                    for cy in [py, py + ph]:
                        for cz in [pz, pz + pd]:
                            extra_candidates.add((cx, cy, cz))
                # Positions at the "shadow" of the box (directly above/below in z)
                extra_candidates.add((px, py, pz + pd))
                extra_candidates.add((px + pw, py, pz + pd))
                extra_candidates.add((px, py + ph, pz + pd))
                extra_candidates.add((px + pw, py + ph, pz + pd))
            
            # Also try grid-like positions
            grid_step = max(1, min(W, H, D) // 20)
            for gx in range(0, W, grid_step):
                for gy in range(0, H, grid_step):
                    for gz in range(0, D, grid_step):
                        extra_candidates.add((gx, gy, gz))
            
            # Combine with standard candidates
            std_candidates = _generate_candidates(placed_list, W, H, D)
            all_candidates = list(set(std_candidates) | extra_candidates)
            all_candidates.sort(key=lambda p: (p[2], p[1], p[0]))
            
            placed_best = None
            placed_rot = None
            placed_score = float('inf')
            
            for rw, rh, rd in rots:
                for cx, cy, cz in all_candidates:
                    if _fits_in_container(cx, cy, cz, rw, rh, rd, W, H, D):
                        if not _check_overlap(cx, cy, cz, rw, rh, rd, placed_list):
                            score = cx + cy + cz
                            if score < placed_score:
                                placed_score = score
                                placed_best = (cx, cy, cz)
                                placed_rot = (rw, rh, rd)
                            break
            
            if placed_best is not None:
                x, y, z = placed_best
                w, h, d = placed_rot
                result[idx] = (0, x, y, z, w, h, d)
                placed_list.append((x, y, z, w, h, d))
    
    # Multiple strategies
    best_result = None
    best_score = 0.0
    
    # Strategy 1: Volume descending
    order1 = sorted(range(n), key=lambda i: volumes[i], reverse=True)
    if _check_time():
        r = _place_with_order(order1, 'best', 'corner')
        if r is not None:
            s = _score_result(r)
            if s > best_score:
                best_score = s
                best_result = r
    
    # Strategy 2: Max dimension descending
    order2 = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    if _check_time():
        r = _place_with_order(order2, 'best', 'corner')
        if r is not None:
            s = _score_result(r)
            if s > best_score:
                best_score = s
                best_result = r
    
    # Strategy 3: Min dimension ascending
    order3 = sorted(range(n), key=lambda i: min(boxes[i]))
    if _check_time():
        r = _place_with_order(order3, 'best', 'corner')
        if r is not None:
            s = _score_result(r)
            if s > best_score:
                best_score = s
                best_result = r
    
    # Strategy 4: Random restarts with diverse sorting
    rng = random.Random(seed)
    max_restarts = 40
    for r in range(max_restarts):
        if not _check_time():
            break
        # Mix of random and semi-structured orders
        if r % 3 == 0:
            order_rand = list(range(n))
            rng.shuffle(order_rand)
        elif r % 3 == 1:
            # Sort by volume but with random tie-breaking
            order_rand = sorted(range(n), key=lambda i: (-volumes[i], rng.random()))
        else:
            # Sort by max dimension with random tie-breaking
            order_rand = sorted(range(n), key=lambda i: (-max(boxes[i]), rng.random()))
        res = _place_with_order(order_rand, 'best', 'corner')
        if res is not None:
            s = _score_result(res)
            if s > best_score:
                best_score = s
                best_result = res
    
    # Strategy 5: Layer-based packing
    if _check_time():
        best_layer_result = _try_layer_packing(boxes, container, volumes, box_rotations, deadline)
        if best_layer_result is not None:
            s = _score_result(best_layer_result)
            if s > best_score:
                best_score = s
                best_result = best_layer_result
    
    # Strategy 6: Additional diverse sorting strategies
    if _check_time():
        additional_strategies = [
            sorted(range(n), key=lambda i: volumes[i] / max(1, max(boxes[i])), reverse=True),
            sorted(range(n), key=lambda i: min(boxes[i]) * max(boxes[i]), reverse=True),
            sorted(range(n), key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2] / (max(boxes[i])**3), reverse=True),
        ]
        for order in additional_strategies:
            if not _check_time():
                break
            r = _place_with_order(order, 'best', 'corner')
            if r is not None:
                s = _score_result(r)
                if s > best_score:
                    best_score = s
                    best_result = r
    
    if best_result is None:
        best_result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_result

def _try_layer_packing(boxes, container, volumes, box_rotations, deadline):
    """
    Time-sliced layer decomposition: discretize container height into slabs,
    pack boxes into each slab using 2D shelf heuristic.
    """
    W, H, D = container
    n = len(boxes)
    
    if n == 0:
        return []
    
    start_time = time.time()
    
    def _check_time():
        return time.time() < deadline - 5.0
    
    # Sort boxes by volume descending
    order = sorted(range(n), key=lambda i: volumes[i], reverse=True)
    
    # Try different layer heights
    best_result = None
    best_score = 0.0
    container_vol = W * H * D
    
    # Generate candidate layer heights based on box dimensions
    layer_heights = set()
    layer_heights.add(H)  # Single layer
    if H >= 2:
        layer_heights.add(H // 2)
    if H >= 3:
        layer_heights.add(H // 3)
    
    # Add box heights as potential layer heights
    for i in range(n):
        for rw, rh, rd in box_rotations[i]:
            if rh > 0 and rh <= H:
                layer_heights.add(rh)
    
    # Filter valid layer heights
    layer_heights = sorted([lh for lh in layer_heights if lh > 0 and lh <= H], reverse=True)
    
    # Limit number of layer heights to try
    layer_heights = layer_heights[:12]
    
    for layer_h in layer_heights:
        if not _check_time():
            break
        
        placed = []
        result = [None] * n
        current_z = 0
        
        # Process boxes in order
        for idx in order:
            if not _check_time():
                return None
            
            ow, oh, od = boxes[idx]
            rots = box_rotations[idx]
            
            # Find a rotation and position in current layer
            best_pos = None
            best_rot = None
            best_score_local = float('inf')
            
            for rw, rh, rd in rots:
                # Check if this rotation fits in current layer
                if rh > layer_h:
                    continue
                
                # Generate candidate x,y positions in current layer
                candidates = [(0, 0)]
                for px, py, pz, pw, ph, pd in placed:
                    if pz + pd <= current_z + layer_h and current_z <= pz + pd:
                        candidates.append((px + pw, py))
                        candidates.append((px, py + ph))
                        candidates.append((px + pw, py + ph))
                
                seen = set()
                unique_cands = []
                for cx, cy in candidates:
                    if (cx, cy) not in seen:
                        seen.add((cx, cy))
                        unique_cands.append((cx, cy))
                
                unique_cands.sort(key=lambda p: (p[1], p[0]))
                
                for cx, cy in unique_cands:
                    if cx + rw > W or cy + rd > D:
                        continue
                    
                    overlap = False
                    for px, py, pz, pw, ph, pd in placed:
                        if pz + pd <= current_z or current_z + layer_h <= pz:
                            continue
                        if not (cx + rw <= px or px + pw <= cx or
                                cy + rd <= py or py + ph <= cy):
                            overlap = True
                            break
                    
                    if not overlap:
                        score = cy + cx
                        if score < best_score_local:
                            best_score_local = score
                            best_pos = (cx, cy, current_z)
                            best_rot = (rw, rh, rd)
                        break
            
            if best_pos is not None:
                x, y, z = best_pos
                w, h, d = best_rot
                result[idx] = (0, x, y, z, w, h, d)
                placed.append((x, y, z, w, h, d))
            else:
                # Try to start a new layer
                new_z = current_z + layer_h
                if new_z + layer_h <= H:
                    placed_in_new_layer = False
                    for rw, rh, rd in rots:
                        if rh > layer_h:
                            continue
                        if rw <= W and rd <= D:
                            # Check overlap in new layer
                            overlap = False
                            for px, py, pz, pw, ph, pd in placed:
                                if pz + pd <= new_z or new_z + layer_h <= pz:
                                    continue
                                if not (0 + rw <= px or px + pw <= 0 or
                                        0 + rd <= py or py + ph <= 0):
                                    overlap = True
                                    break
                            if not overlap:
                                result[idx] = (0, 0, 0, new_z, rw, rh, rd)
                                placed.append((0, 0, new_z, rw, rh, rd))
                                current_z = new_z
                                placed_in_new_layer = True
                                break
                    if not placed_in_new_layer:
                        result[idx] = (-1, 0, 0, 0, ow, oh, od)
                else:
                    result[idx] = (-1, 0, 0, 0, ow, oh, od)
        
        for i in range(n):
            if result[i] is None:
                ow, oh, od = boxes[i]
                result[i] = (-1, 0, 0, 0, ow, oh, od)
        
        filled = sum(w * h * d for b, x, y, z, w, h, d in result if b == 0)
        score = filled / container_vol if container_vol > 0 else 0
        
        if score > best_score:
            best_score = score
            best_result = result
    
    return best_result

def _try_place_diverse(boxes, container, seed=0):
    """Try diverse strategies and return the best result."""
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    def _check_time():
        return time.time() < deadline
    
    # Get rotations
    box_rotations = []
    volumes = []
    for w, h, d in boxes:
        box_rotations.append(_get_rotations(w, h, d))
        volumes.append(w * h * d)
    
    container_vol = W * H * D
    if container_vol == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    def _score_result(placement_result):
        filled = sum(w * h * d for b, x, y, z, w, h, d in placement_result if b == 0)
        return filled / container_vol
    
    best_result = None
    best_score = 0.0
    
    # Multiple sorting strategies
    strategies = [
        sorted(range(n), key=lambda i: volumes[i], reverse=True),
        sorted(range(n), key=lambda i: max(boxes[i]), reverse=True),
        sorted(range(n), key=lambda i: min(boxes[i])),
        sorted(range(n), key=lambda i: boxes[i][0] * boxes[i][1], reverse=True),
        sorted(range(n), key=lambda i: boxes[i][0] * boxes[i][2], reverse=True),
        sorted(range(n), key=lambda i: boxes[i][1] * boxes[i][2], reverse=True),
    ]
    
    for order in strategies:
        if not _check_time():
            break
        
        placed = []
        result = [None] * n
        
        for idx in order:
            if not _check_time():
                return None
            
            ow, oh, od = boxes[idx]
            rots = box_rotations[idx]
            
            best_pos = None
            best_rot = None
            best_score_local = float('inf')
            
            for rw, rh, rd in rots:
                candidates = _generate_candidates(placed, W, H, D)
                candidates.sort(key=lambda p: (p[2], p[1], p[0]))
                
                for cx, cy, cz in candidates:
                    if _fits_in_container(cx, cy, cz, rw, rh, rd, W, H, D):
                        if not _check_overlap(cx, cy, cz, rw, rh, rd, placed):
                            score = cx + cy + cz
                            if score < best_score_local:
                                best_score_local = score
                                best_pos = (cx, cy, cz)
                                best_rot = (rw, rh, rd)
                            break
            
            if best_pos is not None:
                x, y, z = best_pos
                w, h, d = best_rot
                result[idx] = (0, x, y, z, w, h, d)
                placed.append((x, y, z, w, h, d))
            else:
                result[idx] = (-1, 0, 0, 0, ow, oh, od)
        
        for i in range(n):
            if result[i] is None:
                ow, oh, od = boxes[i]
                result[i] = (-1, 0, 0, 0, ow, oh, od)
        
        s = _score_result(result)
        if s > best_score:
            best_score = s
            best_result = result
    
    # Random restarts
    rng = random.Random(seed)
    for r in range(20):
        if not _check_time():
            break
        
        order_rand = list(range(n))
        rng.shuffle(order_rand)
        
        placed = []
        result = [None] * n
        
        for idx in order_rand:
            if not _check_time():
                break
            
            ow, oh, od = boxes[idx]
            rots = box_rotations[idx]
            
            best_pos = None
            best_rot = None
            best_score_local = float('inf')
            
            for rw, rh, rd in rots:
                candidates = _generate_candidates(placed, W, H, D)
                candidates.sort(key=lambda p: (p[2], p[1], p[0]))
                
                for cx, cy, cz in candidates:
                    if _fits_in_container(cx, cy, cz, rw, rh, rd, W, H, D):
                        if not _check_overlap(cx, cy, cz, rw, rh, rd, placed):
                            score = cx + cy + cz
                            if score < best_score_local:
                                best_score_local = score
                                best_pos = (cx, cy, cz)
                                best_rot = (rw, rh, rd)
                            break
            
            if best_pos is not None:
                x, y, z = best_pos
                w, h, d = best_rot
                result[idx] = (0, x, y, z, w, h, d)
                placed.append((x, y, z, w, h, d))
            else:
                result[idx] = (-1, 0, 0, 0, ow, oh, od)
        
        for i in range(n):
            if result[i] is None:
                ow, oh, od = boxes[i]
                result[i] = (-1, 0, 0, 0, ow, oh, od)
        
        s = _score_result(result)
        if s > best_score:
            best_score = s
            best_result = result
    
    if best_result is None:
        best_result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_result

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Try the main algorithm
    try:
        result = _try_place(boxes, container, seed=42)
        return result
    except Exception:
        # Fallback to simple greedy
        placements = []
        x, y, z = 0, 0, 0
        row_h, layer_d = 0, 0
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
        return placements
# EVOLVE-BLOCK-END
