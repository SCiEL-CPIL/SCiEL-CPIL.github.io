"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np
from itertools import product

def _try_place(boxes, container, seed=0):
    """
    Attempt to place boxes into the container using a multi-strategy approach.
    Returns a list of placements: (bin, x, y, z, w, h, d) for each box.
    bin=0 means loaded at (x,y,z) with dimensions (w,h,d).
    bin=-1 means left out (dimensions are the original box dims).
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115.0  # 115s safety margin out of 120s
    
    # Preprocess boxes: for each box, consider all 6 rotations
    box_rotations = []
    for w, h, d in boxes:
        rots = set()
        rots.add((w, h, d))
        rots.add((w, d, h))
        rots.add((h, w, d))
        rots.add((h, d, w))
        rots.add((d, w, h))
        rots.add((d, h, w))
        box_rotations.append(list(rots))
    
    box_indices = list(range(n))
    volumes = [w*h*d for w, h, d in boxes]
    
    def _check_time():
        return time.time() < deadline
    
    def _place_single_strategy(order, rotation_choice):
        """
        Place boxes in given order with given rotation choice strategy.
        rotation_choice: 0=original, 1=first fit, 2=best fit (minimize sum of coords)
        """
        if not _check_time():
            return None
        
        placed_boxes = []
        placement_result = [None] * n
        
        def _overlaps(x, y, z, w, h, d):
            for px, py, pz, pw, ph, pd in placed_boxes:
                if not (x + w <= px or px + pw <= x or
                        y + h <= py or py + ph <= y or
                        z + d <= pz or pz + pd <= z):
                    return True
            return False
        
        def _fits(x, y, z, w, h, d):
            return (x >= 0 and y >= 0 and z >= 0 and
                    x + w <= W and y + h <= H and z + d <= D)
        
        def _find_position(w, h, d):
            """Find a valid position for a box with given dimensions using corner-based heuristic."""
            candidates = set()
            candidates.add((0, 0, 0))
            
            for px, py, pz, pw, ph, pd in placed_boxes:
                for dx in [0, pw]:
                    for dy in [0, ph]:
                        for dz in [0, pd]:
                            candidates.add((px + dx, py + dy, pz + dz))
            
            # Sort candidates by z, y, x to prefer lower, more compact positions
            unique_candidates = sorted(candidates, key=lambda p: (p[2], p[1], p[0]))
            
            for cx, cy, cz in unique_candidates:
                if _fits(cx, cy, cz, w, h, d) and not _overlaps(cx, cy, cz, w, h, d):
                    return (cx, cy, cz)
            
            return None
        
        for idx in order:
            if not _check_time():
                return None
            
            orig_w, orig_h, orig_d = boxes[idx]
            
            if rotation_choice == 0:
                w, h, d = orig_w, orig_h, orig_d
                pos = _find_position(w, h, d)
                if pos is None:
                    placement_result[idx] = (-1, 0, 0, 0, orig_w, orig_h, orig_d)
                else:
                    placement_result[idx] = (0, pos[0], pos[1], pos[2], w, h, d)
                    placed_boxes.append((pos[0], pos[1], pos[2], w, h, d))
            elif rotation_choice == 1:
                best_rot = None
                for rot in box_rotations[idx]:
                    w, h, d = rot
                    pos = _find_position(w, h, d)
                    if pos is not None:
                        best_rot = (w, h, d, pos)
                        break
                if best_rot is None:
                    placement_result[idx] = (-1, 0, 0, 0, orig_w, orig_h, orig_d)
                else:
                    w, h, d, pos = best_rot
                    placement_result[idx] = (0, pos[0], pos[1], pos[2], w, h, d)
                    placed_boxes.append((pos[0], pos[1], pos[2], w, h, d))
            else:
                best_rot = None
                best_score = float('inf')
                for rot in box_rotations[idx]:
                    w, h, d = rot
                    pos = _find_position(w, h, d)
                    if pos is not None:
                        score = pos[0] + pos[1] + pos[2]
                        if score < best_score:
                            best_score = score
                            best_rot = (w, h, d, pos)
                if best_rot is None:
                    placement_result[idx] = (-1, 0, 0, 0, orig_w, orig_h, orig_d)
                else:
                    w, h, d, pos = best_rot
                    placement_result[idx] = (0, pos[0], pos[1], pos[2], w, h, d)
                    placed_boxes.append((pos[0], pos[1], pos[2], w, h, d))
        
        for i in range(n):
            if placement_result[i] is None:
                orig_w, orig_h, orig_d = boxes[i]
                placement_result[i] = (-1, 0, 0, 0, orig_w, orig_h, orig_d)
        
        return placement_result
    
    def _score_result(result):
        if result is None:
            return 0
        filled_vol = sum(w*h*d for bin_id, x, y, z, w, h, d in result if bin_id == 0)
        container_vol = W * H * D
        return filled_vol / container_vol if container_vol > 0 else 0
    
    # Strategy 1: Sort by volume descending
    order1 = sorted(box_indices, key=lambda i: volumes[i], reverse=True)
    result1 = _place_single_strategy(order1, 2)
    score1 = _score_result(result1)
    
    # Strategy 2: Sort by max dimension descending
    order2 = sorted(box_indices, key=lambda i: max(boxes[i]), reverse=True)
    result2 = _place_single_strategy(order2, 2) if _check_time() else None
    score2 = _score_result(result2)
    
    # Strategy 3: Sort by minimum dimension ascending
    order3 = sorted(box_indices, key=lambda i: min(boxes[i]))
    result3 = _place_single_strategy(order3, 2) if _check_time() else None
    score3 = _score_result(result3)
    
    # Strategy 4: Sort by aspect ratio (max/min) descending
    order4 = sorted(box_indices, key=lambda i: max(boxes[i]) / max(min(boxes[i]), 1e-9), reverse=True)
    result4 = _place_single_strategy(order4, 2) if _check_time() else None
    score4 = _score_result(result4)
    
    # Strategy 5: Sort by surface area descending
    order5 = sorted(box_indices, key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True)
    result5 = _place_single_strategy(order5, 2) if _check_time() else None
    score5 = _score_result(result5)
    
    best_result = None
    best_score = max(score1, score2, score3, score4, score5)
    
    for res, sc in [(result1, score1), (result2, score2), (result3, score3), (result4, score4), (result5, score5)]:
        if res is not None and sc == best_score:
            best_result = res
            break
    
    # Try random restarts if we have time
    rng = random.Random(seed)
    max_restarts = 30
    for r in range(max_restarts):
        if not _check_time():
            break
        order_rand = box_indices[:]
        rng.shuffle(order_rand)
        result_r = _place_single_strategy(order_rand, 2)
        if result_r is not None:
            score_r = _score_result(result_r)
            if score_r > best_score:
                best_score = score_r
                best_result = result_r
    
    if best_result is None:
        best_result = []
        for w, h, d in boxes:
            best_result.append((-1, 0, 0, 0, w, h, d))
    
    return best_result

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    try:
        result = _try_place(boxes, container, seed=42)
        return result
    except Exception:
        placements, x, y, z, row_h, layer_d = [], 0, 0, 0, 0, 0
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
