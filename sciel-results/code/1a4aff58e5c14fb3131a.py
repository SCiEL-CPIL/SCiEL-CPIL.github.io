"""Rewritten program for 3D Container Loading."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np
from itertools import permutations

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    # 150s limit, keep 15s margin for safety/overhead
    deadline = start_time + 135.0
    
    # Precompute rotations for each unique box size
    rotation_cache = {}
    def get_rotations(w, h, d):
        key = (w, h, d)
        if key not in rotation_cache:
            rots = set()
            for perm in permutations([w, h, d]):
                rots.add(perm)
            rotation_cache[key] = list(rots)
        return rotation_cache[key]
    
    def pack_with_strategy(boxes, container, sort_key, candidate_order, time_limit=135.0):
        """
        Pack boxes using a given sorting strategy and candidate position ordering.
        
        Args:
            boxes: list of (w, h, d)
            container: (W, H, D)
            sort_key: function to sort boxes (higher first)
            candidate_order: 'zyx', 'xyz', or 'mixed'
        
        Returns:
            placements: list of (bin, x, y, z, w, h, d)
        """
        W, H, D = container
        n = len(boxes)
        start_time = time.time()
        
        # Preprocess: calculate volume and sort
        box_data = []
        for i, (w, h, d) in enumerate(boxes):
            vol = w * h * d
            box_data.append((vol, i, w, h, d))
        
        # Sort by the given key
        box_data.sort(key=sort_key, reverse=True)
        
        placements = [None] * n
        placed_boxes = []
        
        def can_place(x, y, z, bw, bh, bd):
            if x < 0 or y < 0 or z < 0:
                return False
            if x + bw > W or y + bh > H or z + bd > D:
                return False
            for (px, py, pz, pw, ph, pd) in placed_boxes:
                if (x < px + pw and x + bw > px and
                    y < py + ph and y + bh > py and
                    z < pz + pd and z + bd > pz):
                    return False
            return True
        
        def find_best_position(bw, bh, bd):
            """Find the best position based on candidate_order strategy."""
            best_pos = None
            best_score = None
            
            # Generate candidate positions
            candidate_positions = set()
            candidate_positions.add((0, 0, 0))
            
            for (px, py, pz, pw, ph, pd) in placed_boxes:
                candidate_positions.add((px + pw, py, pz))
                candidate_positions.add((px, py + ph, pz))
                candidate_positions.add((px, py, pz + pd))
                candidate_positions.add((px + pw, py + ph, pz))
                candidate_positions.add((px + pw, py, pz + pd))
                candidate_positions.add((px, py + ph, pz + pd))
                candidate_positions.add((px + pw, py + ph, pz + pd))
            
            # Add grid-based candidates
            step = max(1, min(W, H, D) // 25)
            for gx in range(0, W, step):
                for gy in range(0, H, step):
                    for gz in range(0, D, step):
                        candidate_positions.add((gx, gy, gz))
            
            # Score candidates based on ordering strategy
            scored_candidates = []
            for (cx, cy, cz) in candidate_positions:
                if cx + bw > W or cy + bh > H or cz + bd > D:
                    continue
                if not can_place(cx, cy, cz, bw, bh, bd):
                    continue
                
                if candidate_order == 'zyx':
                    # Prioritize lower z, then y, then x
                    score = cz * 1000000 + cy * 1000 + cx
                elif candidate_order == 'xyz':
                    # Prioritize lower x, then y, then z
                    score = cx * 1000000 + cy * 1000 + cz
                else:  # 'mixed'
                    # Balanced scoring
                    score = cx + cy + cz + 0.1 * (cx * cy + cy * cz + cx * cz)
                
                scored_candidates.append((score, (cx, cy, cz)))
            
            if not scored_candidates:
                return None
            
            # Sort by score and pick best
            scored_candidates.sort(key=lambda x: x[0])
            return scored_candidates[0][1]
        
        # Process boxes
        for vol, idx, w, h, d in box_data:
            if time.time() > start_time + time_limit:
                break
            
            rotations = get_rotations(w, h, d)
            best_placement = None
            best_rot_score = None
            
            for (rw, rh, rd) in rotations:
                pos = find_best_position(rw, rh, rd)
                if pos is not None:
                    if candidate_order == 'zyx':
                        score = pos[2] * 1000000 + pos[1] * 1000 + pos[0]
                    elif candidate_order == 'xyz':
                        score = pos[0] * 1000000 + pos[1] * 1000 + pos[2]
                    else:
                        score = pos[0] + pos[1] + pos[2]
                    
                    if best_rot_score is None or score < best_rot_score:
                        best_rot_score = score
                        best_placement = (pos, (rw, rh, rd))
            
            if best_placement is not None:
                (px, py, pz), (pw, ph, pd) = best_placement
                placed_boxes.append((px, py, pz, pw, ph, pd))
                placements[idx] = (0, px, py, pz, pw, ph, pd)
            else:
                placements[idx] = (-1, 0, 0, 0, w, h, d)
        
        # Fill any remaining unplaced boxes
        for i in range(n):
            if placements[i] is None:
                placements[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
        
        return placements
    
    def placed_volume(placements, boxes):
        """Calculate total placed volume."""
        total = 0
        for i, (bin_id, x, y, z, w, h, d) in enumerate(placements):
            if bin_id == 0:
                total += w * h * d
        return total
    
    # Strategy 1: Volume descending, ZYX ordering (prioritize bottom, then back, then left)
    strategy1_key = lambda x: x[0]  # volume
    result1 = pack_with_strategy(boxes, container, strategy1_key, 'zyx', 135.0)
    vol1 = placed_volume(result1, boxes)
    best_result = result1
    best_vol = vol1
    
    if time.time() < deadline - 10:
        # Strategy 2: Max dimension descending, XYZ ordering (prioritize left, then back, then bottom)
        strategy2_key = lambda x: max(x[2], x[3], x[4])
        result2 = pack_with_strategy(boxes, container, strategy2_key, 'xyz', 135.0)
        vol2 = placed_volume(result2, boxes)
        if vol2 > best_vol:
            best_result = result2
            best_vol = vol2
    
    if time.time() < deadline - 10:
        # Strategy 3: Volume descending, XYZ ordering
        strategy3_key = lambda x: x[0]
        result3 = pack_with_strategy(boxes, container, strategy3_key, 'xyz', 135.0)
        vol3 = placed_volume(result3, boxes)
        if vol3 > best_vol:
            best_result = result3
            best_vol = vol3
    
    if time.time() < deadline - 10:
        # Strategy 4: Volume descending, mixed scoring
        strategy4_key = lambda x: x[0]
        result4 = pack_with_strategy(boxes, container, strategy4_key, 'mixed', 135.0)
        vol4 = placed_volume(result4, boxes)
        if vol4 > best_vol:
            best_result = result4
            best_vol = vol4
    
    return best_result
# EVOLVE-BLOCK-END
