"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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
    # 120s limit, keep 10s margin for safety/overhead
    deadline = start_time + 110.0
    
    # Preprocess: calculate volume and sort by volume descending
    box_data = []
    for i, (w, h, d) in enumerate(boxes):
        vol = w * h * d
        box_data.append((vol, i, w, h, d))
    
    # Sort by volume descending (best-first)
    box_data.sort(key=lambda x: -x[0])
    
    # Get all 6 unique rotations for a box
    def get_rotations(w, h, d):
        rotations = set()
        for perm in permutations([w, h, d]):
            rotations.add(perm)
        return list(rotations)
    
    # Best-fit decreasing with rotation search
    # We'll use a simple grid-based approach with backtracking-lite
    
    # For efficiency, use a simple placement strategy:
    # Try to place each box (sorted by volume) in the best position
    # that minimizes wasted space
    
    placements = [None] * n
    placed_count = 0
    
    # Track used space with a 3D occupancy grid
    # But grid might be too large, so use a list of placed boxes for overlap check
    
    placed_boxes = []  # list of (x, y, z, w, h, d)
    
    def can_place(x, y, z, bw, bh, bd):
        # Check bounds
        if x < 0 or y < 0 or z < 0:
            return False
        if x + bw > W or y + bh > H or z + bd > D:
            return False
        # Check overlap with placed boxes
        for (px, py, pz, pw, ph, pd) in placed_boxes:
            if (x < px + pw and x + bw > px and
                y < py + ph and y + bh > py and
                z < pz + pd and z + bd > pz):
                return False
        return True
    
    def find_best_position(bw, bh, bd):
        """Find the position that minimizes the max coordinate (closest to origin)."""
        best_pos = None
        best_score = None
        
        # Generate candidate positions:
        # 1. Origin
        # 2. Along edges of existing placed boxes
        candidate_positions = set()
        candidate_positions.add((0, 0, 0))
        
        # Add positions adjacent to existing boxes
        for (px, py, pz, pw, ph, pd) in placed_boxes:
            # Right of box
            candidate_positions.add((px + pw, py, pz))
            # Above box
            candidate_positions.add((px, py + ph, pz))
            # Front of box
            candidate_positions.add((px, py, pz + pd))
            # Combinations
            candidate_positions.add((px + pw, py + ph, pz))
            candidate_positions.add((px + pw, py, pz + pd))
            candidate_positions.add((px, py + ph, pz + pd))
            candidate_positions.add((px + pw, py + ph, pz + pd))
        
        # Also add some grid-based candidates
        step = max(1, min(W, H, D) // 20)
        for gx in range(0, W, step):
            for gy in range(0, H, step):
                for gz in range(0, D, step):
                    candidate_positions.add((gx, gy, gz))
        
        # Filter and evaluate candidates
        for (cx, cy, cz) in candidate_positions:
            if cx + bw > W or cy + bh > H or cz + bd > D:
                continue
            if not can_place(cx, cy, cz, bw, bh, bd):
                continue
            # Score: prefer positions closer to origin (minimize max extent)
            score = cx + cy + cz
            if best_score is None or score < best_score:
                best_score = score
                best_pos = (cx, cy, cz)
        
        return best_pos
    
    # Process boxes in order of decreasing volume
    for vol, idx, w, h, d in box_data:
        if time.time() > deadline:
            break
        
        rotations = get_rotations(w, h, d)
        best_placement = None
        best_rot_score = None
        
        for (rw, rh, rd) in rotations:
            pos = find_best_position(rw, rh, rd)
            if pos is not None:
                # Score the placement: prefer lower positions
                score = pos[0] + pos[1] + pos[2]
                if best_rot_score is None or score < best_rot_score:
                    best_rot_score = score
                    best_placement = (pos, (rw, rh, rd))
        
        if best_placement is not None:
            (px, py, pz), (pw, ph, pd) = best_placement
            placed_boxes.append((px, py, pz, pw, ph, pd))
            placements[idx] = (0, px, py, pz, pw, ph, pd)
            placed_count += 1
        else:
            placements[idx] = (-1, 0, 0, 0, w, h, d)
    
    # Fill any remaining unplaced boxes
    for i in range(n):
        if placements[i] is None:
            placements[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
    
    return placements
# EVOLVE-BLOCK-END
