"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import itertools
import numpy as np

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 135.0  # 15s margin
    
    # Get all unique rotations for each box
    def get_rotations(w, h, d):
        rots = set()
        for perm in itertools.permutations((w, h, d)):
            rots.add(perm)
        return list(rots)
    
    # Precompute rotations
    box_rots = [get_rotations(w, h, d) for w, h, d in boxes]
    
    # Skyline representation: list of (x, y, z_max) where z_max is the max height occupied
    # at that (x,y) position. We maintain a list of "skyline" entries.
    # Actually, for 3D, we need a more sophisticated approach.
    # Let's use a list of placed boxes and for each new box, find the best position
    # by checking against all placed boxes.
    
    # For efficiency, we'll use a grid-based or sorted approach.
    # Given the small number of boxes (60), we can afford O(n^2) checks.
    
    def find_best_position(box_dims, placed, W, H, D):
        """Find the best (x, y, z) position for a box with given dimensions."""
        bw, bh, bd = box_dims
        best_pos = None
        best_score = -1
        
        # Candidate positions: check corners of existing boxes and container corners
        # For each placed box, the new box can be placed adjacent to it in 6 directions
        # Also check (0,0,0)
        
        candidate_positions = set()
        candidate_positions.add((0, 0, 0))
        
        for (px, py, pz, pw, ph, pd) in placed:
            # Place to the right, left, above, below, front, back
            # Right: x = px + pw
            candidate_positions.add((px + pw, py, pz))
            # Left: x = px - bd (but must be >= 0)
            if px >= bd:
                candidate_positions.add((px - bd, py, pz))
            # Above: y = py + ph
            candidate_positions.add((px, py + ph, pz))
            # Below: y = py - bh
            if py >= bh:
                candidate_positions.add((px, py - bh, pz))
            # Front: z = pz + pd
            candidate_positions.add((px, py, pz + pd))
            # Back: z = pz - bd
            if pz >= bd:
                candidate_positions.add((px, py, pz - bd))
            
            # Also try combinations of edges
            candidate_positions.add((px + pw, py + ph, pz))
            candidate_positions.add((px + pw, py, pz + pd))
            candidate_positions.add((px, py + ph, pz + pd))
        
        # Filter and evaluate candidates
        valid_candidates = []
        for (cx, cy, cz) in candidate_positions:
            if cx < 0 or cy < 0 or cz < 0:
                continue
            if cx + bw > W or cy + bh > H or cz + bd > D:
                continue
            
            # Check overlap with all placed boxes
            overlap = False
            for (px, py, pz, pw, ph, pd) in placed:
                if (cx < px + pw and cx + bw > px and
                    cy < py + ph and cy + bh > py and
                    cz < pz + pd and cz + bd > pz):
                    overlap = True
                    break
            if not overlap:
                # Score: prefer lower positions (minimize z, then y, then x)
                score = -(cz * 10000 + cy * 100 + cx)
                valid_candidates.append((score, (cx, cy, cz)))
        
        if not valid_candidates:
            return None
        
        valid_candidates.sort(reverse=True)
        return valid_candidates[0][1]
    
    # Greedy packing with different orderings
    def pack_with_order(order, boxes, box_rots, container):
        W, H, D = container
        placed = []
        placements = [None] * len(boxes)
        
        for idx in order:
            w, h, d = boxes[idx]
            best_rot = None
            best_pos = None
            best_score = -1
            
            for rot in box_rots[idx]:
                pos = find_best_position(rot, placed, W, H, D)
                if pos is not None:
                    # Score the placement
                    # Prefer lower z, then y, then x
                    score = -(pos[2] * 10000 + pos[1] * 100 + pos[0])
                    if score > best_score:
                        best_score = score
                        best_rot = rot
                        best_pos = pos
            
            if best_rot is not None:
                bw, bh, bd = best_rot
                cx, cy, cz = best_pos
                placed.append((cx, cy, cz, bw, bh, bd))
                placements[idx] = (0, cx, cy, cz, bw, bh, bd)
            else:
                placements[idx] = (-1, 0, 0, 0, w, h, d)
        
        return placements
    
    # Compute score (volume filled)
    def compute_score(placements, container):
        W, H, D = container
        total_vol = W * H * D
        if total_vol == 0:
            return 0.0
        filled_vol = 0
        for p in placements:
            if p[0] == 0:
                filled_vol += p[4] * p[5] * p[6]
        return filled_vol / total_vol
    
    # Generate different orderings
    def generate_order(boxes, strategy='volume'):
        n = len(boxes)
        indices = list(range(n))
        if strategy == 'volume':
            indices.sort(key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True)
        elif strategy == 'max_dim':
            indices.sort(key=lambda i: max(boxes[i]), reverse=True)
        elif strategy == 'min_dim':
            indices.sort(key=lambda i: min(boxes[i]), reverse=True)
        elif strategy == 'surface':
            indices.sort(key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True)
        elif strategy == 'random':
            random.shuffle(indices)
        elif strategy == 'shuffled_volume':
            # Sort by volume but add some randomness
            indices.sort(key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2] + random.uniform(0, 10), reverse=True)
        return indices
    
    # Try multiple strategies and keep the best
    best_placements = None
    best_score = -1
    
    strategies = ['volume', 'max_dim', 'min_dim', 'surface']
    
    for strat in strategies:
        if time.time() > deadline:
            break
        order = generate_order(boxes, strat)
        placements = pack_with_order(order, boxes, box_rots, container)
        score = compute_score(placements, container)
        if score > best_score:
            best_score = score
            best_placements = placements
    
    # Try some random orderings if time permits
    num_random_tries = 10
    for i in range(num_random_tries):
        if time.time() > deadline:
            break
        order = generate_order(boxes, 'random')
        placements = pack_with_order(order, boxes, box_rots, container)
        score = compute_score(placements, container)
        if score > best_score:
            best_score = score
            best_placements = placements
    
    # If no placement found, return all left out
    if best_placements is None:
        best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_placements
# EVOLVE-BLOCK-END
