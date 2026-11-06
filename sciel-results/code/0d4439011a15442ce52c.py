"""3D Container Loading: Box-Tree (Maximal Empty Rectangles) with Local Search."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Preprocess boxes
    box_data = []
    for i, (w, h, d) in enumerate(boxes):
        box_data.append((i, w, h, d))
        
    # Initial result: all boxes left out
    final_placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    
    def get_rotations(w, h, d):
        rots = set()
        rots.add((w, h, d))
        rots.add((w, d, h))
        rots.add((h, w, d))
        rots.add((h, d, w))
        rots.add((d, w, h))
        rots.add((d, h, w))
        return list(rots)

    def split_space(space, box):
        """
        Splits a maximal empty rectangular space by placing a box at its corner (0,0,0) of the space.
        Returns a list of new maximal empty rectangular spaces.
        space: (x, y, z, w, h, d)
        box: (rw, rh, rd)
        """
        sx, sy, sz, sw, sh, sd = space
        bw, bh, bd = box
        
        # Box is placed at (sx, sy, sz)
        # The remaining space is split into up to 6 parts:
        # 1. Above the box: (sx, sy+bh, sz, sw, sh-bh, sd) if sh > bh
        # 2. Right of the box (in x): (sx+bw, sy, sz, sw-bw, sh, sd) if sw > bw
        # 3. Back of the box (in z): (sx, sy, sz+bd, sw, sh, sd-bd) if sd > bd
        
        new_spaces = []
        if sh > bh:
            new_spaces.append((sx, sy + bh, sz, sw, sh - bh, sd))
        if sw > bw:
            new_spaces.append((sx + bw, sy, sz, sw - bw, sh, sd))
        if sd > bd:
            new_spaces.append((sx, sy, sz + bd, sw, sh, sd - bd))
        
        return new_spaces

    def run_box_tree(order_indices, seed_val):
        """
        Runs packing using the Box-Tree (Maximal Empty Rectangles) representation.
        """
        random.seed(seed_val)
        placed = []
        placements = [None] * n
        filled_vol = 0.0
        
        # Initial space is the whole container
        spaces = [(0, 0, 0, W, H, D)]
        
        for idx in order_indices:
            orig_idx, w, h, d = box_data[idx]
            rots = get_rotations(w, h, d)
            # Randomize rotation order
            random.shuffle(rots)
            
            best_placement = None
            best_space_idx = -1
            best_rot = None
            best_score = float('inf')
            
            # For each rotation, find the best space
            for rw, rh, rd in rots:
                # Find the smallest space that fits this box
                # We want to fill gaps, so we prefer smaller spaces
                # Score: volume of the space (smaller is better)
                # Tie-breaker: position (lower z, lower x, higher y? or just lower z)
                
                for s_idx, (sx, sy, sz, sw, sh, sd) in enumerate(spaces):
                    if sw >= rw and sh >= rh and sd >= rd:
                        space_vol = sw * sh * sd
                        # We want the tightest fit.
                        # A good heuristic is to minimize the volume of the space.
                        # If volumes are equal, pick the one with lowest z, then lowest x.
                        score = (space_vol, sz, sx, -sy)
                        
                        if score < best_score:
                            best_score = score
                            best_placement = (sx, sy, sz)
                            best_space_idx = s_idx
                            best_rot = (rw, rh, rd)
            
            if best_placement is not None:
                x, y, z = best_placement
                rw, rh, rd = best_rot
                
                placed.append((x, y, z, rw, rh, rd))
                placements[orig_idx] = (0, x, y, z, rw, rh, rd)
                filled_vol += rw * rh * rd
                
                # Update spaces
                # Remove the used space
                used_space = spaces[best_space_idx]
                spaces.pop(best_space_idx)
                
                # Generate new spaces from the split
                new_spaces = split_space(used_space, (rw, rh, rd))
                
                # Merge new spaces with existing ones to maintain maximality?
                # Simple approach: just add them. This might lead to non-maximal spaces,
                # which is okay for correctness but might reduce efficiency if spaces are fragmented.
                # However, for 60 boxes, the number of spaces grows linearly or slightly superlinearly.
                # Actually, the number of maximal empty rectangles can grow significantly.
                # But for a greedy algorithm, we just need *a* valid placement.
                # The "Best Fit" on non-maximal spaces is still valid.
                
                # Optimization: Check if new spaces are contained in existing ones?
                # This is expensive. Let's just add them.
                spaces.extend(new_spaces)
                
                # Optional: Merge spaces that are adjacent?
                # This is complex in 3D. Let's skip for now.
                
        return filled_vol, placements

    # Main Loop: Multi-Start
    best_vol = 0
    best_placements = None
    
    # Initial deterministic run (BFD by volume)
    initial_order = sorted(range(n), key=lambda i: box_data[i][1]*box_data[i][2]*box_data[i][3], reverse=True)
    
    vol, pls = run_box_tree(initial_order, 42)
    if vol > best_vol:
        best_vol = vol
        best_placements = pls
    
    # Random restarts
    t_start = time.time()
    max_restarts = 100
    restarts_done = 0
    
    while time.time() < deadline and restarts_done < max_restarts:
        # Create a perturbed order
        order = list(initial_order)
        
        # Randomly swap a few adjacent elements
        for _ in range(5):
            i = random.randint(0, n - 2)
            order[i], order[i+1] = order[i+1], order[i]
            
        # Also randomize rotation preference by changing seed
        seed_val = random.randint(0, 10000)
        
        vol, pls = run_box_tree(order, seed_val)
        
        if vol > best_vol:
            best_vol = vol
            best_placements = pls
            
        restarts_done += 1
        
    # Finalize placements
    if best_placements:
        for i in range(n):
            if best_placements[i] is not None:
                final_placements[i] = best_placements[i]
            else:
                final_placements[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
    else:
        for i in range(n):
            final_placements[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])

    return final_placements
# EVOLVE-BLOCK-END
