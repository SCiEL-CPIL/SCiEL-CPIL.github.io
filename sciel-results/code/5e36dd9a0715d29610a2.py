import numpy as np
import random
import time

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Pre-calculate volumes and sort indices by volume descending
    box_vols = [w * h * d for w, h, d in boxes]
    sorted_indices = sorted(range(n), key=lambda i: box_vols[i], reverse=True)
    
    # Pre-compute unique rotations for each box
    box_rotations = []
    for w, h, d in boxes:
        rots = set()
        for rw, rh, rd in [(w,h,d), (w,d,h), (h,w,d), (h,d,w), (d,w,h), (d,h,w)]:
            if rw <= W and rh <= H and rd <= D:
                rots.add((rw, rh, rd))
        box_rotations.append(list(rots))
        
    best_vol = -1
    best_placements = [(-1, 0, 0, 0, 0, 0, 0) for _ in range(n)]
    
    start_time = time.time()
    max_time = 115.0
    
    # Increased restarts to explore more of the solution space
    num_restarts = 800
    
    for r in range(num_restarts):
        if time.time() - start_time > max_time:
            break
            
        # Initialize grid
        try:
            grid = np.zeros((D, H, W), dtype=bool)
        except MemoryError:
            break
            
        candidates = set([(0, 0, 0)])
        placements = [None] * n
        current_vol = 0
        
        # Process boxes in sorted order (largest first)
        for idx in sorted_indices:
            rotations = box_rotations[idx]
            
            if not rotations:
                placements[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
                continue
                
            placed = False
            
            # Shuffle candidates to introduce diversity
            cand_list = list(candidates)
            random.shuffle(cand_list)
            
            for cx, cy, cz in cand_list:
                if placed:
                    break
                
                for rw, rh, rd in rotations:
                    # Check bounds
                    if cx + rw > W or cy + rh > H or cz + rd > D:
                        continue
                    
                    # Check collision
                    if not grid[cz:cz+rd, cy:cy+rh, cx:cx+rw].any():
                        # Place
                        grid[cz:cz+rd, cy:cy+rh, cx:cx+rw] = True
                        placements[idx] = (0, cx, cy, cz, rw, rh, rd)
                        current_vol += rw * rh * rd
                        placed = True
                        
                        # Add new candidates (front corners)
                        # Adding all 7 corners can lead to a large set.
                        # We add them if they are within bounds.
                        new_cands = [
                            (cx + rw, cy, cz),
                            (cx, cy + rh, cz),
                            (cx, cy, cz + rd),
                            (cx + rw, cy + rh, cz),
                            (cx + rw, cy, cz + rd),
                            (cx, cy + rh, cz + rd),
                            (cx + rw, cy + rh, cz + rd)
                        ]
                        for nc in new_cands:
                            if nc[0] < W and nc[1] < H and nc[2] < D:
                                candidates.add(nc)
                        break
                        
            if not placed:
                placements[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
                
        if current_vol > best_vol:
            best_vol = current_vol
            best_placements = [p if p is not None else (-1, 0, 0, 0, 0, 0, 0) for p in placements]
            
    return best_placements
