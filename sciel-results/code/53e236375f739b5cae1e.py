import time
import numpy as np
from itertools import permutations

def _rotations(w, h, d):
    seen = set()
    result = []
    for perm in permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

def load(boxes, container):
    W, H, D = container
    start_time = time.time()
    timeout = 110.0 # Safety margin
    
    # Pre-compute rotations for all boxes
    box_rots = [_rotations(w, h, d) for w, h, d in boxes]
    
    # Sorting strategies
    # 0: Volume desc
    # 1: Max Dim desc
    # 2: Surface Area desc
    
    best_placements = None
    best_score = -1
    
    indices = list(range(len(boxes)))
    
    strategies = [
        sorted(indices, key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True),
        sorted(indices, key=lambda i: max(boxes[i]), reverse=True),
        sorted(indices, key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True)
    ]
    
    for strat_idx, order in enumerate(strategies):
        if time.time() - start_time > timeout / 2:
            break
            
        placed = [] # list of (x, y, z, w, h, d)
        Xs = {0}
        Ys = {0}
        
        current_score = 0.0
        
        for i in order:
            if time.time() - start_time > timeout:
                break
                
            w, h, d = boxes[i]
            rots = box_rots[i]
            
            best_pos = None
            best_cost = float('inf')
            
            for bw, bh, bd in rots:
                for x in Xs:
                    if x + bw > W:
                        continue
                    for y in Ys:
                        if y + bh > H:
                            continue
                        
                        # Find min z
                        min_z = 0
                        for px, py, pz, pw, ph, pd in placed:
                            # Check overlap in X and Y
                            if x < px + pw and px < x + bw:
                                if y < py + ph and py < y + bh:
                                    if pz + pd > min_z:
                                        min_z = pz + pd
                                    
                        if min_z + bd <= D:
                            # Valid position
                            # Cost: prefer lower z, then lower x, then lower y
                            # Using a weighted sum to break ties
                            cost = min_z * 10000 + x * 100 + y
                            if cost < best_cost:
                                best_cost = cost
                                best_pos = (x, y, min_z, bw, bh, bd)
            
            if best_pos:
                placed.append(best_pos)
                # Update candidate sets
                Xs.add(best_pos[0] + best_pos[3])
                Ys.add(best_pos[1] + best_pos[4])
                current_score += best_pos[3] * best_pos[4] * best_pos[5]
            else:
                # Box didn't fit
                pass
        
        if current_score > best_score:
            best_score = current_score
            # Reconstruct final placements list
            # We need to map back to original indices
            # 'placed' contains tuples (x,y,z,w,h,d) but we lost the index.
            # We need to track which box was placed where.
            
            # Let's re-run the best strategy to record indices? 
            # Or modify the loop to store indices.
            
            # To save time, I'll just store the index in the placed list during the run.
            # Let's refine the inner loop to store (index, x, y, z, w, h, d)
            
            # Since I already ran it, I can't easily get indices without re-running.
            # I will re-run the best strategy at the end to generate the output, 
            # or I should have stored them. 
            # Given time constraints, I will store them in the main loop.
            
            # Let's restart the logic to store indices properly.
            
            pass # Placeholder
            
    # The above structure is a bit messy for returning the correct format.
    # I will implement a single pass that stores indices, and try the best strategy found.
    # Actually, simpler: just run the first strategy (Volume) well, and if time permits, try others.
    
    # Let's rewrite the core loop to be clean.
    
    def run_strategy(order):
        placed = [] # (idx, x, y, z, w, h, d)
        Xs = {0}
        Ys = {0}
        
        for i in order:
            if time.time() - start_time > timeout:
                return None, 0.0 # Timeout
            
            w, h, d = boxes[i]
            rots = box_rots[i]
            
            best_pos = None
            best_cost = float('inf')
            
            for bw, bh, bd in rots:
                for x in Xs:
                    if x + bw > W:
                        continue
                    for y in Ys:
                        if y + bh > H:
                            continue
                        
                        min_z = 0
                        for _, px, py, pz, pw, ph, pd in placed:
                            if x < px + pw and px < x + bw:
                                if y < py + ph and py < y + bh:
                                    if pz + pd > min_z:
                                        min_z = pz + pd
                                    
                        if min_z + bd <= D:
                            cost = min_z * 10000 + x * 100 + y
                            if cost < best_cost:
                                best_cost = cost
                                best_pos = (x, y, min_z, bw, bh, bd)
            
            if best_pos:
                placed.append((i, best_pos[0], best_pos[1], best_pos[2], best_pos[3], best_pos[4], best_pos[5]))
                Xs.add(best_pos[0] + best_pos[3])
                Ys.add(best_pos[1] + best_pos[4])
            
        # Calculate score
        vol = sum(p[4]*p[5]*p[6] for p in placed)
        return placed, vol

    best_vol = 0
    best_placed = []
    
    for order in strategies:
        placed, vol = run_strategy(order)
        if placed is None:
            continue
        if vol > best_vol:
            best_vol = vol
            best_placed = placed
            
    # Construct output
    # Initialize with -1
    result = [(-1, 0, 0, 0, 0, 0, 0) for _ in boxes]
    
    for p in best_placed:
        idx = p[0]
        x, y, z, w, h, d = p[1], p[2], p[3], p[4], p[5], p[6]
        result[idx] = (0, x, y, z, w, h, d)
        
    return result
