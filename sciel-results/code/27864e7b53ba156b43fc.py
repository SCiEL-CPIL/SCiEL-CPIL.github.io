"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import heapq
from collections import defaultdict

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 135.0  # 15s margin before 150s kill
    
    # Precompute all rotations for each box (6 permutations)
    def get_rotations(w, h, d):
        rots = []
        seen = set()
        dims = [w, h, d]
        for i in range(3):
            for j in range(3):
                if j == i:
                    continue
                for k in range(3):
                    if k == i or k == j:
                        continue
                    r = (dims[i], dims[j], dims[k])
                    if r not in seen:
                        seen.add(r)
                        rots.append(r)
        return rots
    
    box_rots = [get_rotations(w, h, d) for w, h, d in boxes]
    
    # Box-Tree representation: maximal empty rectangular subspaces
    # Each node: (x, y, z, w, h, d) representing an empty box region
    # We use a priority queue to find best-fit positions
    
    def try_place(boxes_list, container_dims, order, deadline):
        """Try to place boxes in given order using box-tree splitting."""
        W, H, D = container_dims
        
        # Maximal empty regions as a list
        # Each region: [x, y, z, w, h, d]
        regions = [[0, 0, 0, W, H, D]]
        
        placements = [None] * len(boxes_list)
        
        for idx in order:
            w, h, d = boxes_list[idx]
            if time.time() > deadline:
                break
            
            best_pos = None
            best_score = None
            best_region_idx = -1
            
            for ri, (rx, ry, rz, rw, rh, rd) in enumerate(regions):
                if rw < w or rh < h or rd < d:
                    continue
                
                # Try all rotations
                for (rw2, rh2, rd2) in box_rots[idx]:
                    if rw2 > rw or rh2 > rh or rd2 > rd:
                        continue
                    
                    # Score: prefer positions that minimize wasted space
                    # Use a simple heuristic: minimize sum of distances to origin
                    # and prefer tighter fits
                    score = (rx + ry + rz) + 0.1 * (rw - rw2 + rh - rh2 + rd - rd2)
                    
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pos = (rx, ry, rz, rw2, rh2, rd2)
                        best_region_idx = ri
            
            if best_pos is None:
                placements[idx] = (-1, 0, 0, 0, w, h, d)
                continue
            
            px, py, pz, pw, ph, pd = best_pos
            placements[idx] = (0, px, py, pz, pw, ph, pd)
            
            # Split the region
            rx, ry, rz, rw, rh, rd = regions[best_region_idx]
            
            # Remove the used region and add split regions
            del regions[best_region_idx]
            
            # Split into up to 6 regions
            # Region to the right (in x)
            if px + pw < rx + rw:
                regions.append([px + pw, ry, rz, rx + rw - px - pw, rh, rd])
            # Region above (in y)
            if py + ph < ry + rh:
                regions.append([rx, py + ph, rz, rw, ry + rh - py - ph, rd])
            # Region in front (in z)
            if pz + pd < rz + rd:
                regions.append([rx, ry, pz + pd, rw, rh, rz + rd - pz - pd])
            # Region to the right and above
            if px + pw < rx + rw and py + ph < ry + rh:
                regions.append([px + pw, py + ph, rz, rx + rw - px - pw, ry + rh - py - ph, rd])
            # Region to the right and in front
            if px + pw < rx + rw and pz + pd < rz + rd:
                regions.append([px + pw, ry, pz + pd, rx + rw - px - pw, rh, rz + rd - pz - pd])
            # Region above and in front
            if py + ph < ry + rh and pz + pd < rz + rd:
                regions.append([rx, py + ph, pz + pd, rw, ry + rh - py - ph, rz + rd - pz - pd])
            # Region to the right, above, and in front
            if px + pw < rx + rw and py + ph < ry + rh and pz + pd < rz + rd:
                regions.append([px + pw, py + ph, pz + pd, rx + rw - px - pw, ry + rh - py - ph, rz + rd - pz - pd])
        
        # Fill in any unplaced boxes
        for i in range(len(boxes_list)):
            if placements[i] is None:
                w, h, d = boxes_list[i]
                placements[i] = (-1, 0, 0, 0, w, h, d)
        
        return placements
    
    # Strategy 1: Sort by volume descending
    order1 = sorted(range(n), key=lambda i: -(boxes[i][0] * boxes[i][1] * boxes[i][2]))
    result1 = try_place(boxes, container, order1, deadline)
    
    # Calculate score
    def calc_score(placements):
        total_vol = 0
        for p in placements:
            if p[0] == 0:
                total_vol += p[4] * p[5] * p[6]
        return total_vol / (W * H * D)
    
    best_result = result1
    best_score = calc_score(result1)
    
    # Strategy 2: Sort by max dimension descending
    order2 = sorted(range(n), key=lambda i: -max(boxes[i]))
    if time.time() < deadline:
        result2 = try_place(boxes, container, order2, deadline)
        score2 = calc_score(result2)
        if score2 > best_score:
            best_result = result2
            best_score = score2
    
    # Strategy 3: Random restarts if time permits
    if time.time() < deadline - 5:
        for restart in range(20):
            if time.time() > deadline - 2:
                break
            # Random perturbation of order
            order3 = list(range(n))
            random.shuffle(order3)
            # Bias towards larger boxes first
            order3.sort(key=lambda i: -boxes[i][0] * boxes[i][1] * boxes[i][2] * random.uniform(0.5, 1.5))
            result3 = try_place(boxes, container, order3, deadline)
            score3 = calc_score(result3)
            if score3 > best_score:
                best_result = result3
                best_score = score3
    
    return best_result
# EVOLVE-BLOCK-END
