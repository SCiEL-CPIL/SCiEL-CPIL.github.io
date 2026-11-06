"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 145  # 145s limit to be safe within 150s
    
    # Precompute container volume
    container_vol = W * H * D
    
    # Sort boxes by volume descending (Best-Fit-Decreasing strategy)
    indexed_boxes = [(i, w * h * d, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1], reverse=True)
    
    def evaluate_solution(placements):
        """Compute the score: mean B + 0.1*(1 - last-bin occupied volume fraction)"""
        if not placements:
            return 0
        # Count occupied bins
        bin_ids = set(p[0] for p in placements)
        B = len(bin_ids)
        if B == 0:
            return 0
        # Find the last bin (largest label)
        last_bin_id = max(bin_ids)
        last_vol = sum(p[4]*p[5]*p[6] for p in placements if p[0] == last_bin_id)
        if container_vol == 0:
            return B
        fraction = last_vol / container_vol
        return B + 0.1 * (1 - fraction)

    # Local search / restart strategy
    best_score = float('inf')
    best_placements = None
    
    # Number of restarts: we can afford more because grid checks are fast
    num_restarts = 150
    
    for restart in range(num_restarts):
        if time.time() > deadline - 2:  # Stop if close to deadline
            break
        
        # Create a randomized ordering
        if restart == 0:
            order = list(range(n))
        else:
            order = list(range(n))
            random.shuffle(order)
        
        # Sort by volume but with some randomness for restarts > 0
        if restart == 0:
            randomized_boxes = indexed_boxes
        else:
            randomized_boxes = []
            for idx in order:
                i, vol, w, h, d = indexed_boxes[idx]
                # Add noise to volume to perturb ordering
                randomized_vol = vol * random.uniform(0.8, 1.2)
                randomized_boxes.append((i, randomized_vol, w, h, d))
            randomized_boxes.sort(key=lambda x: x[1], reverse=True)
        
        bins = []  # List of lists of (x, y, z, w, h, d)
        grids = [] # List of 3D numpy arrays for occupancy
        placements = [None] * n
        
        for i, vol, w, h, d in randomized_boxes:
            best_bin_idx = -1
            best_pos = None
            best_cost = float('inf')
            
            # Try existing bins
            for b_idx in range(len(bins)):
                grid = grids[b_idx]
                bin_boxes = bins[b_idx]
                
                # Generate candidate positions
                # 1. Origin
                candidates = [(0, 0, 0)]
                
                # 2. Surfaces of existing boxes
                for bx, by, bz, bw, bh, bd in bin_boxes:
                    # Top face
                    candidates.append((bx, by, bz + bd))
                    # Right face
                    candidates.append((bx + bw, by, bz))
                    # Front face
                    candidates.append((bx, by + bh, bz))
                    
                    # Corners (often good for packing)
                    candidates.append((bx + bw, by, bz + bd))
                    candidates.append((bx, by + bh, bz + bd))
                    candidates.append((bx + bw, by + bh, bz))
                    candidates.append((bx, by, bz + bd))
                    
                # Deduplicate candidates
                unique_candidates = list(set(candidates))
                
                # Filter candidates that are out of bounds
                valid_candidates = []
                for cx, cy, cz in unique_candidates:
                    if cx + w <= W and cy + h <= H and cz + d <= D:
                        valid_candidates.append((cx, cy, cz))
                
                if not valid_candidates:
                    continue
                
                # Check each candidate
                found_in_bin = False
                for cx, cy, cz in valid_candidates:
                    # Check overlap using grid
                    if grid[cx:cx+w, cy:cy+h, cz:cz+d].any():
                        continue
                    
                    # Valid placement!
                    # Calculate cost: we want to "fill" the bin.
                    # Heuristic: Prefer lower Z, then lower Y, then lower X (pack to bottom-left-back)
                    cost = cz * 10000 + cy * 100 + cx
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_bin_idx = b_idx
                        best_pos = (cx, cy, cz)
                        found_in_bin = True
                
                if found_in_bin:
                    break
            
            if best_bin_idx == -1:
                # No existing bin fits, open a new one
                best_bin_idx = len(bins)
                bins.append([])
                # Initialize grid for new bin
                grids.append(np.zeros((W, H, D), dtype=np.uint8))
                best_pos = (0, 0, 0)
            
            # Place the box
            bx, by, bz = best_pos
            bins[best_bin_idx].append((bx, by, bz, w, h, d))
            # Update grid
            grids[best_bin_idx][bx:bx+w, by:by+h, bz:bz+d] = 1
            placements[i] = (best_bin_idx, bx, by, bz, w, h, d)
        
        # Evaluate this solution
        score = evaluate_solution(placements)
        
        if score < best_score:
            best_score = score
            best_placements = [tuple(p) for p in placements]
    
    # Return the best solution found
    if best_placements is not None:
        return best_placements
    else:
        # Fallback: simple FFD
        bins = []
        grids = []
        placements = [None] * n
        for i, vol, w, h, d in indexed_boxes:
            best_bin_idx = -1
            best_pos = None
            best_cost = float('inf')
            
            for b_idx in range(len(bins)):
                grid = grids[b_idx]
                bin_boxes = bins[b_idx]
                candidates = [(0, 0, 0)]
                for bx, by, bz, bw, bh, bd in bin_boxes:
                    candidates.append((bx, by, bz + bd))
                    candidates.append((bx + bw, by, bz))
                    candidates.append((bx, by + bh, bz))
                
                unique_candidates = list(set(candidates))
                for cx, cy, cz in unique_candidates:
                    if cx + w > W or cy + h > H or cz + d > D:
                        continue
                    if grid[cx:cx+w, cy:cy+h, cz:cz+d].any():
                        continue
                    cost = cz * 10000 + cy * 100 + cx
                    if cost < best_cost:
                        best_cost = cost
                        best_bin_idx = b_idx
                        best_pos = (cx, cy, cz)
                        break
            
            if best_bin_idx == -1:
                best_bin_idx = len(bins)
                bins.append([])
                grids.append(np.zeros((W, H, D), dtype=np.uint8))
                best_pos = (0, 0, 0)
            
            bx, by, bz = best_pos
            bins[best_bin_idx].append((bx, by, bz, w, h, d))
            grids[best_bin_idx][bx:bx+w, by:by+h, bz:bz+d] = 1
            placements[i] = (best_bin_idx, bx, by, bz, w, h, d)
        
        return [tuple(p) for p in placements]
# EVOLVE-BLOCK-END
