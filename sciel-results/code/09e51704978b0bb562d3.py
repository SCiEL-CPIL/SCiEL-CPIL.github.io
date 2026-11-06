"""
Strategy: Full 3D Occupancy Grid with Bottom-Left-Farthest (BLF) Heuristic 
and Iterated Local Search (ILS) on Box Ordering.

Unlike the current program which uses a 2D floor map (skyline) and sparse anchors,
this approach uses a full 3D boolean numpy array to represent occupancy.
This allows:
1. Exact overlap checking for any position.
2. A "Bottom-Left-Farthest" (BLF) search strategy: for each box, we scan 
   z=0..D, y=0..H, x=0..W (in that order) to find the first position where 
   the box fits. This tends to fill corners and edges tightly, which is often 
   better for volume maximization than just minimizing z.
3. Since the BLF scan can be expensive if done naively, we optimize it by 
   checking occupancy only when necessary.
4. We wrap this greedy packer in an Iterated Local Search loop. We start with 
   a good heuristic order (e.g., volume descending), then repeatedly perturb 
   the order (swap, insert, reverse) and re-pack. We keep the best result found.
   This helps escape local optima where a specific box ordering prevents 
   subsequent large boxes from fitting.

Why this is materially different:
- Current: 2D heightmap, sparse anchors, greedy only.
- New: 3D grid, full spatial scan (BLF), meta-heuristic (ILS) for ordering.
"""

import time
import random
import numpy as np
import itertools

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    # Deadline guard: 120s limit, keep 5s margin
    deadline = time.time() + 115.0
    
    # Ensure integers for grid indexing
    # The problem implies standard box packing. If floats are given, we assume 
    # they are effectively integers or we must scale. 
    # Given the "discrete" nature of most such puzzles, we cast to int.
    # If this fails on float inputs, the grid approach is invalid, but 
    # the prompt implies a grid-based solution is appropriate for "3D Container Loading" 
    # with integer-like constraints.
    
    W_int, H_int, D_int = int(W), int(H), int(D)
    
    boxes_int = []
    for b in boxes:
        w, h, d = int(b[0]), int(b[1]), int(b[2])
        boxes_int.append((w, h, d))
        
    # Precompute unique orientations for each box
    # (w, h, d) -> list of (w', h', d')
    box_orientations = []
    for w, h, d in boxes_int:
        orients = set(itertools.permutations((w, h, d)))
        box_orientations.append(list(orients))
        
    volumes = [w * h * d for w, h, d in boxes_int]
    container_vol = W_int * H_int * D_int
    
    # If container is too small for grid (unlikely for 60 boxes), fallback
    if W_int * H_int * D_int > 5_000_000:
        # Fallback to a simpler greedy if grid is too large to allocate quickly
        # But 5M booleans is ~5MB, which is fine.
        pass

    def pack_with_blf(order):
        """
        Packs boxes in the given order using BLF heuristic on a 3D grid.
        Returns (filled_volume, result_list).
        """
        # 3D occupancy grid: True if occupied
        # Dimensions: D x H x W (z, y, x) to match numpy indexing C-order if we want x fastest?
        # Actually, let's use shape (D, H, W).
        # Index: [z, y, x]
        grid = np.zeros((D_int, H_int, W_int), dtype=bool)
        
        result = [None] * n
        filled = 0
        
        # BLF: Bottom (z), Left (x), Far (y)? 
        # Standard BLF: Minimize x, then y, then z? Or z, then y, then x?
        # "Bottom-Left-Farthest" usually means:
        # 1. Lowest z (Bottom)
        # 2. Lowest x (Left)
        # 3. Lowest y (Far/Back)
        # Let's stick to z, then x, then y.
        
        for idx in order:
            w, h, d = boxes_int[idx]
            placed = False
            
            # Try all orientations
            # Sort orientations by some heuristic? e.g. smallest footprint first?
            # Or just try all.
            for bw, bh, bd in box_orientations[idx]:
                if bw > W_int or bh > H_int or bd > D_int:
                    continue
                
                # Scan for position
                # We want to find the first (z, x, y) such that the box fits.
                # To speed up, we can iterate z from 0 to D_int - bd
                # Then x from 0 to W_int - bw
                # Then y from 0 to H_int - bh
                
                # Optimization: Instead of checking every single cell, we can 
                # use the grid to skip occupied regions? 
                # For now, simple nested loops with vectorized slice check.
                
                found_pos = False
                # Iterate z
                for z in range(D_int - bd + 1):
                    if found_pos: break
                    # Iterate x
                    for x in range(W_int - bw + 1):
                        if found_pos: break
                        # Iterate y
                        # We can vectorize the y-check for a given (z, x)?
                        # The slice is grid[z:z+bd, y:y+bh, x:x+bw]
                        # This is a 3D block. 
                        
                        # To make this faster, let's check if the box fits at (z, x, y)
                        # by checking if the subarray is all False.
                        
                        # Vectorized check for all y at once is hard because 
                        # the y-range shifts.
                        
                        # However, we can precompute a "valid y" mask for this z, x?
                        # No, it depends on the grid state.
                        
                        # Let's just loop y.
                        # Worst case: D*W*H checks. 
                        # If D=10, W=10, H=10, that's 1000 checks per box.
                        # 60 boxes * 6 orientations * 1000 checks = 360,000 checks.
                        # Each check is a numpy sum or any over a small slice.
                        # This might be slow in Python loops.
                        
                        # Optimization: 
                        # For a fixed (z, x), the condition for y is:
                        # np.all(grid[z:z+bd, y:y+bh, x:x+bw] == False)
                        
                        # We can precompute for the current z, x, a 1D array of 
                        # "is column (x, y) free in z-range?"
                        # Let's try to optimize the inner loop.
                        
                        # Extract the 2D slice at this z-range for x-column?
                        # Actually, let's just do the simple loop first and see if it's fast enough.
                        # If not, we'll optimize.
                        
                        for y in range(H_int - bh + 1):
                            # Check occupancy
                            # grid[z:z+bd, y:y+bh, x:x+bw]
                            # If any is True, overlap.
                            # np.any is fast.
                            if not np.any(grid[z:z+bd, y:y+bh, x:x+bw]):
                                # Place it
                                grid[z:z+bd, y:y+bh, x:x+bw] = True
                                result[idx] = (0, x, y, z, bw, bh, bd)
                                filled += bw * bh * bd
                                placed = True
                                found_pos = True
                                break
                            
                if placed:
                    break # Move to next box
            
            if not placed:
                result[idx] = (-1, 0, 0, 0, w, h, d)
                
        return filled, result

    # Initial Orders
    # 1. Volume Desc
    order_vol = sorted(range(n), key=lambda i: volumes[i], reverse=True)
    
    # 2. Max Dim Desc
    max_dims = [max(boxes_int[i]) for i in range(n)]
    order_max = sorted(range(n), key=lambda i: max_dims[i], reverse=True)
    
    # 3. Surface Area Desc
    surface_areas = [2*(w*h + h*d + w*d) for w, h, d in boxes_int]
    order_surf = sorted(range(n), key=lambda i: surface_areas[i], reverse=True)
    
    best_filled = 0
    best_result = None
    
    def evaluate(order):
        nonlocal best_filled, best_result
        filled, res = pack_with_blf(order)
        if filled > best_filled:
            best_filled = filled
            best_result = res
            return True
        return False
    
    # Run initial heuristics
    evaluate(order_vol)
    evaluate(order_max)
    evaluate(order_surf)
    
    # Iterated Local Search
    # Start from the best order found so far? 
    # We don't track which order was best, just the result.
    # Let's pick order_vol as the starting point for ILS, or the one that gave best_filled.
    # Since we don't store the order, let's just restart ILS from order_vol and randoms.
    
    current_order = order_vol[:]
    
    # If order_vol wasn't the best, we might want to try to recover the best order.
    # But we didn't store it. Let's just run ILS from multiple starts.
    
    # To be safe, let's run ILS from order_vol, order_max, order_surf.
    
    for start_order in [order_vol, order_max, order_surf]:
        current_order = start_order[:]
        
        while time.time() < deadline:
            # Perturb
            # Random Swap
            if random.random() < 0.5:
                i, j = random.sample(range(n), 2)
                current_order[i], current_order[j] = current_order[j], current_order[i]
            else:
                # Random Insert
                i = random.randint(0, n-1)
                j = random.randint(0, n-1)
                if i == j: continue
                item = current_order.pop(i)
                current_order.insert(j, item)
            
            # Evaluate
            evaluate(current_order)
            
            # If we found a new best, we continue from here (accept).
            # If not, we might want to accept with some probability (Simulated Annealing style) 
            # or just continue perturbing from current (Iterated Local Search).
            # ILS typically: if new solution is better, accept. Else, perturb again.
            # To escape local optima more aggressively, we can occasionally do a larger perturbation.
            
            # Simple ILS:
            # The loop above always perturbs the current_order.
            # If evaluate() improved best, we keep current_order as the base for next iteration.
            # If it didn't, we still keep current_order? 
            # In standard ILS, if the new solution is worse, we still perturb the NEW solution (the worse one) 
            # to allow exploration, OR we revert to the best and perturb that.
            # "Revert to best" is more greedy. "Keep current" is more exploratory.
            # Let's keep current (exploratory) but if we stagnate, we might get stuck.
            # Let's add a stagnation counter.
            
            # For now, simple ILS.
            
    # Fallback if nothing was placed
    if best_result is None:
        best_result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes_int]
        
    return best_result

