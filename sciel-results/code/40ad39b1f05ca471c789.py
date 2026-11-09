"""
3D Bin Packing Solver
Strategy:
1. Generate multiple sorting orders for boxes (Volume, Max Dim, Surface Area, etc.).
2. For each order, perform a greedy multi-bin packing using a Best-Fit Decreasing
   heuristic with Bottom-Left-Front (BLF) position selection.
3. Candidate positions are generated from the 'support' points (corners/faces) of
   already placed boxes and container boundaries.
4. Use randomized restarts with noisy ordering to explore different packing structures.
5. Apply a local search phase that attempts to:
   a. Move boxes to earlier bins to reduce the total number of bins (B).
   b. Reposition boxes within the last bin to maximize its volume fill.
6. Vectorized overlap checks using NumPy for speed.
7. Deadline guard ensures we return within time limits.
"""
import time
import sys
import random
import numpy as np

# EVOLVE-BLOCK-START

def _check_overlap(placed_boxes, new_box, W, H, D):
    """
    Check if new_box (x,y,z,w,h,d) overlaps with any box in placed_boxes 
    or goes out of bounds.
    Returns True if placement is VALID (no overlap, in bounds).
    """
    x, y, z, w, h, d = new_box
    
    # Bounds check
    if x < 0 or y < 0 or z < 0:
        return False
    if x + w > W or y + h > H or z + d > D:
        return False
    
    if len(placed_boxes) == 0:
        return True
    
    # Vectorized overlap check
    # placed_boxes is a list of tuples (x, y, z, w, h, d)
    # Convert to numpy array for fast broadcasting
    pb = np.array(placed_boxes)
    px, py, pz, pw, ph, pd = pb[:,0], pb[:,1], pb[:,2], pb[:,3], pb[:,4], pb[:,5]
    
    # Separation conditions:
    # Overlap in X if NOT (x1+w1 <= x2 OR x2+w2 <= x1)
    # We want to find if there is ANY box that overlaps.
    # A box overlaps if it is NOT separated in X, AND NOT separated in Y, AND NOT separated in Z.
    
    sep_x = (x + w <= px) | (px + pw <= x)
    sep_y = (y + h <= py) | (py + ph <= y)
    sep_z = (z + d <= pz) | (pz + pd <= z)
    
    # If any box is not separated in any axis, it overlaps.
    # ~(sep_x | sep_y | sep_z) is True for overlapping boxes.
    overlaps = np.any(~(sep_x | sep_y | sep_z))
    
    return not overlaps

def _generate_positions(placed_boxes, w, h, d, W, H, D):
    """
    Generate candidate positions for a box of size (w,h,d) in a bin with placed_boxes.
    We use a 'support' based approach:
    1. Start with (0,0,0).
    2. For each placed box, consider positions adjacent to its faces.
    3. Filter valid positions (in bounds, no overlap).
    4. Sort by BLF heuristic (min y, then min z, then min x).
    """
    candidates = set()
    
    # Origin
    candidates.add((0, 0, 0))
    
    # Add positions based on placed boxes
    for pb in placed_boxes:
        px, py, pz, pw, ph, pd = pb
        
        # X candidates: Left of pb, Right of pb
        cx1 = px - w
        cx2 = px + pw
        
        # Y candidates: Front of pb, Back of pb
        cy1 = py - h
        cy2 = py + ph
        
        # Z candidates: Below pb, Above pb
        cz1 = pz - d
        cz2 = pz + pd
        
        # We don't want to generate all 8 corners for every box as it's O(N^2) and creates many duplicates.
        # Instead, we add specific "aligned" positions that are likely to be useful.
        
        # Positions aligned with the "top" of the box (z + pd)
        # Try to place on top of this box, aligned with its x and y
        candidates.add((px, py, cz2))
        
        # Try to place to the right, aligned with y and z
        candidates.add((cx2, py, pz))
        
        # Try to place behind, aligned with x and z
        candidates.add((px, cy2, pz))
        
        # Also try the "corners" that might fit in gaps
        # (cx1, py, pz), (cx2, py, pz), (px, cy1, pz), (px, cy2, pz)
        # (px, py, cz1), (px, py, cz2)
        
        # Add a few more strategic points
        candidates.add((cx2, cy2, cz2)) # Far corner
        candidates.add((cx1, cy1, cz1)) # Near corner (might be negative)
        
    # Filter and sort
    valid_candidates = []
    for c in candidates:
        if c[0] < 0 or c[1] < 0 or c[2] < 0:
            continue
        if c[0] + w > W or c[1] + h > H or c[2] + d > D:
            continue
            
        # Check overlap
        if _check_overlap(placed_boxes, (c[0], c[1], c[2], w, h, d), W, H, D):
            valid_candidates.append(c)
            
    # Sort by BLF: Bottom (y), Left (x), Front (z) ? 
    # Standard BLF is usually (z, y, x) or (y, z, x) depending on axis definitions.
    # Here: y is height (bottom-up), z is depth (front-back), x is width (left-right).
    # We want to pack to the bottom (low y), front (low z), left (low x).
    valid_candidates.sort(key=lambda p: (p[1], p[2], p[0]))
    
    return valid_candidates

def _pack_greedy(boxes, container, order):
    """
    Perform greedy packing for a specific order of box indices.
    Returns a list of placements: [(bin_id, x, y, z, w, h, d), ...]
    """
    W, H, D = container
    n = len(boxes)
    bins = [] # List of lists of placed boxes (x,y,z,w,h,d)
    result = [None] * n
    
    for idx in order:
        w, h, d = boxes[idx]
        
        best_bin_idx = -1
        best_pos = None
        best_score = float('inf')
        
        # Try to place in existing bins
        for b_idx in range(len(bins)):
            placed_in_bin = bins[b_idx]
            
            # Generate candidates
            candidates = _generate_positions(placed_in_bin, w, h, d, W, H, D)
            
            if not candidates:
                continue
                
            # Pick the first valid candidate (best BLF)
            # Since _generate_positions returns sorted valid candidates, the first one is best for this bin
            pos = candidates[0]
            
            # Calculate a score for this placement to compare across bins
            # We prefer bins that are "fuller" or where the box fits "better"
            # Simple heuristic: prefer lower y, z, x. Also prefer earlier bins to keep bins closed.
            # Actually, just picking the first valid bin with the best BLF position is standard.
            # To improve, we can check if the box fits in an earlier bin better.
            
            # Let's use a simple score: y + z + x (normalized)
            # But we also want to minimize the number of bins. 
            # Standard BFD: Try bins in order 0, 1, 2... Place in the first one that fits.
            # If we do that, we don't need to compare scores across bins, just find the first fit.
            # However, "Best Fit" implies finding the position that minimizes leftover space.
            # For simplicity and speed, "First Fit Decreasing" with BLF position selection is very strong.
            
            best_bin_idx = b_idx
            best_pos = pos
            break # Found a fit in the earliest possible bin
        
        if best_bin_idx != -1:
            pos = best_pos
            new_box = (pos[0], pos[1], pos[2], w, h, d)
            bins[best_bin_idx].append(new_box)
            result[idx] = (best_bin_idx, pos[0], pos[1], pos[2], w, h, d)
        else:
            # Open new bin
            new_bin_idx = len(bins)
            bins.append([])
            new_box = (0, 0, 0, w, h, d)
            bins[new_bin_idx].append(new_box)
            result[idx] = (new_bin_idx, 0, 0, 0, w, h, d)
            
    return result

def _calc_score(result, boxes, container):
    """Calculate the score for a given result."""
    if not result or any(r is None for r in result):
        return float('inf')
        
    W, H, D = container
    container_vol = W * H * D
    
    num_bins = max(r[0] for r in result) + 1
    last_bin_id = max(r[0] for r in result)
    last_bin_vol = sum(r[4]*r[5]*r[6] for r in result if r[0] == last_bin_id)
    
    return num_bins + 0.1 * (1 - last_bin_vol / container_vol)

def _improve_result(result, boxes, container, deadline):
    """
    Robust local search that preserves state for reverts.
    """
    W, H, D = container
    n = len(boxes)
    
    # Store original positions to allow reverts
    orig_positions = [r[1:] for r in result] # (x,y,z,w,h,d)
    orig_bins = [r[0] for r in result]
    
    # Build bins dict
    bins = {}
    for i, r in enumerate(result):
        b_id = r[0]
        if b_id not in bins:
            bins[b_id] = []
        bins[b_id].append(i)
        
    current_score = _calc_score(result, boxes, container)
    rng = random.Random(999)
    
    max_iters = 10000
    iters = 0
    
    while iters < max_iters:
        iters += 1
        if time.time() > deadline:
            break
            
        bin_ids = sorted(bins.keys())
        if len(bin_ids) <= 1:
            break
            
        # Pick a box from a higher bin
        # Bias towards last bin
        src_bin_id = rng.choice(bin_ids[1:])
        if src_bin_id not in bins or not bins[src_bin_id]:
            continue
            
        box_idx = rng.choice(bins[src_bin_id])
        w, h, d = boxes[box_idx]
        
        # Try to move to lower bins
        for dst_bin_id in bin_ids:
            if dst_bin_id >= src_bin_id:
                continue
                
            placed = [result[i] for i in bins[dst_bin_id]]
            candidates = _generate_positions(placed, w, h, d, W, H, D)
            
            if candidates:
                pos = candidates[0]
                
                # Save state
                old_pos = result[box_idx]
                
                # Perform move
                bins[src_bin_id].remove(box_idx)
                if not bins[src_bin_id]:
                    del bins[src_bin_id]
                bins[dst_bin_id].append(box_idx)
                result[box_idx] = (dst_bin_id, pos[0], pos[1], pos[2], w, h, d)
                
                new_score = _calc_score(result, boxes, container)
                
                if new_score < current_score:
                    current_score = new_score
                else:
                    # Revert
                    bins[dst_bin_id].remove(box_idx)
                    if not bins[dst_bin_id]:
                        del bins[dst_bin_id]
                    bins[src_bin_id].append(box_idx)
                    result[box_idx] = old_pos
                    
                break # Try next box if we found a spot (good or bad)
                
    return result

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    if n == 0:
        return []
        
    t_start = time.time()
    DEADLINE = 115.0  # Safety margin from 120s
    
    # Define sorting orders
    indices = list(range(n))
    orders = []
    
    # 1. Volume descending
    orders.append(sorted(indices, key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True))
    
    # 2. Max dimension descending
    orders.append(sorted(indices, key=lambda i: max(boxes[i]), reverse=True))
    
    # 3. Surface area descending
    orders.append(sorted(indices, key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True))
    
    # 4. Min dimension descending (to fill gaps?)
    orders.append(sorted(indices, key=lambda i: min(boxes[i]), reverse=True))
    
    # 5. Height descending
    orders.append(sorted(indices, key=lambda i: boxes[i][1], reverse=True))
    
    best_result = None
    best_score = float('inf')
    
    # Run greedy for each deterministic order
    for order in orders:
        if time.time() - t_start > DEADLINE:
            break
            
        result = _pack_greedy(boxes, container, order)
        score = _calc_score(result, boxes, container)
        
        if score < best_score:
            best_score = score
            best_result = result
            
    # Randomized restarts
    rng = random.Random(123)
    max_restarts = 200
    
    for r in range(max_restarts):
        if time.time() - t_start > DEADLINE:
            break
            
        # Create a noisy order based on volume
        weights = []
        for i in range(n):
            vol = boxes[i][0]*boxes[i][1]*boxes[i][2]
            noise = rng.random() * 0.2 * vol
            weights.append(vol + noise)
            
        rand_order = sorted(indices, key=lambda i: weights[i], reverse=True)
        
        result = _pack_greedy(boxes, container, rand_order)
        score = _calc_score(result, boxes, container)
        
        if score < best_score:
            best_score = score
            best_result = result
            
    # Local Search on the best result
    if best_result is not None and time.time() - t_start < DEADLINE - 5:
        # We need a more robust local search that preserves original positions for revert
        # Let's implement a simpler, safer local search here
        polished = _improve_result(best_result, boxes, container, DEADLINE)
        polished_score = _calc_score(polished, boxes, container)
        if polished_score < best_score:
            best_result = polished
            best_score = polished_score
            
    return best_result

# EVOLVE-BLOCK-END
