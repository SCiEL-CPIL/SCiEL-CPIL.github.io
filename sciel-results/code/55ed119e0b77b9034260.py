"""
3D Bin Packing Solver
Strategy:
1. Generate multiple sorting orders for boxes (Volume, Max Dim, Height, etc.).
2. For each order, perform a greedy multi-bin packing using a Bottom-Left-Front 
   (BLF) style heuristic with exact overlap checks.
3. Candidate positions are generated from the corners of already placed boxes 
   and container boundaries, ensuring tight packing.
4. Select the solution with the best score (lowest B + 0.1*(1 - fill)).
5. Deadline guard ensures we return within time limits.
"""
import time
import sys

# EVOLVE-BLOCK-START

def _boxes_overlap(b1, b2):
    """Check if two boxes (x,y,z,w,h,d) overlap."""
    x1, y1, z1, w1, h1, d1 = b1
    x2, y2, z2, w2, h2, d2 = b2
    
    # Check separation in any axis
    if x1 + w1 <= x2 or x2 + w2 <= x1:
        return False
    if y1 + h1 <= y2 or y2 + h2 <= y1:
        return False
    if z1 + d1 <= z2 or z2 + d2 <= z1:
        return False
    
    return True

def _can_place(placed_boxes, new_box, W, H, D):
    """Check if new_box (x,y,z,w,h,d) fits in container and doesn't overlap placed_boxes."""
    x, y, z, w, h, d = new_box
    if x < 0 or y < 0 or z < 0:
        return False
    if x + w > W or y + h > H or z + d > D:
        return False
    
    for pb in placed_boxes:
        if _boxes_overlap(new_box, pb):
            return False
    return True

def _generate_candidates(placed_boxes, w, h, d, W, H, D):
    """
    Generate candidate positions for a box of size (w,h,d).
    Positions are derived from the 'corners' of existing boxes and container boundaries.
    We prefer positions that are lower (z), then further back (y), then further left (x) 
    to maintain a compact, bottom-left-front structure.
    """
    candidates = set()
    
    # Always try (0,0,0)
    candidates.add((0, 0, 0))
    
    # Add boundaries
    candidates.add((W - w, 0, 0))
    candidates.add((0, H - h, 0))
    candidates.add((0, 0, D - d))
    
    # Generate candidates from placed boxes
    for pb in placed_boxes:
        px, py, pz, pw, ph, pd = pb
        
        # X candidates: left of pb, right of pb
        cx1 = px - w
        cx2 = px + pw
        # Y candidates: front of pb, back of pb
        cy1 = py - h
        cy2 = py + ph
        # Z candidates: below pb, above pb
        cz1 = pz - d
        cz2 = pz + pd
        
        # Combine these to create a small set of likely spots
        # We don't want to explode the search space, so we pick specific combinations
        # 1. Directly below/above/left/right/front/back
        # 2. Corners relative to the box
        
        # Simple heuristic: try all combinations of x from {cx1, cx2}, y from {cy1, cy2}, z from {cz1, cz2}
        # This is 8 points per box. With 40 boxes, that's 320 candidates max. Very fast.
        for cx in [cx1, cx2]:
            for cy in [cy1, cy2]:
                for cz in [cz1, cz2]:
                    if cx >= 0 and cy >= 0 and cz >= 0:
                        if cx + w <= W and cy + h <= H and cz + d <= D:
                            candidates.add((cx, cy, cz))
        
        # Also try the "face" alignments
        # e.g. same x, same y, different z
        candidates.add((px, py, cz2)) # Above
        candidates.add((px, py, cz1)) # Below
        candidates.add((cx2, py, pz)) # Right
        candidates.add((cx1, py, pz)) # Left
        candidates.add((px, cy2, pz)) # Back
        candidates.add((px, cy1, pz)) # Front

    # Filter valid candidates and sort by heuristic score (z, then y, then x)
    # Lower z is better (bottom), then lower y (back? usually y is up or forward, let's assume y is vertical height in container? 
    # Wait, standard convention: x=width, y=depth, z=height? Or x,y=base, z=height?
    # Task says: (bin, x, y, z, w, h, d). Container is W, H, D.
    # Usually W->x, H->y (or z?), D->z (or y?).
    # Let's assume:
    # x in [0, W]
    # y in [0, H]
    # z in [0, D]
    # Volume = W*H*D.
    # If H is height, then z is likely depth or y is height.
    # Let's look at the seed: `x + w > W`, `y + h > H`, `z + d > D`.
    # So w corresponds to x, h to y, d to z.
    # So H is the extent in the y-direction, D is the extent in the z-direction.
    # Usually "Height" is the vertical axis. If H is height, then y is vertical.
    # If D is depth, z is depth.
    # Let's assume y is vertical (Height) and z is Depth.
    # "Bottom" means low y. "Front" means low z?
    # Let's sort by y (height) ascending, then z (depth) ascending, then x (width) ascending.
    valid_candidates = []
    for c in candidates:
        if c[0] >= 0 and c[1] >= 0 and c[2] >= 0:
            if c[0] + w <= W and c[1] + h <= H and c[2] + d <= D:
                valid_candidates.append(c)
                
    # Sort: Prefer lowest y (bottom), then lowest z (front), then lowest x (left)
    valid_candidates.sort(key=lambda p: (p[1], p[2], p[0]))
    return valid_candidates

def pack(boxes, container):
    W, H, D = container
    container_vol = W * H * D
    n = len(boxes)
    
    if n == 0:
        return []
    
    t_start = time.time()
    DEADLINE = 100.0  # Safety margin
    
    # Define different sorting orders
    # boxes: list of (w, h, d)
    indices = list(range(n))
    
    orders = []
    
    # 1. Volume descending
    orders.append(sorted(indices, key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True))
    
    # 2. Max dimension descending
    orders.append(sorted(indices, key=lambda i: max(boxes[i]), reverse=True))
    
    # 3. Height (h) descending - packs tall boxes first to establish vertical structure
    orders.append(sorted(indices, key=lambda i: boxes[i][1], reverse=True))
    
    # 4. Depth (d) descending
    orders.append(sorted(indices, key=lambda i: boxes[i][2], reverse=True))
    
    # 5. Width (w) descending
    orders.append(sorted(indices, key=lambda i: boxes[i][0], reverse=True))
    
    # 6. Aspect ratio (max/min) descending - "square" or "cubic" boxes first?
    def aspect(i):
        w,h,d = boxes[i]
        return max(w,h,d)/min(w,h,d)
    orders.append(sorted(indices, key=aspect, reverse=True))
    
    # 7. Volume ascending (sometimes helps fill gaps? Unlikely for first pass, but cheap to try)
    orders.append(sorted(indices, key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2]))
    
    best_result = None
    best_score = float('inf')
    
    for order in orders:
        if time.time() - t_start > DEADLINE:
            break
            
        # Greedy Packing
        bins = [] # List of lists of placed boxes (x,y,z,w,h,d)
        result = [None] * n
        
        for idx in order:
            w, h, d = boxes[idx]
            placed = False
            
            # Try to place in existing bins
            # We iterate bins in order of "fullness" or just index? 
            # Best fit decreasing usually tries bins that are "most full" or just first fit.
            # Let's try all existing bins and pick the one that yields the best position score,
            # or just the first valid position in the sorted candidate list.
            
            best_bin_idx = -1
            best_pos = None
            best_pos_score = None
            
            for b_idx in range(len(bins)):
                placed_in_bin = bins[b_idx]
                
                # Generate candidates
                candidates = _generate_candidates(placed_in_bin, w, h, d, W, H, D)
                
                for pos in candidates:
                    new_box = (pos[0], pos[1], pos[2], w, h, d)
                    if _can_place(placed_in_bin, new_box, W, H, D):
                        # Heuristic score: lower y (height) is better, then lower z, then lower x
                        # We want to fill the bin tightly.
                        score = pos[1] * 10000 + pos[2] * 100 + pos[0]
                        
                        if best_pos_score is None or score < best_pos_score:
                            best_pos_score = score
                            best_bin_idx = b_idx
                            best_pos = pos
                        break # Found a valid spot in this bin, check other bins for better spot?
                            # Or just take the first valid spot in the best bin?
                            # "Best Fit" usually means the position that leaves the least empty space.
                            # Our candidate generation is already biased towards bottom-left-front.
                            # Taking the first valid candidate from the sorted list is a good BLF heuristic.
                            # Comparing across bins: we prefer the bin where the box fits "lower" (smaller y).
                            # So we should check all bins and pick the global minimum score.
            
            if best_bin_idx != -1:
                w, h, d = boxes[idx]
                pos = best_pos
                new_box = (pos[0], pos[1], pos[2], w, h, d)
                bins[best_bin_idx].append(new_box)
                result[idx] = (best_bin_idx, pos[0], pos[1], pos[2], w, h, d)
                placed = True
            else:
                # Open new bin
                new_bin_idx = len(bins)
                bins.append([])
                new_box = (0, 0, 0, w, h, d)
                bins[new_bin_idx].append(new_box)
                result[idx] = (new_bin_idx, 0, 0, 0, w, h, d)
                placed = True
        
        # Calculate score for this result
        if all(r is not None for r in result):
            num_bins = max(r[0] for r in result) + 1
            # Find last bin occupied
            last_bin_id = max(r[0] for r in result)
            last_bin_vol = sum(r[4]*r[5]*r[6] for r in result if r[0] == last_bin_id)
            
            score = num_bins + 0.1 * (1 - last_bin_vol / container_vol)
            
            if score < best_score:
                best_score = score
                best_result = result
    
    # If no valid result found (should not happen), fallback to simple shelf
    if best_result is None:
        # Fallback: simple shelf packing
        result = []
        b = 0
        x, y, z = 0, 0, 0
        row_h = 0
        layer_d = 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            result.append((b, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        best_result = result
    
    return best_result

# EVOLVE-BLOCK-END
