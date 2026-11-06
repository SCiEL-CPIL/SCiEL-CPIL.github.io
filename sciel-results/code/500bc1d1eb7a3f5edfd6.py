"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def pack(boxes, container):
    """
    3D Bin Packing using Randomized Best-Fit Decreasing with Skyline representation.
    Returns list of (bin_id, x, y, z, w, h, d) for each box.
    """
    deadline = time.time() + 110.0  # Safety margin for 120s limit
    W, H, D = container
    n = len(boxes)
    
    # Pre-sort indices by volume descending to prioritize large items (BFD)
    # We will use this order for placement attempts in each restart
    indices_by_vol = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    best_score = float('inf')
    best_placements = None
    
    # Multi-restart loop
    # Each restart uses a slightly different randomization in tie-breaking or order
    # to explore different local optima
    while time.time() < deadline:
        # Initialize bins
        # Each bin has a list of placed boxes and a skyline
        # Skyline: list of (z, y, x_start, x_end) representing top surfaces
        # Actually, a simpler skyline for 3D is often complex. 
        # Let's use a simpler "max height per cell" grid if dimensions are small? 
        # No, dimensions can be large. 
        # Let's use a list of placed boxes and check overlaps, but that is O(N^2).
        # For 40 boxes, O(N^2) is 1600 checks per box, total 64000 checks per bin.
        # With multiple bins, it's manageable.
        
        # However, to be efficient and "different" from shelf, let's use a 
        # simple "corner candidate" generation based on existing boxes.
        # Candidates: (0,0,0) and (x+w, y, z), (x, y+h, z), (x, y, z+d) for each box.
        
        bins = []  # List of lists of placed boxes (bin_idx, x, y, z, w, h, d)
        
        # Shuffle the indices slightly to break ties and explore variants?
        # BFD is usually deterministic on sorted list. 
        # To get different results in restarts, we can randomly permute boxes 
        # of similar volume or just add small noise to volume for sorting.
        
        # Let's stick to strict BFD but randomize the tie-breaking for equal volumes
        # and maybe randomize the choice among candidate positions with equal "best fit" score.
        
        # Define a scoring function for a placement in a bin
        # "Best Fit" usually means the placement that minimizes wasted space or 
        # is closest to the origin. 
        # Let's define "best" as minimizing (x, y, z) lexicographically, 
        # or minimizing distance to origin.
        # Or, we can try to fill the "last bin" more densely.
        
        # Strategy:
        # 1. Sort boxes by volume desc.
        # 2. For each box, try to place it in existing bins.
        #    - Generate candidate positions (corners).
        #    - Check validity (overlap, bounds).
        #    - Score valid positions.
        #    - Pick best position in best bin.
        # 3. If no existing bin works, open a new bin.
        
        # To improve score: 
        # Score = B + 0.1 * (1 - last_bin_fill)
        # We want to minimize B. 
        # If B is fixed, we want to maximize last_bin_fill.
        # This suggests that when B is minimized, we should try to pack the last bin as full as possible.
        # However, BFD naturally tends to fill bins.
        
        # Let's implement a simple BFD with candidate generation.
        
        # To allow exploration, we will randomize the order of boxes with the same volume
        # and randomize the selection among equally good candidate positions.
        
        # 1. Prepare order
        # Group by volume, shuffle within groups
        boxes_info = [(i, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
        boxes_info.sort(key=lambda x: x[1]*x[2]*x[3], reverse=True)
        
        # Shuffle ties
        for i in range(1, len(boxes_info)):
            if boxes_info[i][1]*boxes_info[i][2]*boxes_info[i][3] == boxes_info[i-1][1]*boxes_info[i-1][2]*boxes_info[i-1][3]:
                # Swap with random previous index in the same volume group
                j = random.randint(0, i)
                while boxes_info[j][1]*boxes_info[j][2]*boxes_info[j][3] != boxes_info[i][1]*boxes_info[i][2]*boxes_info[i][3]:
                    j -= 1
                boxes_info[i], boxes_info[j] = boxes_info[j], boxes_info[i]
        
        placements = [None] * n
        
        for idx, w, h, d in boxes_info:
            placed = False
            best_bin_idx = -1
            best_pos = None
            best_score_val = float('inf')
            
            # Try existing bins
            for b_idx in range(len(bins)):
                bin_boxes = bins[b_idx]
                
                # Generate candidates
                # Start with (0,0,0)
                candidates = [(0, 0, 0)]
                for (bx, by, bz, bw, bh, bd) in bin_boxes:
                    candidates.append((bx + bw, by, bz))
                    candidates.append((bx, by + bh, bz))
                    candidates.append((bx, by, bz + bd))
                
                # Deduplicate candidates
                candidates = list(set(candidates))
                
                # Check each candidate
                for (cx, cy, cz) in candidates:
                    # Check bounds
                    if cx + w > W or cy + h > H or cz + d > D:
                        continue
                    
                    # Check overlap
                    valid = True
                    for (bx, by, bz, bw, bh, bd) in bin_boxes:
                        if (cx < bx + bw and cx + w > bx and
                            cy < by + bh and cy + h > by and
                            cz < bz + bd and cz + d > bz):
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    # Score the position
                    # We want to pack tightly. 
                    # A common heuristic is to minimize the "new bounding box" volume or 
                    # distance to origin.
                    # Let's use distance to origin: x+y+z
                    # To encourage filling the last bin, we might bias towards 
                    # placing in the last bin if it's the only one that fits?
                    # No, BFD minimizes bins first.
                    
                    score = cx + cy + cz
                    
                    # Tie-breaking randomization
                    if score < best_score_val:
                        best_score_val = score
                        best_bin_idx = b_idx
                        best_pos = (cx, cy, cz)
                    elif score == best_score_val:
                        # Randomly choose among ties
                        if random.random() < 0.5:
                            best_bin_idx = b_idx
                            best_pos = (cx, cy, cz)
            
            if best_bin_idx == -1:
                # Open new bin
                best_bin_idx = len(bins)
                bins.append([])
                best_pos = (0, 0, 0)
                best_score_val = 0
            
            # Place box
            bx, by, bz = best_pos
            bins[best_bin_idx].append((bx, by, bz, w, h, d))
            placements[idx] = (best_bin_idx, bx, by, bz, w, h, d)
            
            # Early termination check? No, must place all.
            
        # Calculate score for this restart
        if not bins:
            continue
            
        # Find max bin index used
        max_bin = max(p[0] for p in placements)
        B = max_bin + 1
        
        # Calculate last bin fill
        last_bin_idx = max_bin
        last_bin_volume = 0
        for (bx, by, bz, bw, bh, bd) in bins[last_bin_idx]:
            last_bin_volume += bw * bh * bd
            
        container_vol = W * H * D
        if container_vol > 0:
            fill_fraction = last_bin_volume / container_vol
        else:
            fill_fraction = 0.0
            
        score = B + 0.1 * (1 - fill_fraction)
        
        if score < best_score:
            best_score = score
            best_placements = [p.copy() for p in placements]
            
        # If B is 1 and fill is 1.0, we can't do better (theoretical minimum)
        if best_score <= 1.0:
            break

    # Fallback: if no valid placement found (should not happen), return naive
    if best_placements is None:
        # Fallback to simple shelf packing
        W, H, D = container
        placements, b, x, y, z, row_h, layer_d = [], 0, 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            placements.append((b, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        return placements

    return best_placements
# EVOLVE-BLOCK-END
