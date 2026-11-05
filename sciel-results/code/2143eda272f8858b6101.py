"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    container_vol = W * H * D
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Sort boxes by volume descending for better packing
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[4], reverse=True)
    
    # bins: list of lists, each inner list is list of (orig_idx, x, y, z, w, h, d)
    bins = []
    bin_vols = []  # cumulative volume in each bin
    
    def find_position(bin_placements, w, h, d):
        """Find best (lowest z, then y, then x) position for a box in a bin."""
        x_candidates = [0]
        y_candidates = [0]
        z_candidates = [0]
        
        for _, px, py, pz, pw, ph, pd in bin_placements:
            if px + pw <= W:
                x_candidates.append(px + pw)
            if py + ph <= H:
                y_candidates.append(py + ph)
            if pz + pd <= D:
                z_candidates.append(pz + pd)
        
        x_candidates = sorted(set(x for x in x_candidates if x + w <= W))
        y_candidates = sorted(set(y for y in y_candidates if y + h <= H))
        z_candidates = sorted(set(z for z in z_candidates if z + d <= D))
        
        best_pos = None
        best_score = None
        
        for z in z_candidates:
            for y in y_candidates:
                for x in x_candidates:
                    # Check overlap
                    overlap = False
                    for _, px, py, pz, pw, ph, pd in bin_placements:
                        if (x < px + pw and x + w > px and
                            y < py + ph and y + h > py and
                            z < pz + pd and z + d > pz):
                            overlap = True
                            break
                    if not overlap:
                        score = (z, y, x)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
        
        return best_pos
    
    def compute_score(bins, bin_vols):
        """Compute the objective: B + 0.1*(1 - last_bin_vol/container_vol)"""
        if not bins:
            return float('inf')
        B = len(bins)
        last_bin_vol = bin_vols[-1]
        return B + 0.1 * (1 - last_bin_vol / container_vol)
    
    # Phase 1: Initial greedy placement (Best Fit Decreasing)
    for idx, (orig_i, w, h, d, vol) in enumerate(indexed_boxes):
        if time.time() > deadline:
            break
        
        best_bin = -1
        best_fit_score = None
        
        for b_idx, bin_placements in enumerate(bins):
            pos = find_position(bin_placements, w, h, d)
            if pos is not None:
                # Best fit: minimize z-coordinate (prefer lower placement)
                score = pos[2]
                if best_fit_score is None or score < best_fit_score:
                    best_fit_score = score
                    best_bin = b_idx
        
        if best_bin >= 0:
            x, y, z = find_position(bins[best_bin], w, h, d)
            bins[best_bin].append((orig_i, x, y, z, w, h, d))
            bin_vols[best_bin] += vol
        else:
            bins.append([(orig_i, 0, 0, 0, w, h, d)])
            bin_vols.append(vol)
    
    # Build initial result
    result = [None] * n
    for b_idx, bin_placements in enumerate(bins):
        for orig_i, x, y, z, w, h, d in bin_placements:
            result[orig_i] = (b_idx, x, y, z, w, h, d)
    
    # Fallback for any unplaced boxes
    next_bin = len(bins)
    for i in range(n):
        if result[i] is None:
            w, h, d = boxes[i]
            result[i] = (next_bin, 0, 0, 0, w, h, d)
            next_bin += 1
    
    best_score = compute_score(bins, bin_vols)
    best_result = [tuple(r) for r in result]
    
    # Phase 2: Local search - try to move boxes from last bin to earlier bins
    # This can reduce B or increase last bin volume fraction
    num_moves = 0
    max_moves = 5000
    
    while time.time() < deadline - 5.0 and num_moves < max_moves:
        if len(bins) < 2:
            break
        
        last_bin_idx = len(bins) - 1
        last_bin = bins[last_bin_idx]
        
        if not last_bin:
            break
        
        # Try to move each box in the last bin to an earlier bin
        moved_any = False
        for li in range(len(last_bin) - 1, -1, -1):
            if time.time() > deadline - 5.0:
                break
            
            orig_i, x, y, z, w, h, d = last_bin[li]
            vol = w * h * d
            
            # Try moving to each earlier bin
            best_target_bin = -1
            best_pos = None
            best_target_score = None
            
            for b_idx in range(last_bin_idx):
                if time.time() > deadline - 5.0:
                    break
                
                # Temporarily remove box from last bin for testing
                # (We'll just test placement in other bins)
                pos = find_position(bins[b_idx], w, h, d)
                if pos is not None:
                    # Check if this move improves the score
                    # After move: last bin loses vol, target bin gains vol
                    new_last_vol = bin_vols[last_bin_idx] - vol
                    new_target_vol = bin_vols[b_idx] + vol
                    
                    # If last bin becomes empty, B decreases by 1
                    if new_last_vol == 0:
                        new_B = len(bins) - 1
                        # The new last bin is the one before
                        if new_B > 0:
                            new_last_vol = bin_vols[new_B - 1]
                        else:
                            new_last_vol = 0
                    else:
                        new_B = len(bins)
                    
                    new_score = new_B + 0.1 * (1 - new_last_vol / container_vol)
                    
                    if best_target_score is None or new_score < best_target_score:
                        best_target_score = new_score
                        best_target_bin = b_idx
                        best_pos = pos
            
            if best_target_bin >= 0:
                # Execute the move
                last_bin.pop(li)
                bin_vols[last_bin_idx] -= vol
                
                if bin_vols[last_bin_idx] == 0:
                    # Remove empty last bin
                    bins.pop(last_bin_idx)
                    bin_vols.pop(last_bin_idx)
                    last_bin_idx = len(bins) - 1
                    if last_bin_idx < 0:
                        break
                    # Update result references
                    # We'll rebuild at the end
                else:
                    # Update result for this box
                    result[orig_i] = None  # Will be rebuilt
                
                x, y, z = best_pos
                bins[best_target_bin].append((orig_i, x, y, z, w, h, d))
                bin_vols[best_target_bin] += vol
                
                # Update result
                result[orig_i] = (best_target_bin, x, y, z, w, h, d)
                
                moved_any = True
                num_moves += 1
        
        if not moved_any:
            break
    
    # Rebuild result from bins
    final_result = [None] * n
    for b_idx, bin_placements in enumerate(bins):
        for orig_i, x, y, z, w, h, d in bin_placements:
            final_result[orig_i] = (b_idx, x, y, z, w, h, d)
    
    # Fallback for any unplaced boxes
    next_bin = len(bins)
    for i in range(n):
        if final_result[i] is None:
            w, h, d = boxes[i]
            final_result[i] = (next_bin, 0, 0, 0, w, h, d)
            next_bin += 1
    
    # Compare with initial result
    if compute_score(bins, bin_vols) < best_score:
        best_result = [tuple(r) for r in final_result]
    
    return best_result
# EVOLVE-BLOCK-END
