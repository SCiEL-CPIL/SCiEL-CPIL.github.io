import time
import math
import random

def _box_volume(w, h, d):
    return w * h * d

def _overlaps(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                y1 + h1 <= y2 or y2 + h2 <= x1 or
                z1 + d1 <= z2 or z2 + d2 <= x1)

def _fits_in_container(x, y, z, w, h, d, W, H, D):
    return x >= 0 and y >= 0 and z >= 0 and x + w <= W and y + h <= H and z + d <= D

def _get_candidates(placed, W, H, D):
    """
    Generate candidate positions for a new box in a bin with 'placed' boxes.
    Returns a list of (x, y, z) tuples.
    """
    candidates = set()
    candidates.add((0, 0, 0))
    
    for (px, py, pz, pw, ph, pd) in placed:
        # Faces
        candidates.add((px + pw, py, pz))
        candidates.add((px, py + ph, pz))
        candidates.add((px, py, pz + pd))
        
        # Edges
        candidates.add((px + pw, py + ph, pz))
        candidates.add((px + pw, py, pz + pd))
        candidates.add((px, py + ph, pz + pd))
        
        # Corner
        candidates.add((px + pw, py + ph, pz + pd))
        
    # Filter valid positions
    valid = []
    for (cx, cy, cz) in candidates:
        if _fits_in_container(cx, cy, cz, 1, 1, 1, W, H, D): # Check bounds for the corner itself
             # We need to check if the *box* fits, but we don't know the box size here.
             # So we just return all valid corner positions within the container bounds.
             if cx < W and cy < H and cz < D:
                 valid.append((cx, cy, cz))
    return valid

def _find_best_position(placed, w, h, d, W, H, D):
    """
    Find the best position for a box of size (w,h,d) in a bin with 'placed' boxes.
    Returns (x, y, z) or None.
    """
    candidates = _get_candidates(placed, W, H, D)
    
    best_pos = None
    best_score = None
    
    for (cx, cy, cz) in candidates:
        if not _fits_in_container(cx, cy, cz, w, h, d, W, H, D):
            continue
            
        # Check overlap
        overlap = False
        for (px, py, pz, pw, ph, pd) in placed:
            if _overlaps(cx, cy, cz, w, h, d, px, py, pz, pw, ph, pd):
                overlap = True
                break
        if overlap:
            continue
            
        # Score: Minimize Z, then Y, then X
        score = (cz, cy, cx)
        if best_score is None or score < best_score:
            best_score = score
            best_pos = (cx, cy, cz)
            
    return best_pos

def _calculate_score(bins, W, H, D):
    """
    Calculate the score for a given set of bins.
    bins: list of lists of (x, y, z, w, h, d)
    """
    if not bins:
        return float('inf')
        
    B = len([b for b in bins if b]) # Count non-empty bins
    
    # Find the bin with max volume
    max_vol = 0
    for b in bins:
        if not b: continue
        vol = sum(_box_volume(w, h, d) for (_, _, _, w, h, d) in b)
        if vol > max_vol:
            max_vol = vol
            
    container_vol = W * H * D
    if container_vol == 0:
        return B
        
    # We can choose which bin is the "last" one. We choose the one with max_vol.
    # So the secondary term is based on max_vol.
    secondary = 0.1 * (1 - max_vol / container_vol)
    
    return B + secondary

def pack(boxes, container):
    W, H, D = container
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Initial Solution: Best Fit Decreasing
    # Sort indices by volume descending
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda ib: _box_volume(ib[1][0], ib[1][1], ib[1][2]), reverse=True)
    
    bins = [] # List of lists of (x, y, z, w, h, d)
    
    for orig_idx, (w, h, d) in indexed_boxes:
        if time.time() > deadline:
            break
            
        placed_in_bin = False
        best_bin_idx = -1
        best_pos = None
        best_bin_score = None
        
        # Try existing bins
        for bi, placed in enumerate(bins):
            if time.time() > deadline:
                break
                
            pos = _find_best_position(placed, w, h, d, W, H, D)
            if pos is not None:
                # Score for this bin: We prefer bins that are already "full" in some sense?
                # Or just take the first one that fits?
                # Best Fit usually implies choosing the bin that leaves the least space.
                # Here, we just pick the first valid bin found? Or the one with the best position?
                # Let's pick the bin where the box is placed at the lowest Z.
                pos_score = (pos[2], pos[1], pos[0])
                if best_bin_score is None or pos_score < best_bin_score:
                    best_bin_score = pos_score
                    best_bin_idx = bi
                    best_pos = pos
        
        # If no existing bin fits, create new
        if best_bin_idx == -1:
            if w <= W and h <= H and d <= D:
                best_bin_idx = len(bins)
                best_pos = (0, 0, 0)
        
        if best_pos is not None:
            if best_bin_idx == len(bins):
                bins.append([])
            bx, by, bz = best_pos
            bins[best_bin_idx].append((bx, by, bz, w, h, d))
        else:
            # Fallback: Should not happen if input is valid
            new_bin = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            
    # 2. Local Search (Simulated Annealing)
    # We represent the solution as a list of bins.
    # We need to track which box is in which bin to allow moves/swaps.
    # Let's create a mapping: box_index -> (bin_index, position_index_in_bin)
    # This is complex to maintain.
    
    # Alternative: Represent solution as a list of (bin_id, x, y, z, w, h, d) for each box.
    # This is easier to manipulate.
    
    # Convert bins to flat list of placements
    placements = [None] * len(boxes)
    for bi, bin_content in enumerate(bins):
        for (x, y, z, w, h, d) in bin_content:
             # We lost the original index!
             pass
             
    # We need to keep track of original indices.
    # Let's restart the initial solution generation to keep indices.
    
    placements = [None] * len(boxes)
    bins = []
    
    for orig_idx, (w, h, d) in indexed_boxes:
        if time.time() > deadline:
            break
            
        placed_in_bin = False
        best_bin_idx = -1
        best_pos = None
        best_bin_score = None
        
        for bi, placed in enumerate(bins):
             if time.time() > deadline:
                 break
             pos = _find_best_position(placed, w, h, d, W, H, D)
             if pos is not None:
                 pos_score = (pos[2], pos[1], pos[0])
                 if best_bin_score is None or pos_score < best_bin_score:
                     best_bin_score = pos_score
                     best_bin_idx = bi
                     best_pos = pos
                     
        if best_bin_idx == -1:
            if w <= W and h <= H and d <= D:
                best_bin_idx = len(bins)
                best_pos = (0, 0, 0)
                
        if best_pos is not None:
            if best_bin_idx == len(bins):
                bins.append([])
            bx, by, bz = best_pos
            bins[best_bin_idx].append((bx, by, bz, w, h, d, orig_idx)) # Store orig_idx
            placements[orig_idx] = (best_bin_idx, bx, by, bz, w, h, d)
        else:
            new_bin = len(bins)
            bins.append([(0, 0, 0, w, h, d, orig_idx)])
            placements[orig_idx] = (new_bin, 0, 0, 0, w, h, d)
            
    # Helper to calculate score from placements
    def _score_placements(placements):
        bin_vols = {}
        for i, (bi, x, y, z, w, h, d) in enumerate(placements):
            if placements[i] is None: continue
            vol = w*h*d
            bin_vols[bi] = bin_vols.get(bi, 0) + vol
            
        if not bin_vols:
            return float('inf')
            
        B = len(bin_vols)
        max_vol = max(bin_vols.values())
        container_vol = W * H * D
        if container_vol == 0: return B
        return B + 0.1 * (1 - max_vol / container_vol)
        
    current_score = _score_placements(placements)
    best_placements = [p[:] if p else None for p in placements] # Deep copy
    best_score = current_score
    
    # Local Search
    T = 1.0
    cooling = 0.999
    no_improve_count = 0
    
    # Build bin structures for fast access
    # bin_contents[bi] = list of (box_idx, x, y, z, w, h, d)
    
    def _build_bins(placements):
        bins_dict = defaultdict(list)
        for i, p in enumerate(placements):
            if p is not None:
                bi = p[0]
                bins_dict[bi].append((i, p[1], p[2], p[3], p[4], p[5], p[6]))
        return bins_dict
        
    bins_dict = _build_bins(placements)
    
    while time.time() < deadline:
        # Generate a move
        move_type = random.choice(['move', 'swap'])
        
        if move_type == 'move':
            # Pick a random box
            box_idx = random.randint(0, len(boxes) - 1)
            if placements[box_idx] is None: continue
            cur_bin = placements[box_idx][0]
            
            # Pick a target bin
            # Prefer bins that already exist
            if len(bins_dict) > 0:
                target_bin = random.choice(list(bins_dict.keys()))
                if target_bin == cur_bin: continue
                
                # Try to place box in target_bin
                # Get boxes in target_bin
                target_boxes = [b for b in bins_dict[target_bin] if b[0] != box_idx]
                # Note: box_idx is not in target_bin, so no need to filter, but safe.
                
                pos = _find_best_position([(b[1], b[2], b[3], b[4], b[5], b[6]) for b in target_boxes], 
                                          boxes[box_idx][0], boxes[box_idx][1], boxes[box_idx][2], W, H, D)
                
                if pos is not None:
                    # Perform move
                    new_placements = [p[:] if p else None for p in placements]
                    new_placements[box_idx] = (target_bin, pos[0], pos[1], pos[2], boxes[box_idx][0], boxes[box_idx][1], boxes[box_idx][2])
                    
                    new_score = _score_placements(new_placements)
                    delta = new_score - current_score
                    
                    if delta < 0 or random.random() < math.exp(-delta / T):
                        # Accept
                        placements = new_placements
                        current_score = new_score
                        no_improve_count = 0
                        if current_score < best_score:
                            best_score = current_score
                            best_placements = [p[:] if p else None for p in placements]
                            bins_dict = _build_bins(placements)
                    else:
                        no_improve_count += 1
            else:
                no_improve_count += 1
                
        elif move_type == 'swap':
            if len(bins_dict) < 2:
                no_improve_count += 1
                continue
                
            # Pick two distinct bins
            bin_list = list(bins_dict.keys())
            if len(bin_list) < 2: continue
            bin_a, bin_b = random.sample(bin_list, 2)
            
            # Pick a box from bin_a and a box from bin_b
            if not bins_dict[bin_a] or not bins_dict[bin_b]: continue
            box_a_idx = random.choice(bins_dict[bin_a])[0]
            box_b_idx = random.choice(bins_dict[bin_b])[0]
            
            # Try to place box_a in bin_b (without box_b) and box_b in bin_a (without box_a)
            boxes_a_in_b = [b for b in bins_dict[bin_b] if b[0] != box_b_idx]
            boxes_b_in_a = [b for b in bins_dict[bin_a] if b[0] != box_a_idx]
            
            pos_a = _find_best_position([(b[1], b[2], b[3], b[4], b[5], b[6]) for b in boxes_a_in_b],
                                        boxes[box_a_idx][0], boxes[box_a_idx][1], boxes[box_a_idx][2], W, H, D)
            pos_b = _find_best_position([(b[1], b[2], b[3], b[4], b[5], b[6]) for b in boxes_b_in_a],
                                        boxes[box_b_idx][0], boxes[box_b_idx][1], boxes[box_b_idx][2], W, H, D)
            
            if pos_a is not None and pos_b is not None:
                new_placements = [p[:] if p else None for p in placements]
                new_placements[box_a_idx] = (bin_b, pos_a[0], pos_a[1], pos_a[2], boxes[box_a_idx][0], boxes[box_a_idx][1], boxes[box_a_idx][2])
                new_placements[box_b_idx] = (bin_a, pos_b[0], pos_b[1], pos_b[2], boxes[box_b_idx][0], boxes[box_b_idx][1], boxes[box_b_idx][2])
                
                new_score = _score_placements(new_placements)
                delta = new_score - current_score
                
                if delta < 0 or random.random() < math.exp(-delta / T):
                    placements = new_placements
                    current_score = new_score
                    no_improve_count = 0
                    if current_score < best_score:
                        best_score = current_score
                        best_placements = [p[:] if p else None for p in placements]
                        bins_dict = _build_bins(placements)
                else:
                    no_improve_count += 1
        else:
            no_improve_count += 1
            
        T *= cooling
        
        # Restart if stuck
        if no_improve_count > 1000:
            # Perturb: Randomly move a box
            box_idx = random.randint(0, len(boxes) - 1)
            if placements[box_idx] is not None:
                cur_bin = placements[box_idx][0]
                # Move to a random other bin if it fits, or just keep it
                pass # Simple restart: just continue with lower T
            no_improve_count = 0
            T = max(T, 0.1) # Don't let T go too low
            
    # Post-processing: Relabel bins
    # Find the bin with max volume in best_placements
    bin_vols = {}
    for p in best_placements:
        if p is not None:
            bi = p[0]
            vol = p[4]*p[5]*p[6]
            bin_vols[bi] = bin_vols.get(bi, 0) + vol
            
    if not bin_vols:
        return best_placements
        
    # Sort bins by volume descending
    sorted_bins = sorted(bin_vols.items(), key=lambda item: item[1], reverse=True)
    
    # Assign new labels
    # The fullest bin gets the highest label (B-1)
    # The others get 0..B-2
    B = len(sorted_bins)
    label_map = {}
    for i, (old_label, vol) in enumerate(sorted_bins):
        if i == 0:
            label_map[old_label] = B - 1
        else:
            label_map[old_label] = i - 1
            
    final_placements = []
    for p in best_placements:
        if p is None:
            final_placements.append(None)
        else:
            new_label = label_map[p[0]]
            final_placements.append((new_label, p[1], p[2], p[3], p[4], p[5], p[6]))
            
    return final_placements
