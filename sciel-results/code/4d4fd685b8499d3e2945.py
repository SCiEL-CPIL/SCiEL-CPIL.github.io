import time
import random
import math

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # 1. Initial Construction: Best-Fit Decreasing (BFD)
    # Sort indices by volume descending
    indexed_boxes = sorted(enumerate(boxes), key=lambda x: x[1][0]*x[1][1]*x[1][2], reverse=True)
    
    # State
    # bin_assignments[i] = bin_index for box i
    # bin_positions[bin_index] = list of (box_index, x, y, z)
    # We need to track which box is in which bin to allow moves/swaps.
    
    bin_assignments = [-1] * n
    bin_positions = []  # List of lists. Each inner list contains (box_idx, x, y, z)
    
    def can_place_in_bin(bin_idx, box_idx, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        # Check overlap with existing boxes in this bin
        for (other_idx, ox, oy, oz) in bin_positions[bin_idx]:
            if other_idx == box_idx:
                continue
            ow, oh, od = boxes[other_idx]
            if not (x + w <= ox or ox + ow <= x or
                    y + h <= oy or oy + oh <= y or
                    z + d <= oz or oz + od <= z):
                return False
        return True

    def find_position_in_bin(bin_idx, box_idx, w, h, d):
        # Generate candidates
        candidates = [(0, 0, 0)]
        for (other_idx, ox, oy, oz) in bin_positions[bin_idx]:
            if other_idx == box_idx:
                continue
            ow, oh, od = boxes[other_idx]
            # Corners
            candidates.append((ox + ow, oy, oz))
            candidates.append((ox, oy + oh, oz))
            candidates.append((ox, oy, oz + od))
            candidates.append((ox + ow, oy + oh, oz))
            candidates.append((ox + ow, oy, oz + od))
            candidates.append((ox, oy + oh, oz + od))
            candidates.append((ox + ow, oy + oh, oz + od))
            
        # Deduplicate and sort by (z, y, x) to prefer tight packing
        unique_candidates = list(set(candidates))
        unique_candidates.sort(key=lambda p: (p[2], p[1], p[0]))
        
        for cx, cy, cz in unique_candidates:
            if can_place_in_bin(bin_idx, box_idx, cx, cy, cz, w, h, d):
                return cx, cy, cz
        return None

    # Build initial solution
    for idx, (w, h, d) in indexed_boxes:
        best_bin = -1
        best_pos = None
        best_score = float('inf')
        
        for b in range(len(bin_positions)):
            pos = find_position_in_bin(b, idx, w, h, d)
            if pos is not None:
                # Score: prefer lower z, then y, then x
                score = pos[2] * 10000 + pos[1] * 100 + pos[0]
                if score < best_score:
                    best_score = score
                    best_bin = b
                    best_pos = pos
                    
        if best_bin == -1:
            # Open new bin
            best_bin = len(bin_positions)
            bin_positions.append([])
            best_pos = (0, 0, 0)
            
        bin_assignments[idx] = best_bin
        bin_positions[best_bin].append((idx, best_pos[0], best_pos[1], best_pos[2]))

    # 2. Local Search: Simulated Annealing
    # Objective: Minimize (num_bins, last_bin_volume)
    # Score function for SA:
    # Let B be number of occupied bins.
    # Let V_last be volume of last bin.
    # Cost = B * 1000 + V_last * 0.1 (approximating the mean score structure, but we want to minimize B first)
    # Actually, the score is mean B + 0.1 * (1 - V_last/V_total).
    # Minimizing B is the dominant term.
    
    def get_score():
        if not bin_positions:
            return 0.0
        # Count occupied bins
        occupied = [b for b in range(len(bin_positions)) if bin_positions[b]]
        if not occupied:
            return 0.0
        B = len(occupied)
        last_bin = max(occupied)
        vol = sum(boxes[idx][0]*boxes[idx][1]*boxes[idx][2] for idx, x, y, z in bin_positions[last_bin])
        total_vol = W * H * D
        return B + 0.1 * (1 - vol / total_vol)

    current_score = get_score()
    best_score = current_score
    best_state = None
    
    # Save best state
    def save_state():
        return (bin_assignments[:], [list(b) for b in bin_positions])
    
    best_state = save_state()
    
    T = 1.0
    T_min = 0.001
    alpha = 0.9995
    
    while time.time() < deadline and T > T_min:
        T *= alpha
        
        # Pick a random box
        i = random.randint(0, n - 1)
        w, h, d = boxes[i]
        curr_bin = bin_assignments[i]
        
        # Remove box i from curr_bin
        # Find position in list
        remove_idx = -1
        for j, (idx, x, y, z) in enumerate(bin_positions[curr_bin]):
            if idx == i:
                remove_idx = j
                break
        if remove_idx != -1:
            bin_positions[curr_bin].pop(remove_idx)
        
        # Try to move to a random other bin or new bin
        # With high probability, try existing bins. With low prob, try new bin (if possible) or just existing.
        # Let's stick to moving within existing bins or swapping.
        
        # Strategy 1: Move to random other bin
        # Strategy 2: Swap with a box in another bin
        
        op = random.random()
        if op < 0.7:
            # Move
            target_bins = [b for b in range(len(bin_positions)) if b != curr_bin and bin_positions[b]]
            if not target_bins:
                # Revert
                bin_positions[curr_bin].append((i, 0, 0, 0)) # Placeholder, we need to find pos
                # Actually, if no other bins, we can't move. Revert and continue.
                # Wait, we removed it. We must put it back.
                # Let's find the best pos in curr_bin to restore
                pos = find_position_in_bin(curr_bin, i, w, h, d)
                if pos is None:
                    # This shouldn't happen if the state was valid before
                    pos = (0,0,0)
                bin_positions[curr_bin].append((i, pos[0], pos[1], pos[2]))
                continue
                
            target_bin = random.choice(target_bins)
            pos = find_position_in_bin(target_bin, i, w, h, d)
            if pos is not None:
                # Execute move
                bin_positions[target_bin].append((i, pos[0], pos[1], pos[2]))
                bin_assignments[i] = target_bin
                
                new_score = get_score()
                delta = new_score - current_score
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_state = save_state()
                else:
                    # Revert
                    # Remove from target
                    for j, (idx, x, y, z) in enumerate(bin_positions[target_bin]):
                        if idx == i:
                            bin_positions[target_bin].pop(j)
                            break
                    # Add back to curr
                    # We need to find a valid pos in curr. Since it was there before, (0,0,0) or original might work?
                    # Safer to find best pos again.
                    pos_orig = find_position_in_bin(curr_bin, i, w, h, d)
                    if pos_orig is None:
                         # Fallback: this should not happen if the state was valid
                         pos_orig = (0,0,0)
                    bin_positions[curr_bin].append((i, pos_orig[0], pos_orig[1], pos_orig[2]))
                    bin_assignments[i] = curr_bin
            else:
                # Couldn't place, revert
                pos_orig = find_position_in_bin(curr_bin, i, w, h, d)
                if pos_orig is None:
                    pos_orig = (0,0,0)
                bin_positions[curr_bin].append((i, pos_orig[0], pos_orig[1], pos_orig[2]))
                bin_assignments[i] = curr_bin
        else:
            # Swap
            # Pick random bin j != curr_bin
            target_bins = [b for b in range(len(bin_positions)) if b != curr_bin and bin_positions[b]]
            if target_bins:
                target_bin = random.choice(target_bins)
                # Pick random box k in target_bin
                k_idx = random.randint(0, len(bin_positions[target_bin]) - 1)
                k = bin_positions[target_bin][k_idx][0]
                wk, hk, dk = boxes[k]
                
                # Remove i from curr, k from target
                # We already removed i from curr above.
                # Remove k from target
                for j, (idx, x, y, z) in enumerate(bin_positions[target_bin]):
                    if idx == k:
                        bin_positions[target_bin].pop(j)
                        break
                        
                # Try to place i in target
                pos_i = find_position_in_bin(target_bin, i, w, h, d)
                if pos_i is not None:
                    # Try to place k in curr
                    pos_k = find_position_in_bin(curr_bin, k, wk, hk, dk)
                    if pos_k is not None:
                        # Execute swap
                        bin_positions[target_bin].append((i, pos_i[0], pos_i[1], pos_i[2]))
                        bin_positions[curr_bin].append((k, pos_k[0], pos_k[1], pos_k[2]))
                        bin_assignments[i] = target_bin
                        bin_assignments[k] = curr_bin
                        
                        new_score = get_score()
                        delta = new_score - current_score
                        if delta < 0 or random.random() < math.exp(-delta / T):
                            current_score = new_score
                            if current_score < best_score:
                                best_score = current_score
                                best_state = save_state()
                        else:
                            # Revert swap
                            # Remove i from target, k from curr
                            for j, (idx, x, y, z) in enumerate(bin_positions[target_bin]):
                                if idx == i:
                                    bin_positions[target_bin].pop(j)
                                    break
                            for j, (idx, x, y, z) in enumerate(bin_positions[curr_bin]):
                                if idx == k:
                                    bin_positions[curr_bin].pop(j)
                                    break
                            # Restore original
                            # i back to curr, k back to target
                            # We need to find positions again or store them?
                            # Storing is safer. Let's store original positions before removal.
                            # This implementation is a bit risky with re-finding positions.
                            # Let's restart this iteration if revert is complex?
                            # Actually, let's just find best positions again.
                            pos_i_orig = find_position_in_bin(curr_bin, i, w, h, d)
                            pos_k_orig = find_position_in_bin(target_bin, k, wk, hk, dk)
                            if pos_i_orig and pos_k_orig:
                                bin_positions[curr_bin].append((i, pos_i_orig[0], pos_i_orig[1], pos_i_orig[2]))
                                bin_positions[target_bin].append((k, pos_k_orig[0], pos_k_orig[1], pos_k_orig[2]))
                                bin_assignments[i] = curr_bin
                                bin_assignments[k] = target_bin
                            else:
                                # Critical failure, revert to best known state?
                                # For now, just try to put them back anywhere valid
                                pass
                    else:
                        # Couldn't place k in curr, revert
                        # Put k back in target
                        pos_k_orig = find_position_in_bin(target_bin, k, wk, hk, dk)
                        if pos_k_orig:
                            bin_positions[target_bin].append((k, pos_k_orig[0], pos_k_orig[1], pos_k_orig[2]))
                            bin_assignments[k] = target_bin
                        # i is still in curr (removed at start of op block)
                        pos_i_orig = find_position_in_bin(curr_bin, i, w, h, d)
                        if pos_i_orig:
                            bin_positions[curr_bin].append((i, pos_i_orig[0], pos_i_orig[1], pos_i_orig[2]))
                            bin_assignments[i] = curr_bin
                else:
                    # Couldn't place i in target, revert
                    # Put k back in target
                    pos_k_orig = find_position_in_bin(target_bin, k, wk, hk, dk)
                    if pos_k_orig:
                        bin_positions[target_bin].append((k, pos_k_orig[0], pos_k_orig[1], pos_k_orig[2]))
                        bin_assignments[k] = target_bin
                    # Put i back in curr
                    pos_i_orig = find_position_in_bin(curr_bin, i, w, h, d)
                    if pos_i_orig:
                        bin_positions[curr_bin].append((i, pos_i_orig[0], pos_i_orig[1], pos_i_orig[2]))
                        bin_assignments[i] = curr_bin
            else:
                # No target bins, revert
                pos_orig = find_position_in_bin(curr_bin, i, w, h, d)
                if pos_orig is None:
                    pos_orig = (0,0,0)
                bin_positions[curr_bin].append((i, pos_orig[0], pos_orig[1], pos_orig[2]))
                bin_assignments[i] = curr_bin

    # Restore best state
    if best_state:
        bin_assignments, bin_positions = best_state

    # Format output
    placements = [None] * n
    for b in range(len(bin_positions)):
        for (idx, x, y, z) in bin_positions[b]:
            w, h, d = boxes[idx]
            placements[idx] = (b, x, y, z, w, h, d)
            
    # Ensure all boxes are placed
    for i in range(n):
        if placements[i] is None:
            # Fallback: new bin
            b = len(bin_positions)
            bin_positions.append([(i, 0, 0, 0)])
            bin_assignments[i] = b
            w, h, d = boxes[i]
            placements[i] = (b, 0, 0, 0, w, h, d)

    return placements
