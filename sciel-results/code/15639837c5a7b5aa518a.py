import time
import random
import math

# Global deadline guard
START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin for 150s limit

def _now():
    return time.time()

def _fits(placements, cx, cy, cz, w, h, d, W, H, D):
    """Check if a box fits at (cx, cy, cz) without overlapping existing placements."""
    if cx < 0 or cy < 0 or cz < 0:
        return False
    if cx + w > W or cy + h > H or cz + d > D:
        return False
    for (px, py, pz, pw, ph, pd) in placements:
        if (cx < px + pw and px < cx + w and
                cy < py + ph and py < cy + h and
                cz < pz + pd and pz < cz + d):
            return False
    return True

def _get_anchors(placements, W, H, D):
    """
    Generate candidate positions (anchors) for placing a new box.
    We use a simplified set of anchors to keep SA fast:
    1. Origin
    2. Faces of existing boxes (x+, y+, z+)
    3. Corners (combinations of faces)
    """
    anchors = [(0, 0, 0)]
    for (px, py, pz, pw, ph, pd) in placements:
        # Single face extensions
        if px + pw < W: anchors.append((px + pw, py, pz))
        if py + ph < H: anchors.append((px, py + ph, pz))
        if pz + pd < D: anchors.append((px, py, pz + pd))
        
        # Corner combinations (essential for tight packing)
        if px + pw < W and py + ph < H: anchors.append((px + pw, py + ph, pz))
        if px + pw < W and pz + pd < D: anchors.append((px + pw, py, pz + pd))
        if py + ph < H and pz + pd < D: anchors.append((px, py + ph, pz + pd))
        if px + pw < W and py + ph < H and pz + pd < D: 
            anchors.append((px + pw, py + ph, pz + pd))
            
    return anchors

def _find_spot(placements, w, h, d, W, H, D, rng):
    """
    Try to find a valid position for a box in a bin.
    Returns (x, y, z) or None.
    Uses a randomized selection from valid anchors to allow SA to explore.
    """
    if not placements:
        return (0, 0, 0)
        
    anchors = _get_anchors(placements, W, H, D)
    # Shuffle anchors to introduce randomness in placement choice
    rng.shuffle(anchors)
    
    valid_spots = []
    for (cx, cy, cz) in anchors:
        if _fits(placements, cx, cy, cz, w, h, d, W, H, D):
            valid_spots.append((cx, cy, cz))
            
    if not valid_spots:
        return None
        
    # Pick a random valid spot, or the one with lowest Z (heuristic bias)
    # To make SA effective, we want some randomness. 
    # Let's pick randomly among valid spots, but maybe bias towards lower Z slightly?
    # Actually, pure random among valid spots is good for SA diversity.
    return rng.choice(valid_spots)

def _calculate_score(bins, container):
    """
    Calculate the official score: B + 0.1 * (1 - last_bin_fill).
    bins: list of lists of (idx, x, y, z, w, h, d)
    """
    W, H, D = container
    C = W * H * D
    B = len(bins)
    
    if B == 0:
        return 0.0
        
    # Find the bin with the largest label (index)
    # In our representation, bins are 0-indexed lists.
    # The "last bin" is the one with the highest index that has boxes.
    last_bin = bins[-1]
    
    if not last_bin:
        return B
        
    last_vol = sum(w * h * d for (_, _, _, _, w, h, d) in last_bin)
    fill_ratio = last_vol / C if C > 0 else 0.0
    
    return B + 0.1 * (1.0 - fill_ratio)

def _initial_solution(boxes, container, rng):
    """
    Generate an initial solution using First-Fit Decreasing Volume (FFDV).
    This provides a reasonable starting point for SA.
    """
    W, H, D = container
    n = len(boxes)
    
    # Sort indices by volume descending
    idxs = list(range(n))
    idxs.sort(key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True)
    
    bins = []
    for idx in idxs:
        w, h, d = boxes[idx]
        placed = False
        for bi in range(len(bins)):
            placements = [(x, y, z, w_, h_, d_) for (_, x, y, z, w_, h_, d_) in bins[bi]]
            spot = _find_spot(placements, w, h, d, W, H, D, rng)
            if spot:
                bins[bi].append((idx, spot[0], spot[1], spot[2], w, h, d))
                placed = True
                break
        if not placed:
            # New bin
            bins.append([(idx, 0, 0, 0, w, h, d)])
            
    return bins

def _pack_sa(boxes, container, rng):
    """
    Simulated Annealing to improve the packing.
    """
    W, H, D = container
    n = len(boxes)
    
    # 1. Generate initial solution
    current_bins = _initial_solution(boxes, container, rng)
    current_score = _calculate_score(current_bins, container)
    
    best_bins = [b[:] for b in current_bins]
    best_score = current_score
    
    # SA Parameters
    T = 1.0  # Initial temperature (score difference scale)
    T_min = 0.01
    cooling_rate = 0.995  # Slow cooling to explore more
    max_iter = 50000  # Hard cap on iterations, but time is the real limit
    
    # Precompute box dimensions for quick access
    box_dims = boxes
    
    for i in range(max_iter):
        if _now() > START_TIME + TIME_LIMIT:
            break
            
        # 2. Generate a neighbor solution
        # Strategy: Pick a random box and try to move it to a different bin or position.
        # To keep it simple and fast, we will:
        # a) Pick a random box from a random bin.
        # b) Remove it.
        # c) Try to place it in a random other bin (or same bin at new pos).
        # d) If it fits, calculate new score.
        
        if not current_bins:
            break
            
        total_boxes = sum(len(b) for b in current_bins)
        if total_boxes == 0:
            break
            
        # Pick random box
        r = rng.randint(0, total_boxes - 1)
        cur_bin_idx = 0
        while r >= len(current_bins[cur_bin_idx]):
            r -= len(current_bins[cur_bin_idx])
            cur_bin_idx += 1
            
        box_idx_in_bin = r
        box_data = current_bins[cur_bin_idx][box_idx_in_bin]
        idx, ox, oy, oz, w, h, d = box_data
        
        # Remove box from current bin
        current_bins[cur_bin_idx].pop(box_idx_in_bin)
        
        # Try to place it somewhere else
        # Candidate target bins: all bins except the one we just removed from? 
        # Or include the same bin (to change position)?
        # Let's try all bins, including the original one (to allow re-positioning).
        target_bins = list(range(len(current_bins)))
        rng.shuffle(target_bins)
        
        placed = False
        new_pos = None
        new_bin_idx = -1
        
        for tb in target_bins:
            # Get geometry of target bin
            placements = [(x, y, z, w_, h_, d_) for (_, x, y, z, w_, h_, d_) in current_bins[tb]]
            
            # Try to find a spot
            # To speed up, we can limit the number of anchors tried or use a quick check
            spot = _find_spot(placements, w, h, d, W, H, D, rng)
            
            if spot:
                new_pos = spot
                new_bin_idx = tb
                placed = True
                break
                
        if not placed:
            # Restore original position
            current_bins[cur_bin_idx].append(box_data)
            continue
            
        # Apply move
        current_bins[new_bin_idx].append((idx, new_pos[0], new_pos[1], new_pos[2], w, h, d))
        
        # Calculate new score
        new_score = _calculate_score(current_bins, container)
        
        # Acceptance criteria
        delta = new_score - current_score
        if delta < 0 or rng.random() < math.exp(-delta / T):
            # Accept
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_bins = [b[:] for b in current_bins]
        else:
            # Reject: revert
            # Remove from new bin
            # Find the item we just added
            # Since we just appended, it's the last one? 
            # Wait, if we moved to the same bin, we removed and then appended.
            # If we moved to a different bin, we removed from A and appended to B.
            # We need to remove from B.
            # The item is (idx, new_pos[0], new_pos[1], new_pos[2], w, h, d)
            # It might not be the last one if other items were added? No, we only added one.
            # But if we moved within the same bin, it's the last one.
            # If we moved to a different bin, it's the last one in that bin.
            
            # To be safe, find and remove the specific item we added.
            # Since indices are unique, we can find it by index.
            # But there might be multiple boxes with same index? No, idx is unique.
            
            # Remove from new_bin_idx
            # Find the item with this idx in current_bins[new_bin_idx]
            # Note: If new_bin_idx == cur_bin_idx, we just put it back.
            
            # Efficient removal:
            # We know we appended it. So it's at the end of current_bins[new_bin_idx].
            # Let's verify.
            # If we didn't place it, we restored it.
            # If we placed it and rejected, we need to undo the placement.
            # The placement was: current_bins[new_bin_idx].append(...)
            # So popping the last item should work IF no other changes happened.
            # Since this is a single-step move, popping the last item of the target bin
            # is correct ONLY if we didn't modify other bins. We didn't.
            # However, if new_bin_idx == cur_bin_idx, we popped from cur_bin_idx and appended to cur_bin_idx.
            # So popping from cur_bin_idx (which is new_bin_idx) restores the state.
            
            current_bins[new_bin_idx].pop()
            
            # Wait, if we rejected, we need to put the box back in cur_bin_idx at its original position?
            # No, the state before the move was: Box in cur_bin_idx at box_idx_in_bin.
            # We popped it.
            # If we reject, we must restore it to cur_bin_idx at box_idx_in_bin.
            # But we just popped from new_bin_idx.
            # If new_bin_idx != cur_bin_idx, the box is currently in new_bin_idx (last item).
            # We pop it from new_bin_idx. Now it's nowhere.
            # We need to insert it back into cur_bin_idx at box_idx_in_bin.
            
            # Let's refine the rejection logic.
            
            # Case 1: new_bin_idx == cur_bin_idx
            # We popped from cur_bin_idx, then appended to cur_bin_idx.
            # Reject: Pop from cur_bin_idx (removes the new pos).
            # But we need it at the original position? 
            # The list order doesn't matter for geometry, only for the mapping of which box is where.
            # However, the "box_idx_in_bin" was its position in the list.
            # If we just append, it goes to the end.
            # Does the position in the list matter?
            # No, the bins are just sets of boxes. The order in the list is irrelevant for validity.
            # So, if we reject, we just need the box to be back in cur_bin_idx.
            # If new_bin_idx == cur_bin_idx:
            #   We popped it, then appended it.
            #   Reject: Pop it again. It's back in cur_bin_idx. Correct.
            # If new_bin_idx != cur_bin_idx:
            #   We popped it from cur_bin_idx, then appended it to new_bin_idx.
            #   Reject: Pop it from new_bin_idx. Now it's nowhere.
            #   We need to add it back to cur_bin_idx.
            
            if new_bin_idx != cur_bin_idx:
                # We popped from new_bin_idx above.
                # Now add it back to cur_bin_idx.
                current_bins[cur_bin_idx].append(box_data)
            # If new_bin_idx == cur_bin_idx, we already popped it from the list, 
            # which effectively removed the "new" placement. 
            # But wait. 
            # Start: [A, B, C] (Box is B at idx 1)
            # Pop B: [A, C]
            # Append B (new pos): [A, C, B]
            # Reject: Pop last: [A, C]
            # But B is gone!
            # So for same bin, we need to re-insert B?
            # Actually, if we reject, the state should be identical to before.
            # Before: B was at idx 1.
            # After reject: B is not in the list.
            # So we MUST re-insert B.
            
            # Correction:
            # The "Pop" logic above assumes we are undoing the "Append".
            # If we reject, we undo the "Append".
            # Then we must restore the "Pop".
            
            # So:
            # 1. Pop from new_bin_idx (undo append)
            # 2. Insert back into cur_bin_idx (undo pop)
            
            # We already did step 1 (implicitly in the code structure if we just pop).
            # Let's rewrite the rejection block clearly.
            
            # Actually, the code above `current_bins[new_bin_idx].pop()` is the undo of append.
            # Now we must undo the pop from cur_bin_idx.
            # We can just append to cur_bin_idx.
            
            if new_bin_idx != cur_bin_idx:
                current_bins[cur_bin_idx].append(box_data)
            else:
                # If same bin, we popped from cur_bin_idx, then appended to cur_bin_idx.
                # We just popped from cur_bin_idx (undo append).
                # We still need to undo the initial pop.
                # So we append to cur_bin_idx.
                current_bins[cur_bin_idx].append(box_data)
                
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    return best_bins

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
        
    rng = random.Random(42)
    
    # Run SA
    best_bins = _pack_sa(boxes, container, rng)
    
    # Construct final placements list
    # Map index -> (bin, x, y, z, w, h, d)
    final_placement_map = {}
    for bi, bin_content in enumerate(best_bins):
        for item in bin_content:
            idx, x, y, z, w, h, d = item
            final_placement_map[idx] = (bi, x, y, z, w, h, d)
            
    # Build result in original order
    result = []
    for i in range(n):
        if i in final_placement_map:
            result.append(final_placement_map[i])
        else:
            # Fallback (should not happen)
            result.append((999, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]))
            
    return result
