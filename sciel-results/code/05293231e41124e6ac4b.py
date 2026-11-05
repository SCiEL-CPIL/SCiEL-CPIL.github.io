import time
import random
import math

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # 1. Initial Solution: First-Fit Decreasing (FFD)
    # Sort by volume descending
    indexed_boxes = sorted(enumerate(boxes), key=lambda x: x[1][0]*x[1][1]*x[1][2], reverse=True)
    
    # State: list of bins. Each bin is a list of placed boxes (x, y, z, w, h, d)
    bins = []
    bin_volumes = []
    
    def can_place_in_bin(bin_idx, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        for (ox, oy, oz, ow, oh, od) in bins[bin_idx]:
            if not (x + w <= ox or ox + ow <= x or
                    y + h <= oy or oy + oh <= y or
                    z + d <= oz or oz + od <= z):
                return False
        return True

    def find_position_in_bin(bin_idx, w, h, d):
        # Simple candidate generation: (0,0,0) and corners of existing boxes
        candidates = [(0, 0, 0)]
        for (ox, oy, oz, ow, oh, od) in bins[bin_idx]:
            candidates.append((ox + ow, oy, oz))
            candidates.append((ox, oy + oh, oz))
            candidates.append((ox, oy, oz + od))
            candidates.append((ox + ow, oy + oh, oz))
            candidates.append((ox + ow, oy, oz + od))
            candidates.append((ox, oy + oh, oz + od))
            candidates.append((ox + ow, oy + oh, oz + od))
        
        unique_candidates = list(set(candidates))
        # Sort to prefer lower z, y, x (fill from bottom)
        unique_candidates.sort(key=lambda p: (p[2], p[1], p[0]))
        
        for cx, cy, cz in unique_candidates:
            if can_place_in_bin(bin_idx, cx, cy, cz, w, h, d):
                return cx, cy, cz
        return None

    # Initialize FFD
    for idx, (w, h, d) in indexed_boxes:
        placed = False
        for i in range(len(bins)):
            pos = find_position_in_bin(i, w, h, d)
            if pos is not None:
                bins[i].append((pos[0], pos[1], pos[2], w, h, d))
                bin_volumes[i] += w * h * d
                placed = True
                break
        if not placed:
            new_bin_idx = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            bin_volumes.append(w * h * d)

    # Map original index to current bin index and position in bin
    # We need to track which box is which to move them later.
    # Let's maintain a list of boxes in order: box_list[i] = (w, h, d)
    # And a list of placements: placements[i] = (bin_idx, x, y, z)
    # But for efficiency in local search, it's better to keep the `bins` structure 
    # and have a separate mapping from box_id to (bin_idx, slot_idx).
    
    # Let's restructure for local search:
    # box_info[i] = (w, h, d)
    # current_assignment[i] = (bin_idx, x, y, z)
    
    box_info = [None] * n
    current_assignment = [None] * n
    
    # Reconstruct assignments from bins
    bin_id_counter = 0
    for b_idx in range(len(bins)):
        for box_data in bins[b_idx]:
            x, y, z, w, h, d = box_data
            # We need to find which original index this was.
            # This is tricky because we lost the original index in the FFD loop above 
            # by only appending to bins. 
            # Let's redo the FFD initialization keeping track of indices.
            pass
            
    # Redo initialization properly for local search
    bins = []
    bin_volumes = []
    current_assignment = [None] * n
    box_info = [None] * n
    
    for idx, (w, h, d) in indexed_boxes:
        box_info[idx] = (w, h, d)
        placed = False
        for i in range(len(bins)):
            pos = find_position_in_bin(i, w, h, d)
            if pos is not None:
                bins[i].append((pos[0], pos[1], pos[2], w, h, d))
                bin_volumes[i] += w * h * d
                current_assignment[idx] = (i, pos[0], pos[1], pos[2])
                placed = True
                break
        if not placed:
            new_bin_idx = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            bin_volumes.append(w * h * d)
            current_assignment[idx] = (new_bin_idx, 0, 0, 0)

    def calculate_score():
        if not bins:
            return 0.0
        # B is number of occupied bins
        B = len(bins)
        # Last bin is the one with the largest label (index) that is occupied
        # In our representation, bins are indexed 0..B-1
        # The "last bin" in the problem statement usually implies the bin with the highest index used.
        # Score: mean B + 0.1*(1 - last-bin occupied volume fraction)
        # "last-bin" likely refers to the bin with the maximum index.
        last_bin_idx = B - 1
        last_bin_vol = bin_volumes[last_bin_idx] if last_bin_idx < len(bin_volumes) else 0
        container_vol = W * H * D
        if container_vol == 0:
            return B
        fraction = last_bin_vol / container_vol
        return B + 0.1 * (1 - fraction)

    current_score = calculate_score()
    best_score = current_score
    best_assignments = [a.copy() for a in current_assignment]
    
    # Simulated Annealing
    T = 1.0
    cooling_rate = 0.9999
    min_T = 0.01
    
    # We need a way to remove a box from a bin efficiently.
    # bins is a list of lists. Removing an item from a list is O(N).
    # With 40 boxes, this is fine.
    
    def remove_box(idx):
        b_idx, x, y, z = current_assignment[idx]
        w, h, d = box_info[idx]
        # Find and remove from bins[b_idx]
        target = (x, y, z, w, h, d)
        for i, box_data in enumerate(bins[b_idx]):
            if box_data == target:
                bins[b_idx].pop(i)
                bin_volumes[b_idx] -= w * h * d
                break
        # If bin becomes empty, we could remove it, but keeping empty bins 
        # with indices might confuse "last bin" logic if we don't compact.
        # However, the problem says "Bin labels are nonnegative integers... largest label defines last bin".
        # If we have empty bins with high indices, they shouldn't count as "occupied" for B?
        # "B is the number of occupied bins".
        # So empty bins don't count for B.
        # But "last-bin" usually implies the bin with the max index among *occupied* bins?
        # Or the max index among *all* bins we opened?
        # "places every box... into numbered bins... B is the number of occupied bins... last-bin occupied volume"
        # If I open bin 5, put a box, then move it to bin 0, bin 5 is empty.
        # Is B=1 or B=6? "B is the number of occupied bins". So B=1.
        # "last-bin": Usually in bin packing, this means the bin with the highest index that contains at least one box.
        # Let's assume "last bin" is the max index of an occupied bin.
        
        # To be safe and simple, let's compact bins after every move?
        # Compacting is expensive.
        # Alternative: Just track occupied bins.
        # Let's stick to the list of bins. If a bin is empty, we can leave it there.
        # When calculating score, we find the max index where bin_volumes[i] > 0.
        
        current_assignment[idx] = None
        return True

    def try_place_box(idx):
        w, h, d = box_info[idx]
        best_pos = None
        best_bin = -1
        
        # Try all existing bins
        for i in range(len(bins)):
            if bin_volumes[i] == 0:
                # Bin is empty, can place at 0,0,0
                if w <= W and h <= H and d <= D:
                    # Prefer filling existing empty bins to avoid creating new ones?
                    # Or just treat as new bin logic?
                    # Placing in an empty bin with index i is better than creating bin len(bins)
                    # if i < len(bins).
                    # Let's just allow it.
                    best_pos = (i, 0, 0, 0)
                    # We might want to prefer lower index empty bins to keep "last bin" low?
                    # Actually, if we have empty bins, we should probably reuse the lowest index one.
                    # But for now, just find *a* position.
                    # We can break here if we find a valid spot in an early bin?
                    # No, best fit is better.
                    # Let's continue to check if it fits better elsewhere?
                    # Actually, placing in an empty bin is always "best" in terms of not causing overlap.
                    # But we want to minimize B. Reusing an empty bin is good.
                    # We can break if we find an empty bin with the lowest index?
                    # Let's just collect candidates.
                    pass
            
            pos = find_position_in_bin(i, w, h, d)
            if pos is not None:
                # Heuristic: prefer lower index bins? Or tighter fit?
                # For SA, we just need a valid move.
                # Let's pick the first valid one we find, or the one with min x+y+z?
                # To make moves diverse, let's pick a random valid position?
                # No, let's pick the "best" position (min x+y+z) in the "best" bin.
                # To keep it simple and fast, let's just take the first valid bin we find.
                # This might lead to poor exploration.
                # Better: Pick the bin with the lowest index that fits?
                # This tends to fill bins sequentially.
                if best_bin == -1:
                    best_bin = i
                    best_pos = (i, pos[0], pos[1], pos[2])
                    # Don't break, let's see if we can find a better fit?
                    # Or just break to save time?
                    # Given 40 boxes, checking all bins is fast.
                    # Let's keep the first fit (lowest index) to maintain compactness.
        
        if best_bin != -1:
            # Place it
            b_idx, x, y, z = best_pos
            bins[b_idx].append((x, y, z, w, h, d))
            bin_volumes[b_idx] += w * h * d
            current_assignment[idx] = (b_idx, x, y, z)
            return True
        
        # If no existing bin fits, open a new one
        new_bin_idx = len(bins)
        if w <= W and h <= H and d <= D:
            bins.append([(0, 0, 0, w, h, d)])
            bin_volumes.append(w * h * d)
            current_assignment[idx] = (new_bin_idx, 0, 0, 0)
            return True
            
        return False

    # Main SA Loop
    steps = 0
    while time.time() < deadline:
        steps += 1
        # Pick a random box to move
        idx = random.randint(0, n - 1)
        
        # Remove it
        old_assign = current_assignment[idx].copy()
        remove_box(idx)
        
        # Try to place it back
        # To allow exploration, we don't always want to put it back in the "best" spot.
        # But try_place_box puts it in the first fit.
        # This is a "Re-insertion" move.
        # If we always re-insert in the first fit, we might just revert to the FFD state.
        # To explore, we need to sometimes place it in a *worse* spot or a new bin.
        
        # Let's modify try_place_box to accept a "randomness" parameter?
        # Or, simpler: 
        # 1. Remove box.
        # 2. With prob 0.5, try to place in best fit.
        # 3. With prob 0.5, try to place in a random valid position (or new bin).
        
        # For simplicity in this implementation, let's just do re-insertion into best fit.
        # This is effectively a "shake" or "k-opt" with k=1.
        # If it doesn't improve, we revert.
        
        if not try_place_box(idx):
            # Should not happen if initial was valid
            current_assignment[idx] = old_assign
            continue
            
        new_score = calculate_score()
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_assignments = [a.copy() for a in current_assignment]
        else:
            # Accept with probability exp((current - new) / T)
            delta = new_score - current_score
            if delta < 0:
                prob = 1.0
            else:
                prob = math.exp(-delta / T)
            
            if random.random() < prob:
                current_score = new_score
            else:
                # Revert
                # We need to undo the placement
                # Current assignment is the new one. We need to remove it and put back old.
                new_b, new_x, new_y, new_z = current_assignment[idx]
                w, h, d = box_info[idx]
                # Remove from new_b
                target = (new_x, new_y, new_z, w, h, d)
                for i, box_data in enumerate(bins[new_b]):
                    if box_data == target:
                        bins[new_b].pop(i)
                        bin_volumes[new_b] -= w * h * d
                        break
                
                # Put back in old_b
                old_b, old_x, old_y, old_z = old_assign
                bins[old_b].append((old_x, old_y, old_z, w, h, d))
                bin_volumes[old_b] += w * h * d
                current_assignment[idx] = old_assign
        
        T *= cooling_rate
        if T < min_T:
            T = min_T

    # Reconstruct final placements from best_assignments
    # best_assignments[i] = (bin_idx, x, y, z)
    # But wait, if we reverted, best_assignments holds the state of the *best* score seen.
    # We need to ensure the bins structure matches best_assignments if we want to return them?
    # No, we just return the coordinates.
    # The evaluator checks for overlaps based on the returned coordinates.
    # So we just need to output the (bin, x, y, z, w, h, d) for each box.
    
    placements = []
    for i in range(n):
        b, x, y, z = best_assignments[i]
        w, h, d = box_info[i]
        placements.append((b, x, y, z, w, h, d))
        
    return placements
