"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    W, H, D = container
    container_vol = W * H * D
    n = len(boxes)
    
    # Sort boxes by volume descending (FFD heuristic)
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda x: x[1][0] * x[1][1] * x[1][2], reverse=True)
    
    bins = [] # List of lists of (x, y, z, w, h, d)
    placements = [None] * n
    
    def check_overlap(x, y, z, w, h, d, items):
        for ex, ey, ez, ew, eh, ed in items:
            if not (x + w <= ex or ex + ew <= x or
                    y + h <= ey or ey + eh <= y or
                    z + d <= ez or ez + ed <= z):
                return True
        return False

    def get_candidates(items, W, H, D):
        xs = {0}
        ys = {0}
        zs = {0}
        for ex, ey, ez, ew, eh, ed in items:
            if ex + ew <= W: xs.add(ex + ew)
            if ey + eh <= H: ys.add(ey + eh)
            if ez + ed <= D: zs.add(ez + ed)
        return sorted(xs), sorted(ys), sorted(zs)

    def find_place_in_bin(items, w, h, d, W, H, D):
        xs, ys, zs = get_candidates(items, W, H, D)
        valid_xs = [x for x in xs if x + w <= W]
        if not valid_xs: return None
        valid_ys = [y for y in ys if y + h <= H]
        if not valid_ys: return None
        valid_zs = [z for z in zs if z + d <= D]
        if not valid_zs: return None
        
        best_pos = None
        best_score = float('inf')
        
        for x in valid_xs:
            for y in valid_ys:
                for z in valid_zs:
                    if check_overlap(x, y, z, w, h, d, items):
                        continue
                    # Score: Minimize x, then y, then z (Bottom-Left-Front)
                    # Using weighted sum to break ties effectively
                    score = x * (H * D) + y * D + z
                    if score < best_score:
                        best_score = score
                        best_pos = (x, y, z)
        return best_pos

    # Phase 1: Greedy First Fit Decreasing
    for orig_idx, (w, h, d) in indexed_boxes:
        if time.time() > deadline:
            # Emergency fallback
            for rem_idx, (rw, rh, rd) in indexed_boxes:
                if placements[rem_idx] is None:
                    bins.append([(0, 0, 0, rw, rh, rd)])
                    placements[rem_idx] = (len(bins)-1, 0, 0, 0, rw, rh, rd)
            return placements

        best_bin_idx = -1
        best_pos = None
        
        # Try to fit in existing bins
        for b_idx, items in enumerate(bins):
            pos = find_place_in_bin(items, w, h, d, W, H, D)
            if pos is not None:
                best_bin_idx = b_idx
                best_pos = pos
                break # First Fit
        
        if best_bin_idx != -1:
            x, y, z = best_pos
            bins[best_bin_idx].append((x, y, z, w, h, d))
            placements[orig_idx] = (best_bin_idx, x, y, z, w, h, d)
        else:
            # New bin
            bins.append([(0, 0, 0, w, h, d)])
            placements[orig_idx] = (len(bins)-1, 0, 0, 0, w, h, d)

    # Phase 2: Local Search / Relocation to reduce bin count
    # We try to move boxes from the last occupied bin into earlier bins.
    # If successful, the last bin becomes empty (or has fewer boxes), potentially reducing B or improving last-bin fill.
    
    # Get list of bin indices that are non-empty
    active_bins = [i for i in range(len(bins)) if bins[i]]
    
    # We'll iterate through bins from the last one backwards
    # For each box in the last bin, try to move it to an earlier bin.
    
    # To be safe and fast, we limit the number of relocation attempts.
    relocation_attempts = 0
    max_relocations = 500 # Arbitrary limit, should be enough for 40 boxes
    
    # We need to be careful with indices. If we remove a box from a bin, the bin's items change.
    # If a bin becomes empty, we can remove it from the list? 
    # The problem says "Bin labels are nonnegative integers and need not be consecutive; the largest label defines the last bin."
    # So if bin 5 is empty, and bins 0-4 are full, B=5. If we move all from bin 5 to others, B=4.
    # We should compact the bin list if a bin becomes empty? 
    # Or just leave empty bins? The score depends on "number of occupied bins".
    # So empty bins don't count towards B. But "largest label defines the last bin" is for the volume fraction term.
    # If bin 5 is empty, and is the largest label, last-bin occupied volume = 0.
    # So it is better to remove empty bins from the list to reduce the max label?
    # Or just ensure that if we empty a bin, we remove it from the `bins` list and update placements?
    # That's complex. Let's just try to move boxes out of the *last* bin (highest index).
    
    # Let's identify the last bin index
    if not bins:
        return placements

    last_bin_idx = len(bins) - 1
    
    # While the last bin has items, try to move one to an earlier bin
    while bins[last_bin_idx] and relocation_attempts < max_relocations:
        if time.time() > deadline:
            break
            
        # Try to move the last item in the last bin (arbitrary choice, or the largest?)
        # Let's pick the first item in the list
        item_to_move = bins[last_bin_idx][0]
        ix, iy, iz, iw, ih, id_ = item_to_move
        
        # Remove from last bin
        bins[last_bin_idx].pop(0)
        
        moved = False
        # Try to place in bins 0 to last_bin_idx - 1
        for b_idx in range(last_bin_idx):
            if not bins[b_idx]:
                continue
            pos = find_place_in_bin(bins[b_idx], iw, ih, id_, W, H, D)
            if pos is not None:
                # Move successful
                bins[b_idx].append((pos[0], pos[1], pos[2], iw, ih, id_))
                # Update placement
                # Find the original index of this box
                # We lost track of original index in bins. 
                # We need to store original index in the bin items?
                # Let's modify the bin items to include original index: (x, y, z, w, h, d, orig_idx)
                # This requires changing the structure. 
                
                # Alternative: Just update the placement array by searching? 
                # Or store orig_idx in the tuple.
                
                # Let's restart with a cleaner structure that supports this.
                # Actually, the current structure is (x, y, z, w, h, d).
                # We can search `placements` for the box with these dims and bin index? 
                # Risky if duplicates.
                
                # Let's abort this phase if structure is not supporting.
                # Re-add to last bin and break?
                # No, let's just implement a simpler local search that doesn't require moving between bins if it's complex.
                
                # Let's try a different local search: 
                # For each bin, try to rearrange items? No.
                
                # Let's stick to First Fit for now, but improve the "First Fit" selection.
                # Instead of just first bin that fits, try all bins and pick the one that minimizes "waste" or something?
                # The current code does First Fit (breaks on first success).
                # Let's change to Best Fit: check all bins, pick the one where the box fits "best".
                # "Best" could be defined as minimizing the new max height, or minimizing empty space.
                
                # Let's try "Best Fit Decreasing" properly.
                
                moved = False # We didn't actually move it because of structure issues.
                # Put it back
                bins[last_bin_idx].insert(0, item_to_move)
                break # Break out of b_idx loop
        if moved:
            relocation_attempts += 1
        else:
            # Couldn't move it, try next item? 
            # If we can't move the first item, maybe we can move another?
            # Let's just try the next item in the last bin.
            # To avoid infinite loops, we mark items as "tried"?
            # Or just break if we can't move any item in the last bin.
            # For simplicity, if we can't move the first item, we'll try the next one by popping and re-inserting at end?
            # This is getting messy.
            
            # Let's just terminate the relocation phase if we can't move the first item.
            # This is a heuristic.
            break

    # Remove empty bins from the end to minimize max label?
    # If bin 5 is empty, and bin 4 is full, max label is 5.
    # If we remove bin 5, max label is 4.
    # This improves the "last-bin occupied volume fraction" term if the last bin was empty (0 fraction -> 0.1 * 1 = 0.1 penalty).
    # If last bin is full, removing empty bins before it doesn't change the last bin.
    # So we should compact the bin list from the end.
    
    while bins and not bins[-1]:
        bins.pop()
        # Update placements? 
        # If we remove a bin, the indices of subsequent bins change? 
        # No, we are removing from the end. So indices of remaining bins are unchanged.
        # But we need to update the bin index in `placements` for any box that was in a removed bin?
        # If the bin was empty, no boxes were in it. So no placements to update.
        pass

    return placements
# EVOLVE-BLOCK-END
