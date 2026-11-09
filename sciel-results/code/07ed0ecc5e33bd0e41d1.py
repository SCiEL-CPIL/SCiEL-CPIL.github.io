"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random
import numpy as np

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    W, H, D = container
    container_vol = W * H * D
    n = len(boxes)

    if n == 0:
        return []

    # Sort boxes by volume descending (FFD heuristic)
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda x: x[1][0] * x[1][1] * x[1][2], reverse=True)

    # MEP: Maximal Empty Prisms
    # Each prism is a tuple (x, y, z, w, h, d) representing the free space
    # It is defined as the space from (x, y, z) to (x+w, y+h, z+d)
    
    def insert_prism(prisms, p):
        """Insert a new empty prism, merging with existing compatible ones if possible."""
        # For simplicity and robustness in this heuristic, we don't do complex merging 
        # that requires checking all pairs immediately, but we do avoid duplicates 
        # and keep the list manageable. 
        # A strict MEP algorithm would merge if one contains another or if they align perfectly.
        # Given the time limit, we rely on the slicing to keep them maximal enough.
        prisms.append(p)

    def split_prism(prisms, index, box):
        """
        Split the prism at 'index' to accommodate 'box' placed at (bx, by, bz).
        box: (w, h, d)
        Returns the list of new prisms to replace the one at 'index'.
        """
        px, py, pz, pw, ph, pd = prisms[index]
        bw, bh, bd = box
        
        # Box must be inside the prism
        if not (px <= px + bw <= px + pw and py <= py + bh <= py + ph and pz <= pz + bd <= pz + pd):
             # This should not happen if placement logic is correct
             return []

        bx, by, bz = px, py, pz # Box is placed at the corner of the prism for simplicity in splitting?
        # Actually, the placement logic decides the position. 
        # Let's assume the box is placed at (bx, by, bz).
        # The standard MEP split assumes the box is at the corner (px, py, pz) of the prism.
        # If we allow placement anywhere in the prism, we must split into up to 6 parts.
        # However, standard "shelf" or "corner" heuristics place at (px, py, pz).
        # To keep it simple and fast, we will enforce that we only place boxes at the (x, y, z) corner of the MEP.
        # This is a valid restriction that often works well.
        
        new_prisms = []
        
        # 1. Split along X
        if px + bw < px + pw:
            # Right part
            new_prisms.append((px + bw, py, pz, pw - bw, ph, pd))
            
        # 2. Split along Y (in the remaining X-space)
        if py + bh < py + ph:
            # Back part (full width of original X, but only the Y-strip behind the box)
            # Wait, if we placed at (px, py, pz), the "behind" part in Y is (px, py+bh, pz) with size (pw, ph-bh, pd)
            # But we already carved out the box. 
            # The standard split for a box at corner (px,py,pz) in prism (px,py,pz, pw,ph,pd):
            # 
            # 1. (px+bw, py, pz, pw-bw, ph, pd)  -- Right of box
            # 2. (px, py+bh, pz, bw, ph-bh, pd)  -- Behind box (in Y), within box's X width
            # 3. (px, py, pz+bd, bw, bh, pd-bd)  -- Above box (in Z), within box's X,Y footprint
            # 4. (px+bw, py+bh, pz, pw-bw, ph-bh, pd) -- Right AND Behind
            # 5. (px+bw, py, pz+bd, pw-bw, bh, pd-bd) -- Right AND Above
            # 6. (px, py+bh, pz+bd, bw, ph-bh, pd-bd) -- Behind AND Above
            # 7. (px+bw, py+bh, pz+bd, pw-bw, ph-bh, pd-bd) -- Right, Behind, Above
            
            # Let's verify these 6 (or 7?) cover the rest.
            # The total volume is pw*ph*pd.
            # Box is bw*bh*bd.
            
            # Let's just generate all 6 surrounding boxes and filter out zero-volume ones.
            
            # 1. Right
            if pw - bw > 0:
                new_prisms.append((px + bw, py, pz, pw - bw, ph, pd))
            
            # 2. Behind (in Y) - restricted to box's X width? No, the whole prism minus box.
            # It's easier to think of slicing.
            
            # Slice 1: X dimension. Split at px+bw.
            # Left part (px, py, pz, bw, ph, pd) -> contains box.
            # Right part (px+bw, py, pz, pw-bw, ph, pd) -> Free.
            
            # Now process Left part (px, py, pz, bw, ph, pd).
            # Slice 2: Y dimension. Split at py+bh.
            # Front part (px, py, pz, bw, bh, pd) -> contains box.
            # Back part (px, py+bh, pz, bw, ph-bh, pd) -> Free.
            
            # Now process Front part (px, py, pz, bw, bh, pd).
            # Slice 3: Z dimension. Split at pz+bd.
            # Bottom part (px, py, pz, bw, bh, bd) -> Box.
            # Top part (px, py, pz+bd, bw, bh, pd-bd) -> Free.
            
            # So the free parts are:
            # A. Right of everything: (px+bw, py, pz, pw-bw, ph, pd)
            # B. Behind box (in Y), within box's X: (px, py+bh, pz, bw, ph-bh, pd)
            # C. Above box (in Z), within box's X and Y: (px, py, pz+bd, bw, bh, pd-bd)
            
            # Is that all?
            # Let's check volumes.
            # A: (pw-bw)*ph*pd
            # B: bw*(ph-bh)*pd
            # C: bw*bh*(pd-bd)
            
            # Missing volume?
            # Total = pw*ph*pd
            # Box = bw*bh*bd
            # Sum A+B+C = (pw-bw)ph pd + bw(ph-bh)pd + bw bh (pd-bd)
            # = pw ph pd - bw ph pd + bw ph pd - bw bh pd + bw bh pd - bw bh bd
            # = pw ph pd - bw bh bd
            # Yes! That's exactly the remaining volume.
            
            # So we only need 3 new prisms, not 6, if we assume the box is at the corner.
            
            if ph - bh > 0:
                new_prisms.append((px, py + bh, pz, bw, ph - bh, pd))
            
            if pd - bd > 0:
                new_prisms.append((px, py, pz + bd, bw, bh, pd - bd))
        
        return new_prisms

    def place_box_in_bin(bin_data, box):
        """
        Try to place box in a bin.
        bin_data: list of (prism, orig_idx_of_box_that_created_it? No, just prisms)
        Actually, we need to track which prism to modify.
        We will return the new list of prisms and the placement (x,y,z) if successful.
        """
        w, h, d = box
        best_idx = -1
        best_waste = float('inf')
        
        # Iterate through prisms
        # We want to find the "best fit" prism.
        # Best fit usually means the prism that leaves the least wasted space, or the smallest prism that fits.
        # Let's try: Minimize the volume of the remaining space in the prism after placement.
        # Or simply: Find the first prism that fits (First Fit) or the tightest (Best Fit).
        
        for i, prism in enumerate(bin_data):
            px, py, pz, pw, ph, pd = prism
            if pw >= w and ph >= h and pd >= d:
                # It fits.
                # Calculate waste
                # Waste = prism_volume - box_volume
                # But we split the prism. The "waste" is the sum of the new prisms.
                # This is constant for a given prism and box size.
                # So we just pick the prism with the smallest volume? Or the one that minimizes the max dimension of remaining?
                # Let's stick to: Smallest Prism Volume that fits.
                vol = pw * ph * pd
                if vol < best_waste:
                    best_waste = vol
                    best_idx = i
        
        if best_idx == -1:
            return None, None
            
        # Place the box at the corner of the selected prism
        px, py, pz, pw, ph, pd = bin_data[best_idx]
        placement = (px, py, pz)
        
        # Update the bin's prism list
        new_prisms = list(bin_data)
        new_prisms.pop(best_idx)
        
        # Split
        # We need to pass the box dimensions and the prism.
        # The split function returns the new free prisms.
        new_free_prisms = split_prism([bin_data[best_idx]], 0, (w, h, d))
        
        # Add new prisms to the list
        new_prisms.extend(new_free_prisms)
        
        return placement, new_prisms

    # Data structures
    # bins: List of lists of prisms
    # placements: List of (bin_idx, x, y, z, w, h, d)
    
    bins = []
    placements = [None] * n

    # Phase 1: Greedy Best Fit Decreasing using MEPs
    for orig_idx, (w, h, d) in indexed_boxes:
        if time.time() > deadline:
            # Emergency fallback
            for rem_idx, (rw, rh, rd) in indexed_boxes:
                if placements[rem_idx] is None:
                    bins.append([(0, 0, 0, rw, rh, rd)])
                    placements[rem_idx] = (len(bins) - 1, 0, 0, 0, rw, rh, rd)
            return placements

        placed = False
        # Try existing bins
        # We want to pick the bin that minimizes the volume of the prism we use?
        # Or just the first bin where it fits?
        # Let's try to find the best fit across all bins.
        best_bin_idx = -1
        best_prism_vol = float('inf')
        best_placement = None
        
        for b_idx, bin_prisms in enumerate(bins):
            if not bin_prisms:
                continue
            # Check if it fits in any prism in this bin
            # We need to know which prism gives the best fit.
            # For simplicity, let's just check if it fits in the *smallest* prism that fits.
            # This requires scanning the bin's prisms.
            
            min_prism_vol_in_bin = float('inf')
            min_prism_placement = None
            
            for prism in bin_prisms:
                px, py, pz, pw, ph, pd = prism
                if pw >= w and ph >= h and pd >= d:
                    vol = pw * ph * pd
                    if vol < min_prism_vol_in_bin:
                        min_prism_vol_in_bin = vol
                        min_prism_placement = (px, py, pz)
            
            if min_prism_placement is not None:
                if min_prism_vol_in_bin < best_prism_vol:
                    best_prism_vol = min_prism_vol_in_bin
                    best_bin_idx = b_idx
                    best_placement = min_prism_placement

        if best_bin_idx != -1:
            # Place in best_bin_idx
            x, y, z = best_placement
            bins[best_bin_idx] = place_box_in_bin_update(bins[best_bin_idx], best_bin_idx, x, y, z, w, h, d)
            placements[orig_idx] = (best_bin_idx, x, y, z, w, h, d)
        else:
            # New bin
            bins.append([(0, 0, 0, W, H, D)])
            # Place at 0,0,0
            x, y, z = 0, 0, 0
            # Update the single prism in the new bin
            bins[-1] = place_box_in_bin_update(bins[-1], -1, x, y, z, w, h, d)
            placements[orig_idx] = (len(bins) - 1, x, y, z, w, h, d)

    # Helper to update bin prisms after placing a box at (x,y,z) in a specific bin.
    # We need to find the prism that contains (x,y,z) and was the one we selected.
    # Actually, the logic above found the prism. We should pass the prism index or the prism itself.
    # Let's refactor slightly to be cleaner.

    # Re-implementing Phase 1 with cleaner logic
    bins = []
    placements = [None] * n

    for orig_idx, (w, h, d) in indexed_boxes:
        if time.time() > deadline:
            # Emergency fallback
            for rem_idx, (rw, rh, rd) in indexed_boxes:
                if placements[rem_idx] is None:
                    bins.append([(0, 0, 0, rw, rh, rd)])
                    placements[rem_idx] = (len(bins) - 1, 0, 0, 0, rw, rh, rd)
            return placements

        best_bin_idx = -1
        best_prism_idx = -1
        best_prism_vol = float('inf')
        best_placement = None

        for b_idx, bin_prisms in enumerate(bins):
            for p_idx, prism in enumerate(bin_prisms):
                px, py, pz, pw, ph, pd = prism
                if pw >= w and ph >= h and pd >= d:
                    vol = pw * ph * pd
                    if vol < best_prism_vol:
                        best_prism_vol = vol
                        best_bin_idx = b_idx
                        best_prism_idx = p_idx
                        best_placement = (px, py, pz)

        if best_bin_idx != -1:
            # Remove the used prism
            used_prism = bins[best_bin_idx][best_prism_idx]
            bins[best_bin_idx].pop(best_prism_idx)
            
            # Generate new prisms from splitting
            # Box is at (px, py, pz) of the prism
            px, py, pz, pw, ph, pd = used_prism
            new_prisms = []
            if pw - w > 0:
                new_prisms.append((px + w, py, pz, pw - w, ph, pd))
            if ph - h > 0:
                new_prisms.append((px, py + h, pz, w, ph - h, pd))
            if pd - d > 0:
                new_prisms.append((px, py, pz + d, w, h, pd - d))
            
            bins[best_bin_idx].extend(new_prisms)
            
            x, y, z = best_placement
            placements[orig_idx] = (best_bin_idx, x, y, z, w, h, d)
        else:
            # New bin
            bins.append([(0, 0, 0, W, H, D)])
            x, y, z = 0, 0, 0
            # Split the initial prism
            used_prism = bins[-1][0]
            bins[-1].pop(0)
            px, py, pz, pw, ph, pd = used_prism
            new_prisms = []
            if pw - w > 0:
                new_prisms.append((px + w, py, pz, pw - w, ph, pd))
            if ph - h > 0:
                new_prisms.append((px, py + h, pz, w, ph - h, pd))
            if pd - d > 0:
                new_prisms.append((px, py, pz + d, w, h, pd - d))
            bins[-1].extend(new_prisms)
            
            placements[orig_idx] = (len(bins) - 1, x, y, z, w, h, d)

    # Phase 2: Local Search - Relocation
    # We try to move boxes from the last occupied bin to earlier bins.
    # This requires removing a box from a bin's MEP structure.
    # Removal is complex because we need to merge prisms.
    # Given the complexity and time, we might skip complex removal and just do a simple "re-packing"
    # of the last few bins if time permits.
    
    # Alternative: If we have time, try to re-pack the last bin's items into earlier bins.
    # To do this, we need to "remove" items from the last bin.
    # We can rebuild the last bin's MEPs from scratch? No, that's O(N^2).
    
    # Let's try a simpler improvement:
    # Check if any box in the last bin can fit in an earlier bin.
    # To do this, we need to temporarily remove it from the last bin's MEPs.
    # Since we don't have an efficient remove+merge, we can simulate it by:
    # 1. Reconstructing the last bin's free space from its current items? No, we lost the history.
    # 2. We can just check if the box fits in the *current* free space of earlier bins.
    # 3. If it does, we move it. But we must update the MEPs of the destination bin (easy) 
    #    AND the source bin (hard).
    
    # If we skip updating the source bin's MEPs, we will have "ghost" free space in the source bin,
    # which might allow invalid placements later.
    # However, if we are only moving items OUT of the last bin, and we stop using the last bin for new items,
    # we don't need the source bin's MEPs to be accurate for future placements *into* it, 
    # UNLESS we try to move items back or from other bins into it.
    # Since we are iterating from last to first, and only moving out of the current bin,
    # we won't place new items into the last bin.
    # So we can "ignore" the accuracy of the source bin's MEPs after removal, as long as we don't place into it.
    
    # Let's implement this:
    # 1. Identify the last occupied bin index.
    # 2. List items in that bin.
    # 3. For each item, check if it fits in any earlier bin's current MEPs.
    # 4. If yes, place it in the earlier bin (update MEPs).
    # 5. Mark the item as moved.
    # 6. We don't update the source bin's MEPs, but we don't place into it anymore.
    # 7. If the source bin becomes empty, we can remove it from the list (adjusting indices? No, just leave it empty).
    # 8. Repeat for the new last bin.

    # To know which items are in which bin, we need to track that.
    # We have `placements`.
    # We can build a list of (bin_idx, list_of_orig_idxs).
    
    bin_contents = defaultdict(list)
    for orig_idx, (bin_idx, x, y, z, w, h, d) in enumerate(placements):
        if bin_idx is not None:
            bin_contents[bin_idx].append(orig_idx)

    # Iterate from last bin to first
    max_bin_idx = max(bin_contents.keys()) if bin_contents else -1
    
    for current_max_bin in range(max_bin_idx, 0, -1):
        if time.time() > deadline:
            break
        
        if current_max_bin not in bin_contents or not bin_contents[current_max_bin]:
            continue
            
        items_to_try = list(bin_contents[current_max_bin])
        
        for orig_idx in items_to_try:
            if time.time() > deadline:
                break
                
            # Check if this item is still in this bin
            if orig_idx not in bin_contents[current_max_bin]:
                continue
                
            w, h, d = boxes[orig_idx]
            
            # Try to place in earlier bins
            best_target_bin = -1
            best_target_prism_idx = -1
            best_target_vol = float('inf')
            best_target_pos = None
            
            for t_idx in range(current_max_bin):
                if t_idx not in bin_contents or not bin_contents[t_idx]:
                    continue
                # Check MEPs of bin t_idx
                for p_idx, prism in enumerate(bins[t_idx]):
                    px, py, pz, pw, ph, pd = prism
                    if pw >= w and ph >= h and pd >= d:
                        vol = pw * ph * pd
                        if vol < best_target_vol:
                            best_target_vol = vol
                            best_target_bin = t_idx
                            best_target_prism_idx = p_idx
                            best_target_pos = (px, py, pz)
            
            if best_target_bin != -1:
                # Move the item
                x, y, z = best_target_pos
                # Update destination bin MEPs
                used_prism = bins[best_target_bin][best_target_prism_idx]
                bins[best_target_bin].pop(best_target_prism_idx)
                px, py, pz, pw, ph, pd = used_prism
                new_prisms = []
                if pw - w > 0:
                    new_prisms.append((px + w, py, pz, pw - w, ph, pd))
                if ph - h > 0:
                    new_prisms.append((px, py + h, pz, w, ph - h, pd))
                if pd - d > 0:
                    new_prisms.append((px, py, pz + d, w, h, pd - d))
                bins[best_target_bin].extend(new_prisms)
                
                # Update placements
                placements[orig_idx] = (best_target_bin, x, y, z, w, h, d)
                
                # Update bin_contents
                bin_contents[current_max_bin].remove(orig_idx)
                bin_contents[best_target_bin].append(orig_idx)
                
                # Note: We do NOT update bins[current_max_bin] MEPs.
                # This is safe as long as we don't place into current_max_bin again.
                # Since we are iterating downwards, we won't.

    # Final cleanup: Remove empty bins from the end
    while bins and not bins[-1]:
        bins.pop()
        
    # Also, if the last bin is empty in terms of content but has MEPs?
    # We should check bin_contents.
    # The loop above removes items. If a bin becomes empty, it stays in `bins` but has no items in `bin_contents`.
    # We should remove such bins from the end if they are the highest index.
    
    # Re-compact bin indices?
    # The score depends on the "last bin" label.
    # If bin 5 is empty, and bin 4 is full, the max label is 5.
    # We should remove empty bins from the end of the `bins` list and update `placements`?
    # No, `placements` stores the bin index. If we pop bin 5, bin 4 becomes the last.
    # But if we just pop from the `bins` list, the indices of existing bins don't change.
    # So we just need to pop empty bins from the end of `bins`.
    
    # However, `bins` might have empty lists in the middle?
    # If a bin becomes empty, we don't remove it from the list immediately, just from `bin_contents`.
    # So `bins[i]` might be an empty list `[]`.
    # We should pop from the end while `bins[-1]` is empty.
    
    while bins and not bins[-1]:
        bins.pop()
        
    # What if the last bin (highest index) is not empty in `bins` but empty in `bin_contents`?
    # This can happen if we moved all items out but didn't clear the MEPs.
    # We should check `bin_contents` for the max index.
    
    # Let's rebuild the final bin list to be compact and correct.
    # We only care about occupied bins.
    # Let's create a new list of bins, only including those with items.
    # And re-map the indices.
    
    occupied_bin_indices = [b for b in bin_contents if bin_contents[b]]
    occupied_bin_indices.sort()
    
    if not occupied_bin_indices:
        return placements # All empty? Should not happen if n>0.

    # Create new bins list
    new_bins = []
    bin_index_map = {}
    for old_idx in occupied_bin_indices:
        new_idx = len(new_bins)
        bin_index_map[old_idx] = new_idx
        new_bins.append(bins[old_idx])
        
    # Update placements
    for orig_idx, (bin_idx, x, y, z, w, h, d) in enumerate(placements):
        if bin_idx in bin_index_map:
            new_bin_idx = bin_index_map[bin_idx]
            placements[orig_idx] = (new_bin_idx, x, y, z, w, h, d)
        else:
            # Should not happen
            pass

    return placements

# EVOLVE-BLOCK-END
