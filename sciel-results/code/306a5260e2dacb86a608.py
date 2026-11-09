import time
import math
import random
import sys

# EVOLVE-BLOCK-START
def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    W, H, D = container
    n = len(boxes)
    
    if n == 0:
        return []
    
    # MEP: (x, y, z, w, h, d)
    # Represents a free rectangular prism
    
    def subtract_mep(mep, box):
        """
        Given a free space mep (x, y, z, w, h, d) and a box placed at 
        (bx, by, bz, bw, bh, bd) inside it, return a list of new MEPs 
        that partition the remaining free space.
        """
        x, y, z, w, h, d = mep
        bx, by, bz, bw, bh, bd = box
        
        new_meps = []
        
        # 1. Split along X
        if bx > x:
            new_meps.append((x, y, z, bx - x, h, d))
        if bx + bw < x + w:
            new_meps.append((bx + bw, y, z, x + w - (bx + bw), h, d))
        
        # 2. Split along Y (only in the X-span of the box to avoid duplication)
        if by > y:
            new_meps.append((bx, y, z, bw, by - y, d))
        if by + bh < y + h:
            new_meps.append((bx, by + bh, z, bw, y + h - (by + bh), d))
            
        # 3. Split along Z (only in the X-Y span of the box)
        if bz > z:
            new_meps.append((bx, by, z, bw, bh, bz - z))
        if bz + bd < z + d:
            new_meps.append((bx, by, bz + bd, bw, bh, z + d - (bz + bd)))
            
        return new_meps

    def place_box_in_bin(bin_data, box_dims):
        """
        Tries to place box_dims (w, h, d) into bin_data.
        bin_data is a dict with 'meps' (list) and 'boxes' (list).
        Returns (success, new_meps, placed_coords) or (False, None, None).
        """
        w, h, d = box_dims
        meps = bin_data['meps']
        
        # Best Fit: Find the MEP that minimizes the remaining volume after placement
        best_idx = -1
        best_remaining_vol = float('inf')
        best_pos = None
        
        # Optimization: Sort meps by volume descending? Or just iterate.
        # Iterating all is O(N_meps). N_meps grows with boxes.
        # For 40 boxes, N_meps is small (< 100 usually).
        
        for i, mep in enumerate(meps):
            mx, my, mz, mw, mh, md = mep
            
            # Check if box fits in this MEP
            if w <= mw and h <= mh and d <= md:
                # Try all 8 corners of the MEP? 
                # Or just bottom-left-front? 
                # To be safe and find valid spots, checking corners is good, 
                # but for speed, let's try (mx, my, mz) first, then others if needed.
                # Actually, if it fits in the bounding box of the MEP, 
                # placing it at (mx, my, mz) is always valid relative to other boxes 
                # IF the MEP is truly maximal and non-overlapping.
                # Wait, MEPs are non-overlapping. So if box fits in MEP, 
                # placing it at (mx, my, mz) is valid.
                
                # However, we might want to place it in a "tighter" spot within the MEP?
                # No, the MEP is empty. Placing at (mx,my,mz) is the canonical "best" 
                # for keeping the remaining space contiguous if possible, 
                # but splitting creates new MEPs.
                # The "Best Fit" heuristic usually compares the volume of the *remaining* space 
                # in that specific MEP after the box is placed.
                
                # Volume of MEP
                vol_mep = mw * mh * md
                vol_box = w * h * d
                remaining_vol = vol_mep - vol_box
                
                # Tie-breaker: Prefer smaller MEPs? Or prefer positions that create fewer/ smaller new MEPs?
                # Standard Best Fit: Minimize remaining volume in the *container* or *bin*?
                # Here, we minimize the volume of the *specific MEP* we are using, 
                # which tends to pack tightly into small gaps.
                
                if remaining_vol < best_remaining_vol:
                    best_remaining_vol = remaining_vol
                    best_idx = i
                    best_pos = (mx, my, mz)
                    
        if best_idx == -1:
            return False, None, None
        
        # Perform placement
        mep = meps[best_idx]
        box_pos = (best_pos[0], best_pos[1], best_pos[2], w, h, d)
        
        # Remove old MEP
        new_meps = meps[:best_idx] + meps[best_idx+1:]
        
        # Add new sub-MEPs
        new_meps.extend(subtract_mep(mep, box_pos))
        
        # Optional: Merge adjacent MEPs? 
        # Merging is expensive (O(N^2)) and complex to check adjacency in 3D.
        # For 40 boxes, the fragmentation is manageable without merging.
        
        return True, new_meps, (best_pos[0], best_pos[1], best_pos[2])

    def construct_initial():
        """
        Multi-start BFD construction.
        Returns a list of (bins, placements) candidates.
        """
        candidates = []
        
        # Sort boxes by volume descending
        indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
        indexed_boxes.sort(key=lambda x: x[1]*x[2]*x[3], reverse=True)
        
        # Run a few random permutations or just one?
        # Let's do 5 random restarts with slight noise in sorting?
        # Or just one strong BFD and then SA.
        
        for restart in range(5):
            # Shuffle slightly for diversity in restarts > 0?
            # Or just run BFD once and rely on SA?
            # Let's run BFD once.
            
            bins = [] # List of dicts: {'meps': [], 'boxes': []}
            placements = [None] * n
            
            for i, w, h, d in indexed_boxes:
                placed = False
                # Try existing bins
                for b_idx in range(len(bins)):
                    success, new_meps, pos = place_box_in_bin(bins[b_idx], (w, h, d))
                    if success:
                        # Update bin state
                        bins[b_idx]['meps'] = new_meps
                        bins[b_idx]['boxes'].append((pos[0], pos[1], pos[2], w, h, d, i))
                        placements[i] = (b_idx, pos[0], pos[1], pos[2], w, h, d)
                        placed = True
                        break
                
                if not placed:
                    # Open new bin
                    bin_data = {
                        'meps': [(0, 0, 0, W, H, D)],
                        'boxes': []
                    }
                    success, new_meps, pos = place_box_in_bin(bin_data, (w, h, d))
                    if success:
                        bin_data['meps'] = new_meps
                        bin_data['boxes'].append((pos[0], pos[1], pos[2], w, h, d, i))
                        bins.append(bin_data)
                        placements[i] = (len(bins) - 1, pos[0], pos[1], pos[2], w, h, d)
                    else:
                        # Box doesn't fit in a new empty bin? Should not happen if box <= container
                        # But if box > container, it's invalid. 
                        # Assuming valid inputs.
                        pass
            
            # Convert to standard format for SA if needed, or keep MEP structure?
            # SA needs to move boxes. Moving a box out of a bin requires re-inserting its space into MEPs.
            # This is the hard part. 
            # Removing a box from a bin with MEPs is NOT trivial because 
            # we don't know which MEPs were formed by its subtraction.
            
            # Alternative Strategy for SA:
            # Keep the "boxes" list. 
            # When moving a box, remove it from 'boxes' and REBUILD the MEPs for that bin?
            # Rebuilding MEPs from scratch:
            # 1. Start with full container.
            # 2. Subtract all remaining boxes in that bin.
            # This is O(K^2) where K is boxes in bin. K is small (~10). 
            # So rebuilding is feasible!
            
            candidates.append((bins, placements))
            
        return candidates

    def rebuild_meps(bin_data):
        """Rebuilds the MEP list for a bin from its 'boxes' list."""
        meps = [(0, 0, 0, W, H, D)]
        for box in bin_data['boxes']:
            # box is (x, y, z, w, h, d, idx)
            bx, by, bz, bw, bh, bd, _ = box
            new_meps = []
            for mep in meps:
                # Check if box is inside mep
                mx, my, mz, mw, mh, md = mep
                if bx >= mx and by >= my and bz >= mz and \
                   bx + bw <= mx + mw and by + bh <= my + mh and bz + bd <= mz + md:
                    # Box is inside this MEP, subtract it
                    new_meps.extend(subtract_mep(mep, (bx, by, bz, bw, bh, bd)))
                else:
                    # Box is not in this MEP, keep MEP
                    new_meps.append(mep)
            meps = new_meps
        bin_data['meps'] = meps

    def calculate_score(bins, placements):
        if not bins:
            return 0.0
        B = len(bins)
        last_bin_idx = B - 1
        last_bin = bins[last_bin_idx]
        
        last_bin_vol = sum(b[3]*b[4]*b[5] for b in last_bin['boxes'])
        container_vol = W * H * D
        if container_vol == 0:
            fill = 1.0
        else:
            fill = last_bin_vol / container_vol
        
        return B + 0.1 * (1 - fill)

    # Get initial candidates
    candidates = construct_initial()
    
    best_score = float('inf')
    best_bins = None
    best_placements = None
    
    for bins, placements in candidates:
        score = calculate_score(bins, placements)
        if score < best_score:
            best_score = score
            best_bins = [b.copy() for b in bins] # Shallow copy of dicts, but 'meps' and 'boxes' are lists
            # Need deep copy for safety during SA
            best_bins = []
            for b in bins:
                best_bins.append({
                    'meps': b['meps'][:],
                    'boxes': b['boxes'][:]
                })
            best_placements = placements[:]
            
    if best_bins is None:
        return []
        
    # Simulated Annealing
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    
    current_bins = best_bins
    current_placements = best_placements
    current_score = best_score
    
    # Helper to deep copy state
    def copy_state(bins, placements):
        new_bins = []
        for b in bins:
            new_bins.append({
                'meps': b['meps'][:],
                'boxes': b['boxes'][:]
            })
        return new_bins, placements[:]

    while time.time() < deadline and T > min_T:
        # Save state
        old_bins, old_placements = copy_state(current_bins, current_placements)
        
        move_made = False
        
        # Random move: Relocate or Swap
        if random.random() < 0.7 and len(current_bins) > 1:
            # Relocate
            i = random.randint(0, n - 1)
            curr_bin_idx = current_placements[i][0]
            target_bin_idx = random.randint(0, len(current_bins) - 1)
            
            if curr_bin_idx != target_bin_idx:
                # Get box
                box_entry = None
                for b in current_bins[curr_bin_idx]['boxes']:
                    if b[6] == i:
                        box_entry = b
                        break
                
                if box_entry:
                    # Remove from source
                    current_bins[curr_bin_idx]['boxes'].remove(box_entry)
                    
                    # Try to place in target
                    w, h, d = box_entry[3], box_entry[4], box_entry[5]
                    success, new_meps, pos = place_box_in_bin(current_bins[target_bin_idx], (w, h, d))
                    
                    if success:
                        current_bins[target_bin_idx]['meps'] = new_meps
                        current_bins[target_bin_idx]['boxes'].append((pos[0], pos[1], pos[2], w, h, d, i))
                        current_placements[i] = (target_bin_idx, pos[0], pos[1], pos[2], w, h, d)
                        
                        # Rebuild MEPs for source bin (since we removed a box)
                        rebuild_meps(current_bins[curr_bin_idx])
                        
                        # Compact bins if source is empty
                        if not current_bins[curr_bin_idx]['boxes']:
                            # Remove bin
                            # This shifts indices! Dangerous.
                            # Better: Mark as empty? 
                            # Or re-index all placements? 
                            # Let's just keep empty bins in the list but ignore them in score?
                            # No, score counts "occupied bins".
                            # If we keep empty bins, B is wrong.
                            # We must compact.
                            
                            # Compact logic:
                            new_bins_list = []
                            old_to_new = {}
                            for idx, b in enumerate(current_bins):
                                if b['boxes']:
                                    old_to_new[idx] = len(new_bins_list)
                                    new_bins_list.append(b)
                            
                            if len(new_bins_list) < len(current_bins):
                                # Update placements
                                for i in range(n):
                                    old_idx = current_placements[i][0]
                                    if old_idx in old_to_new:
                                        new_idx = old_to_new[old_idx]
                                        x, y, z, w, h, d = current_placements[i][1], current_placements[i][2], current_placements[i][3], current_placements[i][4], current_placements[i][5]
                                        current_placements[i] = (new_idx, x, y, z, w, h, d)
                                current_bins = new_bins_list
                            
                        move_made = True
                    else:
                        # Rollback
                        current_bins[curr_bin_idx]['boxes'].append(box_entry)
                        rebuild_meps(current_bins[curr_bin_idx])
                        # Restore state fully to be safe
                        current_bins = old_bins
                        current_placements = old_placements
                        
        else:
            # Swap
            if n >= 2:
                i1, i2 = random.sample(range(n), 2)
                bin1_idx = current_placements[i1][0]
                bin2_idx = current_placements[i2][0]
                
                if bin1_idx != bin2_idx:
                    # Find boxes
                    box1 = None
                    box2 = None
                    for b in current_bins[bin1_idx]['boxes']:
                        if b[6] == i1:
                            box1 = b
                    for b in current_bins[bin2_idx]['boxes']:
                        if b[6] == i2:
                            box2 = b
                    
                    if box1 and box2:
                        # Remove both
                        current_bins[bin1_idx]['boxes'].remove(box1)
                        current_bins[bin2_idx]['boxes'].remove(box2)
                        
                        # Try place box1 in bin2
                        w1, h1, d1 = box1[3], box1[4], box1[5]
                        s1, m1, p1 = place_box_in_bin(current_bins[bin2_idx], (w1, h1, d1))
                        
                        if s1:
                            current_bins[bin2_idx]['meps'] = m1
                            current_bins[bin2_idx]['boxes'].append((p1[0], p1[1], p1[2], w1, h1, d1, i1))
                            
                            # Try place box2 in bin1
                            w2, h2, d2 = box2[3], box2[4], box2[5]
                            s2, m2, p2 = place_box_in_bin(current_bins[bin1_idx], (w2, h2, d2))
                            
                            if s2:
                                current_bins[bin1_idx]['meps'] = m2
                                current_bins[bin1_idx]['boxes'].append((p2[0], p2[1], p2[2], w2, h2, d2, i2))
                                
                                # Rebuild MEPs for both bins? 
                                # We just set them from place_box_in_bin which assumes 
                                # the bin state was valid before insertion.
                                # But we removed boxes first! 
                                # So the MEPs in current_bins[bin1_idx] and bin2_idx are STALE 
                                # because we removed box1/box2 without updating MEPs.
                                # 
                                # CORRECTION:
                                # We must rebuild MEPs for both bins after removal, 
                                # BEFORE trying to insert.
                                
                                # Let's rollback and do it correctly.
                                current_bins = old_bins
                                current_placements = old_placements
                                move_made = False
                                
                                # Redo Swap correctly:
                                current_bins, current_placements = copy_state(old_bins, old_placements)
                                
                                # Remove
                                current_bins[bin1_idx]['boxes'].remove(box1)
                                current_bins[bin2_idx]['boxes'].remove(box2)
                                
                                # Rebuild MEPs for both
                                rebuild_meps(current_bins[bin1_idx])
                                rebuild_meps(current_bins[bin2_idx])
                                
                                # Insert box1 into bin2
                                s1, m1, p1 = place_box_in_bin(current_bins[bin2_idx], (w1, h1, d1))
                                if s1:
                                    current_bins[bin2_idx]['meps'] = m1
                                    current_bins[bin2_idx]['boxes'].append((p1[0], p1[1], p1[2], w1, h1, d1, i1))
                                    
                                    # Insert box2 into bin1
                                    s2, m2, p2 = place_box_in_bin(current_bins[bin1_idx], (w2, h2, d2))
                                    if s2:
                                        current_bins[bin1_idx]['meps'] = m2
                                        current_bins[bin1_idx]['boxes'].append((p2[0], p2[1], p2[2], w2, h2, d2, i2))
                                        
                                        # Update placements
                                        current_placements[i1] = (bin2_idx, p1[0], p1[1], p1[2], w1, h1, d1)
                                        current_placements[i2] = (bin1_idx, p2[0], p2[1], p2[2], w2, h2, d2)
                                        
                                        move_made = True
                                    else:
                                        # Rollback
                                        current_bins = old_bins
                                        current_placements = old_placements
                                else:
                                    # Rollback
                                    current_bins = old_bins
                                    current_placements = old_placements

        if move_made:
            new_score = calculate_score(current_bins, current_placements)
            delta = new_score - current_score
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_bins, best_placements = copy_state(current_bins, current_placements)
            else:
                # Rollback
                current_bins = old_bins
                current_placements = old_placements
                current_score = calculate_score(current_bins, current_placements)
                
        T *= cooling_rate

    return best_placements

# EVOLVE-BLOCK-END
