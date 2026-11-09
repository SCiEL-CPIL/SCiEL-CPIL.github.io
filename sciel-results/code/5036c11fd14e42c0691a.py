"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random

def pack(boxes, container):
    W, H, D = container
    if not boxes:
        return []
    
    n = len(boxes)
    container_vol = W * H * D
    
    # Start time for deadline management
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. We target 145s to be safe.
    DEADLINE = start_time + 145.0

    class Bin3D:
        def __init__(self, W, H, D):
            self.W = W
            self.H = H
            self.D = D
            # List of empty boxes: (x, y, z, w, h, d)
            self.empty_boxes = [(0, 0, 0, W, H, D)]
            self.occupied_vol = 0
            self.placed_boxes = []

        def can_fit(self, w, h, d):
            """Check if any empty box can accommodate the item."""
            for ex, ey, ez, ew, eh, ed in self.empty_boxes:
                if w <= ew and h <= eh and d <= ed:
                    return True
            return False

        def get_candidates(self, w, h, d):
            """
            Generate candidate positions.
            For each empty box, the position (ex, ey, ez) is a valid corner.
            We return a list of (score, x, y, z, empty_box_index)
            Score is based on compactness: prefer lower z, then y, then x.
            """
            candidates = []
            for i, (ex, ey, ez, ew, eh, ed) in enumerate(self.empty_boxes):
                if w <= ew and h <= eh and d <= ed:
                    # Score: prefer smaller z (bottom), then y (front), then x (left)
                    # This creates a "shelf-like" filling from the bottom corner.
                    score = (ez, ey, ex)
                    candidates.append((score, ex, ey, ez, i))
            
            if not candidates:
                return []
            
            candidates.sort(key=lambda t: t[0])
            return candidates

        def place(self, w, h, d, x, y, z, empty_box_idx):
            """
            Place a box at (x,y,z) with dims (w,h,d) inside the empty box at empty_box_idx.
            Update the empty_boxes list.
            """
            # Remove the old empty box
            old_ex, old_ey, old_ez, old_ew, old_eh, old_ed = self.empty_boxes[empty_box_idx]
            
            # The placed box is at (x, y, z) with size (w, h, d).
            # It is contained within (old_ex, old_ey, old_ez, old_ew, old_eh, old_ed).
            
            # We need to subtract the placed box from the old empty box.
            # Since the placed box is at the corner (x=old_ex, y=old_ey, z=old_ez) 
            # relative to the empty box? No, x,y,z are absolute coords.
            # But get_candidates returns (ex, ey, ez) as the position.
            # So x == old_ex, y == old_ey, z == old_ez is always true for the chosen candidate.
            
            # Let's verify:
            # candidates.append((score, ex, ey, ez, i))
            # So x=ex, y=ey, z=ez.
            
            # The remainder of the old empty box (old_ex, old_ey, old_ez, old_ew, old_eh, old_ed)
            # after removing (old_ex, old_ey, old_ez, w, h, d) consists of up to 3 disjoint boxes:
            # 1. Right of placed box: (old_ex + w, old_ey, old_ez, old_ew - w, old_eh, old_ed) if w < old_ew
            # 2. Above placed box (in Y): (old_ex, old_ey + h, old_ez, w, old_eh - h, old_ed) if h < old_eh
            #    Wait, the "Above" part in Y only covers the width w.
            #    The space to the right (1) covers the full height old_eh.
            #    So the space "Above" the placed box but "Left" of the right part?
            #    Let's visualize:
            #    Old box: [0, W] x [0, H] x [0, D]
            #    Placed:   [0, w] x [0, h] x [0, d]
            
            #    Remaining volume:
            #    A. x in [w, W], y in [0, H], z in [0, D]  -> (w, 0, 0, W-w, H, D)
            #    B. x in [0, w], y in [h, H], z in [0, D]  -> (0, h, 0, w, H-h, D)
            #    C. x in [0, w], y in [0, h], z in [d, D]  -> (0, 0, d, w, h, D-d)
            
            #    These 3 boxes are disjoint and cover the remainder.
            
            new_empty = []
            
            # 1. Right
            if w < old_ew:
                new_empty.append((old_ex + w, old_ey, old_ez, old_ew - w, old_eh, old_ed))
            
            # 2. Above (in Y direction)
            if h < old_eh:
                new_empty.append((old_ex, old_ey + h, old_ez, w, old_eh - h, old_ed))
            
            # 3. Back (in Z direction)
            if d < old_ed:
                new_empty.append((old_ex, old_ey, old_ez + d, w, h, old_ed - d))
            
            # Remove the old box and add new ones
            # We replace the box at index empty_box_idx with the new ones
            # To keep indices stable or just rebuild? Rebuilding is O(N) but N is small.
            # Better: pop the old one and extend with new ones.
            self.empty_boxes.pop(empty_box_idx)
            self.empty_boxes.extend(new_empty)
            
            self.placed_boxes.append((x, y, z, w, h, d))
            self.occupied_vol += w * h * d

    def pack_heuristic(sort_key_func, random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            
        indexed_boxes = list(enumerate(boxes))
        indexed_boxes.sort(key=lambda t: sort_key_func(t[1]))
        
        bins = [] # List of Bin3D objects
        placement_map = {} # idx -> (bin_idx, x, y, z)
        
        for idx, (w, h, d) in indexed_boxes:
            if w > W or h > H or d > D:
                # Cannot place, skip. (Assume valid inputs per contract, but safe guard)
                continue

            placed = False
            
            # Try existing bins
            # Heuristic: Try bins in order of "tightness" or just index order?
            # For minimizing B, "Best Fit" (bin that fits with least waste) is good.
            # But calculating waste requires trying to place.
            # Let's try bins in order of creation (First Fit) or sorted by occupied volume?
            # Sorted by occupied volume descending might fill full bins first?
            # Actually, for 3D packing, "First Fit Decreasing" is standard.
            # Let's stick to First Fit (bin 0, 1, 2...) as it's simpler and often effective
            # with good item ordering.
            
            for bi in range(len(bins)):
                bin_obj = bins[bi]
                
                # Check if it fits
                if not bin_obj.can_fit(w, h, d):
                    continue
                
                candidates = bin_obj.get_candidates(w, h, d)
                if not candidates:
                    continue
                
                # Pick the best candidate (lowest score)
                # The candidates are already sorted.
                best_score, x, y, z, empty_idx = candidates[0]
                
                # Place it
                bin_obj.place(w, h, d, x, y, z, empty_idx)
                placement_map[idx] = (bi, x, y, z)
                placed = True
                break
            
            if not placed:
                # Open a new bin
                bi = len(bins)
                new_bin = Bin3D(W, H, D)
                bins.append(new_bin)
                # Place at (0,0,0)
                new_bin.place(w, h, d, 0, 0, 0, 0)
                placement_map[idx] = (bi, 0, 0, 0)
        
        # Re-index bins to maximize last-bin volume
        if not bins:
            return []
        
        # Calculate volume of each bin
        bin_volumes = [b.occupied_vol for b in bins]
        
        # Sort bins by volume descending. The bin with the largest volume should get the highest index.
        # We create a mapping from old_bin_index to new_bin_index.
        # We want the bin with the MAX volume to have the MAX index (K-1).
        # The bin with the MIN volume to have the MIN index (0).
        
        # sorted_bin_indices: [idx_of_max_vol, idx_of_2nd_max, ..., idx_of_min_vol]
        sorted_bin_indices = sorted(range(len(bins)), key=lambda i: -bin_volumes[i])
        
        new_index_map = {}
        # We want:
        # old_idx with max vol -> new_idx K-1
        # old_idx with min vol -> new_idx 0
        
        for new_idx, old_idx in enumerate(sorted_bin_indices):
            # sorted_bin_indices[0] is max vol. It should get index K-1.
            # sorted_bin_indices[K-1] is min vol. It should get index 0.
            # So: new_index_map[old_idx] = K - 1 - new_idx
            new_index_map[old_idx] = len(bins) - 1 - new_idx
        
        # Construct final placements
        final_placements = []
        for i in range(n):
            if i in placement_map:
                old_bi, x, y, z = placement_map[i]
                new_bi = new_index_map[old_bi]
                w, h, d = boxes[i]
                final_placements.append((new_bi, x, y, z, w, h, d))
            else:
                # Fallback for unplaced (should not happen)
                final_placements.append((0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]))
        
        return final_placements

    def evaluate(placements):
        if not placements:
            return float('inf')
        
        bins_used = set(p[0] for p in placements)
        B = len(bins_used)
        last_bin = max(bins_used)
        
        last_vol = 0
        for p in placements:
            if p[0] == last_bin:
                last_vol += p[4] * p[5] * p[6]
                
        if container_vol == 0:
            return float('inf')
            
        frac = last_vol / container_vol
        if frac > 1.0: frac = 1.0
        if frac < 0.0: frac = 0.0
        
        score = B + 0.1 * (1 - frac)
        return score

    # Strategies
    def sort_by_vol_desc(box):
        return -(box[0] * box[1] * box[2])
    
    def sort_by_maxdim_desc(box):
        return -max(box)
        
    def sort_by_area_desc(box):
        w, h, d = box
        return -(2*(w*h + h*d + w*d))

    def sort_by_vol_asc(box):
        return (box[0] * box[1] * box[2])

    strategies = [
        ("vol_desc", sort_by_vol_desc, None),
        ("maxdim_desc", sort_by_maxdim_desc, None),
        ("area_desc", sort_by_area_desc, None),
        ("vol_asc", sort_by_vol_asc, None), # Sometimes packing small first helps fill gaps?
    ]
    
    best_placements = None
    best_score = float('inf')
    
    # Run each strategy
    for name, key_func, seed in strategies:
        if time.time() > DEADLINE:
            break
            
        try:
            placements = pack_heuristic(key_func, seed)
            if placements:
                score = evaluate(placements)
                if score < best_score:
                    best_score = score
                    best_placements = placements
        except Exception as e:
            pass

    # If we have time, try a few random restarts with different orderings?
    # The current strategies are deterministic.
    # We can add a randomized strategy: shuffle boxes, then sort by volume?
    # Or just shuffle the input order?
    
    if time.time() < DEADLINE - 5.0:
        # Try a few random permutations
        for attempt in range(5):
            if time.time() > DEADLINE:
                break
            
            # Create a shuffled version of boxes?
            # The pack_heuristic sorts them anyway.
            # To get a different result, we need to change the tie-breaking or the sorting key slightly.
            # Or, we can modify the bin selection order.
            # For now, let's just try "vol_desc" again but with a slight randomization in candidate selection?
            # The current candidate selection is deterministic (lowest score).
            # We can introduce randomness in candidate selection: pick randomly from top 3 candidates.
            
            # Let's implement a variant that picks randomly from top candidates.
            def pack_heuristic_random(sort_key_func, random_seed):
                random.seed(random_seed)
                indexed_boxes = list(enumerate(boxes))
                indexed_boxes.sort(key=lambda t: sort_key_func(t[1]))
                
                bins = []
                placement_map = {}
                
                for idx, (w, h, d) in indexed_boxes:
                    if w > W or h > H or d > D:
                        continue

                    placed = False
                    
                    for bi in range(len(bins)):
                        bin_obj = bins[bi]
                        
                        if not bin_obj.can_fit(w, h, d):
                            continue
                        
                        candidates = bin_obj.get_candidates(w, h, d)
                        if not candidates:
                            continue
                        
                        # Pick randomly from the top 3 candidates (if available)
                        # Candidates are sorted by score.
                        top_k = candidates[:3]
                        # Weighted random or uniform? Uniform is fine.
                        chosen = random.choice(top_k)
                        best_score, x, y, z, empty_idx = chosen
                        
                        bin_obj.place(w, h, d, x, y, z, empty_idx)
                        placement_map[idx] = (bi, x, y, z)
                        placed = True
                        break
                    
                    if not placed:
                        bi = len(bins)
                        new_bin = Bin3D(W, H, D)
                        bins.append(new_bin)
                        new_bin.place(w, h, d, 0, 0, 0, 0)
                        placement_map[idx] = (bi, 0, 0, 0)
                
                if not bins:
                    return []
                
                bin_volumes = [b.occupied_vol for b in bins]
                sorted_bin_indices = sorted(range(len(bins)), key=lambda i: -bin_volumes[i])
                new_index_map = {}
                for new_idx, old_idx in enumerate(sorted_bin_indices):
                    new_index_map[old_idx] = len(bins) - 1 - new_idx
                
                final_placements = []
                for i in range(n):
                    if i in placement_map:
                        old_bi, x, y, z = placement_map[i]
                        new_bi = new_index_map[old_bi]
                        w, h, d = boxes[i]
                        final_placements.append((new_bi, x, y, z, w, h, d))
                    else:
                        final_placements.append((0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]))
                
                return final_placements

            try:
                placements = pack_heuristic_random(sort_by_vol_desc, attempt)
                if placements:
                    score = evaluate(placements)
                    if score < best_score:
                        best_score = score
                        best_placements = placements
            except Exception as e:
                pass

    if best_placements is None:
        # Fallback
        best_placements = []
        for i, (w, h, d) in enumerate(boxes):
            best_placements.append((0, 0, 0, w, h, d))

    return best_placements

# EVOLVE-BLOCK-END
