"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def pack(boxes, container):
    W, H, D = container
    container_vol = W * H * D
    n = len(boxes)
    if n == 0:
        return []
    
    # Deadline guard: return best found before ~115s (safety margin under 120s/150s kill)
    t_start = time.time()
    DEADLINE = 115.0
    
    # Precompute box volumes
    box_vols = [w * h * d for w, h, d in boxes]
    
    # Helper: check if a box at (x,y,z) with dims (w,h,d) overlaps any placed box in bin
    def overlaps(x, y, z, w, h, d, placed):
        for (px, py, pz, pw, ph, pd) in placed:
            # Check overlap on all three axes
            if (x < px + pw and x + w > px and
                y < py + ph and y + h > py and
                z < pz + pd and z + d > pz):
                return True
        return False

    def get_best_blf_position(bin_placed, w, h, d):
        """
        Find the best (x, y, z) position for a box of size (w, h, d) in a bin
        using the Bottom-Left-Front heuristic.
        Returns (x, y, z) or None if no position is found.
        """
        if not bin_placed:
            if w <= W and h <= H and d <= D:
                return (0, 0, 0)
            return None
        
        # Generate candidate positions
        # x candidates: 0 and (px + pw) for all placed boxes
        # y candidates: 0 and (py + ph) for all placed boxes
        # z candidates: 0 and (pz + pd) for all placed boxes
        
        x_candidates = {0}
        y_candidates = {0}
        z_candidates = {0}
        
        for (px, py, pz, pw, ph, pd) in bin_placed:
            x_candidates.add(px + pw)
            y_candidates.add(py + ph)
            z_candidates.add(pz + pd)
        
        # Filter candidates to be within bounds
        valid_x = [x for x in x_candidates if x + w <= W]
        valid_y = [y for y in y_candidates if y + h <= H]
        valid_z = [z for z in z_candidates if z + d <= D]
        
        if not valid_x or not valid_y or not valid_z:
            return None
            
        best_pos = None
        best_key = None
        
        # Iterate through all combinations of valid candidates
        # This is O(N^3) where N is number of boxes in bin, but N is small
        for x in valid_x:
            for y in valid_y:
                for z in valid_z:
                    if not overlaps(x, y, z, w, h, d, bin_placed):
                        # BLF: Minimize z (bottom), then y (left), then x (front)
                        # Actually, standard BLF in 3D is often:
                        # Minimize z (height), then y (depth), then x (width)
                        # Let's stick to (z, y, x) minimization
                        key = (z, y, x)
                        if best_key is None or key < best_key:
                            best_key = key
                            best_pos = (x, y, z)
                            
        return best_pos

    def try_packing(order):
        """
        Try packing boxes in the given order.
        Returns (placements, num_bins, last_bin_vol) or (None, 0, 0) if failed.
        """
        bins = []  # list of lists of (x, y, z, w, h, d)
        bin_volumes = []  # cumulative volume in each bin
        result_placements = [None] * n 
        
        for idx in order:
            w, h, d = boxes[idx]
            placed = False
            
            # Find the best bin to place the box
            # Strategy: Best Fit - place in the bin where it fits with the least remaining space
            # Or simply First Fit for speed, but Best Fit usually yields fewer bins.
            # We will try all bins, find valid positions, and pick the bin+pos that is "best".
            # For simplicity and speed, we will iterate bins, find best BLF pos in each.
            # Then choose the bin that has the highest current volume (Best Fit).
            
            best_bin_idx = -1
            best_pos = None
            best_bin_vol = -1
            
            for b_idx in range(len(bins)):
                pos = get_best_blf_position(bins[b_idx], w, h, d)
                if pos is not None:
                    # This box fits in this bin
                    # We prefer the bin that is already most full (Best Fit)
                    if bin_volumes[b_idx] > best_bin_vol:
                        best_bin_vol = bin_volumes[b_idx]
                        best_bin_idx = b_idx
                        best_pos = pos
            
            if best_bin_idx != -1:
                x, y, z = best_pos
                bins[best_bin_idx].append((x, y, z, w, h, d))
                bin_volumes[best_bin_idx] += w * h * d
                result_placements[idx] = (best_bin_idx, x, y, z, w, h, d)
                placed = True
            else:
                # Open a new bin
                b_idx = len(bins)
                if w <= W and h <= H and d <= D:
                    bins.append([(0, 0, 0, w, h, d)])
                    bin_volumes.append(w * h * d)
                    result_placements[idx] = (b_idx, 0, 0, 0, w, h, d)
                    placed = True
                else:
                    # Box is too big for a single bin (should not happen in valid instances)
                    # But if it does, we can't place it. 
                    # The problem states "places every box", implying they fit.
                    # If a box doesn't fit, the solution is invalid.
                    return None, 0, 0

            if not placed:
                return None, 0, 0

            # Check deadline
            if time.time() - t_start > DEADLINE:
                # If we break here, we have a partial solution.
                # We need to handle this carefully. The task requires placing EVERY box.
                # If we run out of time, we should return the best *complete* solution found.
                # Since this function is called per strategy, if it times out, it returns None.
                return None, 0, 0

        num_bins = len(bins)
        # Compute last bin volume (highest bin index that has boxes)
        # Bins are 0-indexed and created sequentially, so the last bin is len(bins)-1
        last_bin_idx = num_bins - 1
        last_bin_vol = bin_volumes[last_bin_idx]
        
        return result_placements, num_bins, last_bin_vol

    # Generate sorting orders
    orders = []
    
    # 1. Volume descending (BFD) - Primary strategy
    orders.append(sorted(range(n), key=lambda i: box_vols[i], reverse=True))
    
    # 2. Max dimension descending
    orders.append(sorted(range(n), key=lambda i: max(boxes[i][0], boxes[i][1], boxes[i][2]), reverse=True))
    
    # 3. Sum of dimensions descending
    orders.append(sorted(range(n), key=lambda i: sum(boxes[i]), reverse=True))
    
    # 4. Volume ascending (sometimes helps with small gaps?)
    orders.append(sorted(range(n), key=lambda i: box_vols[i]))
    
    # 5. Randomized perturbations of BFD
    rng = random.Random(42)
    for _ in range(3):
        # Start with BFD, then swap a few random pairs
        order = sorted(range(n), key=lambda i: box_vols[i], reverse=True)
        for _ in range(5):
            i, j = rng.sample(range(n), 2)
            order[i], order[j] = order[j], order[i]
        orders.append(order)

    best_placements = None
    best_score = float('inf')
    
    for order in orders:
        if time.time() - t_start > DEADLINE:
            break
        
        placements, num_bins, last_bin_vol = try_packing(order)
        if placements is None:
            continue
        
        # Compute score
        if num_bins == 0:
            score = float('inf')
        else:
            last_bin_frac = last_bin_vol / container_vol if container_vol > 0 else 0
            score = num_bins + 0.1 * (1 - last_bin_frac)
        
        if score < best_score:
            best_score = score
            best_placements = placements
    
    # If no valid placement found (e.g. all timed out or failed), fall back to simple shelf
    if best_placements is None:
        # Fallback: simple shelf packing (guaranteed to finish fast, might be suboptimal)
        placements = []
        b, x, y, z, row_h, layer_d = 0, 0, 0, 0, 0, 0
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
        best_placements = placements
        
    return best_placements
# EVOLVE-BLOCK-END
