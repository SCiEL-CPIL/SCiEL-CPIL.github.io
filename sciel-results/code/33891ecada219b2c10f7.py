import time
import random
import numpy as np

def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    start_time = time.time()
    # Safety margin: The evaluator kills at 150s. We target 145s to be safe.
    deadline = start_time + 145.0

    volumes = np.array([w * h * d for w, h, d in boxes], dtype=np.float64)
    container_vol = float(W * H * D)
    if container_vol <= 0:
        return [(0, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]

    # Precompute box dimensions as a list of tuples for faster access
    box_dims = [(w, h, d) for w, h, d in boxes]

    def is_valid_place(bin_boxes, x, y, z, w, h, d):
        """Check if a box fits at (x,y,z) without overlapping existing boxes in bin_boxes."""
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            # Check overlap: not (x+w <= px or x >= px+pw) means overlap in x
            if not (x + w <= px or x >= px + pw):
                if not (y + h <= py or y >= py + ph):
                    if not (z + d <= pz or z >= pz + pd):
                        return False
        return True

    def get_candidate_positions(bin_boxes, w, h, d):
        """
        Generate candidate positions based on existing boxes.
        We try positions adjacent to existing boxes to maintain compactness.
        """
        cands = [(0, 0, 0)]
        seen = {cands[0]}
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            # Positions to the right, front, and top of existing boxes
            candidates = [
                (px + pw, py, pz),
                (px, py + ph, pz),
                (px, py, pz + pd),
                (px + pw, py + ph, pz),
                (px + pw, py, pz + pd),
                (px, py + ph, pz + pd),
                (px + pw, py + ph, pz + pd),
            ]
            for c in candidates:
                if c not in seen and c[0] + w <= W and c[1] + h <= H and c[2] + d <= D:
                    seen.add(c)
                    cands.append(c)
        return cands

    def find_best_spot(bin_boxes, w, h, d):
        """
        Find the best valid spot in the bin.
        Heuristic: Minimize (z, y, x) to pack towards the origin (Bottom-Left-Back).
        """
        cands = get_candidate_positions(bin_boxes, w, h, d)
        best_pos = None
        best_key = None
        
        for (x, y, z) in cands:
            if is_valid_place(bin_boxes, x, y, z, w, h, d):
                # Key: lower z is better, then lower y, then lower x
                key = (z, y, x)
                if best_key is None or key < best_key:
                    best_key = key
                    best_pos = (x, y, z)
        
        return best_pos

    def compute_score(placements):
        """Calculate the score: B + 0.1 * (1 - last_bin_fill)"""
        if not placements:
            return 0.0
        bin_ids = set()
        last_bin_vol = 0.0
        last_bin_id = -1
        
        for p in placements:
            b = p[0]
            bin_ids.add(b)
            if b > last_bin_id:
                last_bin_id = b
                last_bin_vol = 0.0
            elif b == last_bin_id:
                last_bin_vol += p[4] * p[5] * p[6]
        
        B = len(bin_ids)
        if last_bin_id == -1:
            return float(B)
            
        fill_ratio = last_bin_vol / container_vol if container_vol > 0 else 1.0
        return B + 0.1 * (1.0 - fill_ratio)

    def initial_packing(order):
        """
        Perform initial packing using the given order of box indices.
        """
        bins = []  # List of lists: bins[i] contains (x,y,z,w,h,d) for boxes in bin i
        placements = [None] * n

        for idx in order:
            w, h, d = box_dims[idx]
            best_bin_idx = -1
            best_spot = None
            best_key = None

            # Try to place in existing bins
            for b_idx, bin_boxes in enumerate(bins):
                spot = find_best_spot(bin_boxes, w, h, d)
                if spot is not None:
                    x, y, z = spot
                    # Best Fit Decreasing: Prefer bins that are already fuller?
                    # Or just First Fit? 
                    # Let's use a key that prefers filling current bins to minimize B.
                    # If we can place it, we prefer the one that results in the "tightest" fit?
                    # Actually, just taking the first valid bin (First Fit) is often robust.
                    # But to be safe, let's try to minimize the resulting bounding box or just take the first one.
                    # Given the score depends on B, keeping bins compact is good.
                    # Let's use the key (z, y, x) of the spot to prefer compactness.
                    key = (z, y, x)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_bin_idx = b_idx
                        best_spot = spot

            if best_bin_idx != -1:
                x, y, z = best_spot
                bins[best_bin_idx].append((x, y, z, w, h, d))
                placements[idx] = (best_bin_idx, x, y, z, w, h, d)
            else:
                # Open a new bin
                b_idx = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                placements[idx] = (b_idx, 0, 0, 0, w, h, d)

        return placements

    def get_bin_boxes(placements, bin_id):
        """Helper to extract box dimensions from a specific bin in placements."""
        return [p[1:7] for p in placements if p is not None and p[0] == bin_id]

    def try_move_box(placements, box_idx, target_bin_id):
        """
        Try to move box_idx to target_bin_id.
        Returns new placements if successful and valid, else None.
        """
        w, h, d = placements[box_idx][4], placements[box_idx][5], placements[box_idx][6]
        
        # Get boxes in target bin, excluding the box being moved (if it's already there, though we check src != tgt)
        target_boxes = get_bin_boxes(placements, target_bin_id)
        
        spot = find_best_spot(target_boxes, w, h, d)
        if spot is None:
            return None
        
        new_placements = list(placements)
        new_placements[box_idx] = (target_bin_id, spot[0], spot[1], spot[2], w, h, d)
        return new_placements

    def try_consolidate_bin(placements, bin_to_remove):
        """
        Try to move all boxes from bin_to_remove into other bins.
        Returns new placements if successful, else None.
        """
        boxes_in_bin = [i for i in range(n) if placements[i] is not None and placements[i][0] == bin_to_remove]
        if not boxes_in_bin:
            return placements

        # Sort boxes to move by volume descending (hardest first)
        boxes_in_bin.sort(key=lambda i: -volumes[i])

        # Work on a copy of the bin states
        work_bins = {}
        for b_id in set(p[0] for p in placements):
            if b_id != bin_to_remove:
                work_bins[b_id] = get_bin_boxes(placements, b_id)
        
        # If there are no other bins, we can't consolidate (unless we merge into a new bin 0, but that's just renaming)
        if not work_bins:
            return None

        work_placements = list(placements)
        
        for box_idx in boxes_in_bin:
            w, h, d = box_dims[box_idx]
            placed = False
            
            # Try to place in existing other bins
            # Sort target bins by current volume descending? Or just iterate.
            for b_id in sorted(work_bins.keys()):
                spot = find_best_spot(work_bins[b_id], w, h, d)
                if spot is not None:
                    work_bins[b_id].append((spot[0], spot[1], spot[2], w, h, d))
                    work_placements[box_idx] = (b_id, spot[0], spot[1], spot[2], w, h, d)
                    placed = True
                    break
            
            if not placed:
                return None
        
        return work_placements

    def local_search(placements, current_score):
        """
        Iterative local search to improve the packing.
        """
        improved = True
        pass_count = 0
        max_passes = 100 # Safety cap on passes, time limit is the real constraint

        while improved and pass_count < max_passes:
            improved = False
            pass_count += 1
            if time.time() > deadline - 5:
                break

            last_bin_id = max(p[0] for p in placements)
            all_bin_ids = sorted(set(p[0] for p in placements))

            # Phase 1: Try to eliminate the last bin
            if time.time() < deadline - 10:
                last_bin_boxes = [i for i in range(n) if placements[i] is not None and placements[i][0] == last_bin_id]
                # Only try consolidation if the bin is small enough or we are desperate
                # Actually, always try. If it fails, it's just wasted time.
                new_pl = try_consolidate_bin(placements, last_bin_id)
                if new_pl is not None:
                    new_score = compute_score(new_pl)
                    if new_score < current_score - 1e-12:
                        placements = new_pl
                        current_score = new_score
                        improved = True
                        # Restart loop if bin count changed significantly
                        continue 

            # Phase 2: Single box moves to improve fill or reduce B
            if time.time() < deadline - 10:
                # Sort boxes by volume descending for moves
                order_vol = list(np.argsort(-volumes))
                for i in order_vol:
                    if time.time() > deadline - 10:
                        break
                    if placements[i] is None:
                        continue
                    orig_bin = placements[i][0]
                    if orig_bin == last_bin_id:
                        continue # Moving from last bin to other bins might increase B or keep it same, usually not helpful for B reduction directly, but might help fill?
                        # Actually, moving FROM last bin TO other bins can only increase B or keep it same if it's the only box? No, if we move the last box out, B decreases.
                        # So we should allow moves FROM last bin too.
                        # Let's allow all moves.
                    
                    for tgt_bin in all_bin_ids:
                        if tgt_bin == orig_bin:
                            continue
                        new_pl = try_move_box(placements, i, tgt_bin)
                        if new_pl is not None:
                            new_score = compute_score(new_pl)
                            if new_score < current_score - 1e-12:
                                placements = new_pl
                                current_score = new_score
                                improved = True
                                all_bin_ids = sorted(set(p[0] for p in placements))
                                last_bin_id = max(p[0] for p in placements)
                                break # Restart box loop if improvement found? Or just continue?
                                # Breaking inner loop is good. We don't break outer loop to try other boxes.
                                # But the state changed, so 'orig_bin' is stale.
                                # It's safer to break and restart the whole pass if we made a move.
                                break
                    if improved:
                        break # Restart pass

            # Phase 3: 2-Swap Optimization (if time permits)
            if time.time() < deadline - 20 and len(all_bin_ids) > 1:
                # Try swapping boxes between different bins
                # This is expensive O(N^2 * M). N=40, so 1600 pairs.
                # We limit this to a few iterations or random subsets if slow.
                # For now, we'll do a limited sweep.
                swapped = False
                for i in range(n):
                    if time.time() > deadline - 20:
                        break
                    if placements[i] is None:
                        continue
                    for j in range(i + 1, n):
                        if time.time() > deadline - 20:
                            break
                        if placements[j] is None:
                            continue
                        if placements[i][0] == placements[j][0]:
                            continue
                        
                        # Try swapping i and j
                        w1, h1, d1 = box_dims[i]
                        w2, h2, d2 = box_dims[j]
                        
                        bin1 = placements[i][0]
                        bin2 = placements[j][0]
                        
                        # Remove i from bin1, j from bin2
                        boxes_in_bin1 = [p[1:7] for p in placements if p is not None and p[0] == bin1 and p[4] != w1 or (p[4]==w1 and p[5]!=h1) or (p[4]==w1 and p[5]==h1 and p[6]!=d1)]
                        # This filtering is tricky if dimensions match. 
                        # Better to use indices.
                        
                        # Let's implement a cleaner swap check
                        new_pl = list(placements)
                        
                        # Place box i in bin2 (without box j)
                        boxes_in_bin2_no_j = [p[1:7] for p in placements if p is not None and p[0] == bin2 and p is not placements[j]]
                        spot1 = find_best_spot(boxes_in_bin2_no_j, w1, h1, d1)
                        
                        if spot1 is None:
                            continue
                            
                        # Place box j in bin1 (without box i)
                        boxes_in_bin1_no_i = [p[1:7] for p in placements if p is not None and p[0] == bin1 and p is not placements[i]]
                        spot2 = find_best_spot(boxes_in_bin1_no_i, w2, h2, d2)
                        
                        if spot2 is None:
                            continue
                        
                        # Check if this swap improves score
                        # We need to construct the new placements to compute score
                        # But we can estimate: if it allows consolidation, it's good.
                        # Let's just compute score on a simulated list.
                        
                        new_placements = list(placements)
                        new_placements[i] = (bin2, spot1[0], spot1[1], spot1[2], w1, h1, d1)
                        new_placements[j] = (bin1, spot2[0], spot2[1], spot2[2], w2, h2, d2)
                        
                        new_score = compute_score(new_placements)
                        if new_score < current_score - 1e-12:
                            placements = new_placements
                            current_score = new_score
                            improved = True
                            swapped = True
                            all_bin_ids = sorted(set(p[0] for p in placements))
                            last_bin_id = max(p[0] for p in placements)
                            break # Restart pass
                    if swapped:
                        break

        return placements, current_score

    # ---- Generate Diverse Initial Solutions ----
    
    # Strategy 1: Volume Descending
    order_vol = list(np.argsort(-volumes))
    pl1 = initial_packing(order_vol)
    s1 = compute_score(pl1)
    
    # Strategy 2: Max Dimension Descending
    max_dims = np.array([max(w, h, d) for w, h, d in boxes])
    order_max = list(np.argsort(-max_dims))
    pl2 = initial_packing(order_max)
    s2 = compute_score(pl2)
    
    # Strategy 3: Surface Area Descending
    surf_areas = np.array([2*(w*h + w*d + h*d) for w, h, d in boxes])
    order_surf = list(np.argsort(-surf_areas))
    pl3 = initial_packing(order_surf)
    s3 = compute_score(pl3)
    
    # Strategy 4: "Hardest to Place" (Volume / Min Dimension)
    min_dims = np.array([min(w, h, d) for w, h, d in boxes])
    # Avoid division by zero
    min_dims_safe = np.where(min_dims == 0, 1, min_dims)
    difficulty = volumes / min_dims_safe
    order_diff = list(np.argsort(-difficulty))
    pl4 = initial_packing(order_diff)
    s4 = compute_score(pl4)
    
    best_placements = None
    best_score = float('inf')
    
    candidates = [
        (pl1, s1),
        (pl2, s2),
        (pl3, s3),
        (pl4, s4)
    ]
    
    # Pick the best initial solution
    for pl, s in candidates:
        if s < best_score:
            best_score = s
            best_placements = pl

    # ---- Local Search on Best Initial Solution ----
    if best_placements is not None and time.time() < deadline - 10:
        best_placements, best_score = local_search(best_placements, best_score)

    # ---- Try Local Search on Other Good Initial Solutions if Time Permits ----
    if time.time() < deadline - 30:
        for pl, s in candidates:
            if s == best_score and pl is best_placements:
                continue
            if s < best_score + 0.5: # Only try if it's close to the best
                if time.time() > deadline - 15:
                    break
                new_pl, new_s = local_search(pl, s)
                if new_s < best_score:
                    best_score = new_s
                    best_placements = new_pl

    # ---- Final Verification ----
    # Ensure the result is valid
    def is_valid(placements):
        if len(placements) != n:
            return False
        for p in placements:
            if p is None:
                return False
            b, x, y, z, w, h, d = p
            if b < 0 or x < 0 or y < 0 or z < 0:
                return False
            if x + w > W or y + h > H or z + d > D:
                return False
        # Check overlaps
        for i in range(n):
            for j in range(i + 1, n):
                bi, xi, yi, zi, wi, hi, di = placements[i]
                bj, xj, yj, zj, wj, hj, dj = placements[j]
                if bi == bj:
                    if (xi < xj + wj and xi + wi > xj and
                        yi < yj + hj and yi + hi > yj and
                        zi < zj + dj and zi + di > zj):
                        return False
        return True

    if not is_valid(best_placements):
        # Fallback: Simple Shelf Packing
        best_placements = []
        b, x, y, z, row_h, layer_d = 0, 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            best_placements.append((b, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
            
    return best_placements
