"""3D Bin Packing: parameterized placement with local search optimization."""
# EVOLVE-BLOCK-START
import time
import random
import math

def pack(boxes, container):
    W, H, D = container
    if not boxes:
        return []
    n = len(boxes)
    container_vol = W * H * D
    if container_vol == 0:
        return [(0, 0, 0, w, h, d) for w, h, d in boxes]
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    box_vols = [w * h * d for w, h, d in boxes]
    total_vol = sum(box_vols)

    def is_overlap(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                    y1 + h1 <= y2 or y2 + h2 <= y1 or
                    z1 + d1 <= z2 or z2 + d2 <= z1)

    def can_place(placed, x, y, z, w, h, d):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in placed:
            if is_overlap(x, y, z, w, h, d, px, py, pz, pw, ph, pd):
                return False
        return True

    def get_valid_positions(placed, w, h, d):
        """Generate candidate positions using face extensions."""
        cands = set()
        cands.add((0, 0, 0))
        for (px, py, pz, pw, ph, pd) in placed:
            # Extensions from each face
            cands.add((px + pw, py, pz))
            cands.add((px, py + ph, pz))
            cands.add((px, py, pz + pd))
            # Corners (often covered by combinations, but explicit is safer for gaps)
            cands.add((px + pw, py + ph, pz))
            cands.add((px + pw, py, pz + pd))
            cands.add((px, py + ph, pz + pd))
            cands.add((px + pw, py + ph, pz + pd))
            
        valid = []
        for (cx, cy, cz) in cands:
            if can_place(placed, cx, cy, cz, w, h, d):
                valid.append((cx, cy, cz))
        
        # Sort by compactness (distance to origin) to prefer tight packing
        valid.sort(key=lambda c: c[0] + c[1] + c[2])
        return valid

    def score_assignment(assignments):
        """Compute score: B + 0.1*(1 - last-bin occupied volume fraction)"""
        if not assignments:
            return float('inf')
        bins_used = set(assignments.values())
        B = len(bins_used)
        last_bin = max(bins_used)
        last_vol = sum(box_vols[idx] for idx, b in assignments.items() if b == last_bin)
        frac = last_vol / container_vol
        return B + 0.1 * (1 - frac)

    def greedy_pack(order, bins, assignments):
        """Place boxes in given order using best-fit decreasing style."""
        for idx, (w, h, d) in order:
            if time.time() > deadline:
                break
                
            placed_in_bin = False
            
            if bins:
                best_bin_id = -1
                best_pos = None
                best_metric = float('inf')
                
                for b_id in range(len(bins)):
                    if not bins[b_id]:
                        continue
                    
                    cands = get_valid_positions(bins[b_id], w, h, d)
                    if cands:
                        # Best fit: minimize coordinate sum (compactness)
                        cx, cy, cz = cands[0]
                        metric = cx + cy + cz
                        
                        if metric < best_metric:
                            best_metric = metric
                            best_bin_id = b_id
                            best_pos = (cx, cy, cz)
                
                if best_bin_id >= 0:
                    cx, cy, cz = best_pos
                    bins[best_bin_id].append((cx, cy, cz, w, h, d))
                    assignments[idx] = best_bin_id
                    placed_in_bin = True
            
            if not placed_in_bin:
                b_id = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                assignments[idx] = b_id

    def local_search(bins, assignments):
        """Local search to minimize bins and maximize last-bin fill."""
        if time.time() > deadline:
            return

        improved = True
        iter_count = 0
        max_iters = 2000
        
        while improved and iter_count < max_iters and time.time() < deadline:
            improved = False
            iter_count += 1
            
            # Phase 1: Minimize B by moving boxes from higher bins to lower bins
            # Iterate from highest bin to lowest
            for b_id in range(len(bins) - 1, -1, -1):
                if time.time() > deadline:
                    break
                if not bins[b_id]:
                    continue
                
                boxes_in_bin = list(bins[b_id])
                for local_idx in range(len(boxes_in_bin)):
                    if time.time() > deadline:
                        break
                    
                    # Get the box
                    box_data = boxes_in_bin[local_idx]
                    # We need to know which global index this is.
                    # Current bins structure: list of (x, y, z, w, h, d). We lost the index!
                    # This is a critical flaw in the current code structure if we don't track indices.
                    # The current code tracks indices in `assignments` but `bins` doesn't have them.
                    # We need to recover the index or change the structure.
                    
                    # Let's change bins to store (idx, x, y, z, w, h, d)
                    pass

        # Re-implementing with correct index tracking
        pass

    # --- Corrected Data Structure ---
    # bins: list of lists of (idx, x, y, z, w, h, d)
    
    def greedy_pack_v2(order, bins, assignments):
        for idx, (w, h, d) in order:
            if time.time() > deadline:
                break
                
            placed_in_bin = False
            
            if bins:
                best_bin_id = -1
                best_pos = None
                best_metric = float('inf')
                
                for b_id in range(len(bins)):
                    if not bins[b_id]:
                        continue
                    
                    # Extract placements for overlap check
                    placements = [(x, y, z, ww, hh, dd) for (_, x, y, z, ww, hh, dd) in bins[b_id]]
                    cands = get_valid_positions(placements, w, h, d)
                    if cands:
                        cx, cy, cz = cands[0]
                        metric = cx + cy + cz
                        
                        if metric < best_metric:
                            best_metric = metric
                            best_bin_id = b_id
                            best_pos = (cx, cy, cz)
                
                if best_bin_id >= 0:
                    cx, cy, cz = best_pos
                    bins[best_bin_id].append((idx, cx, cy, cz, w, h, d))
                    assignments[idx] = best_bin_id
                    placed_in_bin = True
            
            if not placed_in_bin:
                b_id = len(bins)
                bins.append([(idx, 0, 0, 0, w, h, d)])
                assignments[idx] = b_id

    def local_search_v2(bins, assignments):
        if time.time() > deadline:
            return

        improved = True
        iter_count = 0
        max_iters = 2000
        
        while improved and iter_count < max_iters and time.time() < deadline:
            improved = False
            iter_count += 1
            
            # Phase 1: Reduce B
            # Try to move boxes from bin i to bin j < i
            for b_id in range(len(bins) - 1, -1, -1):
                if time.time() > deadline:
                    break
                if not bins[b_id]:
                    continue
                
                # Copy list to avoid modification during iteration issues
                boxes_to_try = list(bins[b_id])
                
                for box_entry in boxes_to_try:
                    idx, px, py, pz, pw, ph, pd = box_entry
                    
                    # Check if box is still in this bin (it might have been moved in a previous step of this iteration? No, we only move forward in b_id, but we modify bins. 
                    # If we move a box from b_id to b_id-1, b_id changes.
                    # We should verify the box is still there.
                    if not any(b[0] == idx for b in bins[b_id]):
                        continue
                    
                    # Remove from current bin
                    bins[b_id].remove(box_entry)
                    
                    moved = False
                    # Try to place in any lower bin
                    for target_b in range(b_id):
                        if not bins[target_b]:
                            continue
                        
                        placements = [(x, y, z, ww, hh, dd) for (_, x, y, z, ww, hh, dd) in bins[target_b]]
                        cands = get_valid_positions(placements, pw, ph, pd)
                        
                        if cands:
                            cx, cy, cz = cands[0]
                            bins[target_b].append((idx, cx, cy, cz, pw, ph, pd))
                            assignments[idx] = target_b
                            moved = True
                            improved = True
                            break
                    
                    if not moved:
                        # Put back
                        bins[b_id].append(box_entry)
                        assignments[idx] = b_id
                    
                    # If we moved a box, the bin b_id might now be empty or changed.
                    # If b_id is empty, we can remove it.
                    if not bins[b_id]:
                        # Remove empty bin and renumber? 
                        # Renumbering is expensive. Let's just keep empty bins for now and clean up at the end.
                        # Or, since we iterate downwards, if we empty bin i, we can stop checking it?
                        # No, other boxes might be in it? No, we just emptied it.
                        # But we are iterating b_id from high to low.
                        # If we empty bin 5, we continue to bin 4.
                        # At the end, we will have empty bins in the list.
                        pass
        
        # Cleanup: Remove empty bins and renumber
        new_bins = []
        bin_map = {}
        for i, b in enumerate(bins):
            if b:
                new_id = len(new_bins)
                bin_map[i] = new_id
                new_bins.append(b)
        
        # Update assignments
        for idx in assignments:
            old_bin = assignments[idx]
            if old_bin in bin_map:
                assignments[idx] = bin_map[old_bin]
        
        # Update bins list
        bins[:] = new_bins

        # Phase 2: Maximize volume in the last bin
        if time.time() < deadline and bins:
            last_bin_id = len(bins) - 1
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Try to move boxes from other bins to the last bin
                for b_id in range(last_bin_id):
                    if time.time() > deadline:
                        break
                    if not bins[b_id]:
                        continue
                    
                    boxes_to_try = list(bins[b_id])
                    for box_entry in boxes_to_try:
                        idx, px, py, pz, pw, ph, pd = box_entry
                        
                        if not any(b[0] == idx for b in bins[b_id]):
                            continue
                        
                        placements = [(x, y, z, ww, hh, dd) for (_, x, y, z, ww, hh, dd) in bins[last_bin_id]]
                        cands = get_valid_positions(placements, pw, ph, pd)
                        
                        if cands:
                            cx, cy, cz = cands[0]
                            bins[b_id].remove(box_entry)
                            bins[last_bin_id].append((idx, cx, cy, cz, pw, ph, pd))
                            assignments[idx] = last_bin_id
                            improved = True
                            # We don't break here because we want to fill the last bin as much as possible
                            # But moving a box might open space for another?
                            # Usually, moving to last bin is "filling", so it reduces free space.
                            # So we should break after one move? Or continue?
                            # If we move a small box, it might block a larger one.
                            # Best to try all and keep the best?
                            # For simplicity, just move if it fits.
                            pass

    def build_result(bins, assignments):
        result = [None] * n
        for idx in range(n):
            if idx in assignments:
                b_id = assignments[idx]
                for (i, x, y, z, w, h, d) in bins[b_id]:
                    if i == idx:
                        result[idx] = (b_id, x, y, z, w, h, d)
                        break
                if result[idx] is None:
                    # Fallback
                    result[idx] = (0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
            else:
                # Box wasn't placed
                b_id = len(bins)
                bins.append([(idx, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])])
                assignments[idx] = b_id
                result[idx] = (b_id, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
        
        # Renumber bins to be consecutive
        if result:
            bin_labels = sorted(set(r[0] for r in result if r is not None))
            if bin_labels:
                label_map = {old: new for new, old in enumerate(bin_labels)}
                for i in range(n):
                    if result[i] is not None:
                        old_bin = result[i][0]
                        new_bin = label_map[old_bin]
                        result[i] = (new_bin, result[i][1], result[i][2], result[i][3], result[i][4], result[i][5], result[i][6])
        
        return result

    def validate_result(result):
        bins_boxes = {}
        for idx in range(n):
            if result[idx] is None:
                return False
            b, x, y, z, w, h, d = result[idx]
            if x < 0 or y < 0 or z < 0:
                return False
            if x + w > W or y + h > H or z + d > D:
                return False
            if b not in bins_boxes:
                bins_boxes[b] = []
            bins_boxes[b].append((x, y, z, w, h, d))
        
        for b, bl in bins_boxes.items():
            for i in range(len(bl)):
                for j in range(i + 1, len(bl)):
                    if is_overlap(bl[i][0], bl[i][1], bl[i][2], bl[i][3], bl[i][4], bl[i][5],
                                  bl[j][0], bl[j][1], bl[j][2], bl[j][3], bl[j][4], bl[j][5]):
                        return False
        return True

    def score_from_result(result):
        if not result:
            return float('inf')
        bins_used = set(r[0] for r in result)
        B = len(bins_used)
        last_bin = max(bins_used)
        last_vol = sum(r[4] * r[5] * r[6] for r in result if r[0] == last_bin)
        frac = last_vol / container_vol
        return B + 0.1 * (1 - frac)

    # --- Main Search ---
    
    strategies = [
        lambda t: -(t[1][0] * t[1][1] * t[1][2]),  # Volume desc
        lambda t: -(max(t[1])),                      # Max dim desc
        lambda t: -(t[1][0] + t[1][1] + t[1][2]),   # Sum dim desc
        lambda t: t[1][0] * t[1][1] * t[1][2],      # Volume asc
    ]

    best_score = float('inf')
    best_result = None

    max_restarts = 50
    
    for strat_idx, strat_func in enumerate(strategies):
        if time.time() > deadline:
            break
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            order = list(enumerate(boxes))
            
            if restart == 0:
                order.sort(key=strat_func)
            else:
                # Add noise
                order.sort(key=lambda t: strat_func(t) + random.uniform(-0.5, 0.5) * (t[1][0] * t[1][1] * t[1][2] + 1))
            
            bins = []
            assignments = {}
            
            greedy_pack_v2(order, bins, assignments)
            local_search_v2(bins, assignments)
            
            result = build_result(bins, assignments)
            
            if not validate_result(result):
                continue
            
            score = score_from_result(result)
            
            if score < best_score:
                best_score = score
                best_result = result
                
            # Early exit if we found a perfect score (theoretical minimum)
            # Min B is ceil(total_vol / container_vol)
            min_bins = max(1, math.ceil(total_vol / container_vol))
            if best_score <= min_bins + 0.01:
                return best_result

    if best_result is None:
        # Fallback
        best_result = [None] * n
        for idx, (w, h, d) in enumerate(boxes):
            best_result[idx] = (0, 0, 0, w, h, d)
            
    return best_result
# EVOLVE-BLOCK-END
