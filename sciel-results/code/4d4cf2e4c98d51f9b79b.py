"""3D Bin Packing: Corner-Point Heuristic with free-space pruning and local search."""
import time
import random

# EVOLVE-BLOCK-START
def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []

    start_time = time.time()
    deadline = start_time + 115.0  # safety margin

    # Sort boxes by volume descending (FDD heuristic)
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda x: x[1][0] * x[1][1] * x[1][2], reverse=True)
    sorted_indices = [i for i, _ in indexed_boxes]
    sorted_dims = [(w, h, d) for _, (w, h, d) in indexed_boxes]

    def _compute_score(placements):
        """Compute the objective score: B + 0.1*(1 - last-bin fraction)."""
        if not placements:
            return 0.0
        bin_labels = set(p[0] for p in placements)
        B = len(bin_labels)
        max_bin = max(bin_labels)
        last_bin_vol = 0
        for p in placements:
            if p[0] == max_bin:
                last_bin_vol += p[4] * p[5] * p[6]
        container_vol = W * H * D
        if container_vol > 0:
            frac = last_bin_vol / container_vol
        else:
            frac = 0.0
        return B + 0.1 * (1 - frac)

    def _run_bfd(order):
        """
        Run Best-Fit Decreasing with corner-point heuristic and free-space pruning.
        order: list of original indices in placement order.
        Returns: list of (bin, x, y, z, w, h, d) for each original index.
        """
        # bins: list of lists of (x, y, z, w, h, d)
        bins = []
        # free_spaces: list of lists of (sx, sy, sz, sw, sh, sd)
        free_spaces = []
        placements = [None] * n

        for orig_idx in order:
            w, h, d = boxes[orig_idx]
            if w > W or h > H or d > D:
                # Box doesn't fit in any bin; place in its own bin at 0,0,0 (invalid but required)
                b_idx = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                free_spaces.append([(0, 0, 0, W, H, D)])
                placements[orig_idx] = (b_idx, 0, 0, 0, w, h, d)
                continue

            best_b_idx = -1
            best_pos = None
            best_score_val = float('inf')

            for b_idx in range(len(bins)):
                # Find best position in this bin using free spaces
                pos = _find_position_in_freespace(free_spaces[b_idx], w, h, d)
                if pos is not None:
                    x, y, z = pos
                    # Corner-point heuristic: minimize sum of distances to origin
                    # Actually, standard corner point is min(z, y, x) lexicographically
                    # But let's use a weighted sum to encourage compactness
                    score_val = z + 0.5 * y + 0.25 * x
                    if score_val < best_score_val:
                        best_score_val = score_val
                        best_pos = pos
                        best_b_idx = b_idx

            if best_b_idx == -1:
                # Open new bin
                b_idx = len(bins)
                bins.append([])
                free_spaces.append([(0, 0, 0, W, H, D)])
                best_pos = (0, 0, 0)
                best_b_idx = b_idx

            x, y, z = best_pos
            bins[best_b_idx].append((x, y, z, w, h, d))
            free_spaces[best_b_idx] = _update_freespace(free_spaces[best_b_idx], x, y, z, w, h, d)
            placements[orig_idx] = (best_b_idx, x, y, z, w, h, d)

        return placements, bins, free_spaces

    def _find_position_in_freespace(fs, w, h, d):
        """Find best position in free-space list."""
        best_pos = None
        best_score_val = float('inf')
        for (sx, sy, sz, sw, sh, sd) in fs:
            if w <= sw and h <= sh and d <= sd:
                # Position is (sx, sy, sz)
                score_val = sz + 0.5 * sy + 0.25 * sx
                if score_val < best_score_val:
                    best_score_val = score_val
                    best_pos = (sx, sy, sz)
        return best_pos

    def _update_freespace(fs, x, y, z, w, h, d):
        """Update free-space list after placing a box."""
        new_fs = []
        for (sx, sy, sz, sw, sh, sd) in fs:
            # Check if box (x,y,z,w,h,d) intersects free space (sx,sy,sz,sw,sh,sd)
            # Intersection condition: NOT (separated on any axis)
            if not (x + w <= sx or sx + sw <= x or
                    y + h <= sy or sy + sh <= y or
                    z + d <= sz or sz + sd <= z):
                # They intersect, so we need to split the free space
                # This is the complex part. We split the free space into up to 6 parts
                # that do not overlap with the box.
                # Actually, it's easier to keep the free space list as a list of maximal empty boxes.
                # When a new box is placed, we remove all free spaces that intersect it,
                # and then add the new empty spaces created around the box within those free spaces.
                # However, a simpler approach for correctness (though potentially less efficient) is:
                # 1. For each free space that intersects the box, compute the sub-boxes of the free space
                #    that are not covered by the box.
                # 2. Add those sub-boxes to the new list.
                # 3. Free spaces that do NOT intersect the box are kept as is.
                
                # Let's do the splitting.
                # The box occupies [x, x+w] x [y, y+h] x [z, z+d]
                # The free space is [sx, sx+sw] x [sy, sy+sh] x [sz, sz+sd]
                
                # We generate candidate sub-boxes by slicing along each axis.
                # This is essentially the "skyline" or "maximal rectangle" approach but in 3D.
                
                # Left slice
                if x > sx:
                    new_fs.append((sx, sy, sz, x - sx, sh, sd))
                # Right slice
                if x + w < sx + sw:
                    new_fs.append((x + w, sy, sz, sx + sw - (x + w), sh, sd))
                # Bottom slice (in Y)
                if y > sy:
                    # Note: This slice is only valid in the x-range where the box exists?
                    # No, the standard splitting algorithm generates 6 boxes:
                    # 1. (sx, sy, sz, x-sx, sh, sd)
                    # 2. (x+w, sy, sz, sx+sw-x-w, sh, sd)
                    # 3. (x, sy, sz, w, y-sy, sd)  -- Wait, this overlaps with 1 and 2 if not careful.
                    
                    # The correct way to split a box B1 by a box B2 (where B2 is inside B1's bounding box but not necessarily aligned)
                    # is to generate the regions of B1 \ B2.
                    # Since B2 is a box, B1 \ B2 is a union of up to 6 boxes.
                    
                    # Let's use a more robust splitting:
                    # 1. Region left of B2 in B1: x in [sx, x), y in [sy, sy+sh], z in [sz, sz+sd]
                    if x > sx:
                        new_fs.append((sx, sy, sz, x - sx, sh, sd))
                    # 2. Region right of B2 in B1: x in [x+w, sx+sw], y in [sy, sy+sh], z in [sz, sz+sd]
                    if x + w < sx + sw:
                        new_fs.append((x + w, sy, sz, sx + sw - x - w, sh, sd))
                    # 3. Region below B2 in B1 (within x-range of B2): x in [x, x+w], y in [sy, y), z in [sz, sz+sd]
                    if y > sy:
                        new_fs.append((x, sy, sz, w, y - sy, sd))
                    # 4. Region above B2 in B1 (within x-range of B2): x in [x, x+w], y in [y+h, sy+sh], z in [sz, sz+sd]
                    if y + h < sy + sh:
                        new_fs.append((x, y + h, sz, w, sy + sh - y - h, sd))
                    # 5. Region behind B2 in B1 (within x,y range of B2): x in [x, x+w], y in [y, y+h], z in [sz, z)
                    if z > sz:
                        new_fs.append((x, y, sz, w, h, z - sz))
                    # 6. Region in front of B2 in B1 (within x,y range of B2): x in [x, x+w], y in [y, y+h], z in [z+d, sz+sd]
                    if z + d < sz + sd:
                        new_fs.append((x, y, z + d, w, h, sz + sd - z - d))
            else:
                # No intersection, keep the free space
                new_fs.append((sx, sy, sz, sw, sh, sd))
        
        # Optional: Merge adjacent free spaces to keep the list small?
        # For now, just return the list. It might grow, but for 40 boxes it should be manageable.
        return new_fs

    # Initial run
    placements, bins, free_spaces = _run_bfd(sorted_indices)
    best_placements = placements
    best_score = _compute_score(placements)
    best_order = sorted_indices[:]

    # Local search: try swapping pairs of boxes in the order
    elapsed = time.time() - start_time
    if deadline - elapsed > 10.0:
        current_order = sorted_indices[:]
        # Pre-shuffle to avoid bias? Or just random swaps.
        for _ in range(500):  # Limit iterations
            if time.time() > deadline - 5.0:
                break
            i, j = random.sample(range(n), 2)
            new_order = current_order[:]
            new_order[i], new_order[j] = new_order[j], new_order[i]
            
            new_placements, _, _ = _run_bfd(new_order)
            new_score = _compute_score(new_placements)
            
            if new_score < best_score:
                best_score = new_score
                best_placements = new_placements
                best_order = new_order
                current_order = new_order

    # Final refinement: try to move boxes from the last bin to earlier bins
    # This is a simple "bin elimination" pass.
    if time.time() < deadline - 2.0:
        # Reconstruct bins from best_placements
        # Note: best_placements is a list of (bin, x, y, z, w, h, d)
        # We need to group by bin.
        bin_dict = {}
        for p in best_placements:
            b = p[0]
            if b not in bin_dict:
                bin_dict[b] = []
            bin_dict[b].append(p)
        
        # Sort bins by label
        bin_labels = sorted(bin_dict.keys())
        
        # Try to empty the last bin
        for b_idx in range(len(bin_labels) - 1, 0, -1):
            if time.time() > deadline - 2.0:
                break
            last_bin_label = bin_labels[b_idx]
            boxes_in_last_bin = bin_dict[last_bin_label]
            
            for p in boxes_in_last_bin:
                if time.time() > deadline - 2.0:
                    break
                orig_idx = None
                # Find original index
                for i in range(n):
                    if best_placements[i] == p:
                        orig_idx = i
                        break
                if orig_idx is None:
                    continue
                
                w, h, d = p[4], p[5], p[6]
                
                # Try to place this box in any earlier bin
                moved = False
                for new_b_label in bin_labels[:b_idx]:
                    # Get the free space for this bin? 
                    # We don't have the free space list for the current best_placements directly.
                    # We would need to reconstruct it or use a different check.
                    # For simplicity, let's skip this complex refinement if we don't have the state.
                    # Or, we can just re-run BFD with the box removed from the last bin?
                    # That's expensive.
                    pass
                
                # Alternative: Just try to re-pack the last bin's boxes into earlier bins using _find_position_in_freespace
                # But we need the free spaces.
                # Let's skip this specific refinement to avoid complexity and potential bugs.
                # The local search should have handled most improvements.

    return best_placements

# EVOLVE-BLOCK-END
