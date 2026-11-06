"""3D Bin Packing: Best-fit decreasing with fragmentation-aware bin selection and local search polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin for 150s limit
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Sort boxes by volume descending, then by dimensions for tie-breaking
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: (-x[4], x[1], x[2], x[3]))
    
    # Bins: each bin is a list of placed boxes (x, y, z, w, h, d)
    bins = []
    
    def can_place(bin_boxes, x, y, z, w, h, d):
        """Check if box fits within container and doesn't overlap with existing boxes."""
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
        return True
    
    def find_best_position(bin_boxes, w, h, d):
        """
        Find the best position for a box in a bin.
        Returns (x, y, z) or None if no valid position exists.
        Uses candidate points from box faces and scores by compactness.
        """
        if not bin_boxes:
            if w <= W and h <= H and d <= D:
                return (0, 0, 0)
            return None
        
        # Generate candidate coordinates from existing box faces
        xs = set([0])
        ys = set([0])
        zs = set([0])
        
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            if px + pw <= W:
                xs.add(px + pw)
            if py + ph <= H:
                ys.add(py + ph)
            if pz + pd <= D:
                zs.add(pz + pd)
        
        xs = sorted(xs)
        ys = sorted(ys)
        zs = sorted(zs)
        
        # Limit candidates for performance - keep low values (compact) and some high values
        if len(xs) > 18:
            xs = sorted(set(xs[:13] + xs[-5:]))
        if len(ys) > 18:
            ys = sorted(set(ys[:13] + ys[-5:]))
        if len(zs) > 18:
            zs = sorted(set(zs[:13] + zs[-5:]))
        
        best_score = float('inf')
        best_pos = None
        
        for x in xs:
            if x + w > W:
                continue
            for y in ys:
                if y + h > H:
                    continue
                for z in zs:
                    if z + d > D:
                        continue
                    if can_place(bin_boxes, x, y, z, w, h, d):
                        # Score: prefer compact positions (lower coordinates)
                        # Weight z slightly higher since it's often the limiting dimension
                        score = 0.35*x + 0.35*y + 0.30*z
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
                            # Early exit for performance: first valid in sorted order is often best
                            # But continue a bit to find truly best
                            if score < 1.0:
                                return best_pos
        
        return best_pos
    
    def bin_fragmentation(bin_boxes):
        """
        Estimate fragmentation of a bin.
        Higher fragmentation means more scattered/wasted space.
        """
        if not bin_boxes:
            return 0.0
        # Measure how spread out the boxes are
        max_x = max(px + pw for (px, py, pz, pw, ph, pd) in bin_boxes)
        max_y = max(py + ph for (px, py, pz, pw, ph, pd) in bin_boxes)
        max_z = max(pz + pd for (px, py, pz, pw, ph, pd) in bin_boxes)
        
        # Bounding box volume vs actual volume
        bbox_vol = max_x * max_y * max_z
        actual_vol = sum(pw*ph*pd for (_, _, _, pw, ph, pd) in bin_boxes)
        
        if bbox_vol == 0:
            return 0.0
        return 1.0 - (actual_vol / bbox_vol)
    
    def bin_utilization(bin_boxes):
        """Calculate the volume utilization of a bin."""
        if not bin_boxes:
            return 0.0
        total_vol = sum(pw*ph*pd for (_, _, _, pw, ph, pd) in bin_boxes)
        return total_vol / (W * H * D)
    
    def place_box(w, h, d):
        """
        Place a box in the best existing bin or create a new one.
        Returns (bin_index, x, y, z).
        Uses fragmentation-aware scoring to avoid scattering.
        """
        best_bin = -1
        best_score = float('inf')
        best_pos = None
        
        # Try to fit in existing bins first
        for b_idx, bin_boxes in enumerate(bins):
            if not bin_boxes:
                continue
            pos = find_best_position(bin_boxes, w, h, d)
            if pos is not None:
                x, y, z = pos
                # Combined score: prefer bins with higher utilization, lower fragmentation, and tighter positions
                util = bin_utilization(bin_boxes)
                frag = bin_fragmentation(bin_boxes)
                pos_score = 0.35*x + 0.35*y + 0.30*z
                # We want to minimize pos_score, minimize frag, maximize util
                # score = pos_score + frag_penalty * frag - util_bonus * util
                score = pos_score + 0.8 * frag - 0.6 * util
                if score < best_score:
                    best_score = score
                    best_bin = b_idx
                    best_pos = pos
        
        if best_bin == -1:
            # Create a new bin
            bins.append([])
            best_bin = len(bins) - 1
            best_pos = (0, 0, 0)
        
        x, y, z = best_pos
        bins[best_bin].append((x, y, z, w, h, d))
        return best_bin, x, y, z
    
    # Place boxes in sorted order
    placements = [None] * n
    for idx, w, h, d, vol in indexed_boxes:
        if time.time() > deadline - 5:
            # Emergency: place remaining boxes in new bins (valid but suboptimal)
            for i in range(idx, n):
                _, w2, h2, d2, _ = indexed_boxes[i]
                bin_idx = len(bins)
                bins.append([])
                bins[bin_idx].append((0, 0, 0, w2, h2, d2))
                placements[i] = (bin_idx, 0, 0, 0, w2, h2, d2)
            break
        
        bin_idx, x, y, z = place_box(w, h, d)
        placements[idx] = (bin_idx, x, y, z, w, h, d)
    
    # Local search polish: try to reduce bin count or improve last-bin fill
    if time.time() < deadline - 8:
        if bins:
            # Identify the last bin (highest label)
            last_bin_idx = max(b[0] for b in placements if b is not None)
            
            # Multiple passes of moving boxes to fill last bin better
            for pass_num in range(4):
                if time.time() > deadline - 3:
                    break
                
                last_bin_boxes = bins[last_bin_idx] if last_bin_idx < len(bins) else []
                if not last_bin_boxes:
                    break
                    
                # Try to move small boxes from other bins to last bin
                other_bins = [i for i in range(len(bins)) if i != last_bin_idx and bins[i]]
                if not other_bins:
                    break
                
                moved_any = False
                # Sort other bins by number of boxes (most packed first)
                for ob_idx in sorted(other_bins, key=lambda i: -len(bins[i])):
                    if time.time() > deadline - 2:
                        break
                    if not bins[ob_idx]:
                        continue
                    
                    # Try each box in the other bin (smallest first)
                    box_indices = sorted(range(len(bins[ob_idx])), 
                                       key=lambda j: bins[ob_idx][j][3]*bins[ob_idx][j][4]*bins[ob_idx][j][5])
                    
                    for j in box_indices:
                        if time.time() > deadline - 1:
                            break
                        bx, by, bz, bw, bh, bd = bins[ob_idx][j]
                        
                        # Check if it can fit in last bin
                        pos = find_best_position(last_bin_boxes, bw, bh, bd)
                        if pos is not None:
                            # Move it
                            x, y, z = pos
                            bins[ob_idx].pop(j)
                            last_bin_boxes.append((x, y, z, bw, bh, bd))
                            bins[last_bin_idx] = last_bin_boxes
                            
                            # Update placement
                            for p_idx, p in enumerate(placements):
                                if p is not None and p[0] == ob_idx and p[1] == bx and p[2] == by and p[3] == bz:
                                    placements[p_idx] = (last_bin_idx, x, y, z, bw, bh, bd)
                                    break
                            moved_any = True
                            break
                    if moved_any:
                        break
                
                if not moved_any:
                    break
            
            # Second phase: try to consolidate bins by moving boxes from nearly-empty bins
            if time.time() < deadline - 5:
                for _ in range(2):
                    if time.time() > deadline - 2:
                        break
                    
                    # Find bins with fewest boxes (excluding last bin)
                    non_last_bins = [i for i in range(len(bins)) if i != last_bin_idx and bins[i]]
                    if not non_last_bins:
                        break
                    
                    # Sort by number of boxes (fewest first)
                    non_last_bins.sort(key=lambda i: len(bins[i]))
                    
                    consolidated = False
                    for src_bin in non_last_bins:
                        if time.time() > deadline - 1:
                            break
                        if not bins[src_bin]:
                            continue
                        
                        # Try to move all boxes from this bin to other bins
                        boxes_to_move = list(bins[src_bin])
                        success = True
                        
                        for (bx, by, bz, bw, bh, bd) in boxes_to_move:
                            placed = False
                            # Try to place in last bin first
                            pos = find_best_position(bins[last_bin_idx], bw, bh, bd)
                            if pos is not None:
                                bins[src_bin].remove((bx, by, bz, bw, bh, bd))
                                bins[last_bin_idx].append((pos[0], pos[1], pos[2], bw, bh, bd))
                                # Update placement
                                for p_idx, p in enumerate(placements):
                                    if p is not None and p[0] == src_bin and p[1] == bx and p[2] == by and p[3] == bz:
                                        placements[p_idx] = (last_bin_idx, pos[0], pos[1], pos[2], bw, bh, bd)
                                        break
                                placed = True
                            
                            if not placed:
                                # Try other bins
                                for dst_bin in range(len(bins)):
                                    if dst_bin == src_bin or dst_bin == last_bin_idx:
                                        continue
                                    pos = find_best_position(bins[dst_bin], bw, bh, bd)
                                    if pos is not None:
                                        bins[src_bin].remove((bx, by, bz, bw, bh, bd))
                                        bins[dst_bin].append((pos[0], pos[1], pos[2], bw, bh, bd))
                                        # Update placement
                                        for p_idx, p in enumerate(placements):
                                            if p is not None and p[0] == src_bin and p[1] == bx and p[2] == by and p[3] == bz:
                                                placements[p_idx] = (dst_bin, pos[0], pos[1], pos[2], bw, bh, bd)
                                                break
                                        placed = True
                                        break
                            
                            if not placed:
                                success = False
                                break
                        
                        if success and not bins[src_bin]:
                            consolidated = True
                            break
                    
                    if not consolidated:
                        break
    
    return placements
# EVOLVE-BLOCK-END
