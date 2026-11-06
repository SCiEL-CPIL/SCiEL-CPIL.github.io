"""3D Bin Packing: Best-fit decreasing with tightness-aware positioning and consolidation polish."""
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
    
    # Sort boxes by volume descending, then by max dimension for tie-breaking
    indexed_boxes = [(i, w, h, d, w*h*d, max(w,h,d)) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: (-x[4], x[5], x[1], x[2], x[3]))
    
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
    
    def position_tightness(bin_boxes, x, y, z, w, h, d):
        """
        Score how tightly a position is packed against existing boxes.
        Lower score = tighter fit. Rewards contact with existing boxes on multiple faces.
        """
        if not bin_boxes:
            return 0.0
        
        # Base score: distance from origin (prefer compact packing)
        score = 0.5 * (x + y + z)
        
        # Reward contact with existing boxes (tighter packing)
        contact_score = 0.0
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            # Check if box is adjacent (touching) to any existing box
            # Adjacent in x: x == px+pw or x+w == px
            # Adjacent in y: y == py+ph or y+h == py
            # Adjacent in z: z == pz+pd or z+d == pz
            touches = 0
            if abs(x - (px + pw)) < 1e-9 or abs(x + w - px) < 1e-9:
                touches += 1
            if abs(y - (py + ph)) < 1e-9 or abs(y + h - py) < 1e-9:
                touches += 1
            if abs(z - (pz + pd)) < 1e-9 or abs(z + d - pz) < 1e-9:
                touches += 1
            if touches > 0:
                contact_score += touches
        
        # More contacts = better (lower score)
        score -= 0.1 * contact_score
        
        return score
    
    def find_best_position(bin_boxes, w, h, d):
        """
        Find the best position for a box in a bin.
        Returns (x, y, z) or None if no valid position exists.
        Uses candidate points from box faces and scores by tightness.
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
        
        # Limit candidates for performance
        if len(xs) > 20:
            xs = xs[:20]
        if len(ys) > 20:
            ys = ys[:20]
        if len(zs) > 20:
            zs = zs[:20]
        
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
                        score = position_tightness(bin_boxes, x, y, z, w, h, d)
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
        
        return best_pos
    
    def bin_utilization(bin_boxes):
        """Calculate the volume utilization of a bin."""
        if not bin_boxes:
            return 0.0
        total_vol = sum(pw*ph*pd for (_, _, _, pw, ph, pd) in bin_boxes)
        return total_vol / (W * H * D)
    
    def bin_fragmentation(bin_boxes):
        """
        Estimate fragmentation: how spread out the boxes are.
        Lower = more compact. Uses bounding box volume / actual volume ratio.
        """
        if not bin_boxes:
            return 0.0
        if len(bin_boxes) == 1:
            return 0.0
        
        min_x = min(b[0] for b in bin_boxes)
        min_y = min(b[1] for b in bin_boxes)
        min_z = min(b[2] for b in bin_boxes)
        max_x = max(b[0]+b[3] for b in bin_boxes)
        max_y = max(b[1]+b[4] for b in bin_boxes)
        max_z = max(b[2]+b[5] for b in bin_boxes)
        
        bbox_vol = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)
        actual_vol = sum(b[3]*b[4]*b[5] for b in bin_boxes)
        
        if actual_vol == 0:
            return 0.0
        return bbox_vol / actual_vol - 1.0  # 0 = perfect, higher = more fragmented
    
    def place_box(w, h, d):
        """
        Place a box in the best existing bin or create a new one.
        Returns (bin_index, x, y, z).
        """
        best_bin = -1
        best_score = float('inf')
        best_pos = None
        
        # Try to fit in existing bins first
        for b_idx, bin_boxes in enumerate(bins):
            pos = find_best_position(bin_boxes, w, h, d)
            if pos is not None:
                x, y, z = pos
                # Combined score: prefer bins with higher utilization, lower fragmentation, tighter positions
                util = bin_utilization(bin_boxes)
                frag = bin_fragmentation(bin_boxes)
                pos_tight = position_tightness(bin_boxes, x, y, z, w, h, d)
                
                # We want: high util, low frag, low pos_tight
                # Score to minimize: pos_tight + 0.5*frag - 0.3*util
                score = pos_tight + 0.5 * frag - 0.3 * util
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
    for idx, w, h, d, vol, maxdim in indexed_boxes:
        if time.time() > deadline:
            # Emergency: place remaining boxes in new bins (valid but suboptimal)
            for i in range(idx, n):
                _, w2, h2, d2, _, _ = indexed_boxes[i]
                bin_idx = len(bins)
                bins.append([])
                bins[bin_idx].append((0, 0, 0, w2, h2, d2))
                placements[i] = (bin_idx, 0, 0, 0, w2, h2, d2)
            break
        
        bin_idx, x, y, z = place_box(w, h, d)
        placements[idx] = (bin_idx, x, y, z, w, h, d)
    
    # Final polish: consolidate boxes into fewer bins and improve last bin occupancy
    # This helps with both B (bin count) and the 0.1*(1 - last-bin fraction) term
    if time.time() < deadline - 10 and len(bins) > 1:
        # Try to merge bins: move all boxes from one bin into others
        # Start with the smallest bins (least boxes)
        bin_order = sorted(range(len(bins)), key=lambda i: len(bins[i]))
        
        for src_idx in bin_order:
            if time.time() > deadline - 5:
                break
            if not bins[src_idx]:
                continue
            
            # Try to move all boxes from src_idx to other bins
            src_boxes = list(bins[src_idx])
            src_boxes.sort(key=lambda b: b[3]*b[4]*b[5], reverse=True)  # Largest first
            
            all_moved = True
            for (sx, sy, sz, sw, sh, sd) in src_boxes:
                moved = False
                for dst_idx in range(len(bins)):
                    if dst_idx == src_idx or not bins[dst_idx]:
                        continue
                    pos = find_best_position(bins[dst_idx], sw, sh, sd)
                    if pos is not None:
                        x, y, z = pos
                        bins[dst_idx].append((x, y, z, sw, sh, sd))
                        # Update placement
                        for p_idx, p in enumerate(placements):
                            if p is not None and p[0] == src_idx and p[1] == sx and p[2] == sy and p[3] == sz:
                                placements[p_idx] = (dst_idx, x, y, z, sw, sh, sd)
                                break
                        moved = True
                        break
                if not moved:
                    all_moved = False
                    break
            
            if all_moved:
                bins[src_idx] = []  # Mark as empty
    
    # Second polish: fill the last bin more by moving small boxes from other bins
    if time.time() < deadline - 3 and bins:
        # Find the actual last bin (highest label with boxes)
        occupied_bins = [i for i in range(len(bins)) if bins[i]]
        if occupied_bins:
            last_bin_idx = max(occupied_bins)
            last_bin_boxes = bins[last_bin_idx]
            
            other_bins = [i for i in occupied_bins if i != last_bin_idx]
            
            # Try to move small boxes to last bin to increase its occupancy
            for _ in range(5):  # Multiple rounds
                if time.time() > deadline - 1:
                    break
                moved = False
                for ob_idx in sorted(other_bins, key=lambda i: len(bins[i])):
                    if time.time() > deadline - 1:
                        break
                    if not bins[ob_idx]:
                        continue
                    # Try smallest boxes first
                    box_indices = sorted(range(len(bins[ob_idx])), 
                                       key=lambda j: bins[ob_idx][j][3]*bins[ob_idx][j][4]*bins[ob_idx][j][5])
                    for j in box_indices:
                        bx, by, bz, bw, bh, bd = bins[ob_idx][j]
                        pos = find_best_position(last_bin_boxes, bw, bh, bd)
                        if pos is not None:
                            x, y, z = pos
                            bins[ob_idx].pop(j)
                            bins[last_bin_idx].append((x, y, z, bw, bh, bd))
                            # Update placement
                            for p_idx, p in enumerate(placements):
                                if p is not None and p[0] == ob_idx and p[1] == bx and p[2] == by and p[3] == bz:
                                    placements[p_idx] = (last_bin_idx, x, y, z, bw, bh, bd)
                                    break
                            moved = True
                            break
                    if moved:
                        break
                if not moved:
                    break
    
    return placements
# EVOLVE-BLOCK-END
