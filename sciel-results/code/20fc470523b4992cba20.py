"""3D Bin Packing: Best-fit decreasing with tightness-aware scoring and bin consolidation."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 112.0  # 8s safety margin for 120s limit
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    container_vol = W * H * D
    
    # Sort boxes by volume descending (largest first) for better packing
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: -x[4])
    
    # Bins: each bin is a list of placed boxes (x, y, z, w, h, d)
    bins = []
    
    def can_place(bin_boxes, x, y, z, w, h, d):
        """Check if box fits within container and doesn't overlap with existing boxes."""
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            # No overlap if separated in any dimension
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
        return True
    
    def find_position(bin_boxes, w, h, d):
        """Find the best position for a box in a bin using candidate points with tightness scoring."""
        # Generate candidate positions from all combinations of existing box faces
        xs = [0]
        ys = [0]
        zs = [0]
        
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            if px + pw <= W:
                xs.append(px + pw)
            if py + ph <= H:
                ys.append(py + ph)
            if pz + pd <= D:
                zs.append(pz + pd)
        
        # Remove duplicates and sort
        xs = sorted(set(xs))
        ys = sorted(set(ys))
        zs = sorted(set(zs))
        
        # Score each valid position; prefer positions that minimize coordinates
        # and reward tightness (contact with existing boxes)
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
                        # Base score: prefer lower coordinates (tighter packing toward origin)
                        # Weight z higher since gravity typically stacks downward
                        score = 0.5 * x + 0.5 * y + 1.0 * z
                        
                        # Tightness bonus: reward positions adjacent to existing boxes
                        # This encourages compact packing and reduces fragmentation
                        tightness = 0.0
                        for (px, py, pz, pw, ph, pd) in bin_boxes:
                            # Check adjacency in each dimension
                            if abs(x - (px + pw)) < 1e-9 or abs((x + w) - px) < 1e-9:
                                if not (x + w <= px or px + pw <= x):
                                    if not (y + h <= py or py + ph <= y):
                                        if not (z + d <= pz or pz + pd <= z):
                                            tightness += 1
                            if abs(y - (py + ph)) < 1e-9 or abs((y + h) - py) < 1e-9:
                                if not (y + h <= py or py + ph <= y):
                                    if not (x + w <= px or px + pw <= x):
                                        if not (z + d <= pz or pz + pd <= z):
                                            tightness += 1
                            if abs(z - (pz + pd)) < 1e-9 or abs((z + d) - pz) < 1e-9:
                                if not (z + d <= pz or pz + pd <= z):
                                    if not (x + w <= px or px + pw <= x):
                                        if not (y + h <= py or py + ph <= y):
                                            tightness += 1
                        
                        # Reward tightness: subtract a small amount for each contact
                        score -= 0.1 * tightness
                        
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
        
        return best_pos
    
    def place_box(w, h, d):
        """Place a box in the best existing bin or create a new one."""
        best_bin = -1
        best_score = float('inf')
        best_pos = None
        
        for b_idx, bin_boxes in enumerate(bins):
            pos = find_position(bin_boxes, w, h, d)
            if pos is not None:
                x, y, z = pos
                # Score: prefer positions that minimize wasted space
                # Use the maximum coordinate as a proxy for how "deep" the box is placed
                score = max(x, y, z)
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
        if time.time() > deadline:
            # Emergency: place remaining boxes in a new bin each (valid but suboptimal)
            for i, (w2, h2, d2) in enumerate(boxes[idx:]):
                bin_idx = len(bins)
                bins.append([])
                bins[bin_idx].append((0, 0, 0, w2, h2, d2))
                placements[idx + i] = (bin_idx, 0, 0, 0, w2, h2, d2)
            break
        
        bin_idx, x, y, z = place_box(w, h, d)
        placements[idx] = (bin_idx, x, y, z, w, h, d)
    
    # Bin consolidation pass: try to merge small bins into larger ones to reduce B
    if time.time() < deadline - 5.0:  # Only if we have time
        consolidation_deadline = deadline - 2.0
        max_bins = len(bins)
        
        # Try to move boxes from smaller bins to larger bins
        for target_bin in range(max_bins):
            if time.time() > consolidation_deadline:
                break
            if len(bins[target_bin]) == 0:
                continue
                
            # Try to move each box from other bins to this target bin
            for source_bin in range(max_bins):
                if time.time() > consolidation_deadline:
                    break
                if source_bin == target_bin or len(bins[source_bin]) == 0:
                    continue
                    
                # Try to move boxes from source to target
                moved_any = False
                boxes_to_move = []
                for box_info in bins[source_bin]:
                    bx, by, bz, bw, bh, bd = box_info
                    # Try to find a position in target bin
                    pos = find_position(bins[target_bin], bw, bh, bd)
                    if pos is not None:
                        boxes_to_move.append((box_info, pos))
                
                if boxes_to_move:
                    # Move all boxes that can fit
                    for box_info, pos in boxes_to_move:
                        bx, by, bz, bw, bh, bd = box_info
                        px, py, pz = pos
                        bins[target_bin].append((px, py, pz, bw, bh, bd))
                    bins[source_bin] = []
                    moved_any = True
        
        # Remove empty bins and renumber
        new_bins = [b for b in bins if len(b) > 0]
        bin_remap = {}
        for old_idx, b in enumerate(bins):
            if len(b) > 0:
                bin_remap[old_idx] = new_bins.index(b)
        
        # Update placements with new bin indices
        for i in range(n):
            if placements[i] is not None:
                old_bin = placements[i][0]
                if old_bin in bin_remap:
                    placements[i] = (bin_remap[old_bin],) + placements[i][1:]
        
        bins = new_bins
    
    return placements
# EVOLVE-BLOCK-END
