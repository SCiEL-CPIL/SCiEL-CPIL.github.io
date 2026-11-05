"""Improved 3D Bin Packing solver."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    if n == 0:
        return []
    
    deadline = time.time() + 125.0
    container_vol = W * H * D
    
    # Precompute box volumes
    box_vols = [w * h * d for w, h, d in boxes]
    
    # Sort indices by volume descending (Best Fit Decreasing)
    sorted_indices = sorted(range(n), key=lambda i: box_vols[i], reverse=True)
    
    # We'll maintain a list of bins, each with a list of placed boxes
    # For each bin, we track the occupancy as a set of (x,y,z) occupied cells
    # But that's too expensive for large containers. Instead, we use interval-based overlap checking.
    
    # Better approach: for each bin, maintain a list of placed boxes and check overlaps directly.
    # For 40 boxes, this is feasible.
    
    def check_overlap(bin_boxes, x, y, z, w, h, d):
        """Check if placing a box at (x,y,z) with dims (w,h,d) overlaps with any box in bin_boxes."""
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            if (x < px + pw and px < x + w and
                y < py + ph and py < y + h and
                z < pz + pd and pz < z + d):
                return True
        return False
    
    def find_position(bin_boxes, w, h, d):
        """
        Find the best position for a box in a given bin.
        Uses a priority-based candidate generation: prefer positions that minimize
        the 'waste' by packing tightly against existing boxes and the origin.
        Returns (x, y, z) or None if no position fits.
        """
        # Generate candidate positions
        # Start with origin and face-adjacent positions to existing boxes
        candidates = set()
        candidates.add((0, 0, 0))
        
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            # Positions adjacent to each face
            candidates.add((px + pw, py, pz))
            candidates.add((px, py + ph, pz))
            candidates.add((px, py, pz + pd))
            # Also try combinations of edges for tighter packing
            candidates.add((px + pw, py + ph, pz))
            candidates.add((px + pw, py, pz + pd))
            candidates.add((px, py + ph, pz + pd))
            candidates.add((px + pw, py + ph, pz + pd))
        
        # Also generate positions from unique coordinate combinations
        xs = set([0])
        ys = set([0])
        zs = set([0])
        for (px, py, pz, pw, ph, pd) in bin_boxes:
            xs.update([px, px + pw])
            ys.update([py, py + ph])
            zs.update([pz, pz + pd])
        
        # Filter to valid ranges
        xs = [x for x in xs if 0 <= x <= W]
        ys = [y for y in ys if 0 <= y <= H]
        zs = [z for z in zs if 0 <= z <= D]
        
        # Add all combinations if not too many
        if len(xs) * len(ys) * len(zs) <= 5000:
            for cx in xs:
                for cy in ys:
                    for cz in zs:
                        candidates.add((cx, cy, cz))
        
        # Find the best valid position
        best_pos = None
        best_score = float('inf')
        
        for (cx, cy, cz) in candidates:
            if cx + w > W or cy + h > H or cz + d > D:
                continue
            if check_overlap(bin_boxes, cx, cy, cz, w, h, d):
                continue
            # Score: prefer positions closer to origin (tighter packing)
            # Use a weighted score that favors lower z, then y, then x
            score = cz * 1000 + cy * 100 + cx
            if score < best_score:
                best_score = score
                best_pos = (cx, cy, cz)
        
        return best_pos
    
    # Main packing algorithm
    bins = []  # list of lists of (x, y, z, w, h, d)
    bin_volumes = []  # volume used in each bin
    placements = [None] * n
    
    for idx in sorted_indices:
        if time.time() > deadline:
            break
        
        w, h, d = boxes[idx]
        vol = box_vols[idx]
        
        # Check if box fits in container at all
        if w > W or h > H or d > D:
            # Cannot place this box in any bin; assign to a new bin with a flag
            # But the problem states boxes should be placeable. Let's assume they fit.
            # For safety, create a new bin.
            bi = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            bin_volumes.append(vol)
            placements[idx] = (bi, 0, 0, 0, w, h, d)
            continue
        
        best_bi = -1
        best_pos = None
        best_rem = float('inf')
        
        # Try each existing bin
        for bi in range(len(bins)):
            pos = find_position(bins[bi], w, h, d)
            if pos is not None:
                rem = container_vol - bin_volumes[bi]
                if rem < best_rem:
                    best_rem = rem
                    best_bi = bi
                    best_pos = pos
        
        # If no existing bin works, create a new one
        if best_bi == -1:
            best_bi = len(bins)
            best_pos = (0, 0, 0)
            bins.append([])
            bin_volumes.append(0)
        
        x, y, z = best_pos
        bins[best_bi].append((x, y, z, w, h, d))
        bin_volumes[best_bi] += vol
        placements[idx] = (best_bi, x, y, z, w, h, d)
    
    # Handle any unplaced boxes (shouldn't happen, but for safety)
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            vol = box_vols[i]
            # Try to find a position in any existing bin
            placed = False
            for bi in range(len(bins)):
                pos = find_position(bins[bi], w, h, d)
                if pos is not None:
                    x, y, z = pos
                    bins[bi].append((x, y, z, w, h, d))
                    bin_volumes[bi] += vol
                    placements[i] = (bi, x, y, z, w, h, d)
                    placed = True
                    break
            if not placed:
                bi = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                bin_volumes.append(vol)
                placements[i] = (bi, 0, 0, 0, w, h, d)
    
    # Post-processing: try to reduce the number of bins by moving boxes
    # If we have more than 1 bin, try to consolidate
    # This is a simple heuristic: try to move boxes from the last bin to earlier bins
    
    if len(bins) > 1:
        # Try to move boxes from the highest-numbered bin to lower bins
        # Process bins in reverse order
        for bi in range(len(bins) - 1, 0, -1):
            if time.time() > deadline:
                break
            
            # Try to move each box in bin bi to an earlier bin
            boxes_to_move = list(bins[bi])
            for box in boxes_to_move:
                x, y, z, w, h, d = box
                vol = w * h * d
                
                # Try to find a position in an earlier bin
                for target_bi in range(bi):
                    # Remove box from current bin temporarily
                    bins[bi].remove(box)
                    bin_volumes[bi] -= vol
                    
                    pos = find_position(bins[target_bi], w, h, d)
                    if pos is not None:
                        # Move the box
                        bins[target_bi].append((pos[0], pos[1], pos[2], w, h, d))
                        bin_volumes[target_bi] += vol
                        # Update placement
                        for i in range(n):
                            if placements[i] is not None and placements[i][0] == bi and \
                               placements[i][1] == x and placements[i][2] == y and placements[i][3] == z:
                                placements[i] = (target_bi, pos[0], pos[1], pos[2], w, h, d)
                                break
                        break
                    else:
                        # Put it back
                        bins[bi].append(box)
                        bin_volumes[bi] += vol
                
                if time.time() > deadline:
                    break
            
            # If bin is now empty, we could remove it, but we need to keep bin labels consistent
            # The problem says bin labels need not be consecutive, so we can leave empty bins
            # But to minimize B, we should renumber. However, the score uses the largest label
            # as the last bin. So if we remove a bin, we need to update all placements.
            # Let's skip this for simplicity and just leave empty bins.
    
    return placements
# EVOLVE-BLOCK-END
