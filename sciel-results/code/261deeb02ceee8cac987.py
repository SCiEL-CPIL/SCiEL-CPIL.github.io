"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

# EVOLVE-BLOCK-START
def pack(boxes, container):
    W, H, D = container
    n = len(boxes)
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Sort boxes by volume descending, then by max dimension, for better packing
    indexed = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed.sort(key=lambda x: (-x[4], -max(x[1], x[2], x[3])))
    
    # We'll try to place boxes one by one into bins
    # Each bin maintains a list of placed boxes: (x, y, z, w, h, d)
    bins = []  # list of lists of (x,y,z,w,h,d)
    placement = [None] * n  # (bin_idx, x, y, z) for each original box
    
    def can_place(bin_list, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in bin_list:
            # Check overlap
            if (x < px + pw and x + w > px and
                y < py + ph and y + h > py and
                z < pz + pd and z + d > pz):
                return False
        return True
    
    def find_position(bin_list, w, h, d):
        """Find the best (lowest) position for a box in this bin."""
        # Candidate positions: try to place at origin first, then next to existing boxes
        # For efficiency, generate candidate x, y, z positions
        # x candidates: 0, and for each placed box, px+pw
        # y candidates: 0, and for each placed box, py+ph
        # z candidates: 0, and for each placed box, pz+pd
        
        x_cands = set([0])
        y_cands = set([0])
        z_cands = set([0])
        for (px, py, pz, pw, ph, pd) in bin_list:
            x_cands.add(px + pw)
            y_cands.add(py + ph)
            z_cands.add(pz + pd)
        
        # Sort candidates to prefer lower positions
        x_cands = sorted([x for x in x_cands if x + w <= W])
        y_cands = sorted([y for y in y_cands if y + h <= H])
        z_cands = sorted([z for z in z_cands if z + d <= D])
        
        best_pos = None
        best_score = None
        for x in x_cands:
            for y in y_cands:
                for z in z_cands:
                    if can_place(bin_list, x, y, z, w, h, d):
                        # Score: prefer lower z, then lower y, then lower x
                        score = (z, y, x)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
                        break
                else:
                    continue
                break
            else:
                continue
            break
        
        # If no position found from candidate set, try a more exhaustive search
        if best_pos is None:
            # Try all integer positions? That's too expensive.
            # Instead, try positions where x, y, z are at 0 or at edges of existing boxes
            # Let's be smarter: for each dimension, candidates are 0 and right/top/back of existing boxes
            # But we already did that. Let's try a grid search with step 1 if bin is small
            # Actually, let's just return None if no candidate works
            pass
        
        return best_pos
    
    # Greedy Best Fit Decreasing
    for (orig_idx, w, h, d, vol) in indexed:
        if time.time() > deadline:
            break
        
        placed = False
        # Try existing bins first (best fit: least remaining space or lowest position)
        best_bin_idx = None
        best_pos = None
        best_score = None
        
        for bi, bin_list in enumerate(bins):
            pos = find_position(bin_list, w, h, d)
            if pos is not None:
                x, y, z = pos
                # Score for bin selection: prefer bin where box is placed lowest (z, y, x)
                # Also prefer bins that are already "fuller" to minimize bin count
                bin_vol = sum(pw*ph*pd for (_,_,_,pw,ph,pd) in bin_list)
                score = (z, y, x, bin_vol)  # lower z,y,x is better; higher bin_vol means we prefer it? 
                # Actually, we want to minimize bins, so prefer placing in existing bins.
                # Among existing bins, prefer the one where the box fits "best" (lowest position)
                if best_score is None or (z, y, x) < (best_score[0], best_score[1], best_score[2]):
                    best_score = (z, y, x)
                    best_bin_idx = bi
                    best_pos = pos
        
        if best_bin_idx is not None:
            x, y, z = best_pos
            bins[best_bin_idx].append((x, y, z, w, h, d))
            placement[orig_idx] = (best_bin_idx, x, y, z)
            placed = True
        else:
            # Try a new bin
            new_bin = []
            pos = find_position(new_bin, w, h, d)
            if pos is not None:
                x, y, z = pos
                new_bin.append((x, y, z, w, h, d))
                bins.append(new_bin)
                placement[orig_idx] = (len(bins)-1, x, y, z)
                placed = True
        
        if not placed:
            # Box doesn't fit in any bin (shouldn't happen if single box fits in container)
            # Place in a new bin at origin
            bins.append([(0, 0, 0, w, h, d)])
            placement[orig_idx] = (len(bins)-1, 0, 0, 0)
    
    # Check if all boxes placed
    if any(p is None for p in placement):
        # Fallback: place remaining boxes in new bins
        for i, p in enumerate(placement):
            if p is None:
                w, h, d = boxes[i]
                bins.append([(0, 0, 0, w, h, d)])
                placement[i] = (len(bins)-1, 0, 0, 0)
    
    # Build result
    result = []
    for i in range(n):
        bi, x, y, z = placement[i]
        w, h, d = boxes[i]
        result.append((bi, x, y, z, w, h, d))
    
    return result
# EVOLVE-BLOCK-END
