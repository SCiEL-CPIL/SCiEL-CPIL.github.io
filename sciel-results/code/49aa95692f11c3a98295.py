"""3D Bin Packing with Best-Fit Decreasing + 2D skyline per bin."""
# EVOLVE-BLOCK-START
import time
import math
import random
from functools import lru_cache

def pack(boxes, container):
    """
    Pack boxes into numbered bins. Returns list of (bin, x, y, z, w, h, d).
    Score: mean B + 0.1*(1 - last-bin occupied volume / container volume), lower is better.
    The returned list must correspond to the input boxes by index.
    """
    W, H, D = container
    container_vol = W * H * D
    n = len(boxes)
    
    if n == 0:
        return []
    
    # Sort indices by volume descending (best-fit decreasing)
    idx_sorted = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    # Bin state: list of lists of (x, y, z, w, h, d)
    bins = []  # bins[b] = list of placements in that bin
    # Track original index for each placement to restore order later
    # box_placements[i] will hold (bin, x, y, z, w, h, d) for original box i
    box_placements = [None] * n
    
    def can_place(bin_idx, x, y, z, w, h, d):
        """Check if box fits at (x,y,z) in bin bin_idx without overlap."""
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd) in bins[bin_idx]:
            # Check overlap
            if (x < px + pw and px < x + w and
                y < py + ph and py < y + h and
                z < pz + pd and pz < z + d):
                return False
        return True
    
    def find_position(bin_idx, w, h, d):
        """Find the best (x,y,z) position for a box in the given bin.
        Uses a heuristic: try corners generated from existing box boundaries."""
        # Generate candidate x, y, z positions
        xs = set([0])
        ys = set([0])
        zs = set([0])
        for (px, py, pz, pw, ph, pd) in bins[bin_idx]:
            xs.add(px + pw)
            ys.add(py + ph)
            zs.add(pz + pd)
        
        # Sort candidates to prefer lower positions (compact packing)
        xs = sorted([x for x in xs if x + w <= W])
        ys = sorted([y for y in ys if y + h <= H])
        zs = sorted([z for z in zs if z + d <= D])
        
        best_pos = None
        best_score = None
        for x in xs:
            for y in ys:
                for z in zs:
                    if can_place(bin_idx, x, y, z, w, h, d):
                        # Score: prefer positions that are lower in z, then y, then x
                        # Use a weighted sum to prefer compactness
                        s = z * 1000 + y * 100 + x
                        if best_score is None or s < best_score:
                            best_score = s
                            best_pos = (x, y, z)
        return best_pos
    
    def bin_volume(bin_idx):
        """Total volume of boxes in a bin."""
        return sum(pw*ph*pd for (_, _, _, pw, ph, pd) in bins[bin_idx])
    
    # Place each box
    for i in idx_sorted:
        w, h, d = boxes[i]
        placed = False
        
        # Try existing bins first, in order of decreasing fill (best fit)
        if bins:
            # Sort bin indices by current volume descending
            bin_order = sorted(range(len(bins)), key=lambda b: bin_volume(b), reverse=True)
            for b in bin_order:
                pos = find_position(b, w, h, d)
                if pos is not None:
                    bins[b].append((pos[0], pos[1], pos[2], w, h, d))
                    box_placements[i] = (b, pos[0], pos[1], pos[2], w, h, d)
                    placed = True
                    break
        
        if not placed:
            # Open a new bin
            b_idx = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            box_placements[i] = (b_idx, 0, 0, 0, w, h, d)
    
    # Build internal representation for local search: (bin, x, y, z, w, h, d, orig_idx)
    internal_placements = []
    for i in range(n):
        p = box_placements[i]
        if p is not None:
            internal_placements.append((p[0], p[1], p[2], p[3], p[4], p[5], p[6], i))
    
    def compute_score(placement_list):
        """Compute score from a placement list (8-tuples)."""
        if not placement_list:
            return 0.0
        bin_labels = set(p[0] for p in placement_list)
        B = len(bin_labels)
        max_bin = max(bin_labels)
        last_vol = sum(p[4]*p[5]*p[6] for p in placement_list if p[0] == max_bin)
        fraction = last_vol / container_vol if container_vol > 0 else 0
        return B + 0.1 * (1 - fraction)
    
    current_score = compute_score(internal_placements)
    
    # Local search
    deadline = time.time() + 100  # Leave 20s margin for 120s limit
    best_result = internal_placements[:]
    best_score = current_score
    
    # Build a mutable representation
    # placements[i] = (bin, x, y, z, w, h, d, orig_idx)
    placements = internal_placements[:]
    
    def rebuild_bins(pl):
        """Rebuild bins from placement list."""
        b = {}
        for p in pl:
            bidx = p[0]
            if bidx not in b:
                b[bidx] = []
            b[bidx].append((p[1], p[2], p[3], p[4], p[5], p[6]))
        return b
    
    max_iterations = 5000
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Pick a random box
        i = random.randint(0, len(placements) - 1)
        b, x, y, z, w, h, d, orig_idx = placements[i]
        
        # Get all existing bin labels
        existing_bins = set(p[0] for p in placements if p[0] != b)
        
        if not existing_bins:
            continue
        
        # Try moving to each existing bin
        for target_b in existing_bins:
            # Try to find a position in target_b
            tb = rebuild_bins(placements)
            if target_b not in tb:
                continue
            
            # Generate candidate positions in target_b
            xs = set([0])
            ys = set([0])
            zs = set([0])
            for (px, py, pz, pw, ph, pd) in tb[target_b]:
                xs.add(px + pw)
                ys.add(py + ph)
                zs.add(pz + pd)
            
            xs = sorted([xx for xx in xs if xx + w <= W])
            ys = sorted([yy for yy in ys if yy + h <= H])
            zs = sorted([zz for zz in zs if zz + d <= D])
            
            found = False
            for xx in xs:
                if found:
                    break
                for yy in ys:
                    if found:
                        break
                    for zz in zs:
                        # Check overlap
                        overlap = False
                        for (px, py, pz, pw, ph, pd) in tb[target_b]:
                            if (xx < px + pw and px < xx + w and
                                yy < py + ph and py < yy + h and
                                zz < pz + pd and pz < zz + d):
                                overlap = True
                                break
                        if not overlap:
                            # Compute new score
                            new_pl = placements[:]
                            new_pl[i] = (target_b, xx, yy, zz, w, h, d, orig_idx)
                            new_score = compute_score(new_pl)
                            if new_score < best_score - 1e-9:
                                best_score = new_score
                                best_result = new_pl[:]
                                placements = new_pl
                                found = True
                                break
            if found:
                break
    
    # Construct final result in original order
    final_result = [None] * n
    for p in best_result:
        orig_idx = p[7]
        final_result[orig_idx] = (p[0], p[1], p[2], p[3], p[4], p[5], p[6])
        
    return final_result
# EVOLVE-BLOCK-END
