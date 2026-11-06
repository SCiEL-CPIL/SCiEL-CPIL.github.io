"""3D Bin Packing with multi-start + local search."""
import time
import random
from collections import defaultdict

def pack(boxes, container):
    """
    Hybrid 3D bin packing:
    - Multiple initial strategies (volume-desc, longest-side, randomized)
    - Best-fit-decreasing with front-point candidate generation
    - Local search: try to move boxes to reduce bin count or improve last-bin fill
    Returns list of (bin_id, x, y, z, w, h, d) for each box in original order.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    t_start = time.time()
    deadline = t_start + 140.0  # safety margin before 150s kill
    
    def get_volume(w, h, d):
        return w * h * d
    
    def check_overlap(placed, x, y, z, w, h, d):
        """Check if a box at (x,y,z) with dims (w,h,d) overlaps with any in placed."""
        for (px, py, pz, pw, ph, pd) in placed:
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return True
        return False
        return False
    
    def get_candidate_positions(placed, w, h, d):
        """Generate candidate (x, y, z) positions for a box in a given bin."""
        x_vals = {0}
        y_vals = {0}
        z_vals = {0}
        
        for (px, py, pz, pw, ph, pd) in placed:
            if px + pw <= W:
                x_vals.add(px + pw)
            if py + ph <= H:
                y_vals.add(py + ph)
            if pz + pd <= D:
                z_vals.add(pz + pd)
        
        # Filter to only valid starting positions
        x_vals = [x for x in x_vals if x + w <= W]
        y_vals = [y for y in y_vals if y + h <= H]
        z_vals = [z for z in z_vals if z + d <= D]
        
        if not x_vals or not y_vals or not z_vals:
            return []
        
        # Sort for deterministic behavior
        x_vals.sort()
        y_vals.sort()
        z_vals.sort()
        
        candidates = []
        for x in x_vals:
            for y in y_vals:
                for z in z_vals:
                    if x + w > W or y + h > H or z + d > D:
                        continue
                    if check_overlap(placed, x, y, z, w, h, d):
                        continue
                    candidates.append((x, y, z))
        
        return candidates
    
    def greedy_pack(order, bins_max=None):
        """
        Greedy best-fit-decreasing with given box order.
        order: list of indices into boxes
        Returns (placements, bins) where placements[i] = (bin_idx, x, y, z) for box i
        and bins is list of lists of (x, y, z, w, h, d)
        """
        bins = []  # list of lists of (x, y, z, w, h, d)
        placement = {}  # original index -> (bin_idx, x, y, z)
        
        for idx in order:
            w, h, d = boxes[idx]
            
            best_bin = -1
            best_x, best_y, best_z = 0, 0, 0
            best_cost = float('inf')
            
            for b_idx in range(len(bins)):
                candidates = get_candidate_positions(bins[b_idx], w, h, d)
                for (cx, cy, cz) in candidates:
                    # Cost: prefer tight fit (minimize sum of coordinates)
                    cost = cx + cy + cz
                    if cost < best_cost:
                        best_cost = cost
                        best_bin = b_idx
                        best_x, best_y, best_z = cx, cy, cz
            
            if best_bin == -1:
                # Create new bin
                bins.append([])
                best_bin = len(bins) - 1
                best_x, best_y, best_z = 0, 0, 0
            
            bins[best_bin].append((best_x, best_y, best_z, w, h, d))
            placement[idx] = (best_bin, best_x, best_y, best_z)
        
        return placement, bins
    
    def compute_score(placement, bins):
        """
        Compute score: mean B + 0.1*(1 - last-bin occupied volume / container volume)
        B = number of occupied bins
        """
        if not bins:
            return 0.0
        
        # Find occupied bins
        occupied_bins = set()
        for idx in placement:
            b_idx = placement[idx][0]
            occupied_bins.add(b_idx)
        
        B = len(occupied_bins)
        
        if B == 0:
            return 0.0
        
        # Last bin = largest label
        last_bin = max(occupied_bins)
        
        # Compute occupied volume in last bin
        last_bin_vol = 0
        for (x, y, z, w, h, d) in bins[last_bin]:
            last_bin_vol += w * h * d
        
        container_vol = W * H * D
        if container_vol > 0:
            last_bin_frac = last_bin_vol / container_vol
        else:
            last_bin_frac = 0.0
        
        score = B + 0.1 * (1 - last_bin_frac)
        return score
    
    def local_search(placement, bins, max_iters=500):
        """
        Local search: try to move boxes to other positions or bins to reduce score.
        Returns improved (placement, bins) or original if no improvement found.
        """
        best_score = compute_score(placement, bins)
        best_placement = {k: v for k, v in placement.items()}
        best_bins = [list(b) for b in bins]
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try moving each box to a better position
            for idx in list(placement.keys()):
                b_idx, x, y, z = placement[idx]
                w, h, d = boxes[idx]
                
                # Remove box from current bin
                # Find and remove from bins[b_idx]
                removed = False
                for i, item in enumerate(bins[b_idx]):
                    if item == (x, y, z, w, h, d):
                        bins[b_idx].pop(i)
                        removed = True
                        break
                
                if not removed:
                    continue
                
                # Try placing in all bins (including current)
                best_new_bin = -1
                best_new_x, best_new_y, best_new_z = x, y, z
                best_new_cost = float('inf')
                
                for nb_idx in range(len(bins)):
                    candidates = get_candidate_positions(bins[nb_idx], w, h, d)
                    for (cx, cy, cz) in candidates:
                        cost = cx + cy + cz
                        if cost < best_new_cost:
                            best_new_cost = cost
                            best_new_bin = nb_idx
                            best_new_x, best_new_y, best_new_z = cx, cy, cz
                
                if best_new_bin == -1:
                    # Can't place anywhere, put back
                    bins[b_idx].append((x, y, z, w, h, d))
                    continue
                
                # Place in new position
                bins[best_new_bin].append((best_new_x, best_new_y, best_new_z, w, h, d))
                placement[idx] = (best_new_bin, best_new_x, best_new_y, best_new_z)
                
                # Check if this improved the score
                new_score = compute_score(placement, bins)
                if new_score < best_score - 1e-9:
                    best_score = new_score
                    best_placement = {k: v for k, v in placement.items()}
                    best_bins = [list(b) for b in bins]
                    improved = True
                else:
                    # Revert
                    # Remove from new bin
                    for i, item in enumerate(bins[best_new_bin]):
                        if item == (best_new_x, best_new_y, best_new_z, w, h, d):
                            bins[best_new_bin].pop(i)
                            break
                    # Put back in original
                    bins[b_idx].append((x, y, z, w, h, d))
                    placement[idx] = (b_idx, x, y, z)
            
            if not improved:
                break
        
        return best_placement, best_bins, best_score
    
    # Generate different orderings
    orderings = []
    
    # 1. Volume descending
    vol_order = sorted(range(n), key=lambda i: -get_volume(*boxes[i]))
    orderings.append(vol_order)
    
    # 2. Longest side descending
    longest_order = sorted(range(n), key=lambda i: -max(boxes[i]))
    orderings.append(longest_order)
    
    # 3. Volume + max dim
    mixed_order = sorted(range(n), key=lambda i: -(get_volume(*boxes[i]) + max(boxes[i])))
    orderings.append(mixed_order)
    
    # 4. Randomized (a few)
    for _ in range(3):
        rand_order = list(range(n))
        random.shuffle(rand_order)
        orderings.append(rand_order)
    
    best_overall_score = float('inf')
    best_overall_placement = None
    best_overall_bins = None
    
    for order in orderings:
        if time.time() > deadline - 20:  # Leave time for local search
            break
        
        placement, bins = greedy_pack(order)
        
        # Run local search
        placement, bins, score = local_search(placement, bins, max_iters=200)
        
        if score < best_overall_score:
            best_overall_score = score
            best_overall_placement = placement
            best_overall_bins = bins
    
    # If we didn't find any valid packing (shouldn't happen), fallback
    if best_overall_placement is None:
        # Simple fallback: one box per bin
        best_overall_placement = {}
        best_overall_bins = []
        for i in range(n):
            w, h, d = boxes[i]
            best_overall_bins.append([(0, 0, 0, w, h, d)])
            best_overall_placement[i] = (i, 0, 0, 0)
    
    # Build result in original order
    result = []
    for i in range(n):
        b_idx, x, y, z = best_overall_placement[i]
        w_i, h_i, d_i = boxes[i]
        result.append((b_idx, x, y, z, w_i, h_i, d_i))
    
    return result
