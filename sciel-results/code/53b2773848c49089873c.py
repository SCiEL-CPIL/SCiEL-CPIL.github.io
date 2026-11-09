import time
import itertools

def pack(boxes, container):
    """
    Packs 3D boxes into the minimum number of bins using a First Fit Decreasing (FFD) 
    heuristic with Extreme Point (EP) placement strategy.
    
    Args:
        boxes: List of tuples (w, h, d) representing box dimensions.
        container: Tuple (W, H, D) representing container dimensions.
        
    Returns:
        List of tuples (bin_idx, x, y, z, w, h, d) for each box in the original order.
    """
    start_time = time.time()
    TIME_LIMIT = 110.0  # Leave 10s safety margin
    
    W, H, D = container
    n = len(boxes)
    
    if n == 0:
        return []
        
    # Pre-calculate volumes and sort indices
    # Sort by volume descending, then by max dimension descending
    box_indices = list(range(n))
    def sort_key(i):
        w, h, d = boxes[i]
        vol = w * h * d
        max_dim = max(w, h, d)
        return (-vol, -max_dim)
        
    sorted_indices = sorted(box_indices, key=sort_key)
    
    # Bin representation:
    # Each bin is a dict:
    # 'boxes': list of (w, h, d, x, y, z, orig_idx)
    # 'eps': list of (x, y, z) extreme points
    
    bins = []
    
    def is_feasible(x, y, z, w, h, d, bin_boxes):
        # Check bounds
        if x + w > W or y + h > H or z + d > D:
            return False
        # Check overlap with existing boxes
        for bw, bh, bd, bx, by, bz, _ in bin_boxes:
            # Check if intervals do NOT overlap in any dimension
            if x + w <= bx or x >= bx + bw: continue
            if y + h <= by or y >= by + bh: continue
            if z + d <= bz or z >= bz + bd: continue
            # If we are here, they overlap in all 3 dimensions
            return False
        return True

    def get_new_extreme_points(x, y, z, w, h, d):
        return [
            (x + w, y, z),
            (x, y + h, z),
            (x, y, z + d),
            (x + w, y + h, z),
            (x, y + h, z + d),
            (x + w, y, z + d),
            (x + w, y + h, z + d)
        ]

    def update_extreme_points(bin_data):
        bin_boxes = bin_data['boxes']
        old_eps = bin_data['eps']
        
        # We need to find new EPs.
        # A point is an EP if it is a corner of some placed box and is not strictly inside any other placed box.
        # Also, the origin (0,0,0) is always an EP if valid.
        
        # Candidate points: all corners of all placed boxes.
        candidates = set()
        for bw, bh, bd, bx, by, bz, _ in bin_boxes:
            candidates.add((bx, by, bz))
            candidates.add((bx + bw, by, bz))
            candidates.add((bx, by + bh, bz))
            candidates.add((bx, by, bz + bd))
            candidates.add((bx + bw, by + bh, bz))
            candidates.add((bx, by + bh, bz + bd))
            candidates.add((bx + bw, by, bz + bd))
            candidates.add((bx + bw, by + bh, bz + bd))
            
        # Also keep the old EPs? 
        # Actually, the set of EPs is a subset of the corners of placed boxes.
        # So generating from all corners is sufficient and cleaner.
        
        valid_eps = []
        for p in candidates:
            px, py, pz = p
            # Check if p is strictly inside any box
            is_inside = False
            for bw, bh, bd, bx, by, bz, _ in bin_boxes:
                if bx < px < bx + bw and by < py < by + bh and bz < pz < bz + bd:
                    is_inside = True
                    break
            if not is_inside:
                # Also check bounds? EPs outside container are useless.
                if 0 <= px <= W and 0 <= py <= H and 0 <= pz <= D:
                    valid_eps.append(p)
        
        # Remove dominated points
        # A point p is dominated by q if q.x <= p.x, q.y <= p.y, q.z <= p.z and p != q.
        # We want to keep the "lower-left-front" most points.
        # Sort by x, y, z
        valid_eps.sort()
        
        non_dominated = []
        for p in valid_eps:
            dominated = False
            for q in non_dominated:
                if q[0] <= p[0] and q[1] <= p[1] and q[2] <= p[2]:
                    if q != p:
                        dominated = True
                        break
            if not dominated:
                non_dominated.append(p)
                
        # Final check for duplicates and strict domination in the other direction?
        # The sorted order ensures we check against "smaller" coordinates.
        # If p dominates q (p.x <= q.x etc), then when we processed p, q was not yet in non_dominated.
        # When we process q, we check against p. If p.x <= q.x, p.y <= q.y, p.z <= q.z, then q is dominated by p.
        # So this logic correctly removes dominated points.
        
        bin_data['eps'] = non_dominated

    def place_box_in_bin(bin_idx, box, x, y, z, orig_idx):
        w, h, d = box
        bin_data = bins[bin_idx]
        bin_data['boxes'].append((w, h, d, x, y, z, orig_idx))
        update_extreme_points(bin_data)

    # Main Loop: Place each box
    placements = [None] * n
    
    for idx in sorted_indices:
        w, h, d = boxes[idx]
        
        # Check time
        if time.time() - start_time > TIME_LIMIT:
            # Fallback: place remaining boxes in new bins at 0,0,0 to ensure valid output
            # This is a safety net, though unlikely to be hit if n=40
            for rem_idx in box_indices:
                if placements[rem_idx] is None:
                    rw, rh, rd = boxes[rem_idx]
                    # Find next available bin index
                    used_bins = set(p[0] for p in placements if p is not None)
                    next_bin = max(used_bins) + 1 if used_bins else 0
                    placements[rem_idx] = (next_bin, 0, 0, 0, rw, rh, rd)
            return placements

        placed = False
        
        # Try to fit in existing bins
        best_bin_idx = -1
        best_pos = None
        best_cost = float('inf')
        
        # Heuristic: Try to fill the bin with the most occupied volume first? 
        # Or just try all bins. 
        # To optimize for "last bin utilization", we might prefer filling earlier bins to close them?
        # No, FFD typically just finds the first feasible bin. 
        # However, to minimize B, any feasible bin is good.
        # To maximize last bin utilization, we want to leave the last bin as full as possible.
        # This implies we should prefer placing boxes in *later* bins if they fit? 
        # Actually, standard FFD (first feasible bin) tends to compact the pack into the first bin.
        # This minimizes B. The "last bin" will be the one with the fewest boxes.
        # If we have 2 bins, bin 0 is full, bin 1 has 1 box. Last bin util is low.
        # If we spread out, we might have more bins.
        # So minimizing B is the priority.
        
        for i in range(len(bins)):
            bin_data = bins[i]
            # Try all extreme points
            for x, y, z in bin_data['eps']:
                if is_feasible(x, y, z, w, h, d, bin_data['boxes']):
                    # Cost function:
                    # We want to place the box as close to (0,0,0) as possible to create larger contiguous empty spaces?
                    # Or do we want to "fill" the bin?
                    # Standard EP heuristic: minimize z, then y, then x.
                    cost = z * W * H + y * W + x
                    if cost < best_cost:
                        best_cost = cost
                        best_bin_idx = i
                        best_pos = (x, y, z)
                        
        if best_bin_idx != -1:
            x, y, z = best_pos
            place_box_in_bin(best_bin_idx, (w, h, d), x, y, z, idx)
            placements[idx] = (best_bin_idx, x, y, z, w, h, d)
            placed = True
        else:
            # Create new bin
            bins.append({'boxes': [], 'eps': [(0, 0, 0)]})
            new_bin_idx = len(bins) - 1
            place_box_in_bin(new_bin_idx, (w, h, d), 0, 0, 0, idx)
            placements[idx] = (new_bin_idx, 0, 0, 0, w, h, d)
            placed = True
            
        if not placed:
            # Should not happen
            pass

    # Verify all placements are made
    for i in range(n):
        if placements[i] is None:
            # Emergency placement
            rw, rh, rd = boxes[i]
            used_bins = set(p[0] for p in placements if p is not None)
            next_bin = max(used_bins) + 1 if used_bins else 0
            placements[i] = (next_bin, 0, 0, 0, rw, rh, rd)

    return placements
