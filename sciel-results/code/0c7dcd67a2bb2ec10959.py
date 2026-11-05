"""
Strategy: Multi-Ordering Best-Fit 3D Bin Packing.
Instead of a single greedy pass, we generate multiple orderings of the boxes
(e.g., sorted by volume, by max dimension, random shuffles) and apply a 
Best-Fit Decreasing (BFD) heuristic to each. We keep the solution with the 
best score. This explores the solution space more effectively than a 
single fixed-order shelf packing.
"""
import time
import random
import math

# Safety margin for the 150s limit. We stop searching at 140s.
DEADLINE = time.time() + 140.0

def _fits(placements, cx, cy, cz, w, h, d, W, H, D):
    """Check if a box fits at position (cx, cy, cz) without overlapping."""
    if cx < 0 or cy < 0 or cz < 0:
        return False
    if cx + w > W or cy + h > H or cz + d > D:
        return False
    for (px, py, pz, pw, ph, pd) in placements:
        # Check overlap in all 3 dimensions
        if (cx < px + pw and px < cx + w and
            cy < py + ph and py < cy + h and
            cz < pz + pd and pz < cz + d):
            return False
    return True

def _get_anchors(placements, W, H, D):
    """
    Generate candidate positions (anchors) for placing a new box.
    We use the "corner" method: the bottom-front-left corner of each existing
    box, plus the origin, plus the faces of the container.
    To keep it efficient, we only consider positions that are "supported" 
    (touching the floor, a wall, or another box).
    """
    anchors = set()
    # Origin
    anchors.add((0, 0, 0))
    
    # Faces of the container (for boxes that might slide to the back/top)
    # Note: Standard BFD usually just uses corners of existing boxes.
    # Adding container faces helps if boxes are small and can be pushed back.
    # However, for simplicity and speed, we primarily rely on box corners.
    
    for (px, py, pz, pw, ph, pd) in placements:
        # Corner of the box extending in +x
        anchors.add((px + pw, py, pz))
        # Corner of the box extending in +y
        anchors.add((px, py + ph, pz))
        # Corner of the box extending in +z
        anchors.add((px, py, pz + pd))
        
        # Other corners that might be valid if the box is surrounded
        # But usually, the "front" faces are the most important for BFD.
        # Let's add the 8 corners of each box to be safe, though it increases cost.
        # Actually, just the 3 "positive" corners + origin is often sufficient 
        # for BFD if we sort candidates by (z, y, x) to prefer lower positions.
        
        # Let's add all 8 corners to ensure we don't miss a spot in a "hole"
        for dx in [0, pw]:
            for dy in [0, ph]:
                for dz in [0, pd]:
                    ax, ay, az = px + dx, py + dy, pz + dz
                    if ax <= W and ay <= H and az <= D:
                        anchors.add((ax, ay, az))

    return list(anchors)

def _pack_best_fit(boxes, container, order):
    """
    Pack boxes in the given order using Best-Fit Decreasing heuristic.
    Returns a list of placements (bin_id, x, y, z, w, h, d) or None if failed.
    """
    W, H, D = container
    n = len(boxes)
    
    # bins: list of lists of placements (x, y, z, w, h, d)
    bins = []
    # To speed up fitting, we might need to sort anchors. 
    # We want to prefer lower Z, then lower Y, then lower X.
    
    for idx in order:
        w, h, d = boxes[idx]
        placed = False
        
        # 1. Try to fit in existing bins
        # Sort bins by current volume or just iterate? 
        # Best-fit usually means: find the bin where the box fits with minimal 
        # "wasted" space or just the first bin that fits? 
        # Standard BFD: Fit in the bin that leaves the least remaining space?
        # Simpler: Try bins in order of creation, or sort bins by "fullness"?
        # Let's try bins in order of creation (FIFO) but check all anchors.
        # To improve, we can sort bins by their current max Z (or volume) 
        # to try "fuller" bins first? Or just iterate.
        
        # Optimization: If a bin is "full" (no space), skip.
        # We'll just iterate through all bins.
        
        best_bin_idx = -1
        best_pos = None
        best_score = float('inf')
        
        for b_idx, bin_placements in enumerate(bins):
            anchors = _get_anchors(bin_placements, W, H, D)
            # Sort anchors to prefer lower positions (Z, Y, X)
            # This makes the packing more compact and deterministic
            anchors.sort(key=lambda a: (a[2], a[1], a[0]))
            
            for (ax, ay, az) in anchors:
                if _fits(bin_placements, ax, ay, az, w, h, d, W, H, D):
                    # Calculate a "score" for this position.
                    # We want to minimize the "height" used (Z) to keep the bin flat?
                    # Or minimize the distance from origin?
                    # Let's use (az, ay, ax) as the tie-breaker/score.
                    score = (az, ay, ax)
                    if score < best_score:
                        best_score = score
                        best_bin_idx = b_idx
                        best_pos = (ax, ay, az)
                    # Since anchors are sorted, the first valid one is the best for this bin.
                    # We can break here if we only care about the best position in THIS bin.
                    # But we need to compare across bins.
                    # Actually, if we just want *a* valid spot, the first one in sorted order is best.
                    # But "Best Fit" might imply filling holes.
                    # Let's stick to: Find the globally lowest (z,y,x) position across all bins.
                    break 
        
        if best_bin_idx != -1:
            ax, ay, az = best_pos
            bins[best_bin_idx].append((ax, ay, az, w, h, d))
            placed = True
        else:
            # 2. Try to fit in a new bin
            # Check if it fits in an empty container
            if w <= W and h <= H and d <= D:
                bins.append([(0, 0, 0, w, h, d)])
                placed = True
        
        if not placed:
            return None # Box too large for container

    # Construct final placement list
    # We need to return placements in the order of the original boxes?
    # The task says: "preserving its original ordered dimensions... placements[i] corresponds to boxes[i]"
    # Wait, the inspiration said: "Fixing the return order so that placements[i] corresponds to boxes[i] resolves the orientation error"
    # The current seed program returns placements in the order they were processed.
    # If I process in a different order (e.g., sorted by volume), I must map back to original indices.
    
    # Let's re-structure: The input `boxes` is a list. `order` is a list of indices.
    # We need to output a list `result` where `result[i]` is the placement for `boxes[i]`.
    
    # My `_pack_best_fit` above just packs. I need to track which box went where.
    # Let's modify `_pack_best_fit` to return a dict: index -> (bin, x, y, z)
    
    pass # Placeholder, logic below

def _pack_best_fit_tracked(boxes, container, order):
    W, H, D = container
    bins = [] # List of lists of (x, y, z, w, h, d, original_idx)
    
    for idx in order:
        w, h, d = boxes[idx]
        placed = False
        best_bin_idx = -1
        best_pos = None
        best_score = float('inf')
        
        for b_idx, bin_placements in enumerate(bins):
            # Get anchors
            anchors = set()
            anchors.add((0, 0, 0))
            for (px, py, pz, pw, ph, pd, _) in bin_placements:
                for dx in [0, pw]:
                    for dy in [0, ph]:
                        for dz in [0, pd]:
                            ax, ay, az = px + dx, py + dy, pz + dz
                            if ax <= W and ay <= H and az <= D:
                                anchors.add((ax, ay, az))
            
            # Sort anchors by (z, y, x) to prefer compact packing
            sorted_anchors = sorted(anchors, key=lambda a: (a[2], a[1], a[0]))
            
            for (ax, ay, az) in sorted_anchors:
                if _fits([(p[0], p[1], p[2], p[3], p[4], p[5]) for p in bin_placements], ax, ay, az, w, h, d, W, H, D):
                    score = (az, ay, ax)
                    if score < best_score:
                        best_score = score
                        best_bin_idx = b_idx
                        best_pos = (ax, ay, az)
                    break # First valid anchor in sorted list is best for this bin
            
        if best_bin_idx != -1:
            ax, ay, az = best_pos
            bins[best_bin_idx].append((ax, ay, az, w, h, d, idx))
            placed = True
        else:
            if w <= W and h <= H and d <= D:
                bins.append([(0, 0, 0, w, h, d, idx)])
                placed = True
        
        if not placed:
            return None

    # Map back to original indices
    result = [None] * len(boxes)
    for b_idx, bin_placements in enumerate(bins):
        for (x, y, z, w, h, d, idx) in bin_placements:
            result[idx] = (b_idx, x, y, z, w, h, d)
            
    # Ensure all are placed
    if any(r is None for r in result):
        return None
        
    return result

def _score_solution(placements, boxes, container):
    if not placements:
        return float('inf')
    
    W, H, D = container
    container_vol = W * H * D
    
    # Count bins
    bins = set()
    last_bin_vol = 0
    last_bin_id = -1
    
    for (b, x, y, z, w, h, d) in placements:
        bins.add(b)
        if b > last_bin_id:
            last_bin_id = b
            last_bin_vol = 0 # Reset? No, we need to sum volumes for the max bin ID
    
    # Calculate volume of the last bin (highest ID)
    # Note: Bin IDs might not be consecutive if we skipped some? 
    # In our algo, we start at 0 and increment, so they are consecutive 0..B-1.
    # But the task says "largest label defines the last bin".
    
    for (b, x, y, z, w, h, d) in placements:
        if b == last_bin_id:
            last_bin_vol += w * h * d
            
    B = len(bins)
    
    if container_vol == 0:
        return float('inf')
        
    occupied_fraction = last_bin_vol / container_vol
    score = B + 0.1 * (1 - occupied_fraction)
    
    return score

def _generate_orderings(boxes, n_orders=50):
    """Generate diverse orderings of box indices."""
    n = len(boxes)
    orderings = []
    
    # 1. Volume descending
    vol_order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    orderings.append(vol_order)
    
    # 2. Max dimension descending
    max_dim_order = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    orderings.append(max_dim_order)
    
    # 3. Surface area descending
    sa_order = sorted(range(n), key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True)
    orderings.append(sa_order)
    
    # 4. Original order
    orderings.append(list(range(n)))
    
    # 5. Random shuffles
    rng = random.Random(42)
    for _ in range(n_orders - 4):
        order = list(range(n))
        rng.shuffle(order)
        orderings.append(order)
        
    return orderings

def pack(boxes, container):
    start_time = time.time()
    
    # Generate orderings
    orderings = _generate_orderings(boxes, n_orders=100)
    
    best_score = float('inf')
    best_placement = None
    
    # Try each ordering
    for order in orderings:
        if time.time() > DEADLINE:
            break
            
        placements = _pack_best_fit_tracked(boxes, container, order)
        
        if placements is None:
            continue
            
        score = _score_solution(placements, boxes, container)
        
        if score < best_score:
            best_score = score
            best_placement = placements
            
            # Early exit if we achieve a perfect score? 
            # Min B is 1. Min penalty is 0. Score 1.0 is likely good.
            # But we can't know the theoretical min without solving.
            # If B=1 and last bin is full, score = 1.0.
            # If B=1 and last bin is 90% full, score = 1.01.
            # We'll just keep searching until deadline.
            
    # If no placement found (shouldn't happen if boxes fit in container), return empty or fail?
    # The task implies boxes can be packed.
    if best_placement is None:
        # Fallback: Try original order with simple shelf? 
        # Or just return empty? 
        # Let's try a simple sequential pack as fallback
        W, H, D = container
        placements = []
        b = 0
        x, y, z = 0, 0, 0
        row_h, layer_d = 0, 0
        for i, (w, h, d) in enumerate(boxes):
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d = 0, z + layer_d, 0
                x, row_h = 0, 0
            if z + d > D:
                b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
            placements.append((b, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        return placements

    return best_placement
