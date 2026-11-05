"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _check_overlap(x, y, z, w, h, d, placements):
    """Fast overlap check using interval arithmetic."""
    for (_, ex, ey, ez, ew, eh, ed) in placements:
        if x < ex + ew and ex < x + w and \
           y < ey + eh and ey < y + h and \
           z < ez + ed and ez < z + d:
            return True
    return False

def _find_position(box, placements, W, H, D, strategy='lowest'):
    """Find a valid position for box in given placements.
    
    Uses candidate positions from box boundaries + corner combinations.
    Strategy: 'lowest' = minimize z, then y, then x
              'bestfit' = minimize max remaining dimension
              'corner' = prefer corner positions (maximize contact with boundaries)
    """
    w, h, d = box
    if w > W or h > H or d > D:
        return None
    
    # Build candidate positions from existing box boundaries
    xs = {0}
    ys = {0}
    zs = {0}
    for (_, x, y, z, bw, bh, bd) in placements:
        if x + bw <= W:
            xs.add(x + bw)
        if y + bh <= H:
            ys.add(y + bh)
        if z + bd <= D:
            zs.add(z + bd)
    
    # Also add positions at box edges (for tighter packing)
    for (_, x, y, z, bw, bh, bd) in placements:
        # Positions where new box's face aligns with existing box's face
        if x >= w:
            xs.add(x - w)
        if y >= h:
            ys.add(y - h)
        if z >= d:
            zs.add(z - d)
    
    xs = sorted([x for x in xs if 0 <= x and x + w <= W])
    ys = sorted([y for y in ys if 0 <= y and y + h <= H])
    zs = sorted([z for z in zs if 0 <= z and z + d <= D])
    
    if not xs or not ys or not zs:
        return None
    
    best_pos = None
    best_score = float('inf')
    
    # Sort z first for 'lowest' strategy (check lower z values first)
    if strategy == 'lowest':
        z_order = sorted(zs, key=lambda z: z)
    elif strategy == 'bestfit':
        z_order = sorted(zs, key=lambda z: max(W - (z + d), H, D))
    else:
        z_order = sorted(zs, key=lambda z: z)
    
    for z in z_order:
        for y in ys:
            for x in xs:
                if _check_overlap(x, y, z, w, h, d, placements):
                    continue
                
                if strategy == 'lowest':
                    score = z * (H + 1) * (W + 1) + y * (W + 1) + x
                elif strategy == 'bestfit':
                    rem_x = W - (x + w)
                    rem_y = H - (y + h)
                    rem_z = D - (z + d)
                    score = max(rem_x, rem_y, rem_z) * 1000 + min(rem_x, rem_y, rem_z)
                else:  # corner
                    # Prefer positions that maximize contact with boundaries
                    contact = 0
                    if x == 0: contact += 1
                    if x + w == W: contact += 1
                    if y == 0: contact += 1
                    if y + h == H: contact += 1
                    if z == 0: contact += 1
                    if z + d == D: contact += 1
                    score = -contact * 10000 + z * 100 + y * 10 + x
                
                if score < best_score:
                    best_score = score
                    best_pos = (x, y, z)
    
    return best_pos

def pack(boxes, container):
    W, H, D = container
    start = time.time()
    deadline = start + 140.0  # Safety margin before 150s kill
    
    n = len(boxes)
    container_vol = W * H * D
    
    # Sort by volume descending, with tie-breaking by surface area then max dimension
    indexed = [(i, b) for i, b in enumerate(boxes)]
    indexed.sort(key=lambda t: (
        t[1][0] * t[1][1] * t[1][2],
        2 * (t[1][0]*t[1][1] + t[1][1]*t[1][2] + t[1][0]*t[1][2]),
        max(t[1][0], t[1][1], t[1][2])
    ), reverse=True)
    
    bins = []  # list of lists of placements per bin
    bin_volumes = []  # track volume in each bin for quick sorting
    result = [None] * n
    
    def _try_place_in_bin(box, bi, strategy='lowest'):
        """Try to place box in bin bi using given strategy."""
        pos = _find_position(box, bins[bi], W, H, D, strategy)
        return pos
    
    def _bin_fill_ratio(bi):
        return bin_volumes[bi] / container_vol
    
    # Phase 1: Initial placement with Best-Fit Decreasing
    for orig_idx, box in indexed:
        w, h, d = box
        placed = False
        
        # Try existing bins sorted by fill ratio (most full first)
        if bins:
            bin_order = sorted(range(len(bins)), key=_bin_fill_ratio, reverse=True)
            
            for bi in bin_order:
                if time.time() > deadline:
                    break
                # Try multiple strategies
                for strategy in ['lowest', 'bestfit', 'corner']:
                    pos = _try_place_in_bin(box, bi, strategy)
                    if pos is not None:
                        x, y, z = pos
                        bins[bi].append((bi, x, y, z, w, h, d))
                        bin_volumes[bi] += w * h * d
                        result[orig_idx] = (bi, x, y, z, w, h, d)
                        placed = True
                        break
                if placed:
                    break
        
        if not placed:
            if time.time() > deadline:
                # Emergency: place remaining boxes in new bins
                bi = len(bins)
                for oi, ob in indexed:
                    if result[oi] is None:
                        bins.append([(bi, 0, 0, 0, ob[0], ob[1], ob[2])])
                        bin_volumes.append(ob[0] * ob[1] * ob[2])
                        result[oi] = (bi, 0, 0, 0, ob[0], ob[1], ob[2])
                        bi += 1
                break
            
            # Open new bin
            bi = len(bins)
            bins.append([(bi, 0, 0, 0, w, h, d)])
            bin_volumes.append(w * h * d)
            result[orig_idx] = (bi, 0, 0, 0, w, h, d)
    
    # Phase 2: Local search - try to consolidate boxes into fewer bins
    if time.time() < deadline - 15:
        # Multiple passes of consolidation
        for _pass in range(3):
            if time.time() > deadline - 10:
                break
            
            improved = False
            # Process boxes in order of their current bin (highest bin first)
            for orig_idx in range(n - 1, -1, -1):
                if time.time() > deadline - 10:
                    break
                
                box = boxes[orig_idx]
                w, h, d = box
                current_bin = result[orig_idx][0]
                
                if current_bin == 0:
                    continue
                
                # Try to move to an earlier bin
                moved = False
                for target_bin in range(current_bin - 1, -1, -1):
                    if time.time() > deadline - 10:
                        break
                    
                    # Temporarily remove from current bin
                    cur_placements = bins[current_bin]
                    box_idx = None
                    for i, p in enumerate(cur_placements):
                        if (p[1] == result[orig_idx][1] and 
                            p[2] == result[orig_idx][2] and 
                            p[3] == result[orig_idx][3] and
                            p[4] == w and p[5] == h and p[6] == d):
                            box_idx = i
                            break
                    if box_idx is None:
                        continue
                    
                    cur_placements_without = cur_placements[:box_idx] + cur_placements[box_idx+1:]
                    target_placements = bins[target_bin]
                    
                    # Check if it fits in target bin with multiple strategies
                    pos_found = None
                    for strategy in ['lowest', 'bestfit', 'corner']:
                        pos = _find_position(box, target_placements, W, H, D, strategy)
                        if pos is not None:
                            pos_found = pos
                            break
                    
                    if pos_found is not None:
                        # Move it
                        x, y, z = pos_found
                        bins[target_bin].append((target_bin, x, y, z, w, h, d))
                        bin_volumes[target_bin] += w * h * d
                        bins[current_bin] = cur_placements_without
                        bin_volumes[current_bin] -= w * h * d
                        result[orig_idx] = (target_bin, x, y, z, w, h, d)
                        moved = True
                        improved = True
                        break
                
                if not moved and time.time() > deadline - 10:
                    break
            
            if not improved:
                break
    
    # Phase 3: Optimize last bin fill ratio
    if time.time() < deadline - 5:
        # Find the last bin and try to rearrange boxes to maximize its fill
        if bins:
            max_bin = max(bi for bi in range(len(bins)) if bins[bi])
            if max_bin > 0:
                # Try moving boxes from last bin to earlier bins
                last_bin_boxes = list(range(n))
                last_bin_boxes = [i for i in last_bin_boxes if result[i][0] == max_bin]
                
                for orig_idx in last_bin_boxes:
                    if time.time() > deadline - 3:
                        break
                    
                    box = boxes[orig_idx]
                    w, h, d = box
                    
                    # Try to move to earlier bins
                    for target_bin in range(max_bin - 1, -1, -1):
                        if time.time() > deadline - 3:
                            break
                        
                        cur_placements = bins[max_bin]
                        box_idx = None
                        for i, p in enumerate(cur_placements):
                            if (p[1] == result[orig_idx][1] and 
                                p[2] == result[orig_idx][2] and 
                                p[3] == result[orig_idx][3] and
                                p[4] == w and p[5] == h and p[6] == d):
                                box_idx = i
                                break
                        if box_idx is None:
                            continue
                        
                        cur_without = cur_placements[:box_idx] + cur_placements[box_idx+1:]
                        
                        pos = None
                        for strategy in ['lowest', 'bestfit', 'corner']:
                            pos = _find_position(box, bins[target_bin], W, H, D, strategy)
                            if pos is not None:
                                break
                        
                        if pos is not None:
                            x, y, z = pos
                            bins[target_bin].append((target_bin, x, y, z, w, h, d))
                            bin_volumes[target_bin] += w * h * d
                            bins[max_bin] = cur_without
                            bin_volumes[max_bin] -= w * h * d
                            result[orig_idx] = (target_bin, x, y, z, w, h, d)
                            break
    
    # Ensure all boxes have a placement
    for i in range(n):
        if result[i] is None:
            bi = len(bins)
            w, h, d = boxes[i]
            bins.append([(bi, 0, 0, 0, w, h, d)])
            bin_volumes.append(w * h * d)
            result[i] = (bi, 0, 0, 0, w, h, d)
    
    return result
# EVOLVE-BLOCK-END
