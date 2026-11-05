"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

def _orientations(w, h, d):
    """Yield all 6 distinct orientations of a box with dims (w,h,d)."""
    seen = set()
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            yield perm

def _overlaps(x1, y1, z1, x2, y2, z2, w1, h1, d1, w2, h2, d2):
    """Check if two boxes overlap (strictly)."""
    if x1 + w1 <= x2 or x2 + w2 <= x1:
        return False
    if y1 + h1 <= y2 or y2 + h2 <= y1:
        return False
    if z1 + d1 <= z2 or z2 + d2 <= z1:
        return False
    return True

def _fits(placed, x, y, z, w, h, d, W, H, D):
    """Check if box at (x,y,z) with dims (w,h,d) fits in container and doesn't overlap."""
    if x < 0 or y < 0 or z < 0:
        return False
    if x + w > W or y + h > H or z + d > D:
        return False
    for (px, py, pz, pw, ph, pd) in placed:
        if _overlaps(x, y, z, px, py, pz, w, h, d, pw, ph, pd):
            return False
    return True

def _candidate_positions(placed, W, H, D):
    """Generate candidate (x,y,z) positions to try, sorted by z, y, x."""
    xs = {0}
    ys = {0}
    zs = {0}
    for (px, py, pz, pw, ph, pd) in placed:
        xs.add(px)
        xs.add(px + pw)
        ys.add(py)
        ys.add(py + ph)
        zs.add(pz)
        zs.add(pz + pd)
    # Filter to valid range
    xs = sorted(x for x in xs if 0 <= x <= W)
    ys = sorted(y for y in ys if 0 <= y <= H)
    zs = sorted(z for z in zs if 0 <= z <= D)
    # Generate all combinations, sort by (z, y, x) to prefer bottom-front-left
    cands = []
    for z in zs:
        for y in ys:
            for x in xs:
                cands.append((x, y, z))
    return cands

def _try_pack(boxes, container, order, deadline):
    """Try to pack boxes in given order; return placements list and filled volume."""
    W, H, D = container
    placed = []  # list of (x, y, z, w, h, d)
    result = [None] * len(boxes)  # will store (bin, x, y, z, w, h, d)
    filled_vol = 0.0
    
    for idx in order:
        if time.time() > deadline:
            # Mark remaining as left out
            for j in range(len(boxes)):
                if result[j] is None:
                    w, h, d = boxes[j]
                    result[j] = (-1, 0, 0, 0, w, h, d)
            return result, filled_vol
        
        w, h, d = boxes[idx]
        best_pos = None
        best_orient = None
        
        # Try each orientation
        for ow, oh, od in _orientations(w, h, d):
            # Generate candidate positions
            cands = _candidate_positions(placed, W, H, D)
            for (cx, cy, cz) in cands:
                if _fits(placed, cx, cy, cz, ow, oh, od, W, H, D):
                    best_pos = (cx, cy, cz)
                    best_orient = (ow, oh, od)
                    break
            if best_pos is not None:
                break
        
        if best_pos is not None:
            cx, cy, cz = best_pos
            ow, oh, od = best_orient
            placed.append((cx, cy, cz, ow, oh, od))
            result[idx] = (0, cx, cy, cz, ow, oh, od)
            filled_vol += w * h * d
        else:
            result[idx] = (-1, 0, 0, 0, w, h, d)
    
    # Fill in any None
    for j in range(len(boxes)):
        if result[j] is None:
            w, h, d = boxes[j]
            result[j] = (-1, 0, 0, 0, w, h, d)
    
    return result, filled_vol

def _try_pack_shelf(boxes, container, order, deadline):
    """Shelf-based packing: place boxes on horizontal shelves."""
    W, H, D = container
    result = [None] * len(boxes)
    filled_vol = 0.0
    shelves = []  # list of (z, max_h, used_x) where used_x tracks x-positions
    
    for idx in order:
        if time.time() > deadline:
            for j in range(len(boxes)):
                if result[j] is None:
                    w, h, d = boxes[j]
                    result[j] = (-1, 0, 0, 0, w, h, d)
            return result, filled_vol
        
        w, h, d = boxes[idx]
        best_pos = None
        best_orient = None
        best_shelf_z = None
        
        # Try each orientation
        for ow, oh, od in _orientations(w, h, d):
            # Try to place on existing shelves or create new shelf
            placed_on_shelf = False
            
            for si, (sz, sh, shelf_boxes) in enumerate(shelves):
                # Check if box fits in height on this shelf
                if sz + oh > H:
                    continue
                # Try to find x position on this shelf
                # Collect all x boundaries from boxes on this shelf
                xs = {0}
                for (bx, by, bz, bw, bh, bd) in shelf_boxes:
                    xs.add(bx)
                    xs.add(bx + bw)
                xs = sorted(x for x in xs if 0 <= x <= W)
                
                for cx in xs:
                    # Check if box fits at (cx, sz) with dims (ow, oh, od)
                    if cx + ow > W:
                        continue
                    # Check overlap with boxes on this shelf
                    overlaps = False
                    for (bx, by, bz, bw, bh, bd) in shelf_boxes:
                        if _overlaps(cx, sz, 0, bx, by, bz, ow, oh, od, bw, bh, bd):
                            overlaps = True
                            break
                    if not overlaps:
                        best_pos = (cx, sz, 0)
                        best_orient = (ow, oh, od)
                        best_shelf_z = sz
                        placed_on_shelf = True
                        break
                if placed_on_shelf:
                    break
            
            if not placed_on_shelf:
                # Try to create a new shelf
                # Find the lowest z where we can place a shelf
                existing_zs = [sz for sz, sh, sb in shelves]
                if not existing_zs:
                    new_z = 0
                else:
                    # Try z = 0 first, then above existing shelves
                    candidate_zs = [0] + [sz + sh for sz, sh, sb in shelves]
                    candidate_zs = sorted(set(z for z in candidate_zs if z + oh <= H))
                    if not candidate_zs:
                        continue
                    new_z = candidate_zs[0]
                
                if new_z + oh <= H:
                    # Place at x=0 on new shelf
                    if ow <= W:
                        best_pos = (0, new_z, 0)
                        best_orient = (ow, oh, od)
                        best_shelf_z = new_z
                        placed_on_shelf = True
            
            if placed_on_shelf:
                break
        
        if best_pos is not None:
            cx, cy, cz = best_pos
            ow, oh, od = best_orient
            result[idx] = (0, cx, cy, cz, ow, oh, od)
            filled_vol += w * h * d
            
            # Add to shelf
            found = False
            for si in range(len(shelves)):
                if shelves[si][0] == best_shelf_z:
                    shelves[si][2].append((cx, cy, cz, ow, oh, od))
                    found = True
                    break
            if not found:
                shelves.append((best_shelf_z, oh, [(cx, cy, cz, ow, oh, od)]))
        else:
            result[idx] = (-1, 0, 0, 0, w, h, d)
    
    for j in range(len(boxes)):
        if result[j] is None:
            w, h, d = boxes[j]
            result[j] = (-1, 0, 0, 0, w, h, d)
    
    return result, filled_vol

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    container_vol = W * H * D
    
    # Deadline guard: stop 10s before the 150s kill
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Generate different sort orders
    orders = []
    
    # Order 1: by volume descending
    orders.append(sorted(range(n), key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True))
    
    # Order 2: by max dimension descending
    orders.append(sorted(range(n), key=lambda i: max(boxes[i]), reverse=True))
    
    # Order 3: by sum of dimensions descending
    orders.append(sorted(range(n), key=lambda i: sum(boxes[i]), reverse=True))
    
    # Order 4: by aspect ratio (most cube-like first)
    orders.append(sorted(range(n), key=lambda i: (max(boxes[i]) / min(boxes[i])), reverse=True))
    
    # Order 5: by min dimension descending (prefer taller boxes)
    orders.append(sorted(range(n), key=lambda i: min(boxes[i]), reverse=True))
    
    best_result = None
    best_filled = 0.0
    
    # Try standard packing with different orders
    for order in orders:
        if time.time() > deadline:
            break
        result, filled_vol = _try_pack(boxes, container, order, deadline)
        if filled_vol > best_filled:
            best_filled = filled_vol
            best_result = result
    
    # Try shelf-based packing with different orders
    for order in orders:
        if time.time() > deadline:
            break
        result, filled_vol = _try_pack_shelf(boxes, container, order, deadline)
        if filled_vol > best_filled:
            best_filled = filled_vol
            best_result = result
    
    if best_result is None:
        # Fallback: return all as left out
        best_result = []
        for w, h, d in boxes:
            best_result.append((-1, 0, 0, 0, w, h, d))
    
    return best_result
# EVOLVE-BLOCK-END
