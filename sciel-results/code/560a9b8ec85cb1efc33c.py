import time
import itertools
import numpy as np

def _shelf_pack_2d(items, W, D, deadline):
    """Pack items into a 2D shelf (W x D). items: list of (w, d) sorted.
    Returns list of (x, y) placements for packed items, None for unpacked."""
    placements = [None] * len(items)
    x, y = 0, 0
    shelf_d = 0
    for i, (w, d) in enumerate(items):
        if time.time() > deadline:
            break
        if x + w > W:
            x = 0
            y += shelf_d
            shelf_d = 0
        if y + d > D:
            continue
        placements[i] = (x, y)
        x += w
        shelf_d = max(shelf_d, d)
    return placements

def _try_packing(boxes, container, order, deadline):
    """Try a specific ordering and rotation strategy. Return (filled_volume, placements)."""
    W, H, D = container
    n = len(boxes)
    # For each box, try all 6 rotations, pick the one that fits best in current context
    # We'll do a layer-based approach: process boxes in order, assign to layers
    
    # First, determine rotations for each box based on the order
    # Strategy: for each box, choose rotation that minimizes height (z-dim) to pack more per layer
    # Actually, let's try: for each box, pick rotation that has smallest z-extent first,
    # but also consider w and d for 2D packing
    
    # Let's use a simpler approach: try all 6 rotations for each box,
    # and use a greedy 3D shelf packing
    
    placements = [(-1, 0, 0, 0, 0, 0, 0)] * n
    filled_vol = 0.0
    
    # 3D shelf packing: maintain shelves in x-y plane, stack in z
    # We'll process boxes in the given order
    # For each box, try all 6 rotations, pick the one that fits in the "current position"
    # Current position tracks where we are in the container
    
    # Better approach: layer decomposition
    # Assign boxes to layers based on their height after rotation
    # For each box, choose rotation that minimizes height
    
    # Let's do: sort boxes by chosen rotation's height, then pack each layer with 2D shelf
    # But we need to be careful about which rotation to use
    
    # Simplest robust approach: greedy 3D with best-fit rotation
    x, y, z = 0, 0, 0
    row_h = 0  # height of current row in y
    layer_d = 0  # depth of current layer in z
    
    for idx in order:
        if time.time() > deadline:
            break
        w, h, d = boxes[idx]
        # Try all 6 rotations
        rotations = [
            (w, h, d), (h, w, d), (w, d, h),
            (d, h, w), (d, w, h), (h, d, w)
        ]
        best_rot = None
        best_score = None
        for rw, rh, rd in rotations:
            if rw > W or rh > H or rd > D:
                continue
            # Score: prefer smaller z-extent, then smaller y-extent, then smaller x-extent
            # Actually, we want to fit in current position
            # Check if it fits in current x position
            if x + rw <= W:
                # Fits in current row
                score = (rd, rh, rw)  # minimize depth, then height, then width
                if best_score is None or score < best_score:
                    best_score = score
                    best_rot = (rw, rh, rd)
            else:
                # Would need new row/layer, less preferred
                score = (1000 + rd, 1000 + rh, rw)
                if best_score is None or score < best_score:
                    best_score = score
                    best_rot = (rw, rh, rd)
        
        if best_rot is None:
            continue
        
        rw, rh, rd = best_rot
        
        # Place the box
        if x + rw > W:
            x = 0
            y += row_h
            row_h = 0
        if y + rh > H:
            y = 0
            z += layer_d
            layer_d = 0
            x = 0
            row_h = 0
        if z + rd > D:
            continue
        
        placements[idx] = (0, x, y, z, rw, rh, rd)
        filled_vol += rw * rh * rd
        x += rw
        row_h = max(row_h, rh)
        layer_d = max(layer_d, rd)
    
    return filled_vol, placements

def _get_orderings(boxes):
    """Generate multiple ordering strategies."""
    n = len(boxes)
    indices = list(range(n))
    
    orderings = []
    
    # 1. Volume descending
    vol = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
    orderings.append(sorted(indices, key=lambda i: -vol[i]))
    
    # 2. Max dimension descending
    maxdim = [max(boxes[i]) for i in range(n)]
    orderings.append(sorted(indices, key=lambda i: -maxdim[i]))
    
    # 3. Min dimension descending (tall boxes first)
    mindim = [min(boxes[i]) for i in range(n)]
    orderings.append(sorted(indices, key=lambda i: -mindim[i]))
    
    # 4. Volume / max_dim ratio (flat boxes first)
    ratio = [vol[i] / maxdim[i] if maxdim[i] > 0 else 0 for i in range(n)]
    orderings.append(sorted(indices, key=lambda i: -ratio[i]))
    
    # 5. Surface area descending
    surf = [2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]) for i in range(n)]
    orderings.append(sorted(indices, key=lambda i: -surf[i]))
    
    # 6. Smallest dimension first (thin boxes to fill gaps)
    orderings.append(sorted(indices, key=lambda i: mindim[i]))
    
    # 7. Max dimension ascending
    orderings.append(sorted(indices, key=lambda i: maxdim[i]))
    
    return orderings

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    if n == 0:
        return []
    
    # Check if any box fits at all
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    best_vol = 0.0
    best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    orderings = _get_orderings(boxes)
    
    for order in orderings:
        if time.time() > deadline:
            break
        vol, placements = _try_packing(boxes, container, order, deadline)
        if vol > best_vol:
            best_vol = vol
            best_placements = placements
    
    return best_placements
