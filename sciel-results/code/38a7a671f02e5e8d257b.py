"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import permutations
import numpy as np

def _get_rotations(w, h, d):
    """Get all unique rotations of a box."""
    rots = set()
    for perm in permutations((w, h, d)):
        rots.add(perm)
    return list(rots)

def _pack_boxes(boxes, container, order_indices, time_limit):
    """
    Attempt to place boxes into the container using Extreme Point heuristic.
    Returns (result, filled_volume) where result is a list of placements.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return [], 0
    
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Precompute rotations for each box
    box_rotations = []
    for w, h, d in boxes:
        rots = _get_rotations(w, h, d)
        # Sort rotations: prefer lower height (z-dim) to stack better, then lower depth, then lower width
        rots.sort(key=lambda r: (r[2], r[1], r[0]))
        box_rotations.append(rots)
    
    placed = []  # List of (x, y, z, w, h, d)
    result = [(-1, 0, 0, 0, 0, 0, 0)] * n
    filled_volume = 0
    
    # Candidate points for placement (Extreme Points)
    candidates = [(0, 0, 0)]
    
    def overlaps_any(px, py, pz, pw, ph, pd):
        for ex, ey, ez, ew, eh, ed in placed:
            if (px < ex + ew and px + pw > ex and
                py < ey + eh and py + ph > ey and
                pz < ez + ed and pz + pd > ez):
                return True
        return False

    def add_candidate(x, y, z):
        if 0 <= x <= W and 0 <= y <= H and 0 <= z <= D:
            # Check if this point is strictly inside an existing box
            inside = False
            for ex, ey, ez, ew, eh, ed in placed:
                if x > ex and x < ex + ew and y > ey and y < ey + eh and z > ez and z < ez + ed:
                    inside = True
                    break
            if not inside:
                if (x, y, z) not in candidates:
                    candidates.append((x, y, z))

    for idx in order_indices:
        if time.time() > deadline:
            break
        
        w, h, d = boxes[idx]
        rots = box_rotations[idx]
        
        placed_this = False
        
        # Sort candidates by Z (lowest), then Y (lowest), then X (leftmost)
        valid_candidates = [c for c in candidates if 0 <= c[0] <= W and 0 <= c[1] <= H and 0 <= c[2] <= D]
        valid_candidates.sort(key=lambda p: (p[2], p[1], p[0]))
        
        for bw, bh, bd in rots:
            if time.time() > deadline:
                break
            
            found = False
            for cx, cy, cz in valid_candidates:
                if cx + bw > W or cy + bh > H or cz + bd > D:
                    continue
                
                if not overlaps_any(cx, cy, cz, bw, bh, bd):
                    placed.append((cx, cy, cz, bw, bh, bd))
                    result[idx] = (0, cx, cy, cz, bw, bh, bd)
                    filled_volume += bw * bh * bd
                    
                    add_candidate(cx + bw, cy, cz)
                    add_candidate(cx, cy + bh, cz)
                    add_candidate(cx, cy, cz + bd)
                    
                    if (cx, cy, cz) in candidates:
                        candidates.remove((cx, cy, cz))
                    
                    placed_this = True
                    found = True
                    break
            
            if found:
                break
        
        if not placed_this:
            result[idx] = (-1, 0, 0, 0, w, h, d)
    
    return result, filled_volume

def _layer_pack(boxes, container, order_indices, time_limit):
    """
    Layer-based packing: divide container height into layers, pack 2D shelves in each layer.
    Returns (result, filled_volume).
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return [], 0
    
    start_time = time.time()
    deadline = start_time + time_limit
    
    box_rotations = []
    for w, h, d in boxes:
        rots = _get_rotations(w, h, d)
        # For layer packing, we might want to prefer orientations where the "height" (z-dim) is small
        # to fit more into a layer, or large to reduce number of layers?
        # Let's sort by z-dim ascending to try to fit flatter boxes first in a layer.
        rots.sort(key=lambda r: (r[2], r[1], r[0]))
        box_rotations.append(rots)
    
    result = [(-1, 0, 0, 0, 0, 0, 0)] * n
    filled_volume = 0
    unplaced = list(order_indices)
    
    # We will process layers from bottom to top.
    # In each layer, we use a shelf packing heuristic in 2D (X, Y).
    # The layer height is determined by the tallest box placed in that layer.
    
    current_z = 0
    
    while unplaced and current_z < D:
        if time.time() > deadline:
            break
        
        # Start a new layer
        layer_boxes = []
        layer_max_h = 0
        layer_placed = [] # List of (x, y, w, h, d, box_idx) for this layer
        
        # Try to place boxes from unplaced into this layer
        # We iterate through unplaced boxes. If a box fits in the current layer's shelf structure, we place it.
        # Shelf structure: list of shelves, each shelf is (y_start, y_end, x_cursor)
        # Actually, a simpler "Next Fit" or "First Fit" decreasing by width/height is standard for 2D strip packing.
        # Here we have a fixed width W and height H.
        
        # Let's use a simple shelf approach:
        # Shelves are horizontal strips.
        # We maintain a list of active shelves: [ (y_offset, shelf_height, x_cursor) ]
        # When a box is placed on a shelf, x_cursor += box_width.
        # If it doesn't fit in width, we start a new shelf below the current one? No, shelves are stacked in Y.
        # So we stack shelves in Y direction.
        
        shelves = [] # Each shelf: (y_start, shelf_height, x_cursor)
        y_cursor = 0
        
        # We need to decide the order of boxes to try in this layer.
        # Let's try the unplaced boxes in the given order (which is already sorted by some heuristic).
        
        remaining_in_layer = []
        
        for idx in unplaced:
            if time.time() > deadline:
                break
            
            w, h, d = boxes[idx]
            placed_in_layer = False
            
            # Try all rotations
            for bw, bh, bd in box_rotations[idx]:
                # bw is X-dim, bh is Y-dim, bd is Z-dim (height of box in this layer)
                
                # Check if box height fits in container remaining
                if current_z + bd > D:
                    continue
                
                # Try to fit into an existing shelf
                found_shelf = -1
                for i, (sy, sh, xc) in enumerate(shelves):
                    # Box must fit in shelf height
                    if bh <= sh:
                        # Box must fit in remaining width of shelf
                        if xc + bw <= W:
                            found_shelf = i
                            break
                
                if found_shelf != -1:
                    sy, sh, xc = shelves[found_shelf]
                    # Place box at (xc, sy, current_z)
                    # Wait, shelf y is fixed. Box y is sy.
                    # Box x is xc.
                    # Box z is current_z.
                    
                    # Check overlap with other boxes in this layer?
                    # Since shelves are stacked in Y and we only extend X in a shelf, and Y is distinct per shelf,
                    # and Z is same for all in layer, there is no overlap within the layer if shelf logic is correct.
                    # However, we must ensure the new box doesn't overlap with boxes in *previous* layers.
                    # Previous layers are below current_z. This box is at current_z. So no overlap in Z.
                    
                    # Update shelf
                    shelves[found_shelf] = (sy, sh, xc + bw)
                    
                    # Update layer max height
                    if bd > layer_max_h:
                        layer_max_h = bd
                    
                    # Record placement
                    result[idx] = (0, xc, sy, current_z, bw, bh, bd)
                    filled_volume += bw * bh * bd
                    layer_placed.append(idx)
                    placed_in_layer = True
                    break
                
                # If no existing shelf fits, try to start a new shelf
                # New shelf starts at y_cursor
                if y_cursor + bh <= H:
                    # Start new shelf
                    # But wait, we should check if this box could fit in a *new* shelf that we create now.
                    # The new shelf will have height bh (or potentially larger if we pack others later? No, shelf height is fixed when created).
                    # Actually, in standard shelf packing, the shelf height is determined by the first item placed on it.
                    
                    # Let's create a new shelf for this box.
                    sy = y_cursor
                    sh = bh
                    xc = 0
                    
                    if xc + bw <= W:
                        shelves.append((sy, sh, bw))
                        y_cursor += sh
                        
                        if bd > layer_max_h:
                            layer_max_h = bd
                        
                        result[idx] = (0, xc, sy, current_z, bw, bh, bd)
                        filled_volume += bw * bh * bd
                        layer_placed.append(idx)
                        placed_in_layer = True
                        break
            
            if not placed_in_layer:
                remaining_in_layer.append(idx)
        
        # End of layer processing
        if layer_max_h == 0:
            # No box could be placed in this layer (or layer is empty)
            # If no box was placed, break to avoid infinite loop
            break
            
        # Move to next layer
        current_z += layer_max_h
        
        # Update unplaced list
        unplaced = remaining_in_layer
        
        # If no boxes were placed in this layer, break
        if not layer_placed:
            break
            
    return result, filled_volume

def load(boxes, container):
    """
    Main entry point: place boxes into container.
    Returns list of (bin, x, y, z, w, h, d) for each box.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    box_volumes = [w * h * d for w, h, d in boxes]
    
    # Strategy 1: Volume descending
    order1 = list(range(n))
    order1.sort(key=lambda i: -box_volumes[i])
    
    # Strategy 2: Max dimension descending
    order2 = list(range(n))
    order2.sort(key=lambda i: -max(boxes[i]))
    
    # Strategy 3: Min dimension descending
    order3 = list(range(n))
    order3.sort(key=lambda i: -min(boxes[i]))
    
    # Strategy 4: Sum of dimensions descending
    order4 = list(range(n))
    order4.sort(key=lambda i: -sum(boxes[i]))
    
    # Strategy 5: Volume ascending (smaller first to fill gaps)
    order5 = list(range(n))
    order5.sort(key=lambda i: box_volumes[i])
    
    # Strategy 6: Random shuffle
    random.seed(42)
    order6 = list(range(n))
    random.shuffle(order6)
    
    # Strategy 7: Sort by aspect ratio (max/min)
    order7 = list(range(n))
    order7.sort(key=lambda i: -max(boxes[i]) / max(min(boxes[i]), 1e-9))
    
    # Strategy 8: Sort by surface area
    order8 = list(range(n))
    order8.sort(key=lambda i: -(2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2])))
    
    orders = [order1, order2, order3, order4, order5, order6, order7, order8]
    
    best_result = None
    best_filled = -1
    
    # We will run both _pack_boxes (Extreme Point) and _layer_pack (Layer/Shelf)
    # for each order, and keep the best result.
    
    # To save time, we can interleave or just run all.
    # 8 orders * 2 methods = 16 runs.
    # Each run is fast (O(N^2) or better).
    
    for order in orders:
        if time.time() > deadline:
            break
            
        # Run Extreme Point
        res1, filled1 = _pack_boxes(boxes, container, order, 110.0)
        if filled1 > best_filled:
            best_filled = filled1
            best_result = res1
            
        # Run Layer Pack
        if time.time() > deadline:
            break
        res2, filled2 = _layer_pack(boxes, container, order, 110.0)
        if filled2 > best_filled:
            best_filled = filled2
            best_result = res2
    
    # If we have time left, do more random shuffles
    if best_result is not None and time.time() < deadline - 10.0:
        # Try a few more random shuffles
        for _ in range(20):
            if time.time() > deadline:
                break
            random_order = list(range(n))
            random.shuffle(random_order)
            
            res1, filled1 = _pack_boxes(boxes, container, random_order, 100.0)
            if filled1 > best_filled:
                best_filled = filled1
                best_result = res1
                
            if time.time() > deadline:
                break
            res2, filled2 = _layer_pack(boxes, container, random_order, 100.0)
            if filled2 > best_filled:
                best_filled = filled2
                best_result = res2
    
    if best_result is None:
        # Fallback: return all -1
        best_result = [(-1, 0, 0, 0, 0, 0, 0)] * n
    
    return best_result
# EVOLVE-BLOCK-END
