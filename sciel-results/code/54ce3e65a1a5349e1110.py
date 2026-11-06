import time
import random
import math

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    if W <= 0 or H <= 0 or D <= 0 or n == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s limit
    
    container_vol = W * H * D
    
    # Precompute orientations for each box
    def get_orientations(w, h, d):
        # Return all 6 unique orientations as (ow, oh, od)
        # Sort to avoid duplicates
        opts = []
        seen = set()
        for perm in [(w,h,d), (w,d,h), (h,w,d), (h,d,w), (d,w,h), (d,h,w)]:
            if perm not in seen:
                seen.add(perm)
                opts.append(perm)
        return opts
    
    box_orients = [get_orientations(w, h, d) for w, h, d in boxes]
    
    # Volume of each box
    box_vols = [w * h * d for w, h, d in boxes]
    total_box_vol = sum(box_vols)
    
    # Sort boxes by volume descending for initial heuristic
    order_vol = sorted(range(n), key=lambda i: -box_vols[i])
    
    # Also sort by max dimension descending
    order_max = sorted(range(n), key=lambda i: -max(boxes[i]))
    
    # Sort by surface area
    order_surf = sorted(range(n), key=lambda i: -(2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2])))
    
    def try_place(order, placed_list, remaining):
        """
        Try to place boxes in the given order using a simple shelf-like approach
        with orientation selection. Returns (placements, filled_vol)
        """
        placements = [None] * n
        filled_vol = 0
        
        # Use a list of "free spaces" represented as (x, y, z, w, h, d)
        # Start with the full container
        free_spaces = [(0, 0, 0, W, H, D)]
        
        for idx in order:
            if idx in remaining:
                continue
            if time.time() > deadline:
                break
            
            w, h, d = boxes[idx]
            placed = False
            
            # Try all orientations
            for ow, oh, od in box_orients[idx]:
                # Find a free space where this box fits
                best_space_idx = -1
                best_pos = None
                best_score = -1
                
                for si, (sx, sy, sz, sw, sh, sd) in enumerate(free_spaces):
                    if ow <= sw and oh <= sh and od <= sd:
                        # Score: prefer tightest fit (minimize wasted space)
                        waste = (sw - ow) * sh * sd + (sh - oh) * sd * sw + (sd - od) * sw * sh
                        # Actually simpler: prefer smallest free space that fits
                        space_vol = sw * sh * sd
                        if space_vol < best_score or best_space_idx == -1:
                            best_space_idx = si
                            best_pos = (sx, sy, sz)
                            best_score = space_vol
                
                if best_space_idx >= 0:
                    sx, sy, sz = best_pos
                    # Place box at (sx, sy, sz) with dims (ow, oh, od)
                    placements[idx] = (0, sx, sy, sz, ow, oh, od)
                    filled_vol += ow * oh * od
                    remaining.discard(idx)
                    
                    # Remove this free space and add remaining sub-spaces
                    free_spaces.pop(best_space_idx)
                    sw, sh, sd = free_spaces[best_space_idx] if best_space_idx < len(free_spaces) else (0,0,0)
                    # Wait, we already popped it. Let me restructure.
                    
                    # Actually let's redo this more carefully
                    break
            
            if not placed:
                # Try simpler: just place at first position where it fits in a grid-like manner
                # For now, mark as not placed
                pass
        
        # This approach is getting complicated. Let me use a simpler shelf-based method.
        return placements, filled_vol
    
    # Simpler approach: use a shelf-based packing with multiple sorting orders
    def pack_shelf(order):
        """Shelf-based packing: place boxes on shelves in the given order."""
        placements = [None] * n
        filled_vol = 0
        remaining = set(range(n))
        
        x, y, z = 0, 0, 0
        row_h = 0
        layer_d = 0
        
        for idx in order:
            if idx not in remaining:
                continue
            if time.time() > deadline:
                break
            
            w, h, d = boxes[idx]
            
            # Try all orientations, pick the one that fits best in current position
            best_orient = None
            best_waste = float('inf')
            
            for ow, oh, od in box_orients[idx]:
                # Check if it fits at current position
                if x + ow <= W and y + oh <= H and z + od <= D:
                    waste = (W - x - ow) * oh * od + (H - y - oh) * od * (W - x) + (D - z - od) * (W - x) * (H - y)
                    # Simpler: just use the smallest dimension that fits
                    if best_orient is None or (ow, oh, od) < best_orient:
                        best_orient = (ow, oh, od)
            
            if best_orient is None:
                # Try to find an orientation that fits
                for ow, oh, od in box_orients[idx]:
                    if x + ow <= W and y + oh <= H and z + od <= D:
                        best_orient = (ow, oh, od)
                        break
            
            if best_orient is None:
                # Can't place this box, skip it
                continue
            
            ow, oh, od = best_orient
            placements[idx] = (0, x, y, z, ow, oh, od)
            filled_vol += ow * oh * od
            remaining.discard(idx)
            
            x += ow
            row_h = max(row_h, oh)
            layer_d = max(layer_d, od)
            
            if x + min(boxes[i][0] for i in remaining if i is not None) > W if remaining else False:
                # Move to next shelf
                x = 0
                y += row_h
                row_h = 0
            
            if y + min(boxes[i][1] for i in remaining if i is not None) > H if remaining else False:
                # Move to next layer
                y = 0
                z += layer_d
                layer_d = 0
                x = 0
                row_h = 0
        
        # Mark unplaced boxes
        for i in range(n):
            if placements[i] is None:
                w, h, d = boxes[i]
                placements[i] = (-1, 0, 0, 0, w, h, d)
        
        return placements, filled_vol
    
    # Try multiple sorting orders and keep the best
    best_placements = None
    best_vol = -1
    
    orders_to_try = [
        order_vol,
        order_max,
        order_surf,
        sorted(range(n), key=lambda i: -(boxes[i][0]*boxes[i][1]*boxes[i][2]), reverse=False),
    ]
    
    for order in orders_to_try:
        if time.time() > deadline:
            break
        placements, vol = pack_shelf(order)
        if vol > best_vol:
            best_vol = vol
            best_placements = placements
    
    # Try a few random orders
    num_random = 10
    for _ in range(num_random):
        if time.time() > deadline:
            break
        random_order = list(range(n))
        random.shuffle(random_order)
        placements, vol = pack_shelf(random_order)
        if vol > best_vol:
            best_vol = vol
            best_placements = placements
    
    if best_placements is None:
        # Fallback: all boxes left out
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_placements
