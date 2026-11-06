import numpy as np
from math import ceil
import itertools

def load(boxes, container):
    W, H, D = container
    
    if not boxes:
        return []
    
    # Generate candidate slab heights: use box height dims and divisors of H
    all_dims = set()
    for w, h, d in boxes:
        all_dims.add(w)
        all_dims.add(h)
        all_dims.add(d)
    
    candidate_heights = set(all_dims)
    candidate_heights.add(H)
    # Add common divisors
    for i in range(2, 10):
        if H % i == 0:
            candidate_heights.add(H // i)
    # Also add fractions of H
    for frac in [0.25, 0.33, 0.5, 0.66, 0.75]:
        candidate_heights.add(int(H * frac))
    
    candidate_heights = sorted([h for h in candidate_heights if 0 < h <= H])
    
    best_placements = None
    best_volume = -1
    
    # Precompute all unique rotations for each box
    box_perms = []
    for w, h, d in boxes:
        perms = list(set(itertools.permutations((w, h, d))))
        box_perms.append(perms)
    
    def pack_slab_2d(boxes_in_slab, slab_h, W, D):
        """
        Pack boxes into a 2D slab of width W, depth D, height slab_h.
        Returns list of (box_idx, x, z, w, h, d) where h is the vertical dim.
        Uses shelf-based 2D packing with multiple strategies.
        """
        if not boxes_in_slab:
            return []
        
        best_result = []
        best_vol = -1
        
        # Try multiple sorting strategies
        sort_keys = [
            lambda x: (x[2] * x[3], x[2]),  # area, then width
            lambda x: (x[2], x[3]),          # width, then depth
            lambda x: (x[3], x[2]),          # depth, then width
            lambda x: (max(x[2], x[3]), min(x[2], x[3])),  # max dim
            lambda x: (x[2] + x[3], x[2] * x[3]),  # perimeter, area
        ]
        
        # Try both orientations of the 2D plane
        for orient in [0, 1]:
            for sort_key in sort_keys:
                # Select best rotation for each box
                selected = []
                for idx, a, b, c in boxes_in_slab:
                    # b is the height (vertical), must fit in slab_h
                    if b > slab_h:
                        continue
                    # Choose rotation that minimizes footprint
                    # We keep (a, b, c) where b is vertical
                    # The 2D footprint is a x c
                    selected.append((idx, a, b, c))
                
                if not selected:
                    continue
                
                # Sort
                if orient == 0:
                    # W is horizontal, D is vertical in 2D
                    sorted_items = sorted(selected, key=lambda x: sort_key((x[0], x[1], x[2], x[3])), reverse=True)
                    limit_w = W
                    limit_d = D
                else:
                    # D is horizontal, W is vertical in 2D
                    sorted_items = sorted(selected, key=lambda x: sort_key((x[0], x[3], x[2], x[1])), reverse=True)
                    limit_w = D
                    limit_d = W
                
                # Shelf packing
                shelves = []  # list of (shelf_start, shelf_height, current_pos)
                max_pos = 0
                placed = []
                
                for idx, a, b, c in sorted_items:
                    if orient == 0:
                        len_dim = a
                        h_dim = c
                    else:
                        len_dim = c
                        h_dim = a
                    
                    # Try to fit in existing shelves
                    best_s = -1
                    best_rem = float('inf')
                    
                    for s_idx, (start, sh, cur) in enumerate(shelves):
                        if h_dim <= sh and cur + len_dim <= limit_w:
                            rem = limit_w - (cur + len_dim)
                            if rem < best_rem:
                                best_rem = rem
                                best_s = s_idx
                    
                    if best_s != -1:
                        start, sh, cur = shelves[best_s]
                        new_cur = cur + len_dim
                        shelves[best_s] = (start, sh, new_cur)
                        
                        if orient == 0:
                            x, z = cur, start
                        else:
                            x, z = start, cur
                        
                        placed.append((idx, x, z, a, b, c))
                    else:
                        # New shelf
                        if max_pos + h_dim <= limit_d:
                            start = max_pos
                            max_pos += h_dim
                            shelves.append((start, h_dim, len_dim))
                            
                            if orient == 0:
                                x, z = 0, start
                            else:
                                x, z = start, 0
                            
                            placed.append((idx, x, z, a, b, c))
                
                vol = sum(a * b * c for _, _, _, a, b, c in placed)
                if vol > best_vol:
                    best_vol = vol
                    best_result = placed
        
        return best_result
    
    # Try each slab height
    for slab_h in candidate_heights:
        if slab_h <= 0 or slab_h > H:
            continue
        
        n_slabs = ceil(H / slab_h)
        if n_slabs == 0:
            continue
        
        current_placements = [(-1, 0, 0, 0, 0, 0, 0) for _ in range(len(boxes))]
        total_volume = 0
        placed_boxes = set()
        
        for slab_idx in range(n_slabs):
            y_start = slab_idx * slab_h
            y_end = min(y_start + slab_h, H)
            actual_slab_h = y_end - y_start
            
            if actual_slab_h <= 0:
                continue
            
            # Collect boxes that can fit in this slab height
            boxes_in_slab = []
            for i in range(len(boxes)):
                if i in placed_boxes:
                    continue
                for perm in box_perms[i]:
                    a, b, c = perm
                    if b <= actual_slab_h:
                        boxes_in_slab.append((i, a, b, c))
                        break  # Take first valid rotation
            
            if not boxes_in_slab:
                continue
            
            # Pack the slab
            result = pack_slab_2d(boxes_in_slab, actual_slab_h, W, D)
            
            for idx, x, z, a, b, c in result:
                current_placements[idx] = (0, x, y_start, z, a, b, c)
                placed_boxes.add(idx)
                total_volume += a * b * c
        
        if total_volume > best_volume:
            best_volume = total_volume
            best_placements = [list(p) for p in current_placements]
    
    if best_placements is None:
        best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_placements
