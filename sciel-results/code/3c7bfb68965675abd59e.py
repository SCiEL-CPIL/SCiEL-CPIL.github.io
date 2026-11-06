import numpy as np

def pack(boxes, container):
    W, H, D = container
    if not boxes:
        return []
    
    # Sort indices by descending volume (largest first)
    indexed = list(enumerate(boxes))
    indexed.sort(key=lambda t: t[1][0] * t[1][1] * t[1][2], reverse=True)
    
    # bins is a list of lists of tuples (x, y, z, w, h, d)
    bins = []
    
    def get_bounding_box(items):
        if not items:
            return (0, 0, 0, 0, 0, 0)
        min_x = min(i[0] for i in items)
        min_y = min(i[1] for i in items)
        min_z = min(i[2] for i in items)
        max_x = max(i[0] + i[3] for i in items)
        max_y = max(i[1] + i[4] for i in items)
        max_z = max(i[2] + i[5] for i in items)
        return (min_x, min_y, min_z, max_x, max_y, max_z)

    def can_place(bin_items, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        for (ox, oy, oz, ow, oh, od) in bin_items:
            if (x < ox + ow and x + w > ox and
                y < oy + oh and y + h > oy and
                z < oz + od and z + d > oz):
                return False
        return True

    def find_best_position_in_bin(bin_items, w, h, d):
        # Returns (x, y, z, bb_volume) for the best position, or None
        if not bin_items:
            if w <= W and h <= H and d <= D:
                return (0, 0, 0, w * h * d)
            return None

        xs = set([0])
        ys = set([0])
        zs = set([0])
        for (ox, oy, oz, ow, oh, od) in bin_items:
            xs.add(ox + ow)
            ys.add(oy + oh)
            zs.add(oz + od)
        
        # Filter candidates that are within container bounds
        xs = sorted([x for x in xs if x + w <= W])
        ys = sorted([y for y in ys if y + h <= H])
        zs = sorted([z for z in zs if z + d <= D])
        
        if not xs: xs = [0]
        if not ys: ys = [0]
        if not zs: zs = [0]
        
        best_pos = None
        best_bb_vol = float('inf')
        
        # Current bounding box of the bin
        curr_bb = get_bounding_box(bin_items)
        curr_min_x, curr_min_y, curr_min_z, curr_max_x, curr_max_y, curr_max_z = curr_bb
        
        for x in xs:
            for y in ys:
                for z in zs:
                    if can_place(bin_items, x, y, z, w, h, d):
                        # Calculate new bounding box
                        new_min_x = min(curr_min_x, x)
                        new_min_y = min(curr_min_y, y)
                        new_min_z = min(curr_min_z, z)
                        new_max_x = max(curr_max_x, x + w)
                        new_max_y = max(curr_max_y, y + h)
                        new_max_z = max(curr_max_z, z + d)
                        
                        new_vol = (new_max_x - new_min_x) * (new_max_y - new_min_y) * (new_max_z - new_min_z)
                        
                        if new_vol < best_bb_vol:
                            best_bb_vol = new_vol
                            best_pos = (x, y, z)
                            
        if best_pos:
            return (best_pos[0], best_pos[1], best_pos[2], best_bb_vol)
        return None

    result = [None] * len(boxes)
    
    for orig_idx, (w, h, d) in indexed:
        best_bin_idx = -1
        best_pos = None
        best_bb_vol = float('inf')
        
        for bi in range(len(bins)):
            res = find_best_position_in_bin(bins[bi], w, h, d)
            if res is not None:
                x, y, z, vol = res
                if vol < best_bb_vol:
                    best_bb_vol = vol
                    best_bin_idx = bi
                    best_pos = (x, y, z)
        
        if best_bin_idx != -1:
            x, y, z = best_pos
            bins[best_bin_idx].append((x, y, z, w, h, d))
            result[orig_idx] = (best_bin_idx, x, y, z, w, h, d)
        else:
            # Open new bin
            bins.append([(0, 0, 0, w, h, d)])
            result[orig_idx] = (len(bins) - 1, 0, 0, 0, w, h, d)
            
    return result
