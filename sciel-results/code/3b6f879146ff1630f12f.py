import time

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0
    
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    result = [None] * n
    
    # Sort indices by volume descending
    indexed_boxes = [(i, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    indexed_boxes.sort(key=lambda x: x[1]*x[2]*x[3], reverse=True)
    
    bins = []
    
    def get_candidate_coords(bin_boxes):
        x_set = {0}
        y_set = {0}
        z_set = {0}
        
        for (bx, by, bz, bw, bh, bd) in bin_boxes:
            x_set.add(bx + bw)
            y_set.add(by + bh)
            z_set.add(bz + bd)
            
        x_coords = [x for x in x_set if x < W]
        y_coords = [y for y in y_set if y < H]
        z_coords = [z for z in z_set if z < D]
        
        return x_coords, y_coords, z_coords

    def check_overlap(x, y, z, w, h, d, bin_boxes):
        for (bx, by, bz, bw, bh, bd) in bin_boxes:
            if x < bx + bw and x + w > bx and \
               y < by + bh and y + h > by and \
               z < bz + bd and z + d > bz:
                return True
        return False

    def find_best_pos_in_bin(bin_boxes, w, h, d):
        x_coords, y_coords, z_coords = get_candidate_coords(bin_boxes)
        
        # Sort to ensure first valid found is optimal (min z, min y, min x)
        x_coords.sort()
        y_coords.sort()
        z_coords.sort()
        
        for z in z_coords:
            if z + d > D:
                continue
            for y in y_coords:
                if y + h > H:
                    continue
                for x in x_coords:
                    if x + w > W:
                        continue
                    if not check_overlap(x, y, z, w, h, d, bin_boxes):
                        return (x, y, z)
                        
        return None

    for i, w, h, d in indexed_boxes:
        if time.time() > deadline:
            # Fallback: place in new bins
            b_idx = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            result[i] = (b_idx, 0, 0, 0, w, h, d)
            continue

        best_bin_idx = -1
        best_bin_pos = None
        best_bin_score = None
        
        for b_idx, bin_boxes in enumerate(bins):
            pos = find_best_pos_in_bin(bin_boxes, w, h, d)
            if pos is not None:
                x, y, z = pos
                score = (z, y, x)
                if best_bin_score is None or score < best_bin_score:
                    best_bin_score = score
                    best_bin_idx = b_idx
                    best_bin_pos = pos
        
        if best_bin_idx != -1:
            x, y, z = best_bin_pos
            bins[best_bin_idx].append((x, y, z, w, h, d))
            result[i] = (best_bin_idx, x, y, z, w, h, d)
        else:
            b_idx = len(bins)
            x, y, z = 0, 0, 0
            bins.append([(x, y, z, w, h, d)])
            result[i] = (b_idx, x, y, z, w, h, d)

    return result
