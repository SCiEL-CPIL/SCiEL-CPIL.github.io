import time

def pack(boxes, container):
    """
    Packs boxes into the fewest number of bins using a Best-Fit heuristic.
    Boxes are placed unrotated.
    """
    W, H, D = container
    
    # Sort boxes by volume descending, then by height descending, then width, then depth
    sorted_indices = sorted(range(len(boxes)), key=lambda i: (
        boxes[i][0] * boxes[i][1] * boxes[i][2], 
        boxes[i][1], 
        boxes[i][0], 
        boxes[i][2]
    ), reverse=True)
    
    # bins is a list of lists of tuples (x, y, z, w, h, d)
    bins = []
    
    # results[i] will be (bin_idx, x, y, z, w, h, d) for box i
    results = [None] * len(boxes)
    
    def overlaps(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or 
                    y1 + h1 <= y2 or y2 + h2 <= y1 or 
                    z1 + d1 <= z2 or z2 + d2 <= z1)

    def find_best_position(box, bin_boxes):
        w, h, d = box
        # Candidate coordinates
        x_candidates = [0]
        y_candidates = [0]
        z_candidates = [0]
        
        for bx, by, bz, bw, bh, bd in bin_boxes:
            x_candidates.append(bx + bw)
            y_candidates.append(by + bh)
            z_candidates.append(bz + bd)
        
        x_candidates = sorted(set(x_candidates))
        y_candidates = sorted(set(y_candidates))
        z_candidates = sorted(set(z_candidates))
        
        best_score = float('inf')
        best_pos = None
        
        # Iterate through all candidate combinations
        # Prioritize low z, then low y, then low x
        for z in z_candidates:
            if z + d > D:
                continue
            for y in y_candidates:
                if y + h > H:
                    continue
                for x in x_candidates:
                    if x + w > W:
                        continue
                    
                    valid = True
                    for bx, by, bz, bw, bh, bd in bin_boxes:
                        if overlaps(x, y, z, w, h, d, bx, by, bz, bw, bh, bd):
                            valid = False
                            break
                    
                    if valid:
                        score = z * 10000 + y * 100 + x
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
        
        return best_pos

    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    for idx in sorted_indices:
        if time.time() > deadline:
            # Fallback: if we run out of time, place remaining boxes in a new bin at 0,0,0 
            # or just append to the last bin if it fits? 
            # Given N=40, this is extremely unlikely. 
            # But to be safe, we can just place them in new bins one by one.
            while idx < len(boxes):
                w, h, d = boxes[idx]
                bin_idx = len(bins)
                bins.append([(0, 0, 0, w, h, d)])
                results[idx] = (bin_idx, 0, 0, 0, w, h, d)
                idx += 1
            break
            
        w, h, d = boxes[idx]
        
        best_bin_idx = -1
        best_pos_in_bin = None
        best_global_score = float('inf')
        
        # Try to place in existing bins
        for b_idx, bin_boxes in enumerate(bins):
            pos = find_best_position((w, h, d), bin_boxes)
            if pos is not None:
                x, y, z = pos
                score = z * 10000 + y * 100 + x
                if score < best_global_score:
                    best_global_score = score
                    best_bin_idx = b_idx
                    best_pos_in_bin = pos
        
        if best_bin_idx != -1:
            x, y, z = best_pos_in_bin
            bins[best_bin_idx].append((x, y, z, w, h, d))
            results[idx] = (best_bin_idx, x, y, z, w, h, d)
        else:
            # Open new bin
            bin_idx = len(bins)
            bins.append([(0, 0, 0, w, h, d)])
            results[idx] = (bin_idx, 0, 0, 0, w, h, d)
            
    return results
