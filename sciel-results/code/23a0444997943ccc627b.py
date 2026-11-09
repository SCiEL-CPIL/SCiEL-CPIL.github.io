def load(boxes, container):
    import time
    start_time = time.time()
    DEADLINE = 110
    
    W, H, D = container
    
    def rotations(w, h, d):
        return [
            (w, h, d),
            (w, d, h),
            (h, w, d),
            (h, d, w),
            (d, w, h),
            (d, h, w),
        ]
    
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda ib: ib[1][0]*ib[1][1]*ib[1][2], reverse=True)
    
    placed = []
    result = [None] * len(boxes)
    
    for idx, (w, h, d) in indexed_boxes:
        if time.time() - start_time > DEADLINE:
            break
        
        best_pos = None
        best_score = None
        
        for rw, rh, rd in rotations(w, h, d):
            if rw > W or rh > H or rd > D:
                continue
            
            # Find candidate z positions
            zs = [0]
            for (px, py, pz, pw, ph, pd) in placed:
                z_top = pz + pd
                if z_top + rd <= D:
                    zs.append(z_top)
            zs = sorted(set(zs))
            
            for z in zs:
                # Find candidate x positions
                xs = [0]
                for (px, py, pz, pw, ph, pd) in placed:
                    # Only consider boxes that overlap in y and z
                    if not (py + ph <= z or pz + pd <= z + rd):
                        x_right = px + pw
                        if x_right + rw <= W:
                            xs.append(x_right)
                xs = sorted(set(xs))
                
                # Find candidate y positions
                ys = [0]
                for (px, py, pz, pw, ph, pd) in placed:
                    if not (px + pw <= z or pz + pd <= z + rd):  # This is wrong, should check x overlap
                        pass
                # Let me fix this
                ys = [0]
                for (px, py, pz, pw, ph, pd) in placed:
                    # Check if box overlaps in x and z with the new box at (x, y, z)
                    # But x is not fixed yet, so this is tricky
                    # Let's just use all y positions
                    y_back = py + ph
                    if y_back + rh <= H:
                        ys.append(y_back)
                ys = sorted(set(ys))
                
                for x in xs:
                    for y in ys:
                        # Check if box fits at (x, y, z)
                        fits = True
                        for (px, py, pz, pw, ph, pd) in placed:
                            if not (x + rw <= px or px + pw <= x or
                                    y + rh <= py or py + ph <= y or
                                    z + rd <= pz or pz + pd <= z):
                                fits = False
                                break
                        if fits:
                            score = (z, y, x)
                            if best_score is None or score < best_score:
                                best_score = score
                                best_pos = (x, y, z, rw, rh, rd)
                
                # Early break if we found a position at z=0
                if z == 0 and best_pos is not None:
                    # Check if we found a position at this z
                    # We can break if we have a position at z=0
                    pass
        
        if best_pos is not None:
            x, y, z, rw, rh, rd = best_pos
            placed.append((x, y, z, rw, rh, rd))
            result[idx] = (0, x, y, z, rw, rh, rd)
        else:
            result[idx] = (-1, 0, 0, 0, w, h, d)
    
    for i in range(len(boxes)):
        if result[i] is None:
            w, h, d = boxes[i]
            result[i] = (-1, 0, 0, 0, w, h, d)
    
    return result
