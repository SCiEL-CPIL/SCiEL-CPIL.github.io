import numpy as np

def load(boxes, container):
    W, H, D = container
    
    strategies = [
        lambda b: b[0]*b[1]*b[2],
        lambda b: max(b),
        lambda b: 2*(b[0]*b[1] + b[1]*b[2] + b[0]*b[2]),
    ]
    
    best_result = None
    best_volume = -1
    
    def solve_with_strategy(sort_key):
        indexed_boxes = list(enumerate(boxes))
        indexed_boxes.sort(key=lambda t: sort_key(t[1]), reverse=True)
        
        placed = []  # list of (x, y, z, w, h, d)
        results = [None] * len(boxes)
        total_vol = 0
        
        for orig_idx, (w_orig, h_orig, d_orig) in indexed_boxes:
            orientations = [
                (w_orig, h_orig, d_orig),
                (w_orig, d_orig, h_orig),
                (h_orig, w_orig, d_orig),
                (h_orig, d_orig, w_orig),
                (d_orig, w_orig, h_orig),
                (d_orig, h_orig, w_orig),
            ]
            
            best_pos = None
            best_score = -1
            
            # Generate candidate x, y, z
            # Use unique values from placed boxes + 0
            xs = [0]
            ys = [0]
            zs = [0]
            for (px, py, pz, pw, ph, pd) in placed:
                xs.append(px + pw)
                ys.append(py + ph)
                zs.append(pz + pd)
            
            # Filter and unique
            xs = sorted(set(x for x in xs if x < W))
            ys = sorted(set(y for y in ys if y < H))
            zs = sorted(set(z for z in zs if z < D))
            
            # Generate all combinations
            # To optimize, we can iterate in order of score
            # Score: minimize z, then y, then x
            # So we iterate z from 0, y from 0, x from 0
            
            for ow, oh, od in orientations:
                # Check if orientation fits in container at all
                if ow > W or oh > H or od > D:
                    continue
                
                # Iterate candidates in "best" order
                # We want lowest z, then lowest y, then lowest x
                # So outer loop z, inner y, inner x?
                # Actually, we want the "first" valid one in this order.
                # So we iterate z from 0 to max, y from 0 to max, x from 0 to max.
                
                found = False
                for cz in zs:
                    if cz + od > D:
                        break
                    for cy in ys:
                        if cy + oh > H:
                            break
                        for cx in xs:
                            if cx + ow > W:
                                break
                            
                            # Check overlap with placed boxes
                            # Use vectorized check if placed is not empty
                            if not placed:
                                is_valid = True
                            else:
                                # Convert placed to numpy array for this check?
                                # Or just loop?
                                # With early termination, the number of checks is small.
                                # Let's stick to Python loop for simplicity, as early termination makes it fast.
                                is_valid = True
                                for (px, py, pz, pw, ph, pd) in placed:
                                    if not (cx + ow <= px or px + pw <= cx or
                                            cy + oh <= py or py + ph <= cy or
                                            cz + od <= pz or pz + pd <= cz):
                                        is_valid = False
                                        break
                            
                            if is_valid:
                                best_pos = (cx, cy, cz, ow, oh, od)
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                
                if found:
                    break
            
            if best_pos is not None:
                x, y, z, ow, oh, od = best_pos
                placed.append((x, y, z, ow, oh, od))
                results[orig_idx] = (0, x, y, z, ow, oh, od)
                total_vol += ow * oh * od
            else:
                results[orig_idx] = (-1, 0, 0, 0, w_orig, h_orig, d_orig)
        
        return results, total_vol
    
    for sort_key in strategies:
        res, vol = solve_with_strategy(sort_key)
        if vol > best_volume:
            best_volume = vol
            best_result = res
            
    return best_result
