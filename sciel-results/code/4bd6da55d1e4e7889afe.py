import numpy as np
import random

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    # Pre-calculate orientations for each box
    # orientations[i] is a list of (w, h, d) tuples
    orientations_list = []
    for w, h, d in boxes:
        orients = [
            (w, h, d),
            (w, d, h),
            (h, w, d),
            (h, d, w),
            (d, w, h),
            (d, h, w)
        ]
        # Remove duplicates to save time
        orients = list(set(orients))
        orientations_list.append(orients)
        
    def greedy_pack(order):
        placed = []  # list of (x, y, z, w, h, d)
        results = [None] * n
        total_vol = 0
        
        for idx in order:
            w_orig, h_orig, d_orig = boxes[idx]
            best_pos = None
            
            for ow, oh, od in orientations_list[idx]:
                if ow > W or oh > H or od > D:
                    continue
                
                # Generate candidate coordinates
                # Start with 0
                xs = [0]
                ys = [0]
                zs = [0]
                
                for (px, py, pz, pw, ph, pd) in placed:
                    xs.append(px + pw)
                    ys.append(py + ph)
                    zs.append(pz + pd)
                    
                # Sort and unique
                xs = sorted(set(xs))
                ys = sorted(set(ys))
                zs = sorted(set(zs))
                
                found = False
                # We want the "best" position. 
                # Heuristic: Minimize z, then y, then x.
                # This tends to pack from bottom, back, left.
                
                for cz in zs:
                    if cz + od > D:
                        break
                    for cy in ys:
                        if cy + oh > H:
                            break
                        for cx in xs:
                            if cx + ow > W:
                                break
                            
                            # Check overlap
                            ok = True
                            for (px, py, pz, pw, ph, pd) in placed:
                                if not (cx + ow <= px or px + pw <= cx or
                                        cy + oh <= py or py + ph <= cy or
                                        cz + od <= pz or pz + pd <= cz):
                                    ok = False
                                    break
                            
                            if ok:
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
                results[idx] = (0, x, y, z, ow, oh, od)
                total_vol += ow * oh * od
            else:
                results[idx] = (-1, 0, 0, 0, w_orig, h_orig, d_orig)
                
        return results, total_vol

    best_result = None
    best_volume = -1
    
    # Try a few deterministic sorts
    # 1. Volume desc
    order_vol = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    res, vol = greedy_pack(order_vol)
    if vol > best_volume:
        best_volume = vol
        best_result = res
        
    # 2. Max side desc
    order_max = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    res, vol = greedy_pack(order_max)
    if vol > best_volume:
        best_volume = vol
        best_result = res

    # Try random restarts
    # Limit number of restarts to fit in time
    # 60 boxes, greedy is fast. Let's try 20 restarts.
    for _ in range(20):
        order = list(range(n))
        random.shuffle(order)
        # Bias towards larger boxes first? 
        # A common heuristic is to sort by a random weighted sum of dimensions.
        # Let's just use pure random shuffle for diversity.
        
        res, vol = greedy_pack(order)
        if vol > best_volume:
            best_volume = vol
            best_result = res
            
    return best_result
