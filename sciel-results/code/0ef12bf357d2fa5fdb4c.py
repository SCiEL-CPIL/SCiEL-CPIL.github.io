import numpy as np

def load(boxes, container):
    W, H, D = container
    
    def get_orientations(w, h, d):
        orients = set()
        orients.add((w, h, d))
        orients.add((w, d, h))
        orients.add((h, w, d))
        orients.add((h, d, w))
        orients.add((d, w, h))
        orients.add((d, h, w))
        return list(orients)

    def solve(sort_key):
        indexed_boxes = list(enumerate(boxes))
        indexed_boxes.sort(key=lambda t: sort_key(t[1]), reverse=True)
        
        placed = [] # list of (x, y, z, w, h, d)
        results = [None] * len(boxes)
        total_vol = 0
        
        candidates = [(0, 0, 0)]
        candidate_set = {(0, 0, 0)}
        
        for orig_idx, (w_orig, h_orig, d_orig) in indexed_boxes:
            box_placed = False
            orientations = get_orientations(w_orig, h_orig, d_orig)
            
            # Sort candidates by z, then y, then x to prioritize bottom-left-far
            sorted_candidates = sorted(candidates, key=lambda p: (p[2], p[1], p[0]))
            
            for cx, cy, cz in sorted_candidates:
                if box_placed:
                    break
                
                for ow, oh, od in orientations:
                    if box_placed:
                        break
                    
                    if cx + ow > W or cy + oh > H or cz + od > D:
                        continue
                    
                    overlap = False
                    for (px, py, pz, pw, ph, pd) in placed:
                        if not (cx + ow <= px or px + pw <= cx or
                                cy + oh <= py or py + ph <= cy or
                                cz + od <= pz or pz + pd <= cz):
                            overlap = True
                            break
                    
                    if not overlap:
                        placed.append((cx, cy, cz, ow, oh, od))
                        results[orig_idx] = (0, cx, cy, cz, ow, oh, od)
                        total_vol += ow * oh * od
                        box_placed = True
                        
                        # Generate new candidate points
                        new_points = [
                            (cx + ow, cy, cz),
                            (cx, cy + oh, cz),
                            (cx, cy, cz + od),
                            (cx + ow, cy + oh, cz),
                            (cx + ow, cy, cz + od),
                            (cx, cy + oh, cz + od),
                            (cx + ow, cy + oh, cz + od)
                        ]
                        
                        for np_ in new_points:
                            if np_[0] <= W and np_[1] <= H and np_[2] <= D:
                                if np_ not in candidate_set:
                                    candidates.append(np_)
                                    candidate_set.add(np_)
                        
                        pass

            if not box_placed:
                results[orig_idx] = (-1, 0, 0, 0, w_orig, h_orig, d_orig)
        
        return results, total_vol

    strategies = [
        lambda b: b[0]*b[1]*b[2],
        lambda b: max(b),
        lambda b: sum(b),
        lambda b: b[0]*b[1] + b[1]*b[2] + b[0]*b[2]
    ]
    
    best_result = None
    best_volume = -1
    
    for sort_key in strategies:
        res, vol = solve(sort_key)
        if vol > best_volume:
            best_volume = vol
            best_result = res
            
    return best_result
