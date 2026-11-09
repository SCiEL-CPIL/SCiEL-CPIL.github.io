import numpy as np

def load(boxes, container):
    W, H, D = container
    
    def rotations(w, h, d):
        return [(w,h,d), (w,d,h), (h,w,d), (h,d,w), (d,w,h), (d,h,w)]
    
    boxes_sorted = sorted(boxes, key=lambda b: b[0]*b[1]*b[2], reverse=True)
    
    placed = []
    
    def min_z_for(x, y, rw, rh, rd):
        """Find the minimum z where a box of size (rw,rh,rd) can be placed at (x,y)."""
        min_z = 0
        for px, py, pz, pw, ph, pd in placed:
            # Check if the box at (px,py) overlaps with (x,y) in projection
            if x < px + pw and px < x + rw and y < py + ph and py < y + rh:
                min_z = max(min_z, pz + pd)
        return min_z
    
    def can_place_at(x, y, z, rw, rh, rd):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + rw > W or y + rh > H or z + rd > D:
            return False
        for px, py, pz, pw, ph, pd in placed:
            if (x < px + pw and px < x + rw and
                y < py + ph and py < y + rh and
                z < pz + pd and pz < z + rd):
                return False
        return True
    
    def score(x, y, z, rw, rh, rd):
        # Prefer compact: low z, then low y, then low x
        return z * 10000 + y * 100 + x
    
    result = []
    for w, h, d in boxes_sorted:
        best = None
        best_score = float('inf')
        
        for rw, rh, rd in rotations(w, h, d):
            # Candidate x positions: 0 and right edges of placed boxes
            x_cands = set([0])
            y_cands = set([0])
            for px, py, pz, pw, ph, pd in placed:
                if px + pw <= W:
                    x_cands.add(px + pw)
                if py + ph <= H:
                    y_cands.add(py + ph)
            
            for x in sorted(x_cands):
                if x + rw > W:
                    continue
                for y in sorted(y_cands):
                    if y + rh > H:
                        continue
                    z = min_z_for(x, y, rw, rh, rd)
                    if z + rd > D:
                        continue
                    if can_place_at(x, y, z, rw, rh, rd):
                        s = score(x, y, z, rw, rh, rd)
                        if s < best_score:
                            best_score = s
                            best = (x, y, z, rw, rh, rd)
        
        if best is not None:
            placed.append(best)
            result.append((0, best[0], best[1], best[2], best[3], best[4], best[5]))
        else:
            result.append((-1, 0, 0, 0, w, h, d))
    
    return result
