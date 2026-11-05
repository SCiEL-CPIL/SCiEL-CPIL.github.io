import time
import itertools

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for safety

    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    if n == 0:
        return []

    # 1. Generate all 6 rotations for each box
    def get_rotations(w, h, d):
        return list(set(itertools.permutations((w, h, d))))

    # Sort boxes by volume descending
    indexed_boxes = [(i, w, h, d, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[4], reverse=True)
    
    placements = [None] * n
    placed_boxes = [] # List of (x, y, z, w, h, d)
    
    # Candidate coordinates
    cand_x = [0]
    cand_y = [0]
    cand_z = [0]
    
    # To speed up, we can limit the number of candidates if it grows too large
    # But with N=60, it shouldn't be an issue.
    
    for i, w, h, d, vol in indexed_boxes:
        if time.time() > deadline:
            break
            
        best_pos = None
        best_rot = None
        best_cost = float('inf')
        
        rotations = get_rotations(w, h, d)
        
        for rw, rh, rd in rotations:
            # Check if rotation fits in container
            if rw > W or rh > H or rd > D:
                continue
                
            for cx in cand_x:
                if cx + rw > W:
                    continue
                for cy in cand_y:
                    if cy + rh > H:
                        continue
                    for cz in cand_z:
                        if cz + rd > D:
                            continue
                            
                        # Check collision
                        valid = True
                        for px, py, pz, pw, ph, pd in placed_boxes:
                            # Check overlap
                            if not (cx + rw <= px or px + pw <= cx or
                                    cy + rh <= py or py + ph <= cy or
                                    cz + rd <= pz or pz + pd <= cz):
                                valid = False
                                break
                                
                        if valid:
                            # Heuristic: Minimize x+y+z (Bottom-Left-Farthest)
                            cost = cx + cy + cz
                            if cost < best_cost:
                                best_cost = cost
                                best_pos = (cx, cy, cz)
                                best_rot = (rw, rh, rd)
        
        if best_pos is not None:
            bx, by, bz = best_pos
            bw, bh, bd = best_rot
            placements[i] = (0, bx, by, bz, bw, bh, bd)
            placed_boxes.append((bx, by, bz, bw, bh, bd))
            
            # Update candidates
            nx = bx + bw
            ny = by + bh
            nz = bz + bd
            
            if nx not in cand_x:
                cand_x.append(nx)
            if ny not in cand_y:
                cand_y.append(ny)
            if nz not in cand_z:
                cand_z.append(nz)
                
            # Sort candidates to keep them organized (optional, but helps consistency)
            # Actually, order doesn't matter for correctness, only for which one is found first if costs are equal.
            # We can leave them unsorted to save time.
        else:
            placements[i] = (-1, 0, 0, 0, w, h, d)
            
    # Fill any remaining None (if deadline hit)
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
            
    return placements
