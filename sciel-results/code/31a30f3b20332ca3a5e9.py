import time
import itertools

def load(boxes, container):
    W, H, D = container
    # Sort boxes by volume descending to place large boxes first
    indexed_boxes = [(w, h, d, i) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[0]*x[1]*x[2], reverse=True)
    
    placements = [None] * len(boxes)
    placed = [] # list of (x, y, z, w, h, d)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    for w, h, d, idx in indexed_boxes:
        if time.time() > deadline:
            # If we run out of time, mark remaining as left out
            for i in range(len(boxes)):
                if placements[i] is None:
                    placements[i] = (-1, 0, 0, 0, 0, 0, 0)
            return placements
            
        # Get all 6 rotations
        rotations = [
            (w, h, d),
            (w, d, h),
            (h, w, d),
            (h, d, w),
            (d, w, h),
            (d, h, w)
        ]
        
        best_pos = None
        best_rot = None
        best_score = float('inf')
        
        # Generate candidate coordinates
        # Start with 0, then add offsets from placed boxes
        xs = {0}
        ys = {0}
        zs = {0}
        for px, py, pz, pw, ph, pd in placed:
            xs.add(px + pw)
            ys.add(py + ph)
            zs.add(pz + pd)
            
        # Sort to allow early termination
        xs = sorted([x for x in xs if x <= W])
        ys = sorted([y for y in ys if y <= H])
        zs = sorted([z for z in zs if z <= D])
        
        for rw, rh, rd in rotations:
            if rw > W or rh > H or rd > D:
                continue
                
            # Iterate through candidates
            # Optimization: Only check candidates that are "tight" against existing boxes or origin
            # We iterate all combinations
            
            for cx in xs:
                if cx + rw > W:
                    break
                for cy in ys:
                    if cy + rh > H:
                        break
                    for cz in zs:
                        if cz + rd > D:
                            break
                        
                        # Check overlap
                        overlap = False
                        for px, py, pz, pw, ph, pd in placed:
                            if not (cx + rw <= px or px + pw <= cx or 
                                    cy + rh <= py or py + ph <= cy or 
                                    cz + rd <= pz or pz + pd <= cz):
                                overlap = True
                                break
                                
                        if not overlap:
                            # Score: prefer lower z, then lower y, then lower x
                            # This fills the bottom-left-back corner first
                            score = cz * W * H + cy * W + cx
                            if score < best_score:
                                best_score = score
                                best_pos = (cx, cy, cz)
                                best_rot = (rw, rh, rd)
        
        if best_pos is not None:
            cx, cy, cz = best_pos
            rw, rh, rd = best_rot
            placements[idx] = (0, cx, cy, cz, rw, rh, rd)
            placed.append((cx, cy, cz, rw, rh, rd))
        else:
            placements[idx] = (-1, 0, 0, 0, w, h, d)
            
    # Fill any remaining None (should not happen if loop completed)
    for i in range(len(boxes)):
        if placements[i] is None:
            placements[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
            
    return placements
