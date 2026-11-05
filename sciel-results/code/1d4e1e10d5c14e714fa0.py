import time
import itertools

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    if n == 0:
        return []
        
    # 1. Generate all 6 rotations for each box
    # Rotation is a permutation of (w, h, d)
    # We want to keep the original dimensions for the output, 
    # but use rotated dimensions for placement.
    # Output format: (bin, x, y, z, w, h, d)
    # The task says "any axis rotation allowed". 
    # Usually, the output w,h,d should reflect the placed orientation.
    # Let's assume the output w,h,d should be the dimensions in the container's x,y,z axes.
    
    # Sort boxes by volume descending (largest first)
    # This is a common heuristic.
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda x: x[1][0]*x[1][1]*x[1][2], reverse=True)
    
    placed = [] # list of (x, y, z, w, h, d)
    placements = [None] * n # (bin, x, y, z, w, h, d)
    
    # Candidate points: (x, y, z)
    # Start with origin
    candidates = [(0, 0, 0)]
    
    for i, (orig_idx, (w, h, d)) in enumerate(indexed_boxes):
        if time.time() > deadline:
            break
            
        box_rotations = list(set(itertools.permutations([w, h, d])))
        
        best_pos = None
        best_rot = None
        best_score = None
        
        # Try all rotations
        for rw, rh, rd in box_rotations:
            # Check if rotation fits in container
            if rw > W or rh > H or rd > D:
                continue
                
            # Try all candidate points
            for cx, cy, cz in candidates:
                # Check bounds
                if cx + rw > W or cy + rh > H or cz + rd > D:
                    continue
                    
                # Check overlap with placed boxes
                overlaps = False
                for px, py, pz, pw, ph, pd in placed:
                    # AABB overlap check
                    if not (cx + rw <= px or px + pw <= cx or
                            cy + rh <= py or py + ph <= cy or
                            cz + rd <= pz or pz + pd <= cz):
                        overlaps = True
                        break
                
                if not overlaps:
                    # Score: Bottom-Left-Front
                    # Minimize z, then x, then y
                    score = (cz, cx, cy)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pos = (cx, cy, cz)
                        best_rot = (rw, rh, rd)
                        
        if best_pos is not None:
            x, y, z = best_pos
            rw, rh, rd = best_rot
            placed.append((x, y, z, rw, rh, rd))
            placements[orig_idx] = (0, x, y, z, rw, rh, rd)
            
            # Add new candidate points (corners of the placed box)
            # We add all 8 corners to be safe, though only top-face ones are strictly necessary for support.
            # Adding all ensures we can place boxes next to existing ones on the same level too.
            candidates.extend([
                (x, y, z + rd),
                (x + rw, y, z + rd),
                (x, y + rh, z + rd),
                (x + rw, y + rh, z + rd),
                (x, y, z), # already exists maybe
                (x + rw, y, z),
                (x, y + rh, z),
                (x + rw, y + rh, z)
            ])
        else:
            placements[orig_idx] = (-1, 0, 0, 0, w, h, d)
            
    # Fill any remaining None (in case of timeout)
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
            
    return placements
