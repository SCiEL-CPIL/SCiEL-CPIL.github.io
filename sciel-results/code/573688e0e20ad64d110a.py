import time
import random
import itertools

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    W, H, D = container
    container_volume = W * H * D
    
    # Pre-compute all possible rotations for each box
    # box is (w, h, d)
    # Rotations: (w,h,d), (w,d,h), (h,w,d), (h,d,w), (d,w,h), (d,h,w)
    box_rotations = []
    for w, h, d in boxes:
        rots = set()
        rots.add((w, h, d))
        rots.add((w, d, h))
        rots.add((h, w, d))
        rots.add((h, d, w))
        rots.add((d, w, h))
        rots.add((d, h, w))
        box_rotations.append(list(rots))
        
    def can_place(placed, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        for px, py, pz, pw, ph, pd in placed:
            # Check overlap
            if not (x + w <= px or px + pw <= x or y + h <= py or py + ph <= y or z + d <= pz or pz + pd <= z):
                return False
        return True

    def get_new_corners(placed, x, y, z, w, h, d):
        # Generate new candidate corners based on the newly placed box
        # The new box occupies [x, x+w], [y, y+h], [z, z+d]
        # New corners are the points "exposed" by this box
        candidates = set()
        # Corners on the faces of the new box that are not inside the new box
        # x = x+w
        candidates.add((x + w, y, z))
        candidates.add((x + w, y + h, z))
        candidates.add((x + w, y, z + d))
        candidates.add((x + w, y + h, z + d))
        # y = y+h
        candidates.add((x, y + h, z))
        candidates.add((x, y + h, z + d))
        # z = z+d
        candidates.add((x, y, z + d))
        # Also the "far" corner is already included
        # We should filter these to ensure they are within container bounds
        valid_candidates = set()
        for cx, cy, cz in candidates:
            if 0 <= cx <= W and 0 <= cy <= H and 0 <= cz <= D:
                valid_candidates.add((cx, cy, cz))
        return valid_candidates

    def solve_with_order(order):
        placed = [] # list of (x, y, z, w, h, d)
        placements = [(-1, 0, 0, 0, 0, 0, 0) for _ in range(len(boxes))]
        candidates = {(0, 0, 0)}
        
        for i in order:
            box_id = i
            w_orig, h_orig, d_orig = boxes[box_id]
            best_pos = None
            best_rot = None
            best_score = float('inf')
            
            for rot in box_rotations[box_id]:
                rw, rh, rd = rot
                for cx, cy, cz in candidates:
                    if can_place(placed, cx, cy, cz, rw, rh, rd):
                        # Heuristic: minimize x+y+z to pack tightly
                        score = cx + cy + cz
                        # Secondary heuristic: prefer smaller bounding box growth?
                        # Let's stick with x+y+z for speed and effectiveness
                        if score < best_score:
                            best_score = score
                            best_pos = (cx, cy, cz)
                            best_rot = rot
            
            if best_pos:
                bx, by, bz = best_pos
                bw, bh, bd = best_rot
                placed.append((bx, by, bz, bw, bh, bd))
                placements[box_id] = (0, bx, by, bz, bw, bh, bd)
                new_corners = get_new_corners(placed, bx, by, bz, bw, bh, bd)
                candidates.update(new_corners)
                # Prune candidates that are now invalid? 
                # A candidate (cx, cy, cz) is only useful if it can potentially hold a box.
                # We don't strictly need to prune, but checking can_place will fail anyway.
                # However, the number of candidates can grow. 
                # With 60 boxes, max candidates ~ 60*8 = 480. This is small.
            else:
                # Box cannot be placed
                pass
        
        # Calculate volume filled
        vol = sum(p[4]*p[5]*p[6] for p in placements if p[0] == 0)
        return placements, vol

    # Try multiple sorting strategies
    n = len(boxes)
    indices = list(range(n))
    
    best_placements = None
    best_vol = 0
    
    # Strategy 1: Volume Descending
    order_vol = sorted(indices, key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    p, v = solve_with_order(order_vol)
    if v > best_vol:
        best_vol = v
        best_placements = p
        
    # Strategy 2: Max Dimension Descending
    order_max = sorted(indices, key=lambda i: max(boxes[i]), reverse=True)
    p, v = solve_with_order(order_max)
    if v > best_vol:
        best_vol = v
        best_placements = p
        
    # Strategy 3: Random Restarts
    # We have plenty of time. Let's do ~50 random restarts.
    # Each solve is fast.
    max_restarts = 100
    for _ in range(max_restarts):
        if time.time() > deadline:
            break
        random.shuffle(indices)
        p, v = solve_with_order(indices)
        if v > best_vol:
            best_vol = v
            best_placements = p
            
    if best_placements is None:
        # Fallback: return all -1
        best_placements = [(-1, 0, 0, 0, 0, 0, 0) for _ in range(n)]
        
    return best_placements
