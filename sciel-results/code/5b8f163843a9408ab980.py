"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for 120s limit
    
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    if n == 0:
        return []
    
    # Generate all 6 unique rotations for each box
    def get_rotations(w, h, d):
        rots = set()
        rots.add((w, h, d))
        rots.add((w, d, h))
        rots.add((h, w, d))
        rots.add((h, d, w))
        rots.add((d, w, h))
        rots.add((d, h, w))
        return list(rots)
    
    # For each box, precompute all valid rotations
    box_rotations = []
    for i, (w, h, d) in enumerate(boxes):
        rots = get_rotations(w, h, d)
        # Filter rotations that fit in container
        valid_rots = [(rw, rh, rd) for rw, rh, rd in rots if rw <= W and rh <= H and rd <= D]
        if not valid_rots:
            valid_rots = [(w, h, d)]  # Keep original even if it doesn't fit
        box_rotations.append(valid_rots)
    
    # Sort boxes by volume descending (larger boxes first)
    box_volumes = [(i, w*h*d) for i, (w, h, d) in enumerate(boxes)]
    box_volumes.sort(key=lambda x: x[1], reverse=True)
    sorted_indices = [i for i, _ in box_volumes]
    
    # Placements: list of (bin, x, y, z, w, h, d)
    placements = [None] * n
    placed = [False] * n
    
    # Track occupied space using a list of placed boxes
    # Each entry: (x, y, z, w, h, d, box_index)
    placed_boxes = []
    
    def can_place(x, y, z, w, h, d):
        """Check if a box can be placed at (x,y,z) with dimensions (w,h,d) without overlap."""
        if x + w > W or y + h > H or z + d > D:
            return False
        for (px, py, pz, pw, ph, pd, _) in placed_boxes:
            # Check for overlap in 3D
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
        return True
    
    def find_best_position(box_idx, rw, rh, rd):
        """Find the best position for a box with given rotation.
        Score positions to prefer tight packing: lower z, then compact footprint, then lower x, then lower y."""
        best_pos = None
        best_score = None
        
        # Generate candidate x positions
        x_candidates = set([0])
        for (px, py, pz, pw, ph, pd, _) in placed_boxes:
            if px + pw <= W:
                x_candidates.add(px + pw)
        
        # Generate candidate y positions
        y_candidates = set([0])
        for (px, py, pz, pw, ph, pd, _) in placed_boxes:
            if py + ph <= H:
                y_candidates.add(py + ph)
        
        # Generate candidate z positions
        z_candidates = set([0])
        for (px, py, pz, pw, ph, pd, _) in placed_boxes:
            if pz + pd <= D:
                z_candidates.add(pz + pd)
        
        # Try all combinations
        for x in sorted(x_candidates):
            if x + rw > W:
                continue
            for y in sorted(y_candidates):
                if y + rh > H:
                    continue
                for z in sorted(z_candidates):
                    if z + rd > D:
                        continue
                    if can_place(x, y, z, rw, rh, rd):
                        # Score: prefer lower z (bottom), then compact (smaller x+y), then lower x, then lower y
                        score = (z, x + y, x, y)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
        
        return best_pos
    
    # Greedy placement with rotation selection
    for box_idx in sorted_indices:
        if time.time() > deadline:
            break
        
        w, h, d = boxes[box_idx]
        best_placement = None
        best_orientation = None
        best_pos_score = None
        
        for rw, rh, rd in box_rotations[box_idx]:
            pos = find_best_position(box_idx, rw, rh, rd)
            if pos is not None:
                x, y, z = pos
                # Score the placement: prefer lower z, then compact footprint
                pos_score = (z, x + y, x, y)
                if best_pos_score is None or pos_score < best_pos_score:
                    best_pos_score = pos_score
                    best_placement = pos
                    best_orientation = (rw, rh, rd)
        
        if best_placement is not None:
            x, y, z = best_placement
            rw, rh, rd = best_orientation
            placements[box_idx] = (0, x, y, z, rw, rh, rd)
            placed[box_idx] = True
            placed_boxes.append((x, y, z, rw, rh, rd, box_idx))
        else:
            placements[box_idx] = (-1, 0, 0, 0, w, h, d)
    
    # Handle any remaining unplaced boxes
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    return placements
# EVOLVE-BLOCK-END
