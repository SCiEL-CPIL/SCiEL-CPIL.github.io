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
        
    # Sort boxes by volume descending (largest first)
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda x: x[1][0]*x[1][1]*x[1][2], reverse=True)
    
    placed = [] # list of (x, y, z, w, h, d)
    placements = [None] * n # (bin, x, y, z, w, h, d)
    
    # Candidate points: (x, y, z)
    # Start with origin
    candidates = [(0, 0, 0)]
    candidate_set = {(0, 0, 0)}
    
    def check_overlap(cx, cy, cz, rw, rh, rd, placed_list):
        for px, py, pz, pw, ph, pd in placed_list:
            if not (cx + rw <= px or px + pw <= cx or
                    cy + rh <= py or py + ph <= cy or
                    cz + rd <= pz or pz + pd <= cz):
                return True
        return False

    def get_orientation_score(rw, rh, rd, cx, cy, cz, placed_list, W, H, D):
        """
        Score a placement: lower is better.
        Best-fit heuristic: prefer positions that minimize wasted space.
        1. Minimize z (bottom)
        2. Minimize x (left)
        3. Minimize y (front)
        4. Prefer positions with more "contact" with existing boxes (reduces fragmentation)
        """
        base = (cz, cx, cy)
        
        contact = 0
        for px, py, pz, pw, ph, pd in placed_list:
            if abs(cz - (pz + pd)) < 1e-9:
                if cx < px + pw and cx + rw > px and cy < py + ph and cy + rh > py:
                    contact += 1
            if abs(cz + rd - pz) < 1e-9:
                if cx < px + pw and cx + rw > px and cy < py + ph and cy + rh > py:
                    contact += 1
            if abs(cx - (px + pw)) < 1e-9:
                if cy < py + ph and cy + rh > py and cz < pz + pd and cz + rd > pz:
                    contact += 1
            if abs(cx + rw - px) < 1e-9:
                if cy < py + ph and cy + rh > py and cz < pz + pd and cz + rd > pz:
                    contact += 1
            if abs(cy - (py + ph)) < 1e-9:
                if cx < px + pw and cx + rw > px and cz < pz + pd and cz + rd > pz:
                    contact += 1
            if abs(cy + rh - py) < 1e-9:
                if cx < px + pw and cx + rw > px and cz < pz + pd and cz + rd > pz:
                    contact += 1
        
        return (base, -contact)

    def try_place(boxes_to_place, placed_list, placements_list, candidates_list, candidate_set_local):
        """
        Attempts to place boxes in boxes_to_place into the current state.
        Returns the number of successfully placed boxes.
        """
        placed_count = 0
        
        for orig_idx, (w, h, d) in boxes_to_place:
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
                for cx, cy, cz in candidates_list:
                    # Check bounds
                    if cx + rw > W or cy + rh > H or cz + rd > D:
                        continue
                        
                    # Check overlap with placed boxes
                    if check_overlap(cx, cy, cz, rw, rh, rd, placed_list):
                        continue
                    
                    # Score: Best-fit
                    score = get_orientation_score(rw, rh, rd, cx, cy, cz, placed_list, W, H, D)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pos = (cx, cy, cz)
                        best_rot = (rw, rh, rd)
                        
            if best_pos is not None:
                x, y, z = best_pos
                rw, rh, rd = best_rot
                placed_list.append((x, y, z, rw, rh, rd))
                placements_list[orig_idx] = (0, x, y, z, rw, rh, rd)
                placed_count += 1
                
                # Add new candidate points (corners of the placed box)
                new_candidates = [
                    (x, y, z + rd),
                    (x + rw, y, z + rd),
                    (x, y + rh, z + rd),
                    (x + rw, y + rh, z + rd),
                    (x + rw, y, z),
                    (x, y + rh, z),
                    (x + rw, y + rh, z),
                ]
                
                for c in new_candidates:
                    if c not in candidate_set_local:
                        candidates_list.append(c)
                        candidate_set_local.add(c)
            else:
                placements_list[orig_idx] = (-1, 0, 0, 0, w, h, d)
                
        return placed_count

    # Pass 1: Main packing (Large to Small)
    try_place(indexed_boxes, placed, placements, candidates, candidate_set)
    
    if time.time() < deadline:
        # Pass 2: Gap filling (Small to Large)
        # Get indices of boxes that were left out
        unplaced_indices = [i for i in range(n) if placements[i][0] == -1]
        
        if unplaced_indices:
            # Sort unplaced boxes by volume ascending
            unplaced_boxes = [(i, boxes[i]) for i in unplaced_indices]
            unplaced_boxes.sort(key=lambda x: x[1][0]*x[1][1]*x[1][2])
            
            try_place(unplaced_boxes, placed, placements, candidates, candidate_set)
            
            remaining_indices = [i for i in range(n) if placements[i][0] == -1]
            if remaining_indices and time.time() < deadline - 5:
                 # Try sorting by surface area ascending (smallest surface first)
                 remaining_boxes = [(i, boxes[i]) for i in remaining_indices]
                 remaining_boxes.sort(key=lambda x: 2*(x[1][0]*x[1][1] + x[1][1]*x[1][2] + x[1][0]*x[1][2]))
                 
                 try_place(remaining_boxes, placed, placements, candidates, candidate_set)
                 
                 # Pass 4: Retry remaining by volume descending?
                 remaining_indices = [i for i in range(n) if placements[i][0] == -1]
                 if remaining_indices and time.time() < deadline - 2:
                     remaining_boxes = [(i, boxes[i]) for i in remaining_indices]
                     remaining_boxes.sort(key=lambda x: x[1][0]*x[1][1]*x[1][2], reverse=True)
                     try_place(remaining_boxes, placed, placements, candidates, candidate_set)
    
    # Fill any remaining None (in case of timeout)
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
            
    return placements
