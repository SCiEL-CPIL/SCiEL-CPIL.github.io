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
    
    def check_overlap(cx, cy, cz, rw, rh, rd, placed_list):
        for px, py, pz, pw, ph, pd in placed_list:
            if not (cx + rw <= px or px + pw <= cx or
                    cy + rh <= py or py + ph <= cy or
                    cz + rd <= pz or pz + pd <= cz):
                return True
        return False

    def try_place(boxes_to_place, placed_list, placements_list, candidates_list):
        """
        Attempts to place boxes in boxes_to_place into the current state.
        Returns the number of successfully placed boxes.
        """
        placed_count = 0
        # Work on a copy of candidates to avoid modifying the main list if we fail?
        # Actually, if we place a box, we MUST update candidates.
        # Since we are doing a sequential greedy pass, we just append.
        # But wait, if we fail to place a box, we don't add candidates.
        # If we place a box, we add candidates.
        # This is stateful.
        
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
                placed_list.append((x, y, z, rw, rh, rd))
                placements_list[orig_idx] = (0, x, y, z, rw, rh, rd)
                placed_count += 1
                
                # Add new candidate points (corners of the placed box)
                candidates_list.extend([
                    (x, y, z + rd),
                    (x + rw, y, z + rd),
                    (x, y + rh, z + rd),
                    (x + rw, y + rh, z + rd),
                    (x + rw, y, z),
                    (x, y + rh, z),
                    (x + rw, y + rh, z)
                ])
            else:
                placements_list[orig_idx] = (-1, 0, 0, 0, w, h, d)
                
        return placed_count

    # Pass 1: Main packing (Large to Small)
    try_place(indexed_boxes, placed, placements, candidates)
    
    if time.time() < deadline:
        # Pass 2: Gap filling (Small to Large)
        # Get indices of boxes that were left out
        unplaced_indices = [i for i in range(n) if placements[i][0] == -1]
        
        if unplaced_indices:
            # Sort unplaced boxes by volume ascending
            unplaced_boxes = [(i, boxes[i]) for i in unplaced_indices]
            unplaced_boxes.sort(key=lambda x: x[1][0]*x[1][1]*x[1][2])
            
            # We need to be careful: placements[i] is already set to -1.
            # If we place them, we update it.
            # But we also need to ensure we don't double place or conflict.
            # The placed list contains all currently placed boxes.
            # The candidates list contains all current candidate points.
            
            try_place(unplaced_boxes, placed, placements, candidates)
            
            # If gap filling succeeded, we might have new "holes" created by the small boxes?
            # No, small boxes fill holes. But maybe we can try again with a different metric?
            # Or maybe just try the unplaced ones again in case the first pass missed some due to ordering?
            # Actually, try_place is greedy. If it failed before, it might fail again unless the state changed.
            # The state changed because we placed some small boxes.
            
            # Let's try a second gap-filling pass with the remaining unplaced boxes,
            # this time sorted by "compactness" or just retry.
            # But simply retrying the same set might not help if the greedy choice was bad.
            
            # A more effective strategy: 
            # The first pass placed large boxes.
            # The second pass placed small boxes in gaps.
            # Are there still unplaced boxes?
            # If so, maybe we can try to place them by sorting them by their smallest dimension?
            
            remaining_indices = [i for i in range(n) if placements[i][0] == -1]
            if remaining_indices and time.time() < deadline - 5:
                 # Try sorting by max dimension descending? Or just retry.
                 # Let's try sorting by surface area ascending (smallest surface first)
                 remaining_boxes = [(i, boxes[i]) for i in remaining_indices]
                 remaining_boxes.sort(key=lambda x: 2*(x[1][0]*x[1][1] + x[1][1]*x[1][2] + x[1][0]*x[1][2]))
                 
                 # We need to reset the placements for these boxes? No, they are -1.
                 # try_place will overwrite if successful.
                 try_place(remaining_boxes, placed, placements, candidates)
    
    # Fill any remaining None (in case of timeout)
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
            
    return placements
