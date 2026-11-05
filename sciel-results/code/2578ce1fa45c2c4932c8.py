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

    def get_orientation_score(rw, rh, rd, cx, cy, cz, placed_list, W, H, D):
        """
        Score a placement: lower is better.
        Priority:
        1. Minimize z (bottom) - pack from bottom up
        2. Minimize x (left) - pack from left
        3. Minimize y (front) - pack from front
        4. Prefer orientations that minimize the "wasted space" above the box
           (i.e., prefer flatter boxes to allow more stacking)
        5. Prefer orientations that align better with container dimensions
        """
        # Primary: position (bottom-left-front)
        # Secondary: prefer orientations where the height (rd) is smaller
        # to allow more boxes to be stacked on top
        # Tertiary: prefer orientations where the box fits more snugly
        # in the available space (less leftover in each dimension)
        
        # Calculate how much space is left above, to the right, and behind
        space_above = D - (cz + rd)
        space_right = W - (cx + rw)
        space_behind = H - (cy + rh)
        
        # Penalize creating large gaps that can't be filled
        # A box that's too tall creates a gap above that might not be usable
        # A box that's too wide creates a gap to the right
        
        # Score components:
        # 1. Position score (primary)
        pos_score = (cz, cx, cy)
        
        # 2. Height penalty: prefer shorter boxes (larger space_above is good for stacking)
        #    But we want to minimize the "wasted" height, so prefer smaller rd
        height_penalty = rd
        
        # 3. Footprint penalty: prefer boxes that leave usable space
        #    Calculate the "wasted" space in each direction
        #    If a box is very wide (rw close to W), it blocks the whole width
        #    If a box is very deep (rd close to D), it blocks the whole depth
        
        # Use a smooth penalty: the more space left, the better
        # But we want to avoid creating small unusable gaps
        # Heuristic: prefer orientations where the remaining space in each dimension
        # is either 0 or large enough to fit another box
        
        # Simple approach: penalize based on the "fragmentation" of remaining space
        # If space_above is small but non-zero, it's wasted
        # If space_right is small but non-zero, it's wasted
        
        # Let's use: min(space_above, min_box_height) as "usable" space
        # But we don't know min_box_height easily, so just prefer larger remaining spaces
        
        # Actually, let's keep it simple and effective:
        # Primary: (cz, cx, cy)
        # Secondary: prefer smaller rd (height) to allow stacking
        # Tertiary: prefer orientations that maximize the minimum remaining dimension
        #           (to avoid creating narrow unusable gaps)
        
        min_remaining = min(space_above, space_right, space_behind)
        
        return (pos_score, height_penalty, -min_remaining)

    def try_place(boxes_to_place, placed_list, placements_list, candidates_list):
        """
        Attempts to place boxes in boxes_to_place into the current state.
        Returns the number of successfully placed boxes.
        """
        placed_count = 0
        
        # Use a set to deduplicate candidates for efficiency
        candidate_set = set(candidates_list)
        
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
                for cx, cy, cz in candidate_set:
                    # Check bounds
                    if cx + rw > W or cy + rh > H or cz + rd > D:
                        continue
                        
                    # Check overlap with placed boxes
                    if check_overlap(cx, cy, cz, rw, rh, rd, placed_list):
                        continue
                    
                    # Score: Bottom-Left-Front, then prefer "flatter" or more compact?
                    # Let's use the new scoring function
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
                # Also add points that are adjacent to the box in each direction
                # to fill gaps more effectively
                
                new_candidates = [
                    # Top face corners (for stacking)
                    (x, y, z + rd),
                    (x + rw, y, z + rd),
                    (x, y + rh, z + rd),
                    (x + rw, y + rh, z + rd),
                    # Right face (for placing boxes to the right)
                    (x + rw, y, z),
                    (x + rw, y + rh, z),
                    (x + rw, y, z + rd),
                    (x + rw, y + rh, z + rd),
                    # Back face (for placing boxes behind)
                    (x, y + rh, z),
                    (x + rw, y + rh, z),
                    (x, y + rh, z + rd),
                    (x + rw, y + rh, z + rd),
                ]
                
                # Add only new candidates to the set
                for nc in new_candidates:
                    candidate_set.add(nc)
            else:
                placements_list[orig_idx] = (-1, 0, 0, 0, w, h, d)
                
        # Update the candidates list to match the set
        candidates_list.extend(candidate_set - set(candidates_list))
        
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
            
            try_place(unplaced_boxes, placed, placements, candidates)
            
            remaining_indices = [i for i in range(n) if placements[i][0] == -1]
            if remaining_indices and time.time() < deadline - 5:
                 # Try sorting by surface area ascending (smallest surface first)
                 remaining_boxes = [(i, boxes[i]) for i in remaining_indices]
                 remaining_boxes.sort(key=lambda x: 2*(x[1][0]*x[1][1] + x[1][1]*x[1][2] + x[1][0]*x[1][2]))
                 
                 try_place(remaining_boxes, placed, placements, candidates)
                 
                 # Pass 4: Retry remaining by volume descending?
                 remaining_indices = [i for i in range(n) if placements[i][0] == -1]
                 if remaining_indices and time.time() < deadline - 2:
                     remaining_boxes = [(i, boxes[i]) for i in remaining_indices]
                     remaining_boxes.sort(key=lambda x: x[1][0]*x[1][1]*x[1][2], reverse=True)
                     try_place(remaining_boxes, placed, placements, candidates)
                     
                     # Pass 5: Try sorting by max dimension ascending (smallest max first)
                     remaining_indices = [i for i in range(n) if placements[i][0] == -1]
                     if remaining_indices and time.time() < deadline - 1:
                         remaining_boxes = [(i, boxes[i]) for i in remaining_indices]
                         remaining_boxes.sort(key=lambda x: max(x[1][0], x[1][1], x[1][2]))
                         try_place(remaining_boxes, placed, placements, candidates)
    
    # Fill any remaining None (in case of timeout)
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
            
    return placements
