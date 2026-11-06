import numpy as np

def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Prepare box options
    box_data = []
    for i, (w, h, d) in enumerate(boxes):
        perms = set()
        perms.add((w, h, d))
        perms.add((w, d, h))
        perms.add((h, w, d))
        perms.add((h, d, w))
        perms.add((d, w, h))
        perms.add((d, h, w))
        
        valid_perms = []
        for p in perms:
            if p[0] <= W and p[1] <= H and p[2] <= D:
                valid_perms.append(p)
        
        if not valid_perms:
            box_data.append((i, [(w, h, d)], -1)) # -1 volume indicates invalid? No, just keep original
            # Actually if no valid perm, it can't be placed.
            # But we must return a result for it.
            # We'll mark it as "cannot place" early?
            # Let's keep it in the list but it will fail placement.
            box_data.append((i, [(w, h, d)], w*h*d))
        else:
            # Sort perms by height descending? Or keep all?
            # We will try all perms during placement.
            box_data.append((i, valid_perms, w*h*d))
            
    # Sort by Volume Descending, then Max Dim Descending
    # To break ties, we look at the max dimension of the box (invariant)
    def sort_key(item):
        idx, perms, vol = item
        w, h, d = boxes[idx]
        max_dim = max(w, h, d)
        return (-vol, -max_dim)
        
    box_data.sort(key=sort_key)
    
    # Shelf Structure
    # Each shelf: { 'y': int, 'height': int, 'items': list of (x, z, w, d, box_idx, h) }
    shelves = []
    results = [None] * len(boxes)
    
    # Helper to check 2D overlap and find position
    def try_place_in_shelf(shelf, bw, bh, bd):
        """
        Try to place a box of size (bw, bh, bd) into the shelf.
        Returns (x, z) if successful, None otherwise.
        Updates shelf if successful.
        """
        # Generate candidate X and Z
        xs = {0}
        zs = {0}
        for (px, pz, pw, pd, pidx, ph) in shelf['items']:
            xs.add(px)
            xs.add(px + pw)
            zs.add(pz)
            zs.add(pz + pd)
            
        xs = sorted(xs)
        zs = sorted(zs)
        
        # Try all combinations
        # Priority: Low Z, then Low X (Back-Left)
        for cz in zs:
            if cz + bd > D:
                continue
            for cx in xs:
                if cx + bw > W:
                    continue
                    
                # Check overlap
                overlap = False
                for (px, pz, pw, pd, pidx, ph) in shelf['items']:
                    # Check X and Z overlap
                    if not (cx + bw <= px or px + pw <= cx):
                        if not (cz + bd <= pz or pz + pd <= cz):
                            overlap = True
                            break
                if not overlap:
                    return (cx, cz)
        return None

    for idx, perms, vol in box_data:
        placed = False
        # Try to place in the current (last) shelf
        if shelves:
            shelf = shelves[-1]
            # Try all orientations
            for (bw, bh, bd) in perms:
                # Check if it fits in height of container from this shelf
                if shelf['y'] + bh > H:
                    continue
                
                # Check if it fits in shelf height? 
                # No, shelf height expands.
                # But we must ensure the box doesn't stick out of the container.
                
                pos = try_place_in_shelf(shelf, bw, bh, bd)
                if pos is not None:
                    cx, cz = pos
                    # Place it
                    shelf['items'].append((cx, cz, bw, bd, idx, bh))
                    shelf['height'] = max(shelf['height'], bh)
                    results[idx] = (0, cx, shelf['y'], cz, bw, bh, bd)
                    placed = True
                    break
            
            if placed:
                continue
        
        # If not placed in current shelf, try to open a new shelf
        # Find the lowest y where a new shelf can start
        # The new shelf starts at max(shelf['y'] + shelf['height'] for all shelves)
        # Actually, shelves are stacked sequentially.
        # So new_y = sum of heights of all previous shelves?
        # No, we just append to the list. The new shelf is at the top.
        
        # Calculate next Y
        next_y = 0
        if shelves:
            last_shelf = shelves[-1]
            next_y = last_shelf['y'] + last_shelf['height']
        else:
            next_y = 0
            
        # Check if any orientation fits in the remaining height
        valid_orientations = []
        for (bw, bh, bd) in perms:
            if next_y + bh <= H:
                valid_orientations.append((bw, bh, bd))
                
        if valid_orientations:
            # Create new shelf
            new_shelf = {'y': next_y, 'height': 0, 'items': []}
            shelves.append(new_shelf)
            
            # Try to place the box in this new (empty) shelf
            # It will always fit in 2D if it fits in W, D
            # We just pick the first valid orientation
            # To be consistent, pick the one with smallest height? Or first?
            # Let's pick the one that minimizes height to leave room for others?
            # Or just the first one.
            # Let's try to find the best orientation for the new shelf start.
            # Usually, orienting with smallest height is good for stacking.
            valid_orientations.sort(key=lambda x: x[1])
            
            bw, bh, bd = valid_orientations[0]
            # Place at 0,0 in the new shelf
            new_shelf['items'].append((0, 0, bw, bd, idx, bh))
            new_shelf['height'] = bh
            results[idx] = (0, 0, next_y, 0, bw, bh, bd)
            placed = True
        else:
            # Cannot place
            results[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
            
    # Fill any remaining None results (should not happen if logic is correct)
    for i in range(len(boxes)):
        if results[i] is None:
            results[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
            
    return results
