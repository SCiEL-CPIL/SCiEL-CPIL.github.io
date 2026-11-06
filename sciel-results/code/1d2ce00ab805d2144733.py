"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Preprocess boxes: store (w, h, d, index)
    # We will sort them.
    box_list = [(w, h, d, i) for i, (w, h, d) in enumerate(boxes)]
    
    # Sort by volume descending (Largest First)
    box_list.sort(key=lambda x: x[0]*x[1]*x[2], reverse=True)
    
    # Result storage
    placements = [(-1, 0, 0, 0, 0, 0, 0) for _ in range(n)]
    
    def get_rotations(w, h, d):
        """Yield unique rotations of a box."""
        seen = set()
        rots = []
        # Permutations of (w, h, d)
        for i in range(3):
            for j in range(3):
                if i != j:
                    k = 6 - i - j
                    r = (w[i], h[j], d[k]) # This is wrong, need to permute dims
        # Correct way to get rotations:
        dims = [w, h, d]
        for i in range(3):
            for j in range(3):
                if i != j:
                    k = 3 - i - j
                    # Wait, indices 0,1,2. 
                    # Permutations of (0,1,2):
                    # (0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)
                    pass
        
        # Simpler explicit list of 6 permutations
        perms = [
            (0, 1, 2),
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0)
        ]
        for p in perms:
            rw = dims[p[0]]
            rh = dims[p[1]]
            rd = dims[p[2]]
            if (rw, rh, rd) not in seen:
                seen.add((rw, rh, rd))
                rots.append((rw, rh, rd))
        return rots

    def pack():
        """
        Attempt to pack all boxes using the Box-Tree method.
        Returns a list of placements for this attempt.
        """
        # Free spaces: list of (x, y, z, w, h, d)
        # Start with the whole container
        free_spaces = [(0, 0, 0, W, H, D)]
        
        current_placements = [(-1, 0, 0, 0, 0, 0, 0) for _ in range(n)]
        
        # Iterate through sorted boxes
        for bw, bh, bd, idx in box_list:
            if time.time() > deadline:
                return current_placements # Return partial result
            
            best_pos = None
            best_space_idx = -1
            best_rotation = None
            best_score = float('inf')
            
            # Try all rotations
            for rw, rh, rd in get_rotations(bw, bh, bd):
                # Check all free spaces
                for i, (sx, sy, sz, sw, sh, sd) in enumerate(free_spaces):
                    if rw <= sw and rh <= sh and rd <= sd:
                        # It fits. Calculate a score to pick the "best" space.
                        # Heuristic: Smallest volume free space that fits (to fill gaps)
                        # Tie-breaker: Smallest sum of coordinates (keep compact)
                        vol = sw * sh * sd
                        coord_sum = sx + sy + sz
                        
                        # We want to minimize volume, then coord_sum
                        if vol < best_score or (vol == best_score and coord_sum < best_coord_sum):
                            best_score = vol
                            best_coord_sum = coord_sum
                            best_space_idx = i
                            best_rotation = (rw, rh, rd)
                            best_pos = (sx, sy, sz)
            
            if best_space_idx != -1:
                # Place the box
                px, py, pz = best_pos
                rw, rh, rd = best_rotation
                current_placements[idx] = (0, px, py, pz, rw, rh, rd)
                
                # Update free spaces
                # Remove the space we used
                sx, sy, sz, sw, sh, sd = free_spaces.pop(best_space_idx)
                
                # Create new sub-spaces
                # 1. Right of the box
                if sw > rw:
                    free_spaces.append((px + rw, py, pz, sw - rw, sh, sd))
                # 2. Above the box
                if sh > rh:
                    free_spaces.append((px, py + rh, pz, sw, sh - rh, sd))
                # 3. Front of the box
                if sd > rd:
                    free_spaces.append((px, py, pz + rd, sw, sh, sd - rd))
                
                # Optional: Merge adjacent free spaces to reduce list size?
                # For N=60, list size is small, so merging is not strictly necessary for performance,
                # but it can improve heuristic quality by creating larger contiguous spaces.
                # However, merging is O(K^2). With K~60, it's fast.
                # Let's implement a simple merge: if two free spaces share a full face and align, merge them.
                # This is complex in 3D. Let's skip merging for now to keep code simple and fast,
                # as the "smallest fit" heuristic works well with fragmented spaces too.
            
            # If no space found, box is left out (already -1)
        
        return current_placements

    best_placements = None
    best_volume = -1
    
    # Run the greedy packing
    # We can try multiple times with slight variations if time permits,
    # but the deterministic "Largest First + Smallest Fit" is a strong baseline.
    
    # To add robustness, let's try a few different sorting orders if time is available.
    # 1. Volume Descending
    # 2. Max Dimension Descending
    # 3. Random (a few times)
    
    orders = [
        lambda x: x[0]*x[1]*x[2],
        lambda x: max(x[0], x[1], x[2]),
        lambda x: x[0]*x[1]*x[2] + 0.1*max(x[0], x[1], x[2])
    ]
    
    for order_func in orders:
        if time.time() > deadline:
            break
            
        # Create a new sorted list
        box_list_copy = [(w, h, d, i) for i, (w, h, d) in enumerate(boxes)]
        box_list_copy.sort(key=lambda x: order_func((x[0], x[1], x[2])), reverse=True)
        
        # Temporarily swap box_list
        original_box_list = box_list
        box_list = box_list_copy
        
        placements_attempt = pack()
        
        # Restore
        box_list = original_box_list
        
        # Calculate volume
        vol = 0
        for p in placements_attempt:
            if p[0] == 0:
                vol += p[4] * p[5] * p[6]
        
        if vol > best_volume:
            best_volume = vol
            best_placements = placements_attempt
    
    # If we still have time, do a few random restarts
    if time.time() < deadline - 5.0:
        for _ in range(10):
            if time.time() > deadline:
                break
            box_list_copy = [(w, h, d, i) for i, (w, h, d) in enumerate(boxes)]
            # Sort by volume but add random noise to break ties or slightly perturb order
            box_list_copy.sort(key=lambda x: x[0]*x[1]*x[2] + random.uniform(0, 10), reverse=True)
            
            original_box_list = box_list
            box_list = box_list_copy
            
            placements_attempt = pack()
            
            box_list = original_box_list
            
            vol = 0
            for p in placements_attempt:
                if p[0] == 0:
                    vol += p[4] * p[5] * p[6]
            
            if vol > best_volume:
                best_volume = vol
                best_placements = placements_attempt

    if best_placements is None:
        # Fallback to simple sequential if something went wrong
        return [(-1, 0, 0, 0, 0, 0, 0) for _ in range(n)]
        
    return best_placements
# EVOLVE-BLOCK-END
