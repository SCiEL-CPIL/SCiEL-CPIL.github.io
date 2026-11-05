"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import permutations

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    container_vol = W * H * D
    if container_vol <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Deadline guard: keep a safety margin of 10s before the 150s limit
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Generate all 6 rotations for each box
    def get_rotations(w, h, d):
        return list(set(permutations([w, h, d])))
    
    # For each box, store its dimensions and possible rotations
    box_info = []
    for i, (w, h, d) in enumerate(boxes):
        rots = get_rotations(w, h, d)
        box_info.append((i, rots))
    
    # Sort boxes by volume descending (larger boxes first)
    box_info.sort(key=lambda x: x[1][0][0] * x[1][0][1] * x[1][0][2], reverse=True)
    
    # Greedy placement with multiple strategies
    # Strategy 1: Simple shelf packing with rotation selection
    def greedy_shelf_placement(box_info, W, H, D, deadline):
        placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(len(boxes))]
        
        # Shelf-based approach: place boxes in shelves
        # Try to fit boxes into shelves along the W dimension
        shelves = []  # list of (y, z, shelf_h, shelf_d, placed_boxes)
        
        current_y = 0
        current_z = 0
        current_shelf_h = 0
        current_shelf_d = 0
        
        for idx, (orig_idx, rots) in enumerate(box_info):
            if time.time() > deadline:
                break
            
            w, h, d = boxes[orig_idx]
            placed = False
            
            # Try to find a rotation that fits in current shelf or new position
            for rot in rots:
                rw, rh, rd = rot
                # Check if fits in current shelf
                if current_y + rh <= H and current_z + rd <= D:
                    # Try to place in current shelf
                    # We need to track x position per shelf
                    # For simplicity, let's use a different approach
                    
                    pass
            
            # Let's use a simpler greedy: try to place at (0, current_y, current_z) with best rotation
            best_rot = None
            best_score = -1
            for rot in rots:
                rw, rh, rd = rot
                # Can it fit starting from origin in the remaining space?
                # This is a simplification; we'll use a more careful check below
                pass
            
            # Actually, let's use a proper shelf algorithm
            # We'll maintain a list of shelves, each with available x-space
            # For now, let's do a simpler approach: try to place at front-left-bottom
            
            # Find the best position and rotation
            best_pos = None
            best_rot_chosen = None
            best_priority = None
            
            for rot in rots:
                rw, rh, rd = rot
                # Try to place at (0, y, z) for various y, z
                # Simplified: try placing at the "current" position
                # Check if it fits in the container
                if rw <= W and rh <= H and rd <= D:
                    # Score based on how well it fits (prefer tighter fits)
                    # We'll just place it if it fits in the remaining space
                    pass
            
            # Let's use a simpler but effective approach:
            # Maintain a 3D grid or use a shelf-based method properly
            
            # For this implementation, let's use a basic shelf packing:
            # Place boxes in rows along W, stacking rows along H, and layers along D
            
            if not placed:
                # Try each rotation
                for rot in rots:
                    rw, rh, rd = rot
                    if rw > W or rh > H or rd > D:
                        continue
                    
                    # Try to place in the current "cell"
                    # We'll track: x_pos (current x in shelf), shelf_y, shelf_z
                    # shelf_h = height of current shelf, shelf_d = depth of current layer
                    
                    # Check if we can extend current shelf
                    if shelf_y + rh <= H and shelf_z + rd <= D:
                        # Check x-space: we need to know how much x is used
                        # This simple approach doesn't track x well
                        pass
                
                # Fallback: just try to place at (0, 0, 0) with the first fitting rotation
                for rot in rots:
                    rw, rh, rd = rot
                    if rw <= W and rh <= H and rd <= D:
                        placements[orig_idx] = (0, 0, 0, 0, rw, rh, rd)
                        placed = True
                        break
        
        return placements
    
    # Let's implement a proper shelf-based greedy algorithm
    def proper_shelf_placement(box_info, W, H, D, deadline):
        n = len(boxes)
        placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
        
        # Shelf structure: list of shelves
        # Each shelf: (y_start, z_start, height, depth, x_used, boxes_in_shelf)
        # We'll process shelves in order
        
        shelves = []  # list of dicts
        current_shelf_idx = -1
        
        for idx, (orig_idx, rots) in enumerate(box_info):
            if time.time() > deadline:
                break
            
            w, h, d = boxes[orig_idx]
            placed = False
            
            # Try to place in an existing shelf first
            for si, shelf in enumerate(shelves):
                if time.time() > deadline:
                    break
                
                y0, z0, shelf_h, shelf_d, x_used = shelf['y'], shelf['z'], shelf['h'], shelf['d'], shelf['x_used']
                
                for rot in rots:
                    rw, rh, rd = rot
                    # Check if rotation fits in shelf dimensions
                    if rh > shelf_h or rd > shelf_d:
                        continue
                    # Check if x-space available
                    if x_used + rw <= W:
                        # Place it
                        x_pos = x_used
                        placements[orig_idx] = (0, x_pos, y0, z0, rw, rh, rd)
                        shelf['x_used'] = x_used + rw
                        placed = True
                        break
                
                if placed:
                    break
            
            if placed:
                continue
            
            # Try to create a new shelf
            for rot in rots:
                rw, rh, rd = rot
                if rw > W or rh > H or rd > D:
                    continue
                
                # Find a position for a new shelf
                # Try to stack shelves along H and D
                # For simplicity, try placing at (0, y, z) for various y, z
                found_pos = False
                for y in range(0, H - rh + 1, 1):
                    if time.time() > deadline:
                        break
                    for z in range(0, D - rd + 1, 1):
                        # Check if this position is free (simplified: just check bounds)
                        # A more careful check would be needed, but for greedy this is approximate
                        # We'll just try to place it
                        placements[orig_idx] = (0, 0, y, z, rw, rh, rd)
                        shelves.append({
                            'y': y, 'z': z, 'h': rh, 'd': rd, 'x_used': rw
                        })
                        found_pos = True
                        break
                    if found_pos:
                        break
                if found_pos:
                    placed = True
                    break
            
            if not placed:
                # Leave it out
                placements[orig_idx] = (-1, 0, 0, 0, w, h, d)
        
        return placements
    
    # Let's implement a cleaner, more effective greedy algorithm
    def greedy_3d_placement(box_info, W, H, D, deadline):
        n = len(boxes)
        placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
        
        # We'll use a simple approach: try to place each box at the lowest, leftmost, frontmost position
        # that fits, trying all rotations
        
        # Track occupied space using a list of placed boxes
        placed_boxes = []  # list of (x, y, z, w, h, d, orig_idx)
        
        def can_place(x, y, z, rw, rh, rd):
            """Check if a box at (x,y,z) with dims (rw,rh,rd) doesn't overlap and is in bounds."""
            if x < 0 or y < 0 or z < 0:
                return False
            if x + rw > W or y + rh > H or z + rd > D:
                return False
            
            # Check overlap with existing placed boxes
            for (px, py, pz, pw, ph, pd, _) in placed_boxes:
                # Check if boxes overlap
                if (x < px + pw and x + rw > px and
                    y < py + ph and y + rh > py and
                    z < pz + pd and z + rd > pz):
                    return False
            return True
        
        def find_best_position(rw, rh, rd):
            """Find the best (x, y, z) position for a box of size (rw, rh, rd)."""
            best_pos = None
            best_score = -1
            
            # Try to place at various positions
            # Prioritize lower y, then lower z, then lower x
            for y in range(0, H - rh + 1):
                for z in range(0, D - rd + 1):
                    for x in range(0, W - rw + 1):
                        if can_place(x, y, z, rw, rh, rd):
                            # Score: prefer positions closer to origin
                            score = -(x + y + z)
                            if score > best_score:
                                best_score = score
                                best_pos = (x, y, z)
                            # Since we iterate from origin, the first valid is best
                            return (x, y, z)
            return None
        
        for idx, (orig_idx, rots) in enumerate(box_info):
            if time.time() > deadline:
                break
            
            w, h, d = boxes[orig_idx]
            placed = False
            
            # Try each rotation
            for rot in rots:
                rw, rh, rd = rot
                if rw > W or rh > H or rd > D:
                    continue
                
                pos = find_best_position(rw, rh, rd)
                if pos is not None:
                    x, y, z = pos
                    placements[orig_idx] = (0, x, y, z, rw, rh, rd)
                    placed_boxes.append((x, y, z, rw, rh, rd, orig_idx))
                    placed = True
                    break
            
            if not placed:
                placements[orig_idx] = (-1, 0, 0, 0, w, h, d)
        
        return placements
    
    # Run the greedy placement
    result = greedy_3d_placement(box_info, W, H, D, deadline)
    
    # Calculate filled volume
    filled_vol = 0
    for p in result:
        if p[0] == 0:
            filled_vol += p[4] * p[5] * p[6]
    
    return result
# EVOLVE-BLOCK-END
