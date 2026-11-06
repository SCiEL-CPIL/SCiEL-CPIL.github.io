"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np
from itertools import permutations

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    # 150s limit, keep 10s margin for safety/overhead
    deadline = start_time + 140.0
    
    # Preprocess: calculate volume and sort by volume descending
    box_data = []
    for i, (w, h, d) in enumerate(boxes):
        vol = w * h * d
        box_data.append((vol, i, w, h, d))
    
    # Sort by volume descending (best-first)
    box_data.sort(key=lambda x: -x[0])
    
    # Get all 6 unique rotations for a box
    def get_rotations(w, h, d):
        rotations = set()
        for perm in permutations([w, h, d]):
            rotations.add(perm)
        return list(rotations)
    
    # Layer decomposition strategy
    # We will divide the container height H into layers of fixed thickness 'layer_h'
    # For each layer, we pack boxes using a 2D shelf heuristic on the W x D floor.
    # A box fits in a layer if its height (after rotation) <= layer_h.
    
    # Determine layer height. 
    # If we make layers too thin, we waste space. If too thick, we might not pack well.
    # Let's try a few different layer heights and pick the best result.
    # Common heuristic: layer height = min dimension of boxes? Or just a fraction of H?
    # Let's try layer heights based on the box heights themselves.
    # Actually, simpler: try a few fixed layer heights: H/2, H/3, H/4, H/5, H/10, H/20, etc.
    # Or better: try layer heights equal to the height of the largest box that fits?
    
    # Let's try a set of candidate layer heights.
    # To ensure we cover the space, we need to iterate through layers.
    # If a box is taller than the layer, it must be rotated to fit, or it goes in a different layer if possible.
    # But wait, if we fix layer height, a box with h > layer_h cannot be placed in that layer with that orientation.
    
    # Alternative interpretation of "time-sliced layer decomposition":
    # We process the container in horizontal slices.
    # Slice 1: z=0 to z=L1. Pack boxes with height <= L1.
    # Slice 2: z=L1 to z=L1+L2. Pack boxes with height <= L2.
    # This allows mixing boxes of different heights if we choose L1, L2 carefully.
    
    # Let's implement a generic layer packer.
    # Input: list of boxes to place, available width W, depth D, max height for this layer H_layer.
    # Output: placements for this layer.
    
    def pack_layer(boxes_to_place, W, D, H_layer, current_z_offset, placements):
        """
        Packs boxes into a layer of height H_layer starting at current_z_offset.
        Uses a 2D shelf heuristic on the W x D plane.
        Returns list of (idx, x, y, z, w, h, d) for placed boxes.
        """
        placed_in_layer = []
        # Sort boxes by volume descending again? Or keep global order?
        # Keeping global order (largest first) is usually good.
        
        # 2D Shelf Heuristic State
        # Shelves are horizontal strips in the W x D plane.
        # Each shelf has a y-coordinate (start of shelf), a height (depth used), and a current x-cursor.
        # Actually, standard shelf packing:
        # - Start a new shelf at y=0, height = max height of boxes in shelf? No, shelf height is determined by the tallest box in it.
        # - Place boxes in the shelf from left to right.
        # - When a box doesn't fit in the current shelf, start a new shelf at y = current_y + current_shelf_height.
        
        # We need to track shelves.
        # Shelf: (y_start, shelf_height, x_cursor)
        # Initially, no shelves.
        
        shelves = [] # list of [y_start, shelf_height, x_cursor]
        
        for vol, idx, w, h, d in boxes_to_place:
            # Try all rotations to find one that fits in H_layer and has best 2D fit
            rotations = get_rotations(w, h, d)
            best_rot = None
            best_shelf_idx = -1
            best_x = None
            best_y = None
            best_score = None # Minimize y_start + x_cursor? Or just y_start?
            
            for (rw, rh, rd) in rotations:
                if rh > H_layer:
                    continue
                
                # Try to fit (rw, rd) into the 2D shelf structure
                # Check existing shelves
                found_shelf = False
                for si, shelf in enumerate(shelves):
                    y_start, s_h, x_cur = shelf
                    # Box must fit in shelf height? 
                    # In strict shelf packing, the shelf height is fixed when the first box is placed, or it grows?
                    # Standard: Shelf height is determined by the first box. Subsequent boxes must fit within that height.
                    # If a new box is taller, it might need a new shelf or expand the shelf (which pushes other shelves down? No, shelves are stacked).
                    # If we expand shelf height, we might invalidate the y_start of subsequent shelves? 
                    # Usually, shelf height is fixed at creation. If a box doesn't fit height, it goes to a new shelf.
                    # However, if we allow "expanding" the shelf height, we must ensure it doesn't overlap with the next shelf.
                    # To keep it simple and robust: Shelf height is fixed. If box height > shelf height, it doesn't fit in this shelf.
                    
                    if rh > s_h:
                        continue
                    
                    # Check if box fits in width
                    if x_cur + rw <= W:
                        # It fits!
                        # Score: Prefer lower shelves (smaller y_start) and left-aligned (smaller x_cur)
                        # Actually, just taking the first fit is standard.
                        # Let's score by y_start + x_cur to prefer "bottom-left"
                        score = y_start * 1000 + x_cur
                        if best_score is None or score < best_score:
                            best_score = score
                            best_rot = (rw, rh, rd)
                            best_shelf_idx = si
                            best_x = x_cur
                            best_y = y_start
                            found_shelf = True
                        # Don't break, we might find a better shelf? 
                        # Usually first fit is fast. Best fit might be better.
                        # Let's stick to first fit for speed, but maybe check all shelves.
                    
                if found_shelf:
                    break
                
                # If not found in existing shelves, try to start a new shelf
                # But we can only start a new shelf if the current "y_cursor" + rh <= D (total depth limit? No, D is depth, H is height).
                # Wait, the layer is W x D x H_layer.
                # The 2D plane is W (width) x D (depth).
                # The shelves are stacked in the Depth (D) direction? Or Height?
                # "Horizontal slabs" usually implies slicing along the vertical axis (Z or H).
                # Inside a slab, we have a W x D footprint.
                # Shelf packing is usually done on a 2D rectangle.
                # Let's assume shelves are stacked along the Depth (D) axis.
                # So y in the 2D packing corresponds to the Depth coordinate.
                
                # Calculate current max y used (bottom of the next new shelf)
                if not shelves:
                    next_y = 0
                else:
                    # Find the shelf with the largest y_start + shelf_height
                    max_y = 0
                    for s in shelves:
                        max_y = max(max_y, s[0] + s[1])
                    next_y = max_y
                
                if next_y + rh <= D:
                    # Can start a new shelf
                    # Score for new shelf: next_y + 0
                    score = next_y * 1000 + 0
                    if best_score is None or score < best_score:
                        best_score = score
                        best_rot = (rw, rh, rd)
                        best_shelf_idx = -1 # Special value for new shelf
                        best_x = 0
                        best_y = next_y
                        found_shelf = True # Reusing flag, but meaning is "can place"
                
                if found_shelf:
                    break
            
            if best_rot is not None:
                rw, rh, rd = best_rot
                if best_shelf_idx >= 0:
                    # Place in existing shelf
                    shelves[best_shelf_idx][2] += rw # Update x cursor
                else:
                    # Start new shelf
                    shelves.append([best_y, rh, rw]) # y_start, shelf_height, x_cursor (starts at 0, so cursor is rw)
                
                # Record placement
                # z = current_z_offset
                # x = best_x
                # y = best_y (this is the depth coordinate)
                # w = rw, h = rh, d = rd
                placements[idx] = (0, best_x, best_y, current_z_offset, rw, rh, rd)
                placed_in_layer.append(idx)
            # else: Box doesn't fit in this layer, leave it out for now (or try in next layer if possible? 
            # But our loop structure processes boxes in order. If it doesn't fit here, it's skipped for this layer.
            # We need a mechanism to carry over unplaced boxes to the next layer.
            # This function only handles the boxes passed to it.
            
        return placed_in_layer

    # Main Logic:
    # We need to decide on layer heights.
    # Let's try a few strategies for layer heights:
    # 1. Uniform layers: H/2, H/3, H/4
    # 2. Adaptive layers: Start with the largest box height, then next, etc.
    
    # Since we have a time limit, let's try a few simple configurations and keep the best one.
    
    best_result = None
    best_volume = 0
    
    # Candidate layer configurations
    # We will define layers as a list of (z_start, z_end)
    # Config 1: 2 layers
    # Config 2: 3 layers
    # Config 3: 4 layers
    # Config 4: Many thin layers (e.g., 10)
    
    num_trials = 5
    for trial in range(num_trials):
        if time.time() > deadline:
            break
            
        # Generate layer heights
        # Let's pick random or systematic splits
        if trial == 0:
            num_layers = 2
            layer_h = H / num_layers
            layer_heights = [layer_h] * num_layers
        elif trial == 1:
            num_layers = 3
            layer_h = H / num_layers
            layer_heights = [layer_h] * num_layers
        elif trial == 2:
            num_layers = 4
            layer_h = H / num_layers
            layer_heights = [layer_h] * num_layers
        elif trial == 3:
            num_layers = 5
            layer_h = H / num_layers
            layer_heights = [layer_h] * num_layers
        else:
            # Random split
            num_layers = random.randint(2, 6)
            # Random proportions
            weights = [random.random() for _ in range(num_layers)]
            total_w = sum(weights)
            layer_heights = [w / total_w * H for w in weights]
        
        # Initialize placements
        placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
        
        # Unplaced boxes
        unplaced = list(box_data) # (vol, idx, w, h, d)
        
        current_z = 0
        
        for lh in layer_heights:
            if time.time() > deadline:
                break
            
            # Filter unplaced boxes to those that can potentially fit in this layer height
            # A box fits if any of its rotations has height <= lh
            # We don't filter here, we let pack_layer handle it.
            # But pack_layer takes a list. We pass all unplaced boxes.
            # However, pack_layer modifies placements directly.
            # It returns the list of indices placed.
            
            placed_indices = pack_layer(unplaced, W, D, lh, current_z, placements)
            
            # Remove placed boxes from unplaced
            if placed_indices:
                placed_set = set(placed_indices)
                unplaced = [b for b in unplaced if b[1] not in placed_set]
            
            current_z += lh
            
            # If no boxes left, break
            if not unplaced:
                break
        
        # Calculate total volume placed
        total_vol = 0
        for i in range(n):
            if placements[i][0] == 0:
                _, x, y, z, w, h, d = placements[i]
                total_vol += w * h * d
        
        if total_vol > best_volume:
            best_volume = total_vol
            best_result = [p for p in placements]
            
    if best_result is None:
        # Fallback: try the original greedy approach if layer packing failed completely?
        # Or just return empty/invalid?
        # Let's return a default invalid placement to be safe, or try a simple greedy.
        # Actually, if best_result is None, it means no box was placed in any trial.
        # Let's just return the initial invalid placements.
        return [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
        
    return best_result
# EVOLVE-BLOCK-END
