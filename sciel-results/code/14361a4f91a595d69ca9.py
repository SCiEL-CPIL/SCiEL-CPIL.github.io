"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    deadline = time.time() + 110.0  # 10s safety margin
    n = len(boxes)
    
    # Initialize placements with all boxes left out
    # Format: (bin, x, y, z, w, h, d)
    # bin=0 for loaded, bin=-1 for left out
    # For left out, w,h,d are original dims, x,y,z=0
    initial_placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    best_placements = initial_placements
    best_volume = 0
    
    def get_orientations(w, h, d):
        """Return unique orientations of a box (w, h, d)."""
        orients = set()
        for perm in itertools.permutations([w, h, d]):
            orients.add(perm)
        return list(orients)
    
    def shelf_pack(ordered_indices):
        """
        Pack boxes using a shelf-based algorithm.
        ordered_indices: list of original box indices to try in this order.
        Returns (placements_list, total_volume).
        """
        placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
        total_volume = 0
        
        # Shelves are organized in layers.
        # Each layer has a fixed y-start (base_y) and a height (layer_h).
        # Within a layer, we have shelves. Each shelf has a start_x and current_x.
        # Actually, a simpler model:
        # We maintain a list of "free spaces" or just iterate layers.
        # Let's use a list of shelves.
        # A shelf is defined by: y_pos, height, next_x.
        # But shelves in the same layer share the same y_pos and height?
        # No, typically shelf packing means:
        # Layer 1: y=0, height=h1. Inside, we place boxes side-by-side.
        # When a row is full or a box doesn't fit, we start a new row (increase y within layer? No, shelf packing usually means horizontal shelves).
        
        # Let's use a more robust "Bottom-Left-Fill" or "Shelf" approach.
        # Standard Shelf Packing:
        # 1. Sort items.
        # 2. Create a shelf at current y.
        # 3. Place items on the shelf from left to right.
        # 4. When an item doesn't fit in width, start a new shelf at y + current_shelf_height.
        # 5. When a shelf exceeds container height, we are done (or overflow).
        
        # However, this simple shelf packing doesn't handle "holes" well.
        # A better approach for 3D is to treat it as a 2D problem in the X-Z plane for each Y-layer?
        # Or simply:
        # We maintain a list of available "slots" in the container.
        # Given the time constraint and simplicity, let's stick to a strict shelf method but optimized.
        
        # Actually, let's use a "Next Fit Decreasing Height" style shelf packing.
        # We maintain a list of shelves. Each shelf has:
        # y: starting y coordinate
        # h: height of the shelf (determined by the first item placed on it? or max item height?)
        # x: next available x coordinate
        # 
        # Wait, if we define shelf height by the first item, subsequent items must be shorter or equal?
        # Or does the shelf height grow?
        # Standard shelf: The shelf height is fixed when the first item is placed. Subsequent items must fit within that height.
        # If an item is taller, it must start a new shelf.
        
        shelves = [] # List of dicts: {'y': float, 'h': float, 'x': float}
        current_y = 0
        
        for idx in ordered_indices:
            if time.time() > deadline:
                break
            
            w, h, d = boxes[idx]
            orients = get_orientations(w, h, d)
            
            # Try to place on existing shelves
            placed = False
            for shelf in shelves:
                # Check if we can fit an orientation of this box on this shelf
                for ow, oh, od in orients:
                    if oh <= shelf['h'] and shelf['x'] + ow <= W:
                        # Place it
                        placements[idx] = (0, int(shelf['x']), int(shelf['y']), 0, ow, oh, od)
                        shelf['x'] += ow
                        total_volume += ow * oh * od
                        placed = True
                        break
                if placed:
                    break
            
            if placed:
                continue
            
            # If not placed on existing shelves, try to start a new shelf
            # Find an orientation that fits in the remaining height
            best_orient = None
            for ow, oh, od in orients:
                if current_y + oh <= H and ow <= W:
                    # Prefer orientations that minimize height to fit more shelves?
                    # Or just take the first valid one?
                    # Let's pick the one with smallest height to maximize potential shelves.
                    if best_orient is None or oh < best_orient[1]:
                        best_orient = (ow, oh, od)
            
            if best_orient:
                ow, oh, od = best_orient
                new_shelf = {'y': current_y, 'h': oh, 'x': ow}
                shelves.append(new_shelf)
                current_y += oh
                placements[idx] = (0, 0, int(current_y - oh), 0, ow, oh, od)
                total_volume += ow * oh * od
            # Else: box cannot fit in remaining height, leave it out
        
        return placements, total_volume
    
    # Generate different sorting strategies
    indexed_boxes = list(range(n))
    
    strategies = [
        lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], # Volume
        lambda i: max(boxes[i][0], boxes[i][1], boxes[i][2]), # Max dim
        lambda i: boxes[i][0] + boxes[i][1] + boxes[i][2], # Sum of dims
        lambda i: boxes[i][1], # Height (assuming H is the stacking axis in shelf pack? No, shelf pack uses H as total height, y is stacking axis)
        lambda i: boxes[i][0], # Width
        lambda i: boxes[i][2], # Depth
    ]
    
    # Note: In my shelf_pack above, the "height" of the shelf corresponds to the 'h' dimension of the box in the orientation.
    # The stacking axis is Y (0 to H).
    # The width axis is X (0 to W).
    # The depth axis is Z (0 to D).
    # Wait, the shelf packing I implemented ignores Z entirely!
    # It treats the problem as 2D (X and Y), assuming depth D is not a constraint or is handled by the box's 'd' dimension?
    # No, the box has dimensions w, h, d.
    # In my placement: (0, x, y, 0, ow, oh, od).
    # The box occupies [x, x+ow] in X, [y, y+oh] in Y, and [0, od] in Z.
    # This assumes all boxes are placed at Z=0. This is a 2D packing in the X-Y plane, with depth 'od' being the extent in Z.
    # This is only valid if we don't care about Z-overlap or if all boxes have the same depth or fit within D.
    # The problem allows rotation.
    # If I place a box with depth od, it occupies Z from 0 to od.
    # If I place another box next to it in X, it also occupies Z from 0 to od2.
    # They don't overlap in Z as long as they are side-by-side in X?
    # No, if Box A is at x=0, w=10, d=5. Box B is at x=10, w=10, d=5.
    # They occupy disjoint X ranges, so they don't overlap regardless of Z.
    # So, placing them side-by-side in X is safe.
    # What if I stack them in Y?
    # Box A at y=0, h=5, d=5. Box B at y=5, h=5, d=5.
    # Disjoint Y ranges, safe.
    # What if I try to place a box in Z?
    # My algorithm doesn't use Z for positioning, it fixes z=0.
    # This means I am only packing in the X-Y plane.
    # This is a valid strategy if the container's Z dimension D is large enough to hold the deepest box, OR if we rotate boxes such that their depth is minimized?
    # Actually, if D is small, this strategy fails for boxes with large depth.
    # However, the prompt says "any axis rotation allowed".
    # If I rotate a box such that its smallest dimension is in Z, it might fit.
    # But my algorithm forces z=0.
    # This is a significant limitation.
    
    # Let's refine the algorithm to be a proper 3D shelf packing or at least handle Z.
    # A simple 3D extension:
    # Instead of just X and Y, we can have "layers" in Z as well?
    # Or, we can treat the container as a set of 2D shelves in the X-Y plane, but we can stack shelves in Z?
    # No, usually "shelf packing" in 3D means:
    # 1. Place boxes on a shelf (X-Y plane at a specific Z? No, shelf is usually horizontal).
    # Let's stick to the 2D X-Y packing but ensure we check if the box fits in Z.
    # If the box's depth (in the chosen orientation) is <= D, it fits in Z (assuming z=0).
    # If multiple boxes are placed, they all start at z=0.
    # So the constraint is: max(od for all placed boxes) <= D.
    # This is a valid packing if we ensure no box exceeds D in depth.
    
    # Let's modify shelf_pack to check this.
    # Also, we should try to minimize the depth used if possible?
    # Or just ensure it fits.
    
    def shelf_pack_2d(ordered_indices):
        placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
        total_volume = 0
        shelves = [] # {'y': float, 'h': float, 'x': float}
        current_y = 0
        max_depth_used = 0
        
        for idx in ordered_indices:
            if time.time() > deadline:
                break
            
            w, h, d = boxes[idx]
            orients = get_orientations(w, h, d)
            
            placed = False
            for shelf in shelves:
                for ow, oh, od in orients:
                    if oh <= shelf['h'] and shelf['x'] + ow <= W and od <= D:
                        # Check if placing this box violates depth?
                        # Since all boxes are at z=0, they share the Z space [0, D].
                        # As long as od <= D, it's fine.
                        # Overlap in X or Y is prevented by shelf logic.
                        placements[idx] = (0, int(shelf['x']), int(shelf['y']), 0, ow, oh, od)
                        shelf['x'] += ow
                        total_volume += ow * oh * od
                        max_depth_used = max(max_depth_used, od)
                        placed = True
                        break
                if placed:
                    break
            
            if placed:
                continue
            
            # Try new shelf
            best_orient = None
            for ow, oh, od in orients:
                if current_y + oh <= H and ow <= W and od <= D:
                    if best_orient is None or oh < best_orient[1]:
                        best_orient = (ow, oh, od)
            
            if best_orient:
                ow, oh, od = best_orient
                new_shelf = {'y': current_y, 'h': oh, 'x': ow}
                shelves.append(new_shelf)
                current_y += oh
                placements[idx] = (0, 0, int(current_y - oh), 0, ow, oh, od)
                total_volume += ow * oh * od
                max_depth_used = max(max_depth_used, od)
        
        return placements, total_volume
    
    # Run strategies
    for key_func in strategies:
        if time.time() > deadline - 10:
            break
        
        ordered = sorted(indexed_boxes, key=key_func, reverse=True)
        p, v = shelf_pack_2d(ordered)
        
        if v > best_volume:
            best_volume = v
            best_placements = p
            
        # Also try ascending? Sometimes smaller first fills gaps? Unlikely for shelf.
    
    # One more strategy: Sort by volume, but try to fit largest first.
    # The current strategies cover volume, max dim, etc.
    
    # If time permits, try a simple "First Fit" 3D approach?
    # Given the 120s limit and 5 instances, we have ~24s per instance.
    # The shelf pack is O(N^2 * 6) roughly. N=60. 60^2 * 6 = 21600 ops. Very fast.
    # We can afford many more strategies.
    
    # Let's add a strategy that prioritizes fitting depth?
    # Sort by depth ascending?
    strategies.append(lambda i: boxes[i][2]) # Depth
    strategies.append(lambda i: boxes[i][1]) # Height
    
    # Re-run with expanded strategies if time allows
    # Actually, the loop above already runs them.
    
    return best_placements
# EVOLVE-BLOCK-END
