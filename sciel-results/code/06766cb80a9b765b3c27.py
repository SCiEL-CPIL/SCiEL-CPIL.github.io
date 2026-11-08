"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def pack(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    W, H, D = container
    container_vol = W * H * D
    
    if not boxes:
        return []

    # Sort boxes by volume descending (BFD heuristic)
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda x: x[1][0] * x[1][1] * x[1][2], reverse=True)
    
    # Structure for each bin:
    # A bin is a list of shelves.
    # Each shelf is a dict: {'y': start_y, 'h': shelf_height, 'z': start_z, 'd': shelf_depth, 'items': [(x, w, h, d)]}
    # Actually, simpler:
    # Bins are lists of "Shelves".
    # A Shelf is represented by: [current_x, current_y, current_z, shelf_height, shelf_depth, list_of_items]
    # But shelves are horizontal layers.
    # Let's define a Shelf as:
    # y: bottom of shelf
    # z: bottom of shelf (depth start)
    # height: height of the shelf (max height of items in it)
    # depth: depth of the shelf (max depth of items in it)
    # current_x: next available x position
    # items: list of (x, w, h, d)
    
    # We will maintain a list of bins.
    # Each bin is a list of shelves.
    # Shelves are sorted by y (bottom to top).
    
    bins = []
    
    def get_bin_volume(bin_shelves):
        vol = 0
        for shelf in bin_shelves:
            for item in shelf['items']:
                vol += item[1] * item[2] * item[3]
        return vol

    def try_place_in_bin(bin_shelves, w, h, d):
        """
        Try to place a box (w, h, d) into a bin represented by a list of shelves.
        Returns (shelf_index, x_pos) if successful, else None.
        Uses Best-Fit: choose the shelf where (w - remaining_x) is minimized, 
        or if no space in x, open a new shelf if it fits in Y/Z.
        """
        best_shelf_idx = -1
        best_x = -1
        best_waste = float('inf')
        
        # Check existing shelves
        for i, shelf in enumerate(bin_shelves):
            # Can the box fit in height and depth of this shelf?
            if h <= shelf['height'] and d <= shelf['depth']:
                # Check if it fits in X
                if shelf['current_x'] + w <= W:
                    waste = W - (shelf['current_x'] + w)
                    if waste < best_waste:
                        best_waste = waste
                        best_shelf_idx = i
                        best_x = shelf['current_x']
        
        if best_shelf_idx != -1:
            return best_shelf_idx, best_x

        # If it doesn't fit in any existing shelf, try to open a new shelf
        # Find the lowest Y position where we can place a new shelf of height h and depth d
        # A new shelf can be placed at Y = max_y_of_lower_shelves
        # And Z = 0 (assuming shelves span full depth or we pack depth too? 
        # Standard shelf packing usually assumes shelves span the full width or depth.
        # Let's assume shelves span full width W and full depth D for simplicity and robustness?
        # No, that wastes space. 
        # Let's stick to: Shelves are horizontal slices. 
        # A shelf has a fixed height and depth. 
        # Items in a shelf are placed side-by-side in X.
        # New shelves are stacked in Y.
        # Depth D is shared? Or do we pack depth too?
        # The task allows 3D placement. 
        # A pure shelf method (2D in XY, fixed Z) might waste Z space if boxes have different depths.
        # Let's use a "Layer" approach:
        # Layers are stacked in Z.
        # Within a layer, we use a 2D shelf packing in XY.
        # This is more complex.
        
        # Alternative: "Bottom-Left-Far" (BLF) heuristic without full grid search.
        # Let's try a simpler, more effective heuristic: 
        # "First-Fit Decreasing Height" (FFDH) variant.
        # Sort by height desc.
        # Place in the bin where it fits with the least remaining space in the "active" region.
        
        # Let's revert to a more robust 3D heuristic: 
        # "Best-Fit" based on a simplified "skyline" or "max-min" approach is hard.
        
        # Let's use a different strategy: 
        # **Multi-Objective Sorting and Best-Fit Bin Selection**
        # 1. Sort boxes by a composite key (e.g., max dimension, then volume).
        # 2. For each box, evaluate all existing bins.
        # 3. Score each bin based on how "tight" the fit is.
        # 4. Use a fast, approximate placement finder: 
        #    - Try placing at (0,0,0).
        #    - If overlap, try shifting x, y, z to the "next" available slot based on existing items.
        #    - This is effectively what the current code does, but slow.
        
        # Let's try **Shelf Packing with Depth Packing**:
        # We will maintain shelves that span the full width W.
        # Each shelf has a height h_shelf and depth d_shelf.
        # Items are placed in X within the shelf.
        # If an item doesn't fit in X, it moves to the next shelf in Y.
        # If it doesn't fit in Y (height), it moves to the next layer in Z.
        # This reduces 3D to 2D (XY) within layers.
        
        # Let's implement this specific "Layered Shelf" heuristic.
        
        pass # Placeholder, we will implement the actual logic below

    # --- Actual Implementation: Layered Shelf Packing ---
    
    # We will sort boxes by height descending, then by depth descending, then by width descending.
    # This helps align shelves.
    indexed_boxes.sort(key=lambda x: (x[1][1], x[1][2], x[1][0]), reverse=True)
    
    # Bin structure: List of Layers.
    # Layer structure: List of Shelves.
    # Shelf structure: { 'y': start_y, 'h': shelf_h, 'current_x': next_x, 'items': [(x, w, h, d)] }
    # Note: All shelves in a layer share the same Z range [0, layer_d].
    # Wait, if shelves have different depths, we can't just stack them in Y easily if we want to pack Z efficiently.
    # Let's assume standard Shelf Packing:
    # - Layers are stacked in Z.
    # - Each layer has a fixed depth D_layer (max depth of items in it).
    # - Within a layer, shelves are stacked in Y.
    # - Each shelf has a fixed height H_shelf (max height of items in it).
    # - Items are placed in X within a shelf.
    
    bins = [] # List of layers-lists
    
    def find_best_bin_and_position(box_w, box_h, box_d):
        """
        Returns (bin_idx, layer_idx, shelf_idx, x_pos) or None.
        """
        best_bin = -1
        best_layer = -1
        best_shelf = -1
        best_x = -1
        best_score = float('inf')
        
        for b_idx, layers in enumerate(bins):
            # Try existing layers
            for l_idx, layer in enumerate(layers):
                # Layer depth must be >= box_d
                if layer['depth'] >= box_d:
                    # Try existing shelves in this layer
                    for s_idx, shelf in enumerate(layer['shelves']):
                        # Shelf height must be >= box_h
                        if shelf['h'] >= box_h:
                            # Check X space
                            if shelf['current_x'] + box_w <= W:
                                # Score: prefer bins/layers/shelves that are more full
                                # Waste in X
                                waste_x = W - (shelf['current_x'] + box_w)
                                # Prefer lower waste
                                score = waste_x
                                if score < best_score:
                                    best_score = score
                                    best_bin = b_idx
                                    best_layer = l_idx
                                    best_shelf = s_idx
                                    best_x = shelf['current_x']
            
            # If no existing shelf fits, can we add a new shelf to an existing layer?
            # Find the lowest Y position available in each layer
            for l_idx, layer in enumerate(layers):
                if layer['depth'] >= box_d:
                    # Current top of layer in Y
                    current_y = layer['current_y']
                    if current_y + box_h <= H:
                        # We can add a new shelf at current_y
                        # Score: prefer layers that are deeper (more filled in Z?) or just consistent
                        # Let's use the same waste metric: 0 waste in X for new shelf? 
                        # Actually, opening a new shelf wastes the remaining X space of the previous shelf? No, shelves are independent in Y.
                        # The "waste" is the unused Y space above the new shelf?
                        # Let's just prefer bins with fewer layers (more compact).
                        score = len(layers) * 1000 + current_y
                        if score < best_score:
                            best_score = score
                            best_bin = b_idx
                            best_layer = l_idx
                            best_shelf = -1 # Signal new shelf
                            best_x = 0
                            
            # If no existing layer fits, can we add a new layer?
            # Find the lowest Z position available
            current_z = layers[-1]['z_end'] if layers else 0
            if current_z + box_d <= D:
                # Can add new layer
                score = len(layers) * 10000 + current_z
                if score < best_score:
                    best_score = score
                    best_bin = b_idx
                    best_layer = -1 # Signal new layer
                    best_shelf = -1
                    best_x = 0

        return best_bin, best_layer, best_shelf, best_x

    placements = [None] * len(boxes)
    
    for orig_idx, (w, h, d) in indexed_boxes:
        if time.time() > deadline:
            # Emergency: place remaining in new bins
            for rem_idx, (rw, rh, rd) in indexed_boxes:
                if placements[rem_idx] is None:
                    b = len(bins)
                    bins.append([{'depth': rd, 'current_y': rh, 'z_end': rd, 'shelves': [{'h': rh, 'current_x': rw, 'items': [(0, rw, rh, rd)]}]}])
                    placements[rem_idx] = (b, 0, 0, 0, rw, rh, rd)
            return placements

        if w > W or h > H or d > D:
            # Should not happen, but handle gracefully
            placements[orig_idx] = (len(bins), 0, 0, 0, w, h, d)
            bins.append([])
            continue

        b_idx, l_idx, s_idx, x_pos = find_best_bin_and_position(w, h, d)
        
        if b_idx == -1:
            # Start new bin
            b_idx = len(bins)
            # Create new layer
            new_layer = {
                'depth': d,
                'current_y': h,
                'z_end': d,
                'shelves': []
            }
            # Create new shelf
            new_shelf = {
                'h': h,
                'current_x': w,
                'items': []
            }
            new_layer['shelves'].append(new_shelf)
            bins.append([new_layer])
            placements[orig_idx] = (b_idx, 0, 0, 0, w, h, d)
        else:
            if l_idx == -1:
                # Add new layer to existing bin
                layers = bins[b_idx]
                current_z = layers[-1]['z_end'] if layers else 0
                new_layer = {
                    'depth': d,
                    'current_y': h,
                    'z_end': current_z + d,
                    'shelves': []
                }
                new_shelf = {
                    'h': h,
                    'current_x': w,
                    'items': []
                }
                new_layer['shelves'].append(new_shelf)
                layers.append(new_layer)
                placements[orig_idx] = (b_idx, 0, current_z, 0, w, h, d)
            else:
                layer = bins[b_idx][l_idx]
                if s_idx == -1:
                    # Add new shelf to existing layer
                    current_y = layer['current_y']
                    new_shelf = {
                        'h': h,
                        'current_x': w,
                        'items': []
                    }
                    layer['shelves'].append(new_shelf)
                    layer['current_y'] += h
                    placements[orig_idx] = (b_idx, 0, current_y, 0, w, h, d)
                else:
                    # Place in existing shelf
                    shelf = layer['shelves'][s_idx]
                    shelf['current_x'] += w
                    shelf['items'].append((x_pos, w, h, d))
                    # The y and z are determined by the shelf/layer position
                    # We need to track the absolute y and z for the placement return
                    # Shelf Y start:
                    shelf_y_start = 0
                    for prev_shelf in layer['shelves'][:s_idx]:
                        shelf_y_start += prev_shelf['h']
                    layer_z_start = 0
                    for prev_layer in bins[b_idx][:l_idx]:
                        layer_z_start += prev_layer['depth']
                    
                    placements[orig_idx] = (b_idx, x_pos, shelf_y_start, layer_z_start, w, h, d)
            
    return placements
# EVOLVE-BLOCK-END
