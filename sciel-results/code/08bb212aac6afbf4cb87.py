"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
import numpy as np

# EVOLVE-BLOCK-START
def load(boxes, container):
    W, H, D = container
    deadline = time.time() + 110.0  # Safety margin before 120s kill
    
    # Normalize boxes: ensure we can try all 6 orientations
    # We'll represent a box by its dimensions (w, h, d)
    # The container is (W, H, D)
    
    # Sort boxes by volume descending (heuristic: place big ones first)
    box_indices = list(range(len(boxes)))
    # We'll keep original indices to map back
    # Let's create a list of (original_index, w, h, d)
    box_list = [(i, boxes[i][0], boxes[i][1], boxes[i][2]) for i in box_indices]
    
    # Strategy:
    # 1. Try a simple shelf/layer packing with sorting heuristics.
    # 2. Try a more sophisticated "Best Fit Decreasing" (BFD) style packing.
    # 3. If time allows, try local search or random restarts.
    
    # Since we need to return placements for ALL boxes (with -1 for unplaced),
    # and the score is mean fraction of container volume filled,
    # we want to maximize the sum of volumes of placed boxes.
    
    # Let's implement a "Bottom-Left-Front" (BLF) heuristic with multiple sorting orders.
    # And also a simple shelf packing.
    
    def try_packing(sorted_indices, orientation_choice=None):
        """
        Try to pack boxes given a specific order.
        Returns a list of placements (bin, x, y, z, w, h, d) for all original boxes.
        """
        placements = [(-1, 0, 0, 0, 0, 0, 0)] * len(boxes)
        # We'll maintain a list of placed boxes for collision checking
        # For efficiency with 60 boxes, O(N^2) collision check is fine (3600 checks)
        
        placed_boxes = []  # list of (x, y, z, w, h, d)
        
        for idx in sorted_indices:
            ow, oh, od = boxes[idx]
            
            # Try all 6 orientations
            # Dimensions can be permuted in 6 ways
            orientations = list(itertools.permutations([ow, oh, od]))
            # Remove duplicates if any dimension is same
            seen = set()
            unique_orientations = []
            for o in orientations:
                if o not in seen:
                    seen.add(o)
                    unique_orientations.append(o)
            
            best_placement = None
            best_score = -1
            
            for w, h, d in unique_orientations:
                if w > W or h > H or d > D:
                    continue
                
                # Try to find a position
                # Simple strategy: Scan a grid or use a "shelf" approach?
                # For BLF, we need to find the lowest, leftmost, frontmost position.
                # Since N is small, we can try a discretized search or use a simple incremental method.
                # Let's use a simple method: try positions based on existing placed boxes' boundaries.
                
                # Candidate x, y, z coordinates:
                # x in {0} U {placed_box.x + placed_box.w for placed_box}
                # y in {0} U {placed_box.y + placed_box.h for placed_box}
                # z in {0} U {placed_box.z + placed_box.d for placed_box}
                
                candidate_xs = [0]
                candidate_ys = [0]
                candidate_zs = [0]
                for px, py, pz, pw, ph, pd in placed_boxes:
                    candidate_xs.append(px + pw)
                    candidate_ys.append(py + ph)
                    candidate_zs.append(pz + pd)
                
                # Sort candidates to prioritize "bottom-left-front"
                # But we also need to check bounds
                valid_xs = [x for x in candidate_xs if x + w <= W]
                valid_ys = [y for y in candidate_ys if y + h <= H]
                valid_zs = [z for z in candidate_zs if z + d <= D]
                
                if not valid_xs or not valid_ys or not valid_zs:
                    continue
                
                # To be efficient, we don't want to check all combinations (N^3).
                # But with 60 boxes, max candidates are ~60 each. 60^3 = 216,000 checks per box.
                # 60 boxes * 216,000 * 6 orientations = 77 million checks. Too slow in Python.
                
                # Optimization: Only check a subset.
                # Or, use a simpler "shelf" packing which is O(N^2).
                
                # Let's stick to a simpler heuristic for speed:
                # 1. Shelf packing (layer by layer, row by row)
                # This is what the seed does, but we can improve sorting.
                
                # Actually, let's implement a faster BLF approximation:
                # Try a limited set of candidate positions.
                # For now, let's just return None to indicate this complex BLF is too slow.
                pass
            
            # Fallback: Use a simple "First Fit" with the first valid orientation
            # This is basically what the seed does, but we can try different sort orders.
            
            # Let's implement a proper but fast "Shelf Packing" here.
            # We'll handle shelf packing separately in the main logic.
            pass
        
        return None # Placeholder

    # Let's implement a robust Shelf Packing algorithm with multiple sorting strategies.
    # Shelf Packing:
    # - We have layers (z-direction).
    # - Within a layer, we have shelves (y-direction).
    # - Within a shelf, we place boxes side-by-side (x-direction).
    # - When a box doesn't fit in the current shelf, we start a new shelf (increase y).
    # - When a shelf is too tall for the container, we start a new layer (increase z).
    
    def shelf_pack(sort_key, orientation_strategy):
        """
        sort_key: function to sort boxes
        orientation_strategy: 'auto' (try to fit), 'fixed' (use as is)
        """
        # Create a list of boxes to place
        # We need to decide orientation for each box.
        # For shelf packing, height (y) is critical for shelf height.
        # Width (x) is critical for shelf width.
        # Depth (z) is critical for layer depth.
        
        # Let's try to minimize the "waste".
        # Simple heuristic: Sort by volume descending.
        
        indices = list(range(len(boxes)))
        indices.sort(key=lambda i: -boxes[i][0]*boxes[i][1]*boxes[i][2])
        
        placements = [(-1, 0, 0, 0, 0, 0, 0)] * len(boxes)
        
        # State
        x = 0.0
        y = 0.0
        z = 0.0
        shelf_h = 0.0
        layer_d = 0.0
        
        for idx in indices:
            ow, oh, od = boxes[idx]
            
            # Determine orientation
            # We want to minimize the impact on shelf_h and layer_d.
            # Try all 6 permutations.
            best_orient = None
            best_metrics = None
            
            # For each orientation, check if it fits in the current shelf/layer.
            # If it fits, we prefer the one that minimizes new shelf height or layer depth.
            
            perms = list(itertools.permutations([ow, oh, od]))
            seen = set()
            unique_perms = []
            for p in perms:
                if p not in seen:
                    seen.add(p)
                    unique_perms.append(p)
            
            # First pass: find an orientation that fits in the current shelf (x, y, z)
            # Condition:
            # 1. x + w <= W
            # 2. y + h <= H
            # 3. z + d <= D
            # Note: In shelf packing, we usually assume boxes in a shelf share the same y range?
            # No, standard shelf packing:
            # - We have a current shelf starting at y=shelf_y, height=shelf_h.
            # - New box must fit in x: x + w <= W.
            # - New box must fit in z: z + d <= D (if layers are uniform depth) or z is free?
            # - Standard shelf: all boxes in a shelf have the same y start.
            #   So new box height h must be <= H - y.
            #   And if h > shelf_h, we update shelf_h = h.
            #   But we also need to ensure the box fits in the remaining space of the shelf.
            #   Actually, if we just extend x, we need x + w <= W.
            #   And we need y + h <= H.
            #   And z + d <= D.
            
            # Let's refine the state:
            # current_x: next x position
            # current_y: bottom of current shelf
            # current_z: bottom of current layer
            # shelf_h: height of current shelf (max h of boxes in it)
            # layer_d: depth of current layer (max d of boxes in it)
            
            # Wait, if layer_d is the max d, then all boxes in the layer must have d <= layer_d?
            # No, if we place a box with d > layer_d, we must start a new layer?
            # Or can we place it if z + d <= D?
            # If we place it, it might overlap with boxes in the same layer if they have different d?
            # No, if they are in the same layer, they share the same z range?
            # In standard 3D shelf packing:
            # - Layer: defined by z. All boxes in a layer start at z=layer_z.
            #   So their d must be <= D - layer_z.
            #   But if one box has d=5 and another has d=10, and both start at z=0,
            #   the box with d=5 ends at z=5, the box with d=10 ends at z=10.
            #   They don't overlap in z if they are in different x/y positions?
            #   Yes, they can coexist.
            #   However, the "layer depth" is not a fixed constraint for all boxes unless we enforce it.
            #   To keep it simple and non-overlapping:
            #   We just need to ensure:
            #   - x + w <= W
            #   - y + h <= H
            #   - z + d <= D
            #   - And no overlap with previously placed boxes.
            
            # The simple shelf algorithm assumes:
            # - We fill a shelf (fixed y, varying x, fixed z range? No, z can vary).
            # - Actually, the seed algorithm does:
            #   x, y, z.
            #   If x+w > W: x=0, y += row_h, row_h=0.
            #   If y+h > H: y=0, z += layer_d, layer_d=0, x=0, row_h=0.
            #   If z+d > D: skip.
            #   Place at (x,y,z).
            #   x += w, row_h = max(row_h, h), layer_d = max(layer_d, d).
            
            # This is a "row" packing.
            # - Row: fixed y, varying x. All boxes in a row have the same y.
            #   So h can vary, but the "row height" is the max h.
            #   Next row starts at y + row_h.
            # - Layer: fixed z, varying y. All boxes in a layer have the same z.
            #   So d can vary, but the "layer depth" is the max d.
            #   Next layer starts at z + layer_d.
            
            # This is a valid non-overlapping packing.
            
            # Let's use this exact logic but with better sorting and orientation.
            
            # For each box, try orientations.
            # We want to minimize the "waste" in the current row/layer.
            
            found = False
            best_orient = None
            best_waste = float('inf')
            
            for w, h, d in unique_perms:
                # Check if it fits in the current row/layer
                if x + w <= W and y + h <= H and z + d <= D:
                    # It fits.
                    # What is the "cost" of this orientation?
                    # It might increase row_h or layer_d.
                    # If h > row_h, the next row will be higher.
                    # If d > layer_d, the next layer will be deeper.
                    # We want to minimize future waste.
                    
                    # Simple heuristic: prefer orientations that don't increase row_h or layer_d.
                    # Or minimize max(row_h, h) + max(layer_d, d)?
                    
                    cost = 0
                    if h > shelf_h:
                        cost += (h - shelf_h) * 10 # Penalize increasing row height
                    if d > layer_d:
                        cost += (d - layer_d) * 10 # Penalize increasing layer depth
                    
                    if cost < best_waste:
                        best_waste = cost
                        best_orient = (w, h, d)
            
            if best_orient:
                w, h, d = best_orient
                placements[idx] = (0, x, y, z, w, h, d)
                x += w
                shelf_h = max(shelf_h, h)
                layer_d = max(layer_d, d)
                found = True
            else:
                # Try to start a new row
                # Reset x to 0, y += shelf_h, shelf_h = 0
                # But we need to check if the new row fits in the container
                new_y = y + shelf_h
                if new_y + min(ow, oh, od) <= H: # At least the smallest dimension must fit
                    # Try orientations again in the new row
                    for w, h, d in unique_perms:
                        if 0 + w <= W and new_y + h <= H and z + d <= D:
                            # It fits in the new row
                            # Cost:
                            cost = 0
                            if h > 0: # shelf_h is 0 now
                                pass
                            # Actually, we just pick the first one that fits, or the one that minimizes h?
                            # Let's pick the one that minimizes h to keep row low.
                            if best_orient is None or h < best_orient[1]:
                                best_orient = (w, h, d)
                    
                    if best_orient:
                        w, h, d = best_orient
                        placements[idx] = (0, 0, new_y, z, w, h, d)
                        x = w
                        shelf_h = h
                        # layer_d remains the same? No, layer_d is for the layer.
                        # If we start a new row, we are still in the same layer.
                        # So z is unchanged.
                        # But we must ensure z + d <= D.
                        found = True
                    else:
                        # Try to start a new layer
                        new_z = z + layer_d
                        new_y2 = 0 # Reset y for new layer
                        new_x2 = 0
                        new_shelf_h2 = 0
                        if new_z + min(ow, oh, od) <= D:
                            for w, h, d in unique_perms:
                                if new_x2 + w <= W and new_y2 + h <= H and new_z + d <= D:
                                    if best_orient is None or h < best_orient[1]:
                                        best_orient = (w, h, d)
                            
                            if best_orient:
                                w, h, d = best_orient
                                placements[idx] = (0, new_x2, new_y2, new_z, w, h, d)
                                x = new_x2
                                y = new_y2
                                z = new_z
                                shelf_h = h
                                layer_d = d
                                found = True
            
            if not found:
                # Cannot place this box
                placements[idx] = (-1, 0, 0, 0, ow, oh, od)
        
        return placements

    # Let's try a few sorting strategies
    strategies = [
        lambda i: -boxes[i][0]*boxes[i][1]*boxes[i][2], # Volume desc
        lambda i: -max(boxes[i]), # Max dim desc
        lambda i: -boxes[i][0]*boxes[i][1], # Area desc
    ]
    
    best_placements = None
    best_volume = -1
    
    for key in strategies:
        # We need to modify shelf_pack to accept the sort key
        # But my shelf_pack above hardcoded the sort.
        # Let's refactor shelf_pack to take a sorted list of indices.
        
        def shelf_pack_indices(sorted_indices):
            placements = [(-1, 0, 0, 0, 0, 0, 0)] * len(boxes)
            x = 0.0
            y = 0.0
            z = 0.0
            shelf_h = 0.0
            layer_d = 0.0
            
            for idx in sorted_indices:
                ow, oh, od = boxes[idx]
                perms = list(itertools.permutations([ow, oh, od]))
                seen = set()
                unique_perms = []
                for p in perms:
                    if p not in seen:
                        seen.add(p)
                        unique_perms.append(p)
                
                found = False
                best_orient = None
                best_waste = float('inf')
                
                # Try current row
                for w, h, d in unique_perms:
                    if x + w <= W and y + h <= H and z + d <= D:
                        cost = 0
                        if h > shelf_h:
                            cost += (h - shelf_h)
                        if d > layer_d:
                            cost += (d - layer_d)
                        if cost < best_waste:
                            best_waste = cost
                            best_orient = (w, h, d)
                
                if best_orient:
                    w, h, d = best_orient
                    placements[idx] = (0, x, y, z, w, h, d)
                    x += w
                    shelf_h = max(shelf_h, h)
                    layer_d = max(layer_d, d)
                    found = True
                else:
                    # Try new row
                    new_y = y + shelf_h
                    if new_y + min(ow, oh, od) <= H:
                        best_orient = None
                        best_h = float('inf')
                        for w, h, d in unique_perms:
                            if w <= W and new_y + h <= H and z + d <= D:
                                if h < best_h:
                                    best_h = h
                                    best_orient = (w, h, d)
                        
                        if best_orient:
                            w, h, d = best_orient
                            placements[idx] = (0, 0, new_y, z, w, h, d)
                            x = w
                            y = new_y
                            shelf_h = h
                            found = True
                
                if not found:
                    # Try new layer
                    new_z = z + layer_d
                    if new_z + min(ow, oh, od) <= D:
                        best_orient = None
                        best_h = float('inf')
                        for w, h, d in unique_perms:
                            if w <= W and h <= H and new_z + d <= D:
                                if h < best_h:
                                    best_h = h
                                    best_orient = (w, h, d)
                        
                        if best_orient:
                            w, h, d = best_orient
                            placements[idx] = (0, 0, 0, new_z, w, h, d)
                            x = w
                            y = 0
                            z = new_z
                            shelf_h = h
                            layer_d = d
                            found = True
                
                if not found:
                    placements[idx] = (-1, 0, 0, 0, ow, oh, od)
            
            return placements

        sorted_indices = list(range(len(boxes)))
        sorted_indices.sort(key=key)
        
        placements = shelf_pack_indices(sorted_indices)
        
        # Calculate volume
        vol = 0
        for p in placements:
            if p[0] == 0:
                vol += p[4] * p[5] * p[6]
        
        if vol > best_volume:
            best_volume = vol
            best_placements = placements
        
        if time.time() > deadline:
            break
    
    # If we still have time, try a random restart with shuffled order
    if time.time() < deadline - 10:
        for _ in range(5):
            if time.time() > deadline:
                break
            random.shuffle(box_indices)
            placements = shelf_pack_indices(box_indices)
            vol = 0
            for p in placements:
                if p[0] == 0:
                    vol += p[4] * p[5] * p[6]
            if vol > best_volume:
                best_volume = vol
                best_placements = placements
    
    if best_placements is None:
        # Fallback to seed
        placements, x, y, z, row_h, layer_d = [], 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d, x, row_h = 0, z + layer_d, 0, 0, 0
            if z + d > D:
                placements.append((-1, 0, 0, 0, w, h, d))
                continue
            placements.append((0, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        return placements
    
    return best_placements
# EVOLVE-BLOCK-END
