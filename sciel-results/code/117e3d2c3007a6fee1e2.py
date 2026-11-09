import time
import random
import itertools

def load(boxes, container):
    start_time = time.time()
    DEADLINE = 118.0  # Safety margin for 120s limit
    
    W, H, D = container
    container_vol = W * H * D
    if container_vol == 0 or not boxes:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        
    n = len(boxes)
    
    # Pre-compute unique orientations for each box
    def get_orientations(w, h, d):
        dims = list(itertools.permutations((w, h, d)))
        seen = set()
        unique = []
        for dim in dims:
            if dim not in seen:
                seen.add(dim)
                unique.append(dim)
        return unique

    box_orientations = []
    box_volumes = []
    for w, h, d in boxes:
        box_orientations.append(get_orientations(w, h, d))
        box_volumes.append(w * h * d)

    # Sort indices by volume descending as a base heuristic
    box_indices_by_vol = sorted(range(n), key=lambda i: box_volumes[i], reverse=True)
    
    def run_shelf_pack(order):
        """
        Shelf-based packing strategy.
        We maintain a list of 'shelves'. Each shelf has:
        - y_bottom: the y-coordinate of the bottom of this shelf
        - y_top: the y-coordinate of the top of this shelf (y_bottom + shelf_height)
        - x_cursor: the current x-position where the next box in this shelf can be placed
        - shelf_height: the height of the current shelf (determined by the first box placed in it)
        
        Actually, a more robust shelf algorithm:
        1. Determine the "current layer" height.
        2. Place boxes in this layer from left to right (x) and front to back (z).
        3. When a box doesn't fit in the current row (z limit), move to next row (z).
        4. When a row is full or no box fits in height, start a new layer (y).
        
        However, simple "Next Fit Decreasing Height" (NFDH) on shelves is often strong.
        Let's implement a variant:
        - We have a list of active shelves.
        - Each shelf is defined by y_start, y_end, and a 2D "skyline" or just a simple x-cursor if we assume single-row shelves?
        - No, shelves can have multiple rows in Z.
        
        Let's use a simpler, effective heuristic: "Next Fit Decreasing" on Y (shelves), and within each shelf, pack in X and Z.
        
        Algorithm:
        1. Sort boxes by height (in chosen orientation) descending? Or just use the given order.
           The 'order' parameter dictates the sequence.
        2. We maintain a list of shelves.
           Each shelf: { y_start, y_end, items: [(x, z, w, d, h)] }
           Actually, to make it fast and simple:
           We try to place each box in the "current" shelf.
           If it doesn't fit, we close the current shelf and start a new one.
           
           Within a shelf, we try to place boxes in rows.
           Row 0: z=0. Place boxes side by side in x.
           If x exceeds W, start Row 1: z = max_depth_of_row_0.
           
           This is the "Shelf" algorithm.
        """
        # State for the current shelf
        shelves = []
        # Each shelf is a dict: {'y_start': float, 'y_end': float, 'rows': [{'z_start': float, 'z_end': float, 'x_cursor': float, 'boxes': []}]}
        
        placed_boxes = [None] * n
        total_volume = 0
        
        for idx in order:
            if time.time() - start_time > DEADLINE - 2:
                # Emergency stop
                for i in range(n):
                    if placed_boxes[i] is None:
                        placed_boxes[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
                break
            
            w, h, d = boxes[idx]
            orientations = box_orientations[idx]
            
            best_placement = None
            best_score = float('inf')
            
            # Try each orientation
            for ow, oh, od in orientations:
                # Try to place in existing shelves
                for s_idx, shelf in enumerate(shelves):
                    shelf_h = shelf['y_end'] - shelf['y_start']
                    if oh > shelf_h:
                        continue
                    
                    # Try to place in existing rows of this shelf
                    for r_idx, row in enumerate(shelf['rows']):
                        row_d = row['z_end'] - row['z_start']
                        # Box must fit in row depth? 
                        # In standard shelf packing, boxes in a row share the same z-range?
                        # Usually, yes, the row height is determined by the first box, but here depth is Z.
                        # Let's assume rows are defined by Z.
                        # If od > row_d, it doesn't fit in this row.
                        if od > row_d:
                            continue
                        
                        # Check if it fits in X
                        if row['x_cursor'] + ow <= W:
                            # It fits!
                            x = row['x_cursor']
                            z = row['z_start']
                            y = shelf['y_start']
                            
                            # Score: Prefer lower y, then lower z, then lower x
                            score = y * 1000 + z * 100 + x
                            
                            if score < best_score:
                                best_score = score
                                best_placement = (s_idx, r_idx, x, y, z, ow, oh, od)
                    
                    # Try to start a new row in this shelf
                    # The new row starts at z = max(z_end of all existing rows)
                    if shelf['rows']:
                        max_z_end = max(r['z_end'] for r in shelf['rows'])
                    else:
                        max_z_end = 0
                        
                    if max_z_end + od <= D:
                        # Fits in Z for a new row
                        # Check X: starts at 0
                        if ow <= W:
                            x = 0
                            z = max_z_end
                            y = shelf['y_start']
                            
                            score = y * 1000 + z * 100 + x
                            if score < best_score:
                                best_score = score
                                best_placement = (s_idx, -1, x, y, z, ow, oh, od) # -1 indicates new row

                # Try to start a new shelf
                # New shelf starts at y = max(y_end of all existing shelves)
                if shelves:
                    max_y_end = max(s['y_end'] for s in shelves)
                else:
                    max_y_end = 0
                    
                if max_y_end + oh <= H:
                    # Fits in Y for a new shelf
                    # Check Z: starts at 0
                    if od <= D and ow <= W:
                        x = 0
                        z = 0
                        y = max_y_end
                        score = y * 1000 + z * 100 + x
                        if score < best_score:
                            best_placement = (s_idx if False else -1, -2, x, y, z, ow, oh, od) # -1, -2 indicates new shelf, new row

            if best_placement is not None:
                s_idx, r_idx, x, y, z, ow, oh, od = best_placement
                
                if s_idx == -1:
                    # New shelf
                    shelf = {
                        'y_start': y,
                        'y_end': y + oh,
                        'rows': []
                    }
                    shelves.append(shelf)
                    s_idx = len(shelves) - 1
                    r_idx = -1 # New row in this new shelf

                if r_idx == -1:
                    # New row in existing shelf
                    # Determine z_start
                    if shelves[s_idx]['rows']:
                        max_z_end = max(r['z_end'] for r in shelves[s_idx]['rows'])
                    else:
                        max_z_end = 0
                    z = max_z_end
                    row = {
                        'z_start': z,
                        'z_end': z + od,
                        'x_cursor': ow,
                        'boxes': []
                    }
                    shelves[s_idx]['rows'].append(row)
                    r_idx = len(shelves[s_idx]['rows']) - 1
                else:
                    # Existing row
                    row = shelves[s_idx]['rows'][r_idx]
                    row['x_cursor'] += ow
                    # Update z_end if needed? No, shelf algorithm usually fixes row depth by first box or max.
                    # If we allow stacking in Z within a row, it's more complex.
                    # Standard shelf: all boxes in a row have same Z range?
                    # If od > row['z_end'] - row['z_start'], we skipped it.
                    # So od <= row depth.
                    # But what if od < row depth? It fits.
                    # What if od > row depth? We skipped.
                    # So we are safe.
                    
                    # However, if we place a box with smaller depth in a row with larger depth,
                    # we might waste space. But it's a valid placement.
                    # To be safe, let's ensure we don't violate the "shelf" assumption if we want simple logic.
                    # Actually, if od <= row_depth, it fits physically.
                    # The "shelf" height is fixed by the first box in the SHELF.
                    # The "row" depth is fixed by the first box in the ROW.
                    # If a subsequent box has smaller depth, it sits at the bottom of the row (z_start).
                    # This is valid.
                    
                    # Update the shelf's y_end?
                    # The shelf height is determined by the FIRST box placed in it.
                    # If we place a box with height > shelf_height, it would violate.
                    # We checked `if oh > shelf_h: continue`.
                    # So we are safe.
                    
                    # However, if we start a new shelf, the height is set by the first box.
                    # If we add more boxes to that shelf, they must fit in that height.
                    # This is enforced.
                    
                placed_boxes[idx] = (0, x, y, z, ow, oh, od)
                total_volume += ow * oh * od
            else:
                placed_boxes[idx] = (-1, 0, 0, 0, w, h, d)
                
        # Fill in any unplaced boxes
        for i in range(n):
            if placed_boxes[i] is None:
                placed_boxes[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
                
        return placed_boxes, total_volume

    best_results = None
    best_volume = -1

    # Strategy 1: Volume descending
    res, vol = run_shelf_pack(box_indices_by_vol)
    if vol > best_volume:
        best_volume = vol
        best_results = res

    # Strategy 2: Height descending (sort by max dimension to maximize shelf height consistency?)
    # Actually, for shelf packing, sorting by the dimension that becomes the "height" of the shelf is key.
    # But orientation is chosen dynamically.
    # Let's try sorting by "aspect ratio" or just random.
    
    # Strategy 3: Random restarts
    random.seed(42)
    for trial in range(100):
        if time.time() - start_time > DEADLINE - 5:
            break
        order = list(range(n))
        random.shuffle(order)
        res, vol = run_shelf_pack(order)
        if vol > best_volume:
            best_volume = vol
            best_results = res

    # Strategy 4: Hybrid - Sort by volume, but with some randomization in the first few items
    for trial in range(50):
        if time.time() - start_time > DEADLINE - 5:
            break
        order = list(box_indices_by_vol)
        # Shuffle the last 50%
        mid = n // 2
        random.shuffle(order[mid:])
        res, vol = run_shelf_pack(order)
        if vol > best_volume:
            best_volume = vol
            best_results = res

    if best_results is None:
        best_results = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        
    return best_results
