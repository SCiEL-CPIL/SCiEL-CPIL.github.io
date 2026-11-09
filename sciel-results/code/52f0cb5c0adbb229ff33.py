import time
import itertools

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 110  # 110s to be safe within 120s limit
    
    W, H, D = container
    n_boxes = len(boxes)
    
    # Pre-compute rotations for each box
    box_rotations = []
    for w, h, d in boxes:
        rots = set(itertools.permutations((w, h, d)))
        box_rotations.append(list(rots))
        
    # Heuristics for sorting boxes
    # 1. Volume descending
    # 2. Max dimension descending
    # 3. Surface area descending
    
    def get_order_by_volume():
        return sorted(range(n_boxes), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
        
    def get_order_by_max_dim():
        return sorted(range(n_boxes), key=lambda i: max(boxes[i]), reverse=True)
        
    def get_order_by_surface():
        return sorted(range(n_boxes), key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True)

    orders = [
        get_order_by_volume(),
        get_order_by_max_dim(),
        get_order_by_surface()
    ]
    
    best_placements = None
    best_volume = 0
    
    def check_overlap(x, y, z, w, h, d, placed):
        for px, py, pz, pw, ph, pd in placed:
            if not (x + w <= px or px + pw <= x):
                if not (y + h <= py or py + ph <= y):
                    if not (z + d <= pz or pz + pd <= z):
                        return True
        return False

    def try_pack(order):
        placed = []
        candidates = [(0, 0, 0)]
        placed_indices = set()
        
        for idx in order:
            w_orig, h_orig, d_orig = boxes[idx]
            placed_flag = False
            
            # Try all rotations
            for w, h, d in box_rotations[idx]:
                # Try all candidate positions
                # We iterate over a snapshot of candidates to avoid modification during iteration issues, 
                # though adding to a set while iterating a list copy is fine.
                # However, the number of candidates can grow. 
                # For efficiency, we can just iterate over the current set.
                for cx, cy, cz in candidates:
                    # Check bounds
                    if cx + w > W or cy + h > H or cz + d > D:
                        continue
                    
                    # Check overlap
                    if check_overlap(cx, cy, cz, w, h, d, placed):
                        continue
                    
                    # If we reach here, the box fits at (cx, cy, cz) with rotation (w, h, d)
                    # We place it here. 
                    # Strategy: First fit is often good. 
                    # To improve, we could evaluate "best" position, but first fit is fast.
                    # Let's stick to first fit to allow more restarts/heuristics.
                    placed.append((cx, cy, cz, w, h, d))
                    placed_indices.add(idx)
                    placed_flag = True
                    
                    # Generate new candidate points
                    # The new box creates new "faces" that can support other boxes.
                    # The relevant extreme points are the corners of the new box that are "outermost".
                    new_points = [
                        (cx + w, cy, cz),
                        (cx, cy + h, cz),
                        (cx, cy, cz + d),
                        (cx + w, cy + h, cz),
                        (cx + w, cy, cz + d),
                        (cx, cy + h, cz + d),
                        (cx + w, cy + h, cz + d)
                    ]
                    for pt in new_points:
                        if pt[0] <= W and pt[1] <= H and pt[2] <= D:
                            candidates.add(pt)
                    break
                
                if placed_flag:
                    break
            
            if not placed_flag:
                # Box could not be placed
                pass
        
        # Calculate volume
        vol = 0
        for _, _, _, w, h, d in placed:
            vol += w * h * d
        return vol, placed, placed_indices

    # Try each heuristic
    for order in orders:
        if time.time() > deadline:
            break
            
        vol, placed, placed_indices = try_pack(order)
        
        if vol > best_volume:
            best_volume = vol
            best_placements = [(-1, 0, 0, 0, 0, 0, 0) for _ in range(n_boxes)]
            # Map placed boxes back to original indices
            # Note: 'placed' list contains (x, y, z, w, h, d) but not the original index.
            # We need to track which box was placed.
            # Let's refactor try_pack to return the mapping.
            
            # Re-run try_pack logic to capture indices if this is the best? 
            # Better to modify try_pack to return indices directly.
            
            # Let's just re-implement the tracking in the main loop for clarity or fix try_pack.
            # I will modify try_pack to return a list of (index, x, y, z, w, h, d) for placed boxes.

    # Refactored try_pack to return detailed placement info
    def try_pack_detailed(order):
        placed = [] # List of (index, x, y, z, w, h, d)
        placed_boxes = [] # List of (x, y, z, w, h, d) for overlap check
        candidates = set([(0, 0, 0)])
        placed_indices = set()
        
        for idx in order:
            w_orig, h_orig, d_orig = boxes[idx]
            placed_flag = False
            
            for w, h, d in box_rotations[idx]:
                # We need to iterate over candidates. 
                # To make it deterministic and potentially better, we can sort candidates by (x, y, z)
                # This tends to pack towards the origin, which is good.
                sorted_candidates = sorted(candidates)
                for cx, cy, cz in sorted_candidates:
                    if cx + w > W or cy + h > H or cz + d > D:
                        continue
                    
                    if check_overlap(cx, cy, cz, w, h, d, placed_boxes):
                        continue
                    
                    # Place it
                    placed.append((idx, cx, cy, cz, w, h, d))
                    placed_boxes.append((cx, cy, cz, w, h, d))
                    placed_indices.add(idx)
                    placed_flag = True
                    
                    new_points = [
                        (cx + w, cy, cz),
                        (cx, cy + h, cz),
                        (cx, cy, cz + d),
                        (cx + w, cy + h, cz),
                        (cx + w, cy, cz + d),
                        (cx, cy + h, cz + d),
                        (cx + w, cy + h, cz + d)
                    ]
                    for pt in new_points:
                        if pt[0] <= W and pt[1] <= H and pt[2] <= D:
                            candidates.add(pt)
                    break
                
                if placed_flag:
                    break
        
        vol = sum(w*h*d for _, _, _, w, h, d in placed)
        return vol, placed

    best_vol = 0
    best_placed = []
    
    for order in orders:
        if time.time() > deadline:
            break
        vol, placed = try_pack_detailed(order)
        if vol > best_vol:
            best_vol = vol
            best_placed = placed
            # Optional: If we find a perfect pack (volume == container volume), we can stop
            if best_vol >= W * H * D:
                break

    # If we have time, try a few random shuffles
    if time.time() < deadline - 10:
        import random
        # Use a fixed seed for reproducibility? Or just random.
        # Let's do 5 random restarts
        for _ in range(5):
            if time.time() > deadline:
                break
            random_order = list(range(n_boxes))
            random.shuffle(random_order)
            vol, placed = try_pack_detailed(random_order)
            if vol > best_vol:
                best_vol = vol
                best_placed = placed

    # Construct final result
    result = [(-1, 0, 0, 0, 0, 0, 0) for _ in range(n_boxes)]
    for idx, x, y, z, w, h, d in best_placed:
        result[idx] = (0, x, y, z, w, h, d)
        
    return result
