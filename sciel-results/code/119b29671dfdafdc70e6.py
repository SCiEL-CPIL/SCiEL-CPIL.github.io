"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def load(boxes, container):
    W, H, D = container
    W, H, D = int(W), int(H), int(D)
    n = len(boxes)
    
    if W <= 0 or H <= 0 or D <= 0 or n == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    deadline = time.time() + 115.0  # Safety margin for 120s limit
    
    # Pre-compute unique orientations for each box
    def get_orientations(box):
        w, h, d = box
        orients = set()
        for perm in itertools.permutations([w, h, d]):
            orients.add(perm)
        return list(orients)
    
    orientations = [get_orientations(b) for b in boxes]
    
    def overlaps(b1, b2):
        # b1, b2 are (x, y, z, w, h, d)
        x1, y1, z1, w1, h1, d1 = b1
        x2, y2, z2, w2, h2, d2 = b2
        if x1 + w1 <= x2 or x2 + w2 <= x1:
            return False
        if y1 + h1 <= y2 or y2 + h2 <= y1:
            return False
        if z1 + d1 <= z2 or z2 + d2 <= z1:
            return False
        return True
    
    def find_best_placement(sorted_indices, placed_boxes):
        # placed_boxes is a list of (x, y, z, w, h, d)
        # Returns the best (x, y, z, w, h, d) for the current box or None
        # We want to pack tightly, so we prefer smaller x, y, z
        best_pos = None
        best_score = float('inf')
        
        # Candidate points: Origin and corners of placed boxes
        candidates = [(0, 0, 0)]
        for pb in placed_boxes:
            x, y, z, w, h, d = pb
            # Add corners that might be "skyline" points
            # We add all 8 corners to be safe, filtering later
            candidates.append((x, y, z))
            candidates.append((x + w, y, z))
            candidates.append((x, y + h, z))
            candidates.append((x, y, z + d))
            candidates.append((x + w, y + h, z))
            candidates.append((x + w, y, z + d))
            candidates.append((x, y + h, z + d))
            candidates.append((x + w, y + h, z + d))
        
        # Unique candidates to avoid redundant checks
        unique_candidates = set(candidates)
        
        for cx, cy, cz in unique_candidates:
            for ow, oh, od in sorted_indices: # Wait, sorted_indices is indices, we need box dims
                pass # This loop structure is wrong. We are placing ONE box.
        
        # Correction: The function should take the box to place.
        return None # Placeholder
    
    def pack_with_order(order):
        placed = []
        result = [None] * n # Store (x, y, z, w, h, d) or None
        current_time = time.time()
        
        for idx in order:
            if current_time > deadline:
                break
            box = boxes[idx]
            orients = orientations[idx]
            
            # Candidate points
            candidates = [(0, 0, 0)]
            for pb in placed:
                x, y, z, w, h, d = pb
                candidates.append((x, y, z))
                candidates.append((x + w, y, z))
                candidates.append((x, y + h, z))
                candidates.append((x, y, z + d))
                candidates.append((x + w, y + h, z))
                candidates.append((x + w, y, z + d))
                candidates.append((x, y + h, z + d))
                candidates.append((x + w, y + h, z + d))
            
            unique_candidates = set(candidates)
            
            best_pos = None
            best_score = float('inf')
            
            for cx, cy, cz in unique_candidates:
                for ow, oh, od in orients:
                    # Check bounds
                    if cx + ow > W or cy + oh > H or cz + od > D:
                        continue
                    
                    # Check overlap
                    new_box = (cx, cy, cz, ow, oh, od)
                    valid = True
                    for pb in placed:
                        if overlaps(new_box, pb):
                            valid = False
                            break
                    
                    if valid:
                        # Score: prefer positions closer to origin (0,0,0)
                        # Using sum of coordinates as a simple metric for "tightness"
                        score = cx + cy + cz
                        if score < best_score:
                            best_score = score
                            best_pos = new_box
            
            if best_pos:
                placed.append(best_pos)
                result[idx] = best_pos
                current_time = time.time()
            else:
                result[idx] = None
        
        return result
    
    # Define sorting orders
    # 1. Volume descending
    order_vol = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    # 2. Max dimension descending
    order_max = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    # 3. Aspect ratio (more cubic first)
    order_aspect = sorted(range(n), key=lambda i: max(boxes[i]) / min(boxes[i]))
    # 4. Surface area descending
    order_surf = sorted(range(n), key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True)
    
    orders = [order_vol, order_max, order_aspect, order_surf]
    
    best_result = None
    best_volume = 0
    
    for order in orders:
        if time.time() > deadline:
            break
        
        res = pack_with_order(order)
        
        vol = 0
        for i in range(n):
            if res[i] is not None:
                x, y, z, w, h, d = res[i]
                vol += w * h * d
        
        if vol > best_volume:
            best_volume = vol
            best_result = res
    
    # If no boxes were placed in any run (shouldn't happen unless all too big), return all -1
    if best_result is None:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Format output
    final_placements = []
    for i in range(n):
        if best_result[i] is not None:
            x, y, z, w, h, d = best_result[i]
            final_placements.append((0, x, y, z, w, h, d))
        else:
            w, h, d = boxes[i]
            final_placements.append((-1, 0, 0, 0, w, h, d))
            
    return final_placements

# EVOLVE-BLOCK-END
