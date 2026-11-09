import time
import random
import itertools
import sys

# EVOLVE-BLOCK-START
def load(boxes, container):
    start_time = time.time()
    DEADLINE = 118.0  # Safety margin for 150s limit (was 120, now 150)
    
    W, H, D = container
    container_vol = W * H * D
    if container_vol == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    
    # Pre-calculate unique orientations for each box
    box_orientations = []
    for w, h, d in boxes:
        perms = set(itertools.permutations((w, h, d)))
        box_orientations.append(list(perms))
        
    volumes = [w*h*d for w, h, d in boxes]
    
    best_fill = 0
    best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    def can_place(x, y, z, w, h, d, placed):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        for px, py, pz, pw, ph, pd in placed:
            if not (x + w <= px or px + pw <= x or 
                    y + h <= py or py + ph <= y or 
                    z + d <= pz or pz + pd <= z):
                return False
        return True

    def find_position(w, h, d, placed):
        # Generate candidate positions based on existing boxes + 0
        # Also add positions offset by full dimensions to allow stacking
        x_cands = set([0])
        y_cands = set([0])
        z_cands = set([0])
        
        for px, py, pz, pw, ph, pd in placed:
            # Next to in x
            if px + pw <= W:
                x_cands.add(px + pw)
            # Next to in y
            if py + ph <= H:
                y_cands.add(py + ph)
            # On top in z
            if pz + pd <= D:
                z_cands.add(pz + pd)
            
        # Convert to sorted lists for deterministic iteration order
        x_cands = sorted(x_cands)
        y_cands = sorted(y_cands)
        z_cands = sorted(z_cands)
        
        best_pos = None
        best_score = float('inf')
        
        # Iterate z first (lowest), then y, then x
        for z in z_cands:
            if z + d > D:
                continue
            for y in y_cands:
                if y + h > H:
                    continue
                for x in x_cands:
                    if x + w > W:
                        continue
                    if can_place(x, y, z, w, h, d, placed):
                        # Heuristic: minimize z, then y, then x
                        # We can also add a slight preference for corners
                        score = z * 10000 + y * 100 + x
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
                            
        return best_pos

    def run_pack(order, seed_override=None):
        # Set local RNG seed if provided for reproducibility in specific runs
        # But we mostly rely on global random for variety in multi-start
        
        placed = []
        placements = [None] * n
        current_vol = 0
        
        for idx in order:
            w, h, d = boxes[idx]
            orients = box_orientations[idx]
            
            best_pos = None
            best_orient = None
            best_score = float('inf')
            
            # Try all orientations
            for ow, oh, od in orients:
                pos = find_position(ow, oh, od, placed)
                if pos is not None:
                    x, y, z = pos
                    # Same scoring as find_position for consistency
                    score = z * 10000 + y * 100 + x
                    if score < best_score:
                        best_score = score
                        best_pos = pos
                        best_orient = (ow, oh, od)
            
            if best_pos is not None:
                x, y, z = best_pos
                ow, oh, od = best_orient
                placed.append((x, y, z, ow, oh, od))
                placements[idx] = (0, x, y, z, ow, oh, od)
                current_vol += ow * oh * od
            else:
                placements[idx] = (-1, 0, 0, 0, w, h, d)
                
        return current_vol, placements

    # Multi-start strategy
    # 1. Sorted by volume descending
    # 2. Sorted by max dimension descending
    # 3. Sorted by aspect ratio (most cube-like first)
    # 4-15: Random permutations
    
    orders = []
    
    # Order 1: Volume descending
    orders.append(sorted(range(n), key=lambda i: volumes[i], reverse=True))
    
    # Order 2: Max dimension descending
    orders.append(sorted(range(n), key=lambda i: max(boxes[i]), reverse=True))
    
    # Order 3: Aspect ratio (volume / max_dim^3, higher is more cubic)
    # We want to pack cubes first? Or long bars first? 
    # Usually, fitting large irregular shapes first is harder. 
    # Let's try "most cubic" first (highest ratio)
    def aspect_ratio(i):
        w, h, d = boxes[i]
        max_d = max(w, h, d)
        if max_d == 0: return 0
        vol = w*h*d
        return vol / (max_d**3)
        
    orders.append(sorted(range(n), key=aspect_ratio, reverse=True))

    # Add random orders
    # We will generate them on the fly in the loop to save memory and allow more iterations if time permits
    # But for the initial batch, let's pre-generate a few
    for _ in range(12):
        random_order = list(range(n))
        random.shuffle(random_order)
        orders.append(random_order)
    
    # Execute all pre-generated orders
    for order in orders:
        if time.time() - start_time > DEADLINE:
            break
            
        vol, placements = run_pack(order)
        
        if vol > best_fill:
            best_fill = vol
            best_placements = [p if p is not None else (-1, 0, 0, 0, w, h, d) for p, (w,h,d) in zip(placements, boxes)]
            # Fix: placements are already set for all indices in run_pack
    
    # If time remains, do more random restarts
    # This allows the algorithm to fill the available time budget, potentially finding better solutions
    # with increased variance.
    
    while time.time() - start_time < DEADLINE:
        random_order = list(range(n))
        random.shuffle(random_order)
        
        vol, placements = run_pack(random_order)
        
        if vol > best_fill:
            best_fill = vol
            best_placements = placements
            
        # Small sleep to avoid burning CPU if it's extremely fast, 
        # though for N=60 it shouldn't be instant.
        # Actually, just checking time is fine.

    # Ensure all placements are valid tuples
    final_placements = []
    for i, p in enumerate(best_placements):
        if p is None:
            w, h, d = boxes[i]
            final_placements.append((-1, 0, 0, 0, w, h, d))
        else:
            final_placements.append(p)
            
    return final_placements
# EVOLVE-BLOCK-END
