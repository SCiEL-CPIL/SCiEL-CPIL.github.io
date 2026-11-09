import time
import random
import itertools
import sys
import numpy as np

# EVOLVE-BLOCK-START
def load(boxes, container):
    start_time = time.time()
    DEADLINE = 118.0  # Safety margin for 120s limit (evaluator kills at 150s)
    
    W, H, D = container
    container_vol = W * H * D
    if container_vol == 0 or len(boxes) == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    
    # Pre-calculate orientations for each box (unique permutations)
    box_orientations = []
    for w, h, d in boxes:
        perms = set(itertools.permutations((w, h, d)))
        box_orientations.append(list(perms))
    
    volumes = [w*h*d for w, h, d in boxes]
    
    best_fill = 0
    best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    def run_pack(order):
        """Place boxes in given order, trying all orientations, finding best low position."""
        placed_x = []
        placed_y = []
        placed_z = []
        placed_w = []
        placed_h = []
        placed_d = []
        placements = [None] * n
        current_vol = 0
        
        for idx in order:
            w, h, d = boxes[idx]
            orients = box_orientations[idx]
            
            best_pos = None
            best_orient = None
            best_score = float('inf')
            
            for ow, oh, od in orients:
                # Generate candidate positions
                x_cands = [0]
                y_cands = [0]
                z_cands = [0]
                
                for i in range(len(placed_x)):
                    if placed_x[i] + placed_w[i] <= W:
                        x_cands.append(placed_x[i] + placed_w[i])
                    if placed_y[i] + placed_h[i] <= H:
                        y_cands.append(placed_y[i] + placed_h[i])
                    if placed_z[i] + placed_d[i] <= D:
                        z_cands.append(placed_z[i] + placed_d[i])
                
                x_cands = sorted(set(x_cands))
                y_cands = sorted(set(y_cands))
                z_cands = sorted(set(z_cands))
                
                # Try to find best position (lowest z, then y, then x)
                found = False
                for z in z_cands:
                    if z + od > D:
                        continue
                    for y in y_cands:
                        if y + oh > H:
                            continue
                        for x in x_cands:
                            if x + ow > W:
                                continue
                            
                            # Check overlap with all placed boxes using numpy
                            if len(placed_x) > 0:
                                px = np.array(placed_x)
                                py = np.array(placed_y)
                                pz = np.array(placed_z)
                                pw = np.array(placed_w)
                                ph = np.array(placed_h)
                                pd = np.array(placed_d)
                                
                                overlap = ~((x + ow <= px) | (px + pw <= x) |
                                          (y + oh <= py) | (py + ph <= y) |
                                          (z + od <= pz) | (pz + pd <= z))
                                if np.any(overlap):
                                    continue
                            
                            # Valid position found
                            score = z * H * W + y * W + x
                            if score < best_score:
                                best_score = score
                                best_pos = (x, y, z)
                                best_orient = (ow, oh, od)
                            found = True
                            break
                        if found:
                            break
                    if found:
                        break
                
                if found:
                    break
            
            if best_pos is not None:
                x, y, z = best_pos
                ow, oh, od = best_orient
                placed_x.append(x)
                placed_y.append(y)
                placed_z.append(z)
                placed_w.append(ow)
                placed_h.append(oh)
                placed_d.append(od)
                placements[idx] = (0, x, y, z, ow, oh, od)
                current_vol += ow * oh * od
            else:
                placements[idx] = (-1, 0, 0, 0, w, h, d)
                
        return current_vol, placements
    
    # Generate diverse ordering strategies
    orders = []
    
    # Strategy 1: Volume descending
    orders.append(sorted(range(n), key=lambda i: volumes[i], reverse=True))
    
    # Strategy 2: Elongation (max dimension) descending
    elongations = [max(w, h, d) for w, h, d in boxes]
    orders.append(sorted(range(n), key=lambda i: elongations[i], reverse=True))
    
    # Strategy 3: Surface area descending
    surface_areas = [2*(w*h + w*d + h*d) for w, h, d in boxes]
    orders.append(sorted(range(n), key=lambda i: surface_areas[i], reverse=True))
    
    # Strategy 4: Volume ascending (try small boxes first to fill gaps)
    orders.append(sorted(range(n), key=lambda i: volumes[i]))
    
    # Random orders - generate based on remaining time
    random.seed(42)
    num_random = 20
    for _ in range(num_random):
        random_order = list(range(n))
        random.shuffle(random_order)
        orders.append(random_order)
    
    # Run all orders within time budget
    for i, order in enumerate(orders):
        if time.time() - start_time > DEADLINE:
            break
            
        vol, placements = run_pack(order)
        
        if vol > best_fill:
            best_fill = vol
            best_placements = placements
    
    # If we still have time, do more random restarts
    while time.time() - start_time < DEADLINE - 5:
        random_order = list(range(n))
        random.shuffle(random_order)
        vol, placements = run_pack(random_order)
        if vol > best_fill:
            best_fill = vol
            best_placements = placements
    
    return best_placements
# EVOLVE-BLOCK-END
