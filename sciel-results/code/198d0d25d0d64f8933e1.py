import time
import itertools
import numpy as np

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    
    if W <= 0 or H <= 0 or D <= 0 or n == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s limit
    
    # Preprocess: generate all valid orientations for each box
    box_orientations = []
    for w, h, d in boxes:
        # Generate all unique permutations
        perms = set(itertools.permutations([w, h, d]))
        valid_perms = [(pw, ph, pd) for pw, ph, pd in perms if pw <= W and ph <= H and pd <= D]
        if valid_perms:
            box_orientations.append(valid_perms)
        else:
            box_orientations.append([(w, h, d)])  # Keep original even if doesn't fit (will be left out)
    
    # Sort boxes by volume descending, then by max dimension
    box_indices = list(range(n))
    box_indices.sort(key=lambda i: (-boxes[i][0]*boxes[i][1]*boxes[i][2], 
                                     -max(boxes[i][0], boxes[i][1], boxes[i][2])))
    
    # 3D grid approach: track occupied cells
    # Use a 3D boolean array for occupancy
    # Grid resolution: 1 unit per integer coordinate
    # Container dimensions W, H, D
    # A box at (x, y, z) with dims (w, h, d) occupies cells [x, x+w) x [y, y+h) x [z, z+d)
    
    # We'll use a 3D numpy array for occupancy
    # To save memory, we only allocate for the container size
    occ = np.zeros((W+1, H+1, D+1), dtype=bool)
    
    placements = [None] * n
    loaded_count = 0
    loaded_volume = 0
    
    # Greedy placement with best-fit heuristic
    # For each box in sorted order, try to place it in the "lowest" possible position
    # We'll try multiple strategies and keep the best result
    
    best_placements = None
    best_volume = -1
    
    # Strategy 1: Simple greedy - place at lowest z, then lowest y, then lowest x
    def try_placement_greedy():
        nonlocal occ, placements, loaded_count, loaded_volume, best_placements, best_volume
        
        occ = np.zeros((W+1, H+1, D+1), dtype=bool)
        placements = [None] * n
        loaded_count = 0
        loaded_volume = 0
        
        for idx in box_indices:
            if time.time() > deadline:
                break
                
            w, h, d = boxes[idx]
            placed = False
            
            # Try all orientations
            for ow, oh, od in box_orientations[idx]:
                if ow > W or oh > H or od > D:
                    continue
                    
                # Find the lowest z, then lowest y, then lowest x position
                # Scan z from 0 to D-od
                placed_here = False
                for z in range(D - od + 1):
                    if placed_here:
                        break
                    for y in range(H - oh + 1):
                        if placed_here:
                            break
                        for x in range(W - ow + 1):
                            # Check if this position is free
                            # Box occupies [x, x+ow) x [y, y+oh) x [z, z+od)
                            region = occ[x:x+ow, y:y+oh, z:z+od]
                            if not region.any():
                                # Place it
                                occ[x:x+ow, y:y+oh, z:z+od] = True
                                placements[idx] = (0, x, y, z, ow, oh, od)
                                loaded_count += 1
                                loaded_volume += ow * oh * od
                                placed = True
                                placed_here = True
                                break
                if placed:
                    break
            
            if not placed:
                placements[idx] = (-1, 0, 0, 0, w, h, d)
        
        if loaded_volume > best_volume:
            best_volume = loaded_volume
            best_placements = [tuple(p) for p in placements]
    
    # Run greedy
    try_placement_greedy()
    
    # Strategy 2: Try a different ordering - by height ascending to fill layers
    box_indices_h = list(range(n))
    box_indices_h.sort(key=lambda i: (boxes[i][1], -boxes[i][0]*boxes[i][1]*boxes[i][2]))
    
    def try_placement_height_order():
        nonlocal occ, placements, loaded_count, loaded_volume, best_placements, best_volume
        
        occ = np.zeros((W+1, H+1, D+1), dtype=bool)
        placements = [None] * n
        loaded_count = 0
        loaded_volume = 0
        
        for idx in box_indices_h:
            if time.time() > deadline:
                break
                
            w, h, d = boxes[idx]
            placed = False
            
            for ow, oh, od in box_orientations[idx]:
                if ow > W or oh > H or od > D:
                    continue
                    
                placed_here = False
                for z in range(D - od + 1):
                    if placed_here:
                        break
                    for y in range(H - oh + 1):
                        if placed_here:
                            break
                        for x in range(W - ow + 1):
                            region = occ[x:x+ow, y:y+oh, z:z+od]
                            if not region.any():
                                occ[x:x+ow, y:y+oh, z:z+od] = True
                                placements[idx] = (0, x, y, z, ow, oh, od)
                                loaded_count += 1
                                loaded_volume += ow * oh * od
                                placed = True
                                placed_here = True
                                break
                if placed:
                    break
            
            if not placed:
                placements[idx] = (-1, 0, 0, 0, w, h, d)
        
        if loaded_volume > best_volume:
            best_volume = loaded_volume
            best_placements = [tuple(p) for p in placements]
    
    try_placement_height_order()
    
    # Strategy 3: Width-first ordering
    box_indices_w = list(range(n))
    box_indices_w.sort(key=lambda i: (boxes[i][0], -boxes[i][0]*boxes[i][1]*boxes[i][2]))
    
    def try_placement_width_order():
        nonlocal occ, placements, loaded_count, loaded_volume, best_placements, best_volume
        
        occ = np.zeros((W+1, H+1, D+1), dtype=bool)
        placements = [None] * n
        loaded_count = 0
        loaded_volume = 0
        
        for idx in box_indices_w:
            if time.time() > deadline:
                break
                
            w, h, d = boxes[idx]
            placed = False
            
            for ow, oh, od in box_orientations[idx]:
                if ow > W or oh > H or od > D:
                    continue
                    
                placed_here = False
                for z in range(D - od + 1):
                    if placed_here:
                        break
                    for y in range(H - oh + 1):
                        if placed_here:
                            break
                        for x in range(W - ow + 1):
                            region = occ[x:x+ow, y:y+oh, z:z+od]
                            if not region.any():
                                occ[x:x+ow, y:y+oh, z:z+od] = True
                                placements[idx] = (0, x, y, z, ow, oh, od)
                                loaded_count += 1
                                loaded_volume += ow * oh * od
                                placed = True
                                placed_here = True
                                break
                if placed:
                    break
            
            if not placed:
                placements[idx] = (-1, 0, 0, 0, w, h, d)
        
        if loaded_volume > best_volume:
            best_volume = loaded_volume
            best_placements = [tuple(p) for p in placements]
    
    try_placement_width_order()
    
    # If we have time, try a depth-first ordering too
    if time.time() < deadline - 20:
        box_indices_d = list(range(n))
        box_indices_d.sort(key=lambda i: (boxes[i][2], -boxes[i][0]*boxes[i][1]*boxes[i][2]))
        
        def try_placement_depth_order():
            nonlocal occ, placements, loaded_count, loaded_volume, best_placements, best_volume
            
            occ = np.zeros((W+1, H+1, D+1), dtype=bool)
            placements = [None] * n
            loaded_count = 0
            loaded_volume = 0
            
            for idx in box_indices_d:
                if time.time() > deadline:
                    break
                    
                w, h, d = boxes[idx]
                placed = False
                
                for ow, oh, od in box_orientations[idx]:
                    if ow > W or oh > H or od > D:
                        continue
                        
                    placed_here = False
                    for z in range(D - od + 1):
                        if placed_here:
                            break
                        for y in range(H - oh + 1):
                            if placed_here:
                                break
                            for x in range(W - ow + 1):
                                region = occ[x:x+ow, y:y+oh, z:z+od]
                                if not region.any():
                                    occ[x:x+ow, y:y+oh, z:z+od] = True
                                    placements[idx] = (0, x, y, z, ow, oh, od)
                                    loaded_count += 1
                                    loaded_volume += ow * oh * od
                                    placed = True
                                    placed_here = True
                                    break
                    if placed:
                        break
                
                if not placed:
                    placements[idx] = (-1, 0, 0, 0, w, h, d)
            
            if loaded_volume > best_volume:
                best_volume = loaded_volume
                best_placements = [tuple(p) for p in placements]
        
        try_placement_depth_order()
    
    # Finalize: ensure all placements are set
    if best_placements is None:
        best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Verify no None values
    result = []
    for i, p in enumerate(best_placements):
        if p is None:
            w, h, d = boxes[i]
            result.append((-1, 0, 0, 0, w, h, d))
        else:
            result.append(p)
    
    return result
