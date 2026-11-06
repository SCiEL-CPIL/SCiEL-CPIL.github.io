"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import heapq
from itertools import permutations

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 135.0  # 150s limit, keep 15s margin
    
    # Generate all unique rotations for each box
    def get_rotations(w, h, d):
        rots = set()
        for perm in permutations((w, h, d)):
            rots.add(perm)
        return list(rots)
    
    box_rots = [get_rotations(w, h, d) for w, h, d in boxes]
    box_vols = [w * h * d for w, h, d in boxes]
    
    container_vol = W * H * D
    
    # Sort boxes by volume descending for better packing
    indices = list(range(n))
    indices.sort(key=lambda i: box_vols[i], reverse=True)
    
    # Try multiple heuristics and keep the best
    def pack_heuristic(order, rotation_choice='first'):
        """
        Place boxes in given order using a simple shelf-based approach.
        rotation_choice: 'first', 'best_fit_x', 'best_fit_y', 'best_fit_z'
        """
        placements = [None] * n
        x, y, z = 0, 0, 0
        row_h = 0
        layer_d = 0
        
        for idx in order:
            if time.time() > deadline:
                break
            best_pos = None
            best_rot = None
            
            w, h, d = boxes[idx]
            rots = box_rots[idx]
            
            if rotation_choice == 'first':
                rw, rh, rd = rots[0]
            elif rotation_choice == 'best_fit_x':
                # Choose rotation that minimizes x extent
                rw, rh, rd = min(rots, key=lambda r: r[0])
            elif rotation_choice == 'best_fit_y':
                rw, rh, rd = min(rots, key=lambda r: r[1])
            elif rotation_choice == 'best_fit_z':
                rw, rh, rd = min(rots, key=lambda r: r[2])
            else:
                rw, rh, rd = rots[0]
            
            # Try to place at current position
            if x + rw <= W and y + rh <= H and z + rd <= D:
                placements[idx] = (0, x, y, z, rw, rh, rd)
                x += rw
                row_h = max(row_h, rh)
                layer_d = max(layer_d, rd)
                continue
            
            # Try to move to next row
            if x == 0 or x + rw > W:
                x, y, row_h = 0, y + row_h, 0
                if y + rh <= H and z + rd <= D:
                    placements[idx] = (0, x, y, z, rw, rh, rd)
                    x += rw
                    row_h = max(row_h, rh)
                    layer_d = max(layer_d, rd)
                    continue
            
            # Try to move to next layer
            if y == 0 or y + rh > H:
                y, z, layer_d, x, row_h = 0, z + layer_d, 0, 0, 0
                if z + rd <= D and x + rw <= W and y + rh <= H:
                    placements[idx] = (0, x, y, z, rw, rh, rd)
                    x += rw
                    row_h = max(row_h, rh)
                    layer_d = max(layer_d, rd)
                    continue
            
            # Try all rotations at current position
            placed = False
            for rw2, rh2, rd2 in rots:
                # Try at current x
                if x + rw2 <= W and y + rh2 <= H and z + rd2 <= D:
                    placements[idx] = (0, x, y, z, rw2, rh2, rd2)
                    x += rw2
                    row_h = max(row_h, rh2)
                    layer_d = max(layer_d, rd2)
                    placed = True
                    break
                # Try at new row
                if (x == 0 or x + rw2 > W) and (y + rh2 <= H) and (z + rd2 <= D):
                    nx, ny, nrh = 0, y + row_h, 0
                    if ny + rh2 <= H and z + rd2 <= D:
                        placements[idx] = (0, nx, ny, z, rw2, rh2, rd2)
                        x, y, row_h = nx, ny, max(nrh, rh2)
                        layer_d = max(layer_d, rd2)
                        placed = True
                        break
                # Try at new layer
                if (y == 0 or y + rh2 > H) and (z + rd2 <= D):
                    nz = z + layer_d
                    if nz + rd2 <= D and rw2 <= W and rh2 <= H:
                        placements[idx] = (0, 0, 0, nz, rw2, rh2, rd2)
                        x, y, z, row_h, layer_d = rw2, 0, nz, rh2, rd2
                        placed = True
                        break
            
            if not placed:
                placements[idx] = (-1, 0, 0, 0, w, h, d)
        
        # Fill any None placements with -1
        for i in range(n):
            if placements[i] is None:
                w, h, d = boxes[i]
                placements[i] = (-1, 0, 0, 0, w, h, d)
        
        return placements
    
    def score_placements(placements):
        """Calculate fraction of container volume filled."""
        filled = 0
        for p in placements:
            if p[0] == 0:
                filled += p[4] * p[5] * p[6]
        return filled / container_vol if container_vol > 0 else 0
    
    # Try multiple strategies
    best_placements = None
    best_score = -1
    
    # Strategy 1: Volume descending, first rotation
    placements = pack_heuristic(indices, 'first')
    s = score_placements(placements)
    if s > best_score:
        best_score = s
        best_placements = placements
    
    # Strategy 2: Volume descending, best fit x
    if time.time() < deadline:
        placements = pack_heuristic(indices, 'best_fit_x')
        s = score_placements(placements)
        if s > best_score:
            best_score = s
            best_placements = placements
    
    # Strategy 3: Volume descending, best fit y
    if time.time() < deadline:
        placements = pack_heuristic(indices, 'best_fit_y')
        s = score_placements(placements)
        if s > best_score:
            best_score = s
            best_placements = placements
    
    # Strategy 4: Volume descending, best fit z
    if time.time() < deadline:
        placements = pack_heuristic(indices, 'best_fit_z')
        s = score_placements(placements)
        if s > best_score:
            best_score = s
            best_placements = placements
    
    # Strategy 5: Try a few random orders
    if time.time() < deadline:
        for _ in range(10):
            if time.time() > deadline:
                break
            random_order = indices[:]
            random.shuffle(random_order)
            placements = pack_heuristic(random_order, 'first')
            s = score_placements(placements)
            if s > best_score:
                best_score = s
                best_placements = placements
    
    # Strategy 6: Try sorting by different criteria
    if time.time() < deadline:
        # Sort by max dimension
        indices_max = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
        placements = pack_heuristic(indices_max, 'first')
        s = score_placements(placements)
        if s > best_score:
            best_score = s
            best_placements = placements
        
        # Sort by min dimension
        indices_min = sorted(range(n), key=lambda i: min(boxes[i]), reverse=True)
        placements = pack_heuristic(indices_min, 'first')
        s = score_placements(placements)
        if s > best_score:
            best_score = s
            best_placements = placements
    
    # Strategy 7: Try a more sophisticated placement with backtracking for small cases
    if time.time() < deadline and n <= 20:
        def pack_backtrack(order, depth, placements, x, y, z, row_h, layer_d):
            if depth == len(order):
                return placements, x, y, z, row_h, layer_d
            
            if time.time() > deadline:
                return None, x, y, z, row_h, layer_d
            
            idx = order[depth]
            w, h, d = boxes[idx]
            rots = box_rots[idx]
            
            for rw, rh, rd in rots:
                # Try at current position
                if x + rw <= W and y + rh <= H and z + rd <= D:
                    placements[idx] = (0, x, y, z, rw, rh, rd)
                    result = pack_backtrack(order, depth + 1, placements, x + rw, y, z, max(row_h, rh), max(layer_d, rd))
                    if result[0] is not None:
                        return result
                    placements[idx] = None
                
                # Try at new row
                if (x == 0 or x + rw > W) and y + row_h + rh <= H and z + rd <= D:
                    placements[idx] = (0, 0, y + row_h, z, rw, rh, rd)
                    result = pack_backtrack(order, depth + 1, placements, rw, y + row_h, z, rh, max(layer_d, rd))
                    if result[0] is not None:
                        return result
                    placements[idx] = None
                
                # Try at new layer
                if (y == 0 or y + row_h + rh > H) and z + layer_d + rd <= D:
                    placements[idx] = (0, 0, 0, z + layer_d, rw, rh, rd)
                    result = pack_backtrack(order, depth + 1, placements, rw, 0, z + layer_d, rh, rd)
                    if result[0] is not None:
                        return result
                    placements[idx] = None
            
            placements[idx] = (-1, 0, 0, 0, w, h, d)
            # Try to place remaining boxes
            result = pack_backtrack(order, depth + 1, placements, x, y, z, row_h, layer_d)
            if result[0] is not None:
                return result
            
            return None, x, y, z, row_h, layer_d
        
        # Try backtracking with volume order
        placements = [None] * n
        result = pack_backtrack(indices, 0, placements, 0, 0, 0, 0, 0)
        if result[0] is not None:
            for i in range(n):
                if placements[i] is None:
                    w, h, d = boxes[i]
                    placements[i] = (-1, 0, 0, 0, w, h, d)
            s = score_placements(placements)
            if s > best_score:
                best_score = s
                best_placements = placements
    
    # Final fallback: if best_placements is still None, use the simple heuristic
    if best_placements is None:
        best_placements = pack_heuristic(indices, 'first')
    
    return best_placements
# EVOLVE-BLOCK-END
