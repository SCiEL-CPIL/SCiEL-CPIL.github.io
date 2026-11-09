"""3D Container Loading – combined grid-accelerated packing with multi-strategy search."""
import time
import itertools
import random
import numpy as np

def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s limit
    
    def check_deadline():
        return time.time() - start_time > DEADLINE
    
    def get_rotations(w, h, d):
        """Return unique axis rotations."""
        seen = set()
        rots = []
        for p in itertools.permutations((w, h, d)):
            if p not in seen:
                seen.add(p)
                rots.append(p)
        return rots
    
    # Pre-compute orientations for all boxes
    n = len(boxes)
    box_orientations = [get_rotations(w, h, d) for w, h, d in boxes]
    box_volumes = [w * h * d for w, h, d in boxes]
    
    # Grid-based occupancy for fast overlap checks
    MAX_GRID_DIM = 200
    
    def make_grid():
        if W <= MAX_GRID_DIM and H <= MAX_GRID_DIM and D <= MAX_GRID_DIM:
            return np.zeros((W, H, D), dtype=bool)
        return None
    
    def grid_check_and_mark(grid, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        if grid[x:x+w, y:y+h, z:z+d].any():
            return False
        grid[x:x+w, y:y+h, z:z+d] = True
        return True
    
    def grid_unmark(grid, x, y, z, w, h, d):
        grid[x:x+w, y:y+h, z:z+d] = False
    
    def list_check(placed, x, y, z, w, h, d):
        for px, py, pz, pw, ph, pd in placed:
            if not (x + w <= px or px + pw <= x or
                    y + h <= py or py + ph <= y or
                    z + d <= pz or pz + pd <= z):
                return False
        return True
    
    def generate_candidates(placed):
        """Generate candidate positions based on placed boxes."""
        cands = set()
        cands.add((0, 0, 0))
        
        for (px, py, pz, pw, ph, pd) in placed:
            cands.add((px + pw, py, pz))
            cands.add((px, py + ph, pz))
            cands.add((px, py, pz + pd))
            cands.add((px + pw, py + ph, pz))
            cands.add((px + pw, py, pz + pd))
            cands.add((px, py + ph, pz + pd))
            cands.add((px + pw, py + ph, pz + pd))
            # Also add boundary-aligned positions
            cands.add((0, py, pz))
            cands.add((px, 0, pz))
            cands.add((px, py, 0))
        
        valid = []
        for (x, y, z) in cands:
            if 0 <= x < W and 0 <= y < H and 0 <= z < D:
                valid.append((x, y, z))
        return valid
    
    def score_pos(x, y, z):
        """Score a position: prefer lower z, then lower y, then lower x."""
        return z * 10000 + y * 100 + x
    
    def try_place(idx, placed, grid):
        """Try to place box idx; returns (x,y,z,rw,rh,rd) or None."""
        w, h, d = boxes[idx]
        rots = box_orientations[idx]
        best = None
        best_sc = float('inf')
        
        cands = generate_candidates(placed)
        
        for (rw, rh, rd) in rots:
            if rw > W or rh > H or rd > D:
                continue
            for (x, y, z) in cands:
                if x + rw > W or y + rh > H or z + rd > D:
                    continue
                if grid is not None:
                    if grid[x:x+rw, y:y+rh, z:z+rd].any():
                        continue
                else:
                    if not list_check(placed, x, y, z, rw, rh, rd):
                        continue
                sc = score_pos(x, y, z)
                if sc < best_sc:
                    best_sc = sc
                    best = (x, y, z, rw, rh, rd)
        return best
    
    def pack_order(order):
        """Pack boxes in given order, return list of placements."""
        placed = []
        result = [None] * n
        grid = make_grid()
        
        for idx in order:
            if check_deadline():
                break
            pos = try_place(idx, placed, grid)
            if pos is not None:
                x, y, z, rw, rh, rd = pos
                placed.append((x, y, z, rw, rh, rd))
                if grid is not None:
                    grid[x:x+rw, y:y+rh, z:z+rd] = True
                result[idx] = (0, x, y, z, rw, rh, rd)
        
        for i in range(n):
            if result[i] is None:
                w, h, d = boxes[i]
                result[i] = (-1, 0, 0, 0, w, h, d)
        return result
    
    def filled_volume(result):
        return sum(w * h * d for (b, x, y, z, w, h, d) in result if b == 0)
    
    # Sorting strategies
    indexed = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    
    def sort_vol(lst):
        return sorted(lst, key=lambda t: (-t[1]*t[2]*t[3], -max(t[1], t[2], t[3])))
    
    def sort_max(lst):
        return sorted(lst, key=lambda t: (-max(t[1], t[2], t[3]), -t[1]*t[2]*t[3]))
    
    def sort_surf(lst):
        return sorted(lst, key=lambda t: (-2*(t[1]*t[2]+t[2]*t[3]+t[1]*t[3]), -t[1]*t[2]*t[3]))
    
    def sort_aspect(lst):
        def aspect(t):
            dims = sorted([t[1], t[2], t[3]])
            return (dims[2]/dims[0] if dims[0] > 0 else 0)
        return sorted(lst, key=lambda t: (aspect(t), -t[1]*t[2]*t[3]))
    
    def sort_base(lst):
        return sorted(lst, key=lambda t: (-t[1]*t[2], -t[1]*t[2]*t[3]))
    
    strategies = [sort_vol, sort_max, sort_surf, sort_aspect, sort_base]
    
    best_result = None
    best_vol = 0
    
    # Deterministic passes
    for sf in strategies:
        if check_deadline():
            break
        order = [t[0] for t in sf(indexed)]
        res = pack_order(order)
        vol = filled_volume(res)
        if vol > best_vol:
            best_vol = vol
            best_result = res
    
    # Random restarts with perturbed volume-descending order
    rng = random.Random(42)
    max_restarts = 500
    
    for _ in range(max_restarts):
        if check_deadline():
            break
        
        # Start with volume-descending order
        order = [t[0] for t in sort_vol(indexed)]
        
        # Apply random perturbations
        for _ in range(rng.randint(1, 8)):
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            order[i], order[j] = order[j], order[i]
        
        res = pack_order(order)
        vol = filled_volume(res)
        if vol > best_vol:
            best_vol = vol
            best_result = res
    
    # Local improvement: try to swap adjacent boxes in the best result
    if best_result is not None and not check_deadline():
        for _ in range(100):
            if check_deadline():
                break
            
            # Find two loaded boxes to potentially swap
            loaded_indices = [i for i in range(n) if best_result[i][0] == 0]
            if len(loaded_indices) < 2:
                break
            
            i, j = rng.sample(loaded_indices, 2)
            
            # Try swapping their positions
            bi, xi, yi, zi, wi, hi, di = best_result[i]
            bj, xj, yj, zj, wj, hj, dj = best_result[j]
            
            # Check if swapping works
            # Remove both, try placing i at j's spot and j at i's spot
            placed = [(x, y, z, w, h, d) for (b, x, y, z, w, h, d) in best_result 
                     if b == 0 and not (x == xi and y == yi and z == zi and w == wi and h == hi and d == di)
                     and not (x == xj and y == yj and z == zj and w == wj and h == hj and d == dj)]
            
            grid = make_grid()
            for (px, py, pz, pw, ph, pd) in placed:
                if grid is not None:
                    grid[px:px+pw, py:py+ph, pz:pz+pd] = True
            
            # Try placing i at j's position
            pos_i = try_place(i, placed, grid)
            if pos_i is None:
                continue
            
            xi2, yi2, zi2, wi2, hi2, di2 = pos_i
            placed.append((xi2, yi2, zi2, wi2, hi2, di2))
            if grid is not None:
                grid[xi2:xi2+wi2, yi2:yi2+hi2, zi2:zi2+di2] = True
            
            # Try placing j at i's position
            pos_j = try_place(j, placed, grid)
            if pos_j is None:
                continue
            
            xj2, yj2, zj2, wj2, hj2, dj2 = pos_j
            
            # Create new result
            new_result = list(best_result)
            new_result[i] = (0, xi2, yi2, zi2, wi2, hi2, di2)
            new_result[j] = (0, xj2, yj2, zj2, wj2, hj2, dj2)
            
            new_vol = filled_volume(new_result)
            if new_vol > best_vol:
                best_vol = new_vol
                best_result = new_result
    
    # Final reinsertion pass
    if best_result is not None and not check_deadline():
        left_out = [i for i in range(n) if best_result[i][0] == -1]
        if left_out:
            placed = [(x, y, z, w, h, d) for (b, x, y, z, w, h, d) in best_result if b == 0]
            grid = make_grid()
            for (px, py, pz, pw, ph, pd) in placed:
                if grid is not None:
                    grid[px:px+pw, py:py+ph, pz:pz+pd] = True
            
            for li_idx in left_out:
                if check_deadline():
                    break
                pos = try_place(li_idx, placed, grid)
                if pos is not None:
                    x, y, z, rw, rh, rd = pos
                    placed.append((x, y, z, rw, rh, rd))
                    if grid is not None:
                        grid[x:x+rw, y:y+rh, z:z+rd] = True
                    best_result[li_idx] = (0, x, y, z, rw, rh, rd)
            
            new_vol = filled_volume(best_result)
            if new_vol > best_vol:
                best_vol = new_vol
    
    if best_result is None:
        best_result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_result
