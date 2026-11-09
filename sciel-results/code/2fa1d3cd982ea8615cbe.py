import time
import itertools
import random
import numpy as np

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0
    
    W, H, D = container
    n_boxes = len(boxes)
    
    MAX_GRID_SIZE = 200
    if W > MAX_GRID_SIZE or H > MAX_GRID_SIZE or D > MAX_GRID_SIZE:
        return _load_fallback(boxes, container, deadline)
    
    height_map = np.zeros((W, H), dtype=np.int32)
    
    def blb_solve(ordered_indices, placement_strategy='blb'):
        results = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n_boxes)]
        placed = []
        candidates = [(0, 0, 0)]
        
        for idx in ordered_indices:
            if time.time() > deadline:
                return results, False
            
            w, h, d = boxes[idx]
            best_pos = None
            best_rot = None
            best_score = float('inf')
            
            rotations = list(set(itertools.permutations((w, h, d))))
            
            for rw, rh, rd in rotations:
                for cx, cy, cz in candidates:
                    if cx + rw > W or cy + rh > H or cz + rd > D:
                        continue
                    
                    # Fast pre-filter using height map
                    if cx + rw <= W and cy + rh <= H:
                        footprint = height_map[cx:cx+rw, cy:cy+rh]
                        if footprint.size > 0:
                            max_h = int(np.max(footprint))
                            if max_h > cz + rd or max_h > D:
                                continue
                    
                    # Exact overlap check
                    valid = True
                    for px, py, pz, pw, ph, pd in placed:
                        if (cx < px + pw and px < cx + rw and
                            cy < py + ph and py < cy + rh and
                            cz < pz + pd and pz < cz + rd):
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    if placement_strategy == 'blb':
                        score = cz * 1000000 + cy * 1000 + cx
                    elif placement_strategy == 'bottom-left':
                        score = cz * 1000000 + cx * 1000 + cy
                    else:
                        score = cy * 1000000 + cz * 1000 + cx
                    
                    if score < best_score:
                        best_score = score
                        best_pos = (cx, cy, cz)
                        best_rot = (rw, rh, rd)
            
            if best_pos is not None:
                bx, by, bz = best_pos
                bw, bh, bd = best_rot
                placed.append((bx, by, bz, bw, bh, bd))
                results[idx] = (0, bx, by, bz, bw, bh, bd)
                
                # Update height map
                height_map[bx:bx+bw, by:by+bh] = np.maximum(
                    height_map[bx:bx+bw, by:by+bh], bz + bd)
                
                # Generate new candidate points
                candidates.append((bx, by, bz + bd))
                candidates.append((bx + bw, by, bz))
                candidates.append((bx, by + bh, bz))
                candidates.append((bx + bw, by + bh, bz))
                candidates.append((bx + bw, by, bz + bd))
                candidates.append((bx, by + bh, bz + bd))
                candidates.append((bx + bw, by + bh, bz + bd))
        
        return results, True
    
    def calculate_score(results):
        return sum(w * h * d for bin_id, x, y, z, w, h, d in results if bin_id == 0)
    
    best_results = None
    best_score = 0
    
    strategies = [
        lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2],
        lambda i: max(boxes[i]),
        lambda i: boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2],
        lambda i: boxes[i][0] * boxes[i][1],
        lambda i: -min(boxes[i]),
    ]
    
    indices = list(range(n_boxes))
    placement_strategies = ['blb', 'bottom-left', 'left-bottom']
    
    for sort_func in strategies:
        if time.time() > deadline:
            break
        sorted_indices = sorted(indices, key=sort_func, reverse=True)
        for ps in placement_strategies:
            if time.time() > deadline:
                break
            height_map[:] = 0
            results, completed = blb_solve(sorted_indices, ps)
            if completed:
                score = calculate_score(results)
                if score > best_score:
                    best_score = score
                    best_results = results
    
    if best_results is not None and time.time() < deadline - 10.0:
        attempts = 0
        while time.time() < deadline - 5.0 and attempts < 40:
            random.shuffle(indices)
            height_map[:] = 0
            ps = placement_strategies[attempts % len(placement_strategies)]
            results, completed = blb_solve(indices, ps)
            attempts += 1
            if completed:
                score = calculate_score(results)
                if score > best_score:
                    best_score = score
                    best_results = results
    
    if best_results is None:
        best_results = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_results

def _load_fallback(boxes, container, deadline):
    W, H, D = container
    n_boxes = len(boxes)
    
    def blb_solve(ordered_indices, placement_strategy='blb'):
        placed = []
        results = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n_boxes)]
        candidates = [(0, 0, 0)]
        
        for idx in ordered_indices:
            if time.time() > deadline:
                return results, False
            
            w, h, d = boxes[idx]
            best_pos = None
            best_rot = None
            best_score = float('inf')
            rotations = list(set(itertools.permutations((w, h, d))))
            
            for rw, rh, rd in rotations:
                for cx, cy, cz in candidates:
                    if cx + rw > W or cy + rh > H or cz + rd > D:
                        continue
                    valid = True
                    for px, py, pz, pw, ph, pd in placed:
                        if (cx < px + pw and px < cx + rw and
                            cy < py + ph and py < cy + rh and
                            cz < pz + pd and pz < cz + rd):
                            valid = False
                            break
                    if not valid:
                        continue
                    if placement_strategy == 'blb':
                        score = cz * 1000000 + cy * 1000 + cx
                    elif placement_strategy == 'bottom-left':
                        score = cz * 1000000 + cx * 1000 + cy
                    else:
                        score = cy * 1000000 + cz * 1000 + cx
                    if score < best_score:
                        best_score = score
                        best_pos = (cx, cy, cz)
                        best_rot = (rw, rh, rd)
            
            if best_pos is not None:
                bx, by, bz = best_pos
                bw, bh, bd = best_rot
                placed.append((bx, by, bz, bw, bh, bd))
                results[idx] = (0, bx, by, bz, bw, bh, bd)
                candidates.append((bx, by, bz + bd))
                candidates.append((bx + bw, by, bz))
                candidates.append((bx, by + bh, bz))
                candidates.append((bx + bw, by + bh, bz))
                candidates.append((bx + bw, by, bz + bd))
                candidates.append((bx, by + bh, bz + bd))
                candidates.append((bx + bw, by + bh, bz + bd))
        
        return results, True
    
    def calculate_score(results):
        return sum(w * h * d for bin_id, x, y, z, w, h, d in results if bin_id == 0)
    
    best_results = None
    best_score = 0
    strategies = [
        lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2],
        lambda i: max(boxes[i]),
        lambda i: boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2],
        lambda i: boxes[i][0] * boxes[i][1],
        lambda i: -min(boxes[i]),
    ]
    indices = list(range(n_boxes))
    placement_strategies = ['blb', 'bottom-left', 'left-bottom']
    
    for sort_func in strategies:
        if time.time() > deadline:
            break
        sorted_indices = sorted(indices, key=sort_func, reverse=True)
        for ps in placement_strategies:
            if time.time() > deadline:
                break
            results, completed = blb_solve(sorted_indices, ps)
            if completed:
                score = calculate_score(results)
                if score > best_score:
                    best_score = score
                    best_results = results
    
    if best_results is not None and time.time() < deadline - 10.0:
        attempts = 0
        while time.time() < deadline - 5.0 and attempts < 30:
            random.shuffle(indices)
            ps = placement_strategies[attempts % len(placement_strategies)]
            results, completed = blb_solve(indices, ps)
            attempts += 1
            if completed:
                score = calculate_score(results)
                if score > best_score:
                    best_score = score
                    best_results = results
    
    if best_results is None:
        best_results = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_results
