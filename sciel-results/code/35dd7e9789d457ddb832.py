import time
import math
import random
from itertools import permutations

def load(boxes, container):
    start_time = time.time()
    DEADLINE = 145.0  # 5s safety margin

    W, H, D = container
    n = len(boxes)
    container_vol = W * H * D
    
    if container_vol == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    # Pre-compute orientations for each box
    box_orientations = []
    for i in range(n):
        w, h, d = boxes[i]
        seen = set()
        orients = []
        for perm in permutations((w, h, d)):
            if perm not in seen:
                seen.add(perm)
                orients.append(perm)
        box_orientations.append(orients)

    def can_place(x, y, z, w, h, d, placed):
        """
        Check if box at (x,y,z) with dims (w,h,d) fits.
        placed is sorted by x.
        """
        if x + w > W or y + h > H or z + d > D:
            return False
        
        for px, py, pz, pw, ph, pd in placed:
            # If the placed box is completely to the left, and sorted by x, 
            # we might still have boxes to the right, so we can't break on px+pw <= x 
            # unless we are sure no further box overlaps. 
            # Actually, if sorted by x, once px >= x + w, we can break.
            if px >= x + w:
                break
            if px + pw <= x:
                continue
                
            # Check y and z overlap
            if not (y + h <= py or py + ph <= y):
                if not (z + d <= pz or pz + pd <= z):
                    return False
        return True

    def get_best_position(ow, oh, od, placed):
        """
        Find the best position for a box with dims (ow, oh, od).
        Priority: Minimize y, then z, then x.
        placed is sorted by x.
        """
        x_cands = set([0])
        y_cands = set([0])
        z_cands = set([0])
        
        for px, py, pz, pw, ph, pd in placed:
            x_cands.add(px + pw)
            y_cands.add(py + ph)
            z_cands.add(pz + pd)
            
        # Filter candidates that fit within container for this box
        x_cands = [x for x in x_cands if x + ow <= W]
        y_cands = [y for y in y_cands if y + oh <= H]
        z_cands = [z for z in z_cands if z + od <= D]
        
        if not x_cands or not y_cands or not z_cands:
            return None
            
        # Sort to prioritize low y, then low z, then low x
        y_cands.sort()
        z_cands.sort()
        x_cands.sort()
        
        # Iterate in priority order
        for y in y_cands:
            for z in z_cands:
                for x in x_cands:
                    if can_place(x, y, z, ow, oh, od, placed):
                        return (x, y, z)
                        
        return None

    def insert_sorted(placed, new_box):
        """Insert new box into placed list maintaining sort by x."""
        x, y, z, w, h, d = new_box
        # Simple insertion sort is fine for N=60
        for i in range(len(placed)):
            if x < placed[i][0]:
                placed.insert(i, new_box)
                return
        placed.append(new_box)

    def greedy_fill(order, box_orientations, boxes):
        """
        Perform greedy packing using the given order of box indices.
        Returns a dict of placements: {index: (x, y, z, w, h, d)}
        """
        placed = []
        result = {}
        
        for idx in order:
            if time.time() - start_time > DEADLINE:
                break
                
            w, h, d = boxes[idx]
            placed_idx = False
            
            # Try all orientations
            for ow, oh, od in box_orientations[idx]:
                pos = get_best_position(ow, oh, od, placed)
                if pos is not None:
                    x, y, z = pos
                    new_box = (x, y, z, ow, oh, od)
                    insert_sorted(placed, new_box)
                    result[idx] = new_box
                    placed_idx = True
                    break
                    
        return result

    def compute_volume(result):
        vol = 0
        for idx, (x, y, z, w, h, d) in result.items():
            vol += w * h * d
        return vol

    # Strategies for sorting
    def sort_by_volume(boxes):
        return sorted(range(len(boxes)), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    def sort_by_max_dim(boxes):
        return sorted(range(len(boxes)), key=lambda i: max(boxes[i]), reverse=True)
        
    def sort_by_min_dim(boxes):
        return sorted(range(len(boxes)), key=lambda i: min(boxes[i]), reverse=True)

    def sort_by_base_area(boxes):
        return sorted(range(len(boxes)), key=lambda i: boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2], reverse=True)

    strategies = [
        sort_by_volume,
        sort_by_max_dim,
        sort_by_min_dim,
        sort_by_base_area,
    ]

    best_result = {}
    best_vol = 0

    # Deterministic runs
    for strat in strategies:
        if time.time() - start_time > DEADLINE:
            break
        order = strat(boxes)
        result = greedy_fill(order, box_orientations, boxes)
        vol = compute_volume(result)
        if vol > best_vol:
            best_vol = vol
            best_result = result

    # Random restarts
    max_restarts = 100
    for i in range(max_restarts):
        if time.time() - start_time > DEADLINE:
            break
        
        if i % 3 == 0:
            order = list(range(n))
            random.shuffle(order)
        elif i % 3 == 1:
            order = sort_by_volume(boxes)
            for _ in range(3):
                a, b = random.randint(0, n-1), random.randint(0, n-1)
                order[a], order[b] = order[b], order[a]
        else:
            order = sort_by_max_dim(boxes)
            for _ in range(3):
                a, b = random.randint(0, n-1), random.randint(0, n-1)
                order[a], order[b] = order[b], order[a]

        result = greedy_fill(order, box_orientations, boxes)
        vol = compute_volume(result)
        if vol > best_vol:
            best_vol = vol
            best_result = result

    # Construct final output
    placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    for idx, (x, y, z, w, h, d) in best_result.items():
        placements[idx] = (0, x, y, z, w, h, d)
        
    return placements
