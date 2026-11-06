"""
Strategy: Skyline/Floor-Map based packing with multi-heuristic orderings and
light simulated annealing on the ordering to escape local optima.

The key improvement over the anchor-based approach is the use of a 2D grid
representation ("skyline") where each cell stores the minimum z-height.
This allows O(1) amortized overlap checks and efficient lowest-position
search, dramatically speeding up each packing trial and enabling many more
restarts within the time budget.
"""
import time
import itertools
import random
import math
import sys

# EVOLVE-BLOCK-START

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    deadline = time.time() + 115.0  # 5s safety margin for 120s limit

    # Precompute orientations for each box (unique permutations)
    all_orients = []
    for i in range(n):
        w, h, d = boxes[i]
        orients = list(set(itertools.permutations((w, h, d))))
        all_orients.append(orients)

    # Precompute volumes
    volumes = [w * h * d for w, h, d in boxes]
    container_vol = W * H * D

    # Skyline representation: 2D grid of min z-height
    # Grid resolution: use exact dimensions for precision
    # But for efficiency, we'll use a coarser grid if dimensions are large
    # For 60 boxes, dimensions are likely moderate; use exact grid
    grid_w = W
    grid_h = H
    # Initialize skyline: all cells at height 0
    skyline = [[0] * grid_w for _ in range(grid_h)]

    def reset_skyline():
        """Reset skyline to all zeros."""
        for y in range(grid_h):
            for x in range(grid_w):
                skyline[y][x] = 0

    def get_skyline_height(x, y, bw, bh):
        """Get the maximum z-height in the rectangle [x, x+bw) x [y, y+bh)."""
        if x + bw > W or y + bh > H or x < 0 or y < 0:
            return -1  # invalid
        max_h = 0
        for yy in range(y, min(y + bh, H)):
            for xx in range(x, min(x + bw, W)):
                if skyline[yy][xx] > max_h:
                    max_h = skyline[yy][xx]
        return max_h

    def set_skyline(x, y, bw, bh, z):
        """Set the skyline height in the rectangle to z (must be >= current)."""
        for yy in range(y, min(y + bh, H)):
            for xx in range(x, min(x + bw, W)):
                if skyline[yy][xx] < z:
                    skyline[yy][xx] = z

    def find_best_position(bw, bh, bd):
        """
        Find the best (lowest, then most compact) position for a box of dims (bw, bh, bd).
        Returns (x, y, z) or None if no fit.
        """
        best_pos = None
        best_score = float('inf')
        
        # Iterate over all possible x, y positions
        for y in range(H - bh + 1):
            for x in range(W - bw + 1):
                z = get_skyline_height(x, y, bw, bh)
                if z >= 0 and z + bd <= D:
                    # Score: prefer low z, then low x, then low y (BLF heuristic)
                    score = z * 1000000 + x * 1000 + y
                    if score < best_score:
                        best_score = score
                        best_pos = (x, y, z)
        
        return best_pos

    def pack_order(order):
        """Pack boxes in the given order using skyline-based placement."""
        reset_skyline()
        placed = []  # list of (x, y, z, bw, bh, bd)
        result = [None] * n
        filled = 0
        
        for idx in order:
            w, h, d = boxes[idx]
            best_pos = None
            best_orient = None
            best_score = float('inf')
            
            # Try all orientations
            for bw, bh, bd in all_orients[idx]:
                if bw > W or bh > H or bd > D:
                    continue
                
                pos = find_best_position(bw, bh, bd)
                if pos is not None:
                    x, y, z = pos
                    # Score: prefer low z, then low x, then low y
                    score = z * 1000000 + x * 1000 + y
                    if score < best_score:
                        best_score = score
                        best_pos = pos
                        best_orient = (bw, bh, bd)
            
            if best_pos is not None:
                bx, by, bz = best_pos
                bw, bh, bd = best_orient
                set_skyline(bx, by, bw, bh, bz + bd)
                placed.append((bx, by, bz, bw, bh, bd))
                result[idx] = (0, bx, by, bz, bw, bh, bd)
                filled += bw * bh * bd
            else:
                result[idx] = (-1, 0, 0, 0, w, h, d)
                
        return filled, result

    # Initial Order: Volume Descending
    initial_order = sorted(range(n), key=lambda i: volumes[i], reverse=True)
    
    best_filled, best_result = pack_order(initial_order)
    
    # If no boxes could be placed, return early
    if best_filled == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Multi-Start with various heuristics
    start_time = time.time()
    
    # Generate diverse initial orderings
    orderings = []
    
    # 1. Volume descending
    orderings.append(sorted(range(n), key=lambda i: volumes[i], reverse=True))
    
    # 2. Volume ascending
    orderings.append(sorted(range(n), key=lambda i: volumes[i]))
    
    # 3. Largest dimension first
    orderings.append(sorted(range(n), key=lambda i: max(boxes[i]), reverse=True))
    
    # 4. Smallest footprint first (w*h)
    orderings.append(sorted(range(n), key=lambda i: boxes[i][0] * boxes[i][1]))
    
    # 5. Random (seeded for reproducibility)
    random.seed(42)
    for _ in range(5):
        o = list(range(n))
        random.shuffle(o)
        orderings.append(o)
    
    # Evaluate all initial orderings
    for order in orderings:
        if time.time() > deadline - 5.0:
            break
        f, r = pack_order(order)
        if f > best_filled:
            best_filled = f
            best_result = r
            if best_filled == container_vol:
                return best_result
    
    # Simulated Annealing on the best ordering found
    # Start from the order that gave the best result
    # We need to track which order gave the best result
    # For simplicity, restart SA from volume-descending
    
    current_order = initial_order[:]
    T = 500.0
    cooling = 0.998
    min_T = 1.0
    
    sa_start = time.time()
    no_improve_count = 0
    
    while T > min_T and time.time() < deadline - 8.0:
        # Pick two random indices to swap
        i, j = random.sample(range(n), 2)
        
        # Swap
        current_order[i], current_order[j] = current_order[j], current_order[i]
        
        new_filled, new_result = pack_order(current_order)
        delta = new_filled - best_filled
        
        if delta > 0:
            # Accept and update best
            best_filled = new_filled
            best_result = new_result
            no_improve_count = 0
            if best_filled == container_vol:
                break
        elif delta >= 0 or random.random() < math.exp(delta / max(T, 0.01)):
            # Accept non-improvement (SA acceptance)
            no_improve_count += 1
        else:
            # Revert
            current_order[i], current_order[j] = current_order[j], current_order[i]
            no_improve_count += 1
        
        T *= cooling
        
        # Early stop if stuck for too long
        if no_improve_count > 200:
            no_improve_count = 0
            # Try a different starting point
            current_order = list(range(n))
            random.shuffle(current_order)
            T = max(T, 100.0)  # Reheat
    
    # If we still have time, try more random restarts
    while time.time() < deadline - 2.0:
        order = list(range(n))
        random.shuffle(order)
        f, r = pack_order(order)
        if f > best_filled:
            best_filled = f
            best_result = r
            if best_filled == container_vol:
                break
    
    # Fallback if something went wrong
    if best_result is None:
        best_result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_result

# EVOLVE-BLOCK-END
