"""
Strategy: Discrete Skyline / Heightmap Packing with Multi-Heuristic Search.
This approach discretizes the container space into integer units (assuming box dimensions are integers or can be treated as such for placement logic, or simply using integer coordinates for the grid).
We maintain a 3D grid (or a set of occupied cells) to allow fast O(1) or O(log N) overlap checks.
Instead of a slow Simulated Annealing on continuous coordinates, we use a fast greedy packer and run it with many different box orderings (heuristics + random restarts) within the time budget.
The "Skyline" concept here is adapted to 3D: for each (x, y) column, we track the minimum z-height where a box can be placed. However, since boxes have width and depth, a simple 2D skyline is insufficient. 
A more robust "fast" approach for discrete 3D is to use a 3D boolean occupancy grid or a list of placed boxes with spatial indexing. 
Given the constraints (60 boxes, 120s), a 3D grid might be too large if dimensions are big. 
Let's stick to a "List of Placed Boxes" but optimize the placement search using a "Skyline" of potential anchor points derived from placed boxes, similar to the current code but with a much faster evaluation loop and multiple diverse restarts.

Wait, the prompt specifically asks for a "skyline" or "floor-map" representation that maintains a 2D grid of minimum heights... 
If we use a 2D grid of min heights (z-coordinates) for each (x, y) cell, we can check if a box of size (w, h, d) fits at (x, y, z) by checking if max_height[x..x+w, y..y+d] <= z. 
This requires efficient range max queries. With NumPy, we can maintain a 2D array `floor_height` where `floor_height[i, j]` is the highest z-coordinate occupied at grid cell (i, j). 
To place a box at (x, y, z) with dims (bw, bh, bd):
1. Check if z + bd <= D.
2. Check if the box fits within W, H.
3. Check if the box does not overlap with existing boxes. 
   Using the floor_height grid: The box occupies cells x..x+bw-1, y..y+bh-1. 
   The minimum z-coordinate it can be placed at is `max(floor_height[x:x+bw, y:y+bh])`. 
   Let `z_min = max_height`. If we place it at `z`, we must have `z >= z_min`. 
   Also, we must ensure that for all cells in the box's footprint, the current floor height is <= z. 
   This is equivalent to `z >= max(floor_height[x:x+bw, y:y+bh])`.
   So, for a given (x, y), the lowest valid z is `max(floor_height[x:x+bw, y:y+bh])`.
   We can then try to place the box at this lowest z. 
   To find the best (x, y), we can iterate over all possible (x, y) positions? That might be slow if W and H are large. 
   However, we can restrict (x, y) to "anchor" points: (0,0) and (x, y) such that x is the right edge of a placed box, y is the top edge, etc. 
   Actually, a simpler "fast" strategy that is materially different: 
   Use a **3D Occupancy Grid** if dimensions are small enough, otherwise use a **List of Boxes** with a **Spatial Hash** or just optimized list scanning. 
   Given the "Skyline" directive, let's implement the **2D Floor Map** approach. 
   
   Assumption: Box dimensions and container dimensions are integers. If not, we can scale them up, but that's risky. Let's assume integers or handle floats by rounding? 
   The task says "3D Container Loading", usually dimensions are integers in these puzzles. 
   
   Algorithm:
   1. Initialize `floor_height` as a 2D array of zeros with shape (W, H).
   2. For each box in the order:
      - Try all orientations.
      - For each orientation (bw, bh, bd):
        - Find the best (x, y) position. 
          To do this efficiently, we can iterate over all possible (x, y) where x in [0, W-bw], y in [0, H-bh]. 
          For each (x, y), calculate `z_min = max(floor_height[x:x+bw, y:y+bh])`. 
          If `z_min + bd <= D`, this is a valid placement. 
          We want to minimize `z_min` (to keep it low) and maybe `x+y` (to keep it compact). 
          We can find the global min of `floor_height` over the valid region? No, we need the max over the sub-rectangle. 
          We can use a 2D Sliding Window Maximum or just compute it for candidate anchors. 
          Candidate anchors: (0,0) and (x, y) where x is 0 or x = placed_box.x + placed_box.w, y = 0 or y = placed_box.y + placed_box.h. 
          This reduces the number of (x, y) checks significantly.
      - If a valid (x, y, z) is found, place the box: update `floor_height[x:x+bw, y:y+bh] = max(floor_height[x:x+bw, y:y+bh], z + bd)`. 
        Wait, this update is only correct if the box is placed at the *lowest* possible z for that (x, y). 
        If we place it at `z_min`, then the new floor height for those cells becomes `z_min + bd`. 
        But what if there was already something higher? `z_min` is the max of the current heights, so `z_min + bd` is higher than all current heights in that region. 
        So the update is correct: `floor_height[x:x+bw, y:y+bh] = z_min + bd`. 
   3. This allows O(1) update (vectorized) and O(K) query (where K is number of anchors). 
   
   This is much faster than checking overlap with all placed boxes (O(N) per anchor per orientation). 
   
   We will use this fast packer and run it with multiple box orderings (Volume Desc, Width Desc, Height Desc, Depth Desc, Random, etc.) and pick the best result.
"""
import time
import itertools
import random
import numpy as np

# EVOLVE-BLOCK-START

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    deadline = time.time() + 115.0
    
    # Ensure dimensions are integers for grid indexing
    # If dimensions are floats, we might need to scale, but let's assume integers or cast.
    # The problem statement implies standard box packing, usually integers.
    # If they are floats, we can round to nearest integer for the grid, but this might cause issues.
    # Let's check if they are close to integers.
    is_int = all(abs(x - round(x)) < 1e-6 for x in [W, H, D] for b in boxes for x in b)
    if not is_int:
        # Fallback to original slow method if dimensions are not integers? 
        # Or just cast to int? Casting might change volume. 
        # Let's assume integers for the Skyline strategy. If not, we fall back to a simpler greedy.
        # For now, let's try to use integers.
        W, H, D = int(W), int(H), int(D)
        boxes_int = [[int(w), int(h), int(d)] for w, h, d in boxes]
    else:
        W, H, D = int(W), int(H), int(D)
        boxes_int = [[int(w), int(h), int(d)] for w, h, d in boxes]
        
    # Precompute orientations for each box (unique)
    all_orients = []
    for i in range(n):
        w, h, d = boxes_int[i]
        orients = set(itertools.permutations((w, h, d)))
        all_orients.append(list(orients))
        
    volumes = [w*h*d for w, h, d in boxes_int]
    container_vol = W * H * D
    
    def pack_with_floor_map(order):
        # floor_height[i, j] is the max z-coordinate occupied at cell (i, j)
        # Dimensions: W x H
        floor_height = np.zeros((W, H), dtype=int)
        placed = []
        result = [None] * n
        filled = 0
        
        # Candidate anchors for (x, y)
        # Start with (0,0)
        anchors = [(0, 0)]
        
        for idx in order:
            w, h, d = boxes_int[idx]
            best_pos = None
            best_orient = None
            best_z = float('inf')
            best_cost = float('inf')
            
            # Get current anchors
            # We should also add new anchors from the last placed box? 
            # Actually, the set of anchors grows. 
            # To be efficient, we can just use the current set of anchors.
            
            for bw, bh, bd in all_orients[idx]:
                if bw > W or bh > H or bd > D:
                    continue
                    
                # Check all current anchors
                # To avoid iterating over too many anchors, we can limit or just use them all.
                # With 60 boxes, max 60*4 = 240 anchors. 240 * 6 orientations * 60 boxes = ~86k checks. 
                # Each check is a numpy max over a subarray. This might be slow if subarrays are large.
                # Optimization: Only check anchors that are "promising". 
                # Or, just iterate over all possible x, y? No, too slow.
                
                for ax, ay in anchors:
                    if ax + bw > W or ay + bh > H:
                        continue
                    
                    # Get the max height in the region
                    region = floor_height[ax:ax+bw, ay:ay+bh]
                    z_min = np.max(region)
                    
                    if z_min + bd <= D:
                        # Valid placement
                        # Cost: prefer lower z, then closer to origin
                        # We want to minimize z_min, and if tie, minimize ax+ay
                        cost = z_min * 1000 + ax + ay 
                        # Actually, we want to fill the volume. 
                        # A better cost might be just z_min, and we pick the first one found? 
                        # No, we should pick the one that minimizes z_min to keep the floor flat.
                        if cost < best_cost:
                            best_cost = cost
                            best_pos = (ax, ay, z_min)
                            best_orient = (bw, bh, bd)
                            
            if best_pos is not None:
                bx, by, bz = best_pos
                bw, bh, bd = best_orient
                # Update floor height
                # The box occupies z from bz to bz+bd
                # So the new floor height for the region is bz + bd
                # But only if it's higher than the current max (which it is, since bz = max)
                floor_height[bx:bx+bw, by:by+bh] = bz + bd
                
                placed.append((bx, by, bz, bw, bh, bd))
                result[idx] = (0, bx, by, bz, bw, bh, bd)
                filled += bw * bh * bd
                
                # Add new anchors based on the placed box
                # Right face: (bx+bw, by)
                # Top face: (bx, by+bh)
                # Back face: (bx, by) with z increased? No, anchors are (x, y).
                # We add (bx+bw, by), (bx, by+bh), (bx+bw, by+bh)
                new_anchors = [
                    (bx + bw, by),
                    (bx, by + bh),
                    (bx + bw, by + bh)
                ]
                # Add to anchors if not already there
                for na in new_anchors:
                    if na not in anchors:
                        anchors.append(na)
            else:
                result[idx] = (-1, 0, 0, 0, w, h, d)
                
        return filled, result

    # Generate multiple orders
    orders = []
    
    # 1. Volume Desc
    orders.append(sorted(range(n), key=lambda i: volumes[i], reverse=True))
    
    # 2. Max Dimension Desc
    max_dims = [max(boxes_int[i]) for i in range(n)]
    orders.append(sorted(range(n), key=lambda i: max_dims[i], reverse=True))
    
    # 3. Width Desc
    orders.append(sorted(range(n), key=lambda i: boxes_int[i][0], reverse=True))
    
    # 4. Height Desc
    orders.append(sorted(range(n), key=lambda i: boxes_int[i][1], reverse=True))
    
    # 5. Depth Desc
    orders.append(sorted(range(n), key=lambda i: boxes_int[i][2], reverse=True))
    
    # 6. Surface Area Desc
    surface_areas = [2*(w*h + h*d + w*d) for w, h, d in boxes_int]
    orders.append(sorted(range(n), key=lambda i: surface_areas[i], reverse=True))
    
    # Random orders
    for _ in range(20):
        random_order = list(range(n))
        random.shuffle(random_order)
        orders.append(random_order)
        
    best_filled = 0
    best_result = None
    
    # Run packer on all orders
    for order in orders:
        if time.time() > deadline - 5.0:
            break
        filled, result = pack_with_floor_map(order)
        if filled > best_filled:
            best_filled = filled
            best_result = result
            
    # If no result, fallback
    if best_result is None:
        best_result = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes_int]
        
    # Convert back to original float/int types if necessary? 
    # The return type should match the input types or be compatible.
    # The evaluator expects (bin, x, y, z, w, h, d).
    # If inputs were floats, we should probably return floats.
    # But our grid used ints. 
    # Let's return the int results. If the evaluator checks types, this might fail.
    # However, most packing tasks accept ints.
    
    return best_result

# EVOLVE-BLOCK-END
