"""3D Container Loading using Skyline/Grid-based Best Fit with Randomized Restarts."""
import time
import random
import numpy as np

def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    if n == 0:
        return []
        
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit
    
    # Determine grid resolution
    # We want cells small enough to capture boundaries but large enough for speed.
    # A max dimension of 200 is a good balance.
    max_dim = max(W, H, D)
    # Ensure resolution is at least 10x10x10 for small containers, but capped.
    # If boxes are very small relative to container, we might need finer grid, 
    # but usually 100-200 is sufficient for "fraction of volume" metrics 
    # unless precision of placement matters more than fill rate.
    # Let's use a resolution that makes the grid size manageable.
    # Grid size = (W/res) * (H/res) * (D/res).
    # Let's target ~100k to 1M cells max.
    # 100^3 = 1M.
    res = max(1, int(max_dim / 100))
    
    # Ensure integer dimensions for grid
    gw = max(1, int(np.ceil(W / res)))
    gh = max(1, int(np.ceil(H / res)))
    gd = max(1, int(np.ceil(D / res)))
    
    # Pre-calculate box volumes for sorting
    box_vols = [w * h * d for w, h, d in boxes]
    
    best_score = -1.0
    best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Define a heuristic function to evaluate a placement
    # Score is total volume placed / container volume
    container_vol = W * H * D
    
    def get_orientations(w, h, d):
        # Return unique orientations
        orientations = set()
        orientations.add((w, h, d))
        orientations.add((w, d, h))
        orientations.add((h, w, d))
        orientations.add((h, d, w))
        orientations.add((d, w, h))
        orientations.add((d, h, w))
        # Filter out duplicates (e.g., if w==h)
        return list(orientations)

    def try_pack(order, seed):
        rng = random.Random(seed)
        # Initialize grid
        # 0 means free, 1 means occupied
        grid = np.zeros((gw, gh, gd), dtype=np.uint8)
        
        placed_volumes = 0.0
        placements = [None] * n
        
        # Map box indices to their position in the order
        # We iterate through the order, picking the box at that index
        
        # To speed up, we can pre-compute the grid indices for each box orientation?
        # No, because position changes.
        
        # Helper to check if a box fits at (x, y, z) with dims (bw, bh, bd)
        # x, y, z are grid coordinates
        def fits(x, y, z, bw, bh, bd):
            if x + bw > gw or y + bh > gh or z + bd > gd:
                return False
            # Check if any cell in the sub-grid is occupied
            # Using slicing is fast in NumPy
            sub = grid[x:x+bw, y:y+bh, z:z+bd]
            if sub.max() > 0:
                return False
            return True

        def place(x, y, z, bw, bh, bd):
            grid[x:x+bw, y:y+bh, z:z+bd] = 1

        # Iterate through boxes in the given order
        for i in order:
            w, h, d = boxes[i]
            # Try all orientations
            found = False
            # Sort orientations by "most compact" or just random? 
            # Randomization in orientation choice can help escape local minima
            orients = get_orientations(w, h, d)
            rng.shuffle(orients)
            
            best_pos = None
            best_score_pos = float('inf')
            
            # We need to find a valid position.
            # Scanning the entire grid for each box is O(N^2 * GridSize) which might be too slow.
            # Optimization: Only check positions where the box *could* fit.
            # Better: Use a "list of empty corners" or just scan from 0,0,0 upwards?
            # Since we want "Best Fit" (lowest z, then y, then x), we can scan systematically.
            
            # To keep it fast, we limit the search space or use a coarser scan?
            # Actually, for 60 boxes, even a full grid scan of 100x100x100 = 1M checks per box * 60 boxes = 60M checks.
            # In Python, this is too slow.
            # We must optimize the search.
            
            # Alternative: Maintain a list of "candidate positions" (corners of empty spaces)?
            # Or, simply scan but break early if found?
            # "First Fit" is fast but suboptimal. "Best Fit" is slow.
            # Let's try a compromise: Scan a subset of positions or use a coarser grid for search?
            
            # Let's stick to a slightly optimized scan.
            # If we use `grid.max` on slices, it's C-speed.
            # But iterating over 1M positions in Python loop is slow.
            
            # Strategy:
            # 1. Generate a list of potential x,y,z? No.
            # 2. Use a "skyline" approach on the Z-axis?
            
            # Let's try a different tactic for speed:
            # Instead of checking every cell, we can maintain a "min height" array for 2D (X,Y) -> Z.
            # This is the "Skyline" or "Floor Map" idea from the prompt.
            # floor_map[x][y] = minimum z where a box can be placed at (x,y).
            
            # Let's implement the Floor Map approach for speed.
            # floor_map is 2D array of size gw x gh.
            # When we place a box at (x,y,z) with dims (bw,bh,bd):
            # For all i in [x, x+bw) and j in [y, y+bh):
            #   floor_map[i][j] = max(floor_map[i][j], z + bd)
            
            # This makes placement O(BW * BH).
            # And checking if a box fits at (x,y) is O(1):
            #   z = floor_map[x][y] (actually max over the footprint)
            #   if z + bd <= gd: fits.
            
            # Let's re-implement `try_pack` using Floor Map.
            
            floor_map = np.zeros((gw, gh), dtype=np.int32)
            
            for i in order:
                w, h, d = boxes[i]
                orients = get_orientations(w, h, d)
                rng.shuffle(orients)
                
                best_pos = None
                best_score_pos = float('inf')
                
                for bw, bh, bd in orients:
                    # Convert to grid units
                    # Note: We are using grid units.
                    # If w/res is 0.5, it becomes 1 cell.
                    bw_g = max(1, int(np.ceil(bw / res)))
                    bh_g = max(1, int(np.ceil(bh / res)))
                    bd_g = max(1, int(np.ceil(bd / res)))
                    
                    if bw_g > gw or bh_g > gh or bd_g > gd:
                        continue
                    
                    # Find best position for this orientation
                    # Scan x, y
                    # To speed up, we can skip scanning if we know it won't fit?
                    # We scan x from 0 to gw-bw_g, y from 0 to gh-bh_g
                    
                    # Optimization: We can pre-compute a list of "candidate" x,y?
                    # For now, let's scan but limit the search if possible.
                    # With gw, gh ~ 100, loop is 10,000 iterations.
                    # 60 boxes * 6 orients * 10,000 = 3.6M iterations.
                    # Python can do ~10M-50M simple ops/sec. This is borderline but likely OK for 115s.
                    # We can optimize by breaking early if we find a "good" position?
                    # "Best Fit" requires checking all.
                    
                    # Let's try to optimize the inner loop using NumPy vectorization where possible?
                    # Hard to do with the "max over footprint" logic easily in one shot for all x,y.
                    
                    # We'll stick to Python loops but keep them tight.
                    
                    for x in range(gw - bw_g + 1):
                        # Optimization: If floor_map[x, y] is already high, maybe skip?
                        # We need max(floor_map[x:x+bw_g, y:y+bh_g])
                        
                        # We can pre-compute the "min z" for each x,y? No, it changes.
                        
                        for y in range(gh - bh_g + 1):
                            # Get the current max height in the footprint
                            # This is a slice max, which is fast.
                            z_min = np.max(floor_map[x:x+bw_g, y:y+bh_g])
                            
                            if z_min + bd_g <= gd:
                                # It fits!
                                # Score: prefer lower z, then lower y, then lower x
                                score = z_min * 10000 + y * 100 + x
                                if score < best_score_pos:
                                    best_score_pos = score
                                    best_pos = (x, y, z_min, bw_g, bh_g, bd_g)
                
                if best_pos:
                    x, y, z, bw_g, bh_g, bd_g = best_pos
                    # Update floor map
                    floor_map[x:x+bw_g, y:y+bh_g] = np.maximum(floor_map[x:x+bw_g, y:y+bh_g], z + bd_g)
                    
                    # Map back to real coordinates
                    # The box occupies [x*res, (x+bw_g)*res] in W?
                    # No, we used ceil(w/res).
                    # To be precise, we should place it at the actual coordinate.
                    # Let's place it at (x*res, y*res, z*res) with dims (w, h, d) corresponding to the orientation.
                    # Wait, the orientation (bw, bh, bd) was chosen.
                    # We need to map bw_g back to the actual dimension used.
                    # Since we used ceil, the actual dimension might be smaller than bw_g*res.
                    # For placement validity, we just need to ensure it doesn't overlap.
                    # If we place it at x*res with width bw, and bw <= bw_g*res, it fits in the grid cell block.
                    
                    # Which orientation was it?
                    # We lost track of which (bw,bh,bd) led to best_pos in the loop above.
                    # We need to store it.
                    
                    # Let's refactor the inner loop to store the orientation.
                    # Actually, `best_pos` currently stores grid dims.
                    # We need to know the real dims to report.
                    
                    # Re-doing the logic to track real dims.
                    
                    # Let's restart the inner logic in a cleaner way.
                    pass # Placeholder, logic needs rework to track real dims

            # The above pass indicates I need to write the final code carefully.
            
        # Since the logic inside try_pack needs to be robust, I will write the final version directly.
        
    # Final Implementation of try_pack with Floor Map
    
    def try_pack_v2(order, seed):
        rng = random.Random(seed)
        floor_map = np.zeros((gw, gh), dtype=np.int32)
        placements = [None] * n
        placed_vol = 0.0
        
        for i in order:
            w, h, d = boxes[i]
            orients = get_orientations(w, h, d)
            rng.shuffle(orients)
            
            best_pos = None
            best_score_pos = float('inf')
            best_real_dims = None
            
            for bw, bh, bd in orients:
                bw_g = max(1, int(np.ceil(bw / res)))
                bh_g = max(1, int(np.ceil(bh / res)))
                bd_g = max(1, int(np.ceil(bd / res)))
                
                if bw_g > gw or bh_g > gh or bd_g > gd:
                    continue
                
                # Scan
                # Optimization: If the box is large, the range of x,y is small.
                # If the box is small, it's large.
                
                # To speed up, we can sample? No, we want best fit.
                # Let's just scan.
                
                for x in range(gw - bw_g + 1):
                    # Skip if we can't beat current best score significantly?
                    # Score is dominated by z. If min_z at this x is already > best_z, skip?
                    # We don't know min_z without checking y.
                    
                    for y in range(gh - bh_g + 1):
                        z_min = np.max(floor_map[x:x+bw_g, y:y+bh_g])
                        
                        if z_min + bd_g <= gd:
                            # Calculate score
                            # We want to minimize z, then y, then x.
                            # Weighting: z is most important for stability/packing efficiency?
                            # Actually, for volume fill, "Best Fit" (tightest) is good.
                            # Usually "Bottom-Left-Farthest" (BLF) or similar.
                            # Let's use: Score = z_min * 10000 + y * 100 + x
                            score = z_min * 10000 + y * 100 + x
                            
                            if score < best_score_pos:
                                best_score_pos = score
                                best_pos = (x, y, z_min)
                                best_real_dims = (bw, bh, bd)
            
            if best_pos:
                x, y, z = best_pos
                bw, bh, bd = best_real_dims
                
                # Update floor map
                # We need to update the grid cells covered by this box.
                # Grid x range: x to x + bw_g
                # Grid y range: y to y + bh_g
                floor_map[x:x+bw_g, y:y+bh_g] = np.maximum(floor_map[x:x+bw_g, y:y+bh_g], z + bd_g)
                
                # Record placement
                # Coordinates:
                # x_real = x * res
                # y_real = y * res
                # z_real = z * res
                # w_real = bw
                # h_real = bh
                # d_real = bd
                
                # Note: The box might not perfectly align with grid cells if w is not multiple of res.
                # But since we reserved bw_g cells, and bw <= bw_g * res, it fits within the reserved space.
                # We place it at the bottom-left corner of the reserved space.
                
                placements[i] = (0, x * res, y * res, z * res, bw, bh, bd)
                placed_vol += bw * bh * bd
            else:
                placements[i] = (-1, 0, 0, 0, w, h, d)
                
        # Calculate score
        score = placed_vol / container_vol
        return score, placements

    # Main loop: Randomized Restarts
    # We will run multiple restarts with different random orders.
    
    # Initial order: Largest Volume First (Largest Side First)
    # Sort indices by volume descending
    base_order = sorted(range(n), key=lambda i: box_vols[i], reverse=True)
    
    # Run first with base order
    score, pl = try_pack_v2(base_order, 42)
    if score > best_score:
        best_score = score
        best_placements = pl
    
    # Time check
    if time.time() - start_time > time_limit:
        return best_placements

    # Now, randomized restarts
    # We will shuffle the order slightly or use different heuristics.
    # Strategy:
    # 1. Largest Volume First (Done)
    # 2. Largest Side First (Max dimension)
    # 3. Random permutations
    
    max_side_order = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    score, pl = try_pack_v2(max_side_order, 43)
    if score > best_score:
        best_score = score
        best_placements = pl
        
    if time.time() - start_time > time_limit:
        return best_placements

    # Random restarts
    # How many can we afford?
    # Each try_pack_v2 takes T seconds.
    # Let's estimate T from the first run.
    t0 = time.time()
    score, pl = try_pack_v2(base_order, 44) # Re-run to estimate? No, just assume.
    # Let's just run as many as possible.
    
    restarts = 0
    while time.time() - start_time < time_limit - 5.0: # 5s safety margin
        # Generate a random order
        # Mix of random and sorted?
        # Pure random is often bad.
        # Let's try: Sort by volume, then swap random pairs (Perturbation)
        order = base_order.copy()
        # Perturb
        for _ in range(n // 2):
            i, j = rng.randint(0, n-1), rng.randint(0, n-1)
            order[i], order[j] = order[j], order[i]
            
        score, pl = try_pack_v2(order, restarts)
        if score > best_score:
            best_score = score
            best_placements = pl
            
        restarts += 1
        
    return best_placements

# EVOLVE-BLOCK-START
# The function above is the implementation.
# We need to ensure the function name is `load` and it matches the signature.
# The code above defines `load`.
# EVOLVE-BLOCK-END
