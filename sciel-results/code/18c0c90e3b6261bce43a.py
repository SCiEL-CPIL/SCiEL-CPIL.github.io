import time
import random
import itertools
import numpy as np

def _rotations(w, h, d):
    """Return all 6 unique orientations of a box."""
    seen = set()
    result = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    W, H, D = container
    container_volume = W * H * D
    n = len(boxes)
    
    if container_volume == 0:
        return [(-1, 0, 0, 0, b[0], b[1], b[2]) for b in boxes]

    # Pre-compute box properties for sorting
    volumes = [b[0]*b[1]*b[2] for b in boxes]
    max_dims = [max(b) for b in boxes]
    min_dims = [min(b) for b in boxes]
    sum_dims = [sum(b) for b in boxes]
    
    # Grid Resolution:
    # Balance between precision and memory/speed.
    # Max 100 voxels per dimension to keep grid size manageable (< 1M voxels).
    # Also ensure min dimension is at least 1 voxel if possible.
    res_x = max(1, min(100, int(W) + 1))
    res_y = max(1, min(100, int(H) + 1))
    res_z = max(1, min(100, int(D) + 1))
    
    # Scaling factors: 1.0 / (res - 1) maps [0, W] to [0, res-1]
    # We use a simple mapping: voxel_index = int(coord / W * (res - 1))
    # To ensure a box of width W fits in res_x voxels, we scale slightly.
    # Let's use a grid of size (res_x, res_y, res_z) representing the container.
    # A box from x to x+w occupies voxels from int(x/W*(res_x-1)) to int((x+w)/W*(res_x-1))
    # To be safe and ensure no overlap is missed, we might need a finer grid or careful rounding.
    # However, for a heuristic, a slightly coarser grid that might allow "micro-overlaps" in the grid
    # but are valid in continuous space is risky.
    # Better: Scale such that 1 unit = 1 voxel if dimensions are small, else downsample.
    
    # Let's stick to a fixed resolution based on container size to ensure consistency.
    # If W is 100, res_x=100. 1 unit = 1 voxel.
    # If W is 1000, res_x=100. 10 units = 1 voxel.
    
    # To handle the mapping correctly:
    # We want to check if a box [x, x+w] overlaps with occupied voxels.
    # We will mark a voxel as occupied if the box covers its center? Or if it covers any part?
    # If we mark a voxel as occupied if the box covers ANY part of it, we are conservative (safe).
    # Voxel i covers [i*W/res_x, (i+1)*W/res_x].
    # Box covers [x, x+w].
    # Overlap if x < (i+1)*W/res_x and x+w > i*W/res_x.
    # This is equivalent to: i < (x+w)*res_x/W and i+1 > x*res_x/W.
    # i_max = int((x+w)*res_x/W) - 1
    # i_min = int(x*res_x/W)
    
    # Let's implement a fast check using NumPy slicing.
    
    def _try_place_grid(order, scoring_func, rng=None):
        """
        Place boxes using a 3D grid for occupancy.
        Returns (placements, volume_filled).
        """
        # Initialize grid
        # Using int8 to save memory, 1 means occupied
        grid = np.zeros((res_x, res_y, res_z), dtype=np.int8)
        
        placed_list = []
        placements = [None] * n
        total_vol = 0
        
        for orig_idx in order:
            if time.time() > deadline:
                break
                
            w, h, d = boxes[orig_idx]
            rots = _rotations(w, h, d)
            
            # Generate candidate positions
            # We can't just use extreme points easily with grid unless we track them.
            # Let's stick to a "lowest-left-farthest" search over a set of candidate coordinates.
            # To speed up, we can sample candidate positions from the grid voxels?
            # No, that's too many.
            
            # Strategy: Use the same Extreme Points logic but check against Grid.
            # We need to maintain the list of extreme points.
            # The grid check is: 
            # sx1 = int(x * res_x / W)
            # sx2 = int((x+w) * res_x / W)
            # If grid[sx1:sx2, sy1:sy2, sz1:sz2].any() > 0: Overlap
            
            # To make this fast, we need to minimize the number of candidates.
            # We will generate candidates from the "frontier" of the grid?
            # Or just use the previous Extreme Points list?
            # The EP list grows. Checking each EP against the grid is O(1) per EP (vectorized slice).
            
            # Let's maintain the EP list.
            if len(placed_list) == 0:
                candidates = [(0, 0, 0)]
            else:
                # Reuse candidates from previous iteration? 
                # The set of extreme points changes as we place boxes.
                # We should update the set.
                # For simplicity in this new strategy, let's just use a heuristic set of points:
                # 1. (0,0,0)
                # 2. Points derived from the last placed box?
                # 3. A coarse grid of points?
                
                # Let's try a hybrid: Keep the EP list but cap its size if it gets too large?
                # Or just generate a new set of candidate points based on the current "skyline"?
                # Skyline in 3D is complex.
                
                # Let's stick to the EP list from the previous step, but update it.
                # Actually, computing EPs is O(N_placed).
                # Checking each EP is O(1) with Grid.
                # So total per box is O(N_placed * 6_rots).
                # This is similar to the original, but the inner loop is vectorized NumPy.
                # The bottleneck in original was Python loop for overlap.
                # Here, the bottleneck is generating EPs and iterating them in Python.
                
                # To really speed up, we should avoid iterating all EPs in Python.
                # But let's try this first: Is the NumPy slice check fast enough to allow more restarts?
                
                pass # Placeholder for EP generation
            
            # Let's refine the candidate generation.
            # We will maintain `extreme_points` list.
            # Initially [(0,0,0)].
            # When placing a box at (x,y,z) with size (w,h,d):
            # Add (x+w, y, z), (x, y+h, z), (x, y, z+d) etc.
            
            # We need to store the current EPs outside the loop.
            # Since this is a single trial, we can manage it locally.
            
            # RE-IMPLEMENTATION of _try_place_grid with EPs
            
            # Reset for this trial
            grid = np.zeros((res_x, res_y, res_z), dtype=np.int8)
            placements = [None] * n
            placed_list = [] # For tracking EPs
            ep_set = set()
            ep_set.add((0, 0, 0))
            eps = [(0, 0, 0)]
            total_vol = 0
            
            for orig_idx in order:
                if time.time() > deadline:
                    break
                
                w, h, d = boxes[orig_idx]
                rots = _rotations(w, h, d)
                
                best_pos = None
                best_dim = None
                best_score = None
                best_candidates = []
                
                # Iterate over current extreme points
                # To keep it fast, we might limit the number of EPs considered if it grows too large?
                # Or just iterate all.
                for (cx, cy, cz) in eps:
                    for (rw, rh, rd) in rots:
                        # Check bounds
                        if cx + rw > W or cy + rh > H or cz + rd > D:
                            continue
                        
                        # Check Grid Overlap
                        # Calculate voxel indices
                        # Using integer arithmetic for speed
                        # sx1 = int(cx / W * (res_x - 1))
                        # sx2 = int((cx + rw) / W * (res_x - 1)) + 1
                        # Note: If W is large and rw is small, sx1 might equal sx2.
                        # We need to ensure we check the voxel containing the box.
                        
                        # Let's use a mapping:
                        # idx = int(coord * res / W)
                        # If idx >= res, idx = res - 1
                        # Slice: grid[idx1:idx2+1, ...]
                        
                        sx1 = int(cx * res_x / W)
                        if sx1 >= res_x: sx1 = res_x - 1
                        sx2 = int((cx + rw) * res_x / W)
                        if sx2 >= res_x: sx2 = res_x - 1
                        
                        sy1 = int(cy * res_y / H)
                        if sy1 >= res_y: sy1 = res_y - 1
                        sy2 = int((cy + rh) * res_y / H)
                        if sy2 >= res_y: sy2 = res_y - 1
                        
                        sz1 = int(cz * res_z / D)
                        if sz1 >= res_z: sz1 = res_z - 1
                        sz2 = int((cz + rd) * res_z / D)
                        if sz2 >= res_z: sz2 = res_z - 1
                        
                        # Ensure sx1 <= sx2
                        if sx1 > sx2: sx1, sx2 = sx2, sx1
                        if sy1 > sy2: sy1, sy2 = sy2, sy1
                        if sz1 > sz2: sz1, sz2 = sz2, sz1
                        
                        # Check occupancy
                        # If any voxel in the range is occupied, it's a conflict
                        # Note: This is a conservative check. If a box spans multiple voxels,
                        # and one is occupied, it fails.
                        # This might reject valid placements if the "occupied" voxel is only partially overlapped
                        # by a previous box that didn't fully fill it? 
                        # No, we mark voxels as occupied if a box covers them.
                        # If we mark a voxel as occupied, it means *some* part of it is filled.
                        # If a new box covers *any* part of that voxel, they overlap.
                        # So this check is correct for "no overlap" if we define overlap as "any intersection".
                        
                        if grid[sx1:sx2+1, sy1:sy2+1, sz1:sz2+1].any():
                            continue
                        
                        # Calculate Score
                        score = scoring_func((cx, cy, cz), (rw, rh, rd))
                        
                        if best_score is None or score < best_score:
                            best_score = score
                            best_pos = (cx, cy, cz)
                            best_dim = (rw, rh, rd)
                            best_candidates = [(best_pos, best_dim)]
                        elif score == best_score:
                            best_candidates.append(((cx, cy, cz), (rw, rh, rd)))
                
                if best_pos is not None:
                    if rng and len(best_candidates) > 1:
                        best_pos, best_dim = rng.choice(best_candidates)
                    
                    bx, by, bz = best_pos
                    bw, bh, bd = best_dim
                    
                    # Update Grid
                    sx1 = int(bx * res_x / W)
                    if sx1 >= res_x: sx1 = res_x - 1
                    sx2 = int((bx + bw) * res_x / W)
                    if sx2 >= res_x: sx2 = res_x - 1
                    if sx1 > sx2: sx1, sx2 = sx2, sx1
                    
                    sy1 = int(by * res_y / H)
                    if sy1 >= res_y: sy1 = res_y - 1
                    sy2 = int((by + bh) * res_y / H)
                    if sy2 >= res_y: sy2 = res_y - 1
                    if sy1 > sy2: sy1, sy2 = sy2, sy1
                    
                    sz1 = int(bz * res_z / D)
                    if sz1 >= res_z: sz1 = res_z - 1
                    sz2 = int((bz + bd) * res_z / D)
                    if sz2 >= res_z: sz2 = res_z - 1
                    if sz1 > sz2: sz1, sz2 = sz2, sz1
                    
                    grid[sx1:sx2+1, sy1:sy2+1, sz1:sz2+1] = 1
                    
                    # Update EPs
                    new_points = [
                        (bx + bw, by, bz),
                        (bx, by + bh, bz),
                        (bx, by, bz + bd),
                        (bx + bw, by + bh, bz),
                        (bx, by + bh, bz + bd),
                        (bx + bw, by, bz + bd),
                        (bx + bw, by + bh, bz + bd),
                    ]
                    
                    for p in new_points:
                        if p[0] <= W and p[1] <= H and p[2] <= D:
                            if p not in ep_set:
                                ep_set.add(p)
                                eps.append(p)
                    
                    placed_list.append((bx, by, bz, bw, bh, bd))
                    placements[orig_idx] = (0, bx, by, bz, bw, bh, bd)
                    total_vol += bw * bh * bd
                else:
                    placements[orig_idx] = (-1, 0, 0, 0, w, h, d)
            
            for i in range(n):
                if placements[i] is None:
                    w, h, d = boxes[i]
                    placements[i] = (-1, 0, 0, 0, w, h, d)
            
            return placements, total_vol

    # Define Scoring Functions
    def score_default(pos, dim):
        return (pos[2], pos[1], pos[0])
        
    def score_sum(pos, dim):
        return (pos[0] + pos[1] + pos[2], pos[2])
        
    def score_z(pos, dim):
        return (pos[2], pos[1], pos[0])
        
    # Strategies
    vol_order = sorted(range(n), key=lambda i: volumes[i], reverse=True)
    maxdim_order = sorted(range(n), key=lambda i: max_dims[i], reverse=True)
    mindim_order = sorted(range(n), key=lambda i: min_dims[i])
    
    strategies = []
    
    # Deterministic
    strategies.append((vol_order, score_default, None))
    strategies.append((maxdim_order, score_default, None))
    strategies.append((mindim_order, score_default, None))
    strategies.append((vol_order, score_sum, None))
    
    # Randomized Restarts
    # We can afford more restarts now if the grid check is fast.
    # Let's aim for ~50-100 restarts depending on speed.
    # We will dynamically determine how many we can run.
    
    rng_master = random.Random(42)
    
    # Add some random orders
    for i in range(20):
        order = list(range(n))
        rng_master.shuffle(order)
        strategies.append((order, score_default, random.Random(i)))
        
    # Add randomized ties with standard orders
    for i in range(20):
        strategies.append((vol_order, score_default, random.Random(100+i)))
        
    best_placements = None
    best_volume = -1
    
    # Run strategies
    for order, scorer, rng in strategies:
        if time.time() > deadline:
            break
            
        placements, vol = _try_place_grid(order, scorer, rng)
        
        if vol > best_volume:
            best_volume = vol
            best_placements = placements
            if best_volume >= container_volume:
                break
    
    if best_placements is None:
        # Fallback: Simple shelf packing
        placements = []
        x, y, z, row_h, layer_d = 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d, x, row_h = 0, z + layer_d, 0, 0, 0
            if z + d > D:
                placements.append((-1, 0, 0, 0, w, h, d))
                continue
            placements.append((0, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        best_placements = placements
        
    return best_placements
