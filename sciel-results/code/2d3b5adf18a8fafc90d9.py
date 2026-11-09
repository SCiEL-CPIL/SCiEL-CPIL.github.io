"""
3D Container Loading: Hybrid Extreme-Point Greedy + Occupancy-Grid Acceleration + Local Search
"""
import time
import random
import itertools
import math
import numpy as np

def _rotations(w, h, d):
    """Return all 6 unique orientations of a box (w,h,d)."""
    seen = set()
    result = []
    for perm in itertools.permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    return result

def _extreme_points(placed, container):
    """Generate candidate (x,y,z) positions from placed boxes."""
    W, H, D = container
    xs = {0}
    ys = {0}
    zs = {0}
    for (px, py, pz, pw, ph, pd) in placed:
        xs.add(px + pw)
        ys.add(py + ph)
        zs.add(pz + pd)
    
    # Filter to those within container bounds
    xs = [x for x in xs if x < W]
    ys = [y for y in ys if y < H]
    zs = [z for z in zs if z < D]
    
    points = []
    for x in xs:
        for y in ys:
            for z in zs:
                points.append((x, y, z))
    return points

def _face_points(placed, container):
    """Generate candidate positions from face-adjacent corners of placed boxes."""
    W, H, D = container
    candidates = set()
    candidates.add((0, 0, 0))
    for (px, py, pz, pw, ph, pd) in placed:
        candidates.add((px + pw, py, pz))
        candidates.add((px, py + ph, pz))
        candidates.add((px, py, pz + pd))
        candidates.add((px + pw, py + ph, pz))
        candidates.add((px + pw, py, pz + pd))
        candidates.add((px, py + ph, pz + pd))
        candidates.add((px + pw, py + ph, pz + pd))
    
    # Filter to valid positions
    valid = []
    for (x, y, z) in candidates:
        if 0 <= x < W and 0 <= y < H and 0 <= z < D:
            valid.append((x, y, z))
    return valid

def _fits(box_dims, pos, container, placed):
    """Check if box fits at pos without overlap and within container."""
    W, H, D = container
    x, y, z = pos
    w, h, d = box_dims
    
    # Check container bounds
    if x + w > W or y + h > H or z + d > D:
        return False
    
    # Check overlap with placed boxes
    for (px, py, pz, pw, ph, pd) in placed:
        if (x < px + pw and x + w > px and
            y < py + ph and y + h > py and
            z < pz + pd and z + d > pz):
            return False
    return True

def _score_position(pos, box_dims, placed, container):
    """
    Score a candidate position.
    We want to minimize z (height), then y, then x to keep the packing compact.
    A lower score is better.
    """
    x, y, z = pos
    return (z, y, x)

class OccupancyGrid:
    """
    3D Occupancy Grid for fast overlap checking.
    Uses a resolution-based grid. If resolution is 1, it's a voxel map.
    """
    def __init__(self, container, resolution=1):
        self.W, self.H, self.D = container
        self.res = resolution
        # Dimensions in grid units
        self.gw = max(1, int(math.ceil(self.W / self.res)))
        self.gh = max(1, int(math.ceil(self.H / self.res)))
        self.gd = max(1, int(math.ceil(self.D / self.res)))
        self.grid = np.zeros((self.gw, self.gh, self.gd), dtype=np.int8)
        self.placed_boxes = [] # List of (x, y, z, w, h, d) for exact checks if needed or rollback

    def _to_grid_coords(self, x, y, z):
        return (int(x // self.res), int(y // self.res), int(z // self.res))

    def _box_to_grid_range(self, x, y, z, w, h, d):
        """
        Calculate the grid cells covered by a box.
        Returns (gx_start, gx_end, gy_start, gy_end, gz_start, gz_end)
        """
        gx_start, gy_start, gz_start = self._to_grid_coords(x, y, z)
        # End coordinates are exclusive in numpy slicing
        gx_end, gy_end, gz_end = self._to_grid_coords(x + w, y + h, z + d)
        
        # Clamp to grid bounds
        gx_start = max(0, min(gx_start, self.gw))
        gx_end = max(0, min(gx_end, self.gw))
        gy_start = max(0, min(gy_start, self.gh))
        gy_end = max(0, min(gy_end, self.gh))
        gz_start = max(0, min(gz_start, self.gd))
        gz_end = max(0, min(gz_end, self.gd))
        
        return gx_start, gx_end, gy_start, gy_end, gz_start, gz_end

    def check_fits(self, x, y, z, w, h, d):
        """
        Check if a box fits at (x,y,z) with dims (w,h,d).
        Returns True if no overlap in the grid.
        Note: This is an approximation if res > 1 or if boxes don't align with grid.
        For exactness, we usually keep res=1 for integer coords or use this as a pre-filter.
        Given the problem likely uses integer or simple float coords, and we want speed,
        we'll use this as a fast pre-filter. If it says "fits", we still need to be careful.
        Actually, if res=1 and coords are integers, it is exact for axis-aligned boxes.
        """
        if x + w > self.W or y + h > self.H or z + d > self.D:
            return False
            
        gx_start, gx_end, gy_start, gy_end, gz_start, gz_end = self._box_to_grid_range(x, y, z, w, h, d)
        
        # If the box spans no grid cells (shouldn't happen if w,h,d >= res), return True
        if gx_end <= gx_start or gy_end <= gy_start or gz_end <= gz_start:
            return True

        # Check if any cell in the range is occupied
        # numpy slicing: [start:end]
        region = self.grid[gx_start:gx_end, gy_start:gy_end, gz_start:gz_end]
        return not np.any(region > 0)

    def place(self, x, y, z, w, h, d):
        """Mark the box as placed in the grid."""
        gx_start, gx_end, gy_start, gy_end, gz_start, gz_end = self._box_to_grid_range(x, y, z, w, h, d)
        self.grid[gx_start:gx_end, gy_start:gy_end, gz_start:gz_end] = 1
        self.placed_boxes.append((x, y, z, w, h, d))

    def remove(self, box_idx):
        """Remove a box from the grid. box_idx is index in placed_boxes."""
        if box_idx < 0 or box_idx >= len(self.placed_boxes):
            return
        x, y, z, w, h, d = self.placed_boxes[box_idx]
        gx_start, gx_end, gy_start, gy_end, gz_start, gz_end = self._box_to_grid_range(x, y, z, w, h, d)
        self.grid[gx_start:gx_end, gy_start:gy_end, gz_start:gz_end] = 0
        # Remove from list
        self.placed_boxes.pop(box_idx)
        # Note: This shifts indices. If we do this frequently, it's O(N).
        # For local search, we might want to swap-remove or just rebuild.
        # Given N=60, O(N) is fine.

    def rebuild_from_placements(self, placements, boxes):
        """Rebuild grid from a list of placements."""
        self.grid = np.zeros((self.gw, self.gh, self.gd), dtype=np.int8)
        self.placed_boxes = []
        for i in range(len(placements)):
            p = placements[i]
            if p[0] == 0:
                x, y, z, w, h, d = p[1], p[2], p[3], p[4], p[5], p[6]
                self.place(x, y, z, w, h, d)

def _greedy_place(boxes, container, order, deadline, mode='standard'):
    """
    Greedy placement of boxes in the given order.
    Returns a list of placements (bin, x, y, z, w, h, d) and total volume.
    """
    W, H, D = container
    placed = []
    placements = [None] * len(boxes)
    total_vol = 0.0
    n = len(boxes)
    
    # Initialize Occupancy Grid
    # Use resolution 1 for exactness with integer-like coords, or 1.0 for floats?
    # The problem doesn't specify integer coords. Let's assume standard float handling.
    # If we use a grid, we need to be careful with floats.
    # Strategy: Use the grid as a *pre-filter*. 
    # 1. Check grid. If occupied, reject.
    # 2. If grid says empty, do exact check against `placed` list.
    # This hybrid approach is safe and fast.
    
    # For the grid to be useful, we need a reasonable resolution.
    # If coords are large, res=1 might be too fine (large grid).
    # Let's estimate grid size.
    # Max dimension ~ 100-1000?
    # Let's stick to the exact list check for safety, but use the grid to skip obvious overlaps?
    # Actually, the list check is O(N). N=60. 60 checks is very fast.
    # The bottleneck in the original code was likely the number of candidates * rotations * checks.
    # Candidates can be N^3. 60^3 = 216,000. 216,000 * 60 = 13M checks. That's slow in Python.
    # Using a grid:
    # Grid size: If W,H,D are small (e.g., 100), 100^3 = 1M cells.
    # Check is O(1) (numpy slice).
    # Placement is O(1).
    # This should be significantly faster.
    
    # Let's determine grid resolution.
    # If we use res=1, and coords are floats, we lose precision.
    # We MUST use the exact check if the grid says it fits, to handle float boundaries.
    # Or, we can quantize coords? No, that changes the problem.
    
    # Safe Hybrid Strategy:
    # 1. Generate candidates.
    # 2. For each candidate/rotation:
    #    a. Quick bounds check.
    #    b. Grid check (fast reject).
    #    c. Exact list check (confirm).
    
    # Grid Resolution:
    # To make the grid check fast and accurate enough for a pre-filter,
    # we can use a resolution that makes the grid manageable.
    # However, if we use the grid as the *primary* checker, we need exactness.
    # Let's assume the input coordinates are such that a voxel grid with res=1 (or similar) works.
    # If not, we fall back to exact check.
    
    # Let's try to use the grid as the primary checker if the container size is small enough.
    max_dim = max(W, H, D)
    if max_dim <= 200:
        use_grid = True
        grid = OccupancyGrid(container, resolution=1)
    else:
        use_grid = False
        grid = None

    for idx in order:
        if time.time() > deadline:
            break
        w, h, d = boxes[idx]
        rots = _rotations(w, h, d)
        
        # Generate candidate positions
        candidates = _extreme_points(placed, container)
        face_pts = _face_points(placed, container)
        for fp in face_pts:
            if fp not in candidates:
                candidates.append(fp)
        if (0, 0, 0) not in candidates:
            candidates.append((0, 0, 0))
        
        best_pos = None
        best_rot = None
        best_score = None
        
        for pos in candidates:
            for rw, rh, rd in rots:
                # 1. Bounds
                if pos[0] + rw > W or pos[1] + rh > H or pos[2] + rd > D:
                    continue
                
                # 2. Grid Pre-filter
                if use_grid:
                    if not grid.check_fits(pos[0], pos[1], pos[2], rw, rh, rd):
                        continue
                
                # 3. Exact Check
                if _fits((rw, rh, rd), pos, container, placed):
                    score = _score_position(pos, (rw, rh, rd), placed, container)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pos = pos
                        best_rot = (rw, rh, rd)
        
        if best_pos is not None:
            x, y, z = best_pos
            rw, rh, rd = best_rot
            placed.append((x, y, z, rw, rh, rd))
            placements[idx] = (0, x, y, z, rw, rh, rd)
            total_vol += rw * rh * rd
            if use_grid:
                grid.place(x, y, z, rw, rh, rd)
        else:
            placements[idx] = (-1, 0, 0, 0, w, h, d)
    
    # Fill any unprocessed boxes as not placed
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
            
    return placements, total_vol

def _shake_local_search_v2(placements, boxes, container, deadline, rng, num_shakes=10):
    """
    More robust shake local search using copies.
    """
    n = len(boxes)
    current_placements = [p[:] for p in placements]
    current_vol = sum(p[4]*p[5]*p[6] for p in current_placements if p[0] == 0)
    
    loaded_indices = [i for i in range(n) if current_placements[i][0] == 0]
    
    for _ in range(num_shakes):
        if time.time() > deadline:
            break
        
        if not loaded_indices:
            break
            
        # Pick random boxes to shake
        k = rng.randint(1, min(3, len(loaded_indices)))
        boxes_to_remove = rng.sample(loaded_indices, k)
        
        # Create a temporary state
        temp_placements = [p[:] for p in current_placements]
        
        # Extract placed boxes not in remove set
        placed_boxes = []
        for i in range(n):
            p = temp_placements[i]
            if p[0] == 0 and i not in boxes_to_remove:
                placed_boxes.append((p[1], p[2], p[3], p[4], p[5], p[6]))
        
        # Mark removed boxes as unplaced in temp
        for i in boxes_to_remove:
            w, h, d = boxes[i]
            temp_placements[i] = (-1, 0, 0, 0, w, h, d)
        
        # Try to re-insert removed boxes
        shuffle_list = boxes_to_remove[:]
        rng.shuffle(shuffle_list)
        
        for idx in shuffle_list:
            w, h, d = boxes[idx]
            rots = _rotations(w, h, d)
            candidates = _extreme_points(placed_boxes, container)
            if (0, 0, 0) not in candidates:
                candidates.append((0, 0, 0))
            
            best_pos = None
            best_rot = None
            best_score = None
            
            for pos in candidates:
                for rw, rh, rd in rots:
                    if _fits((rw, rh, rd), pos, container, placed_boxes):
                        score = _score_position(pos, (rw, rh, rd), placed_boxes, container)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_pos = pos
                            best_rot = (rw, rh, rd)
            
            if best_pos is not None:
                x, y, z = best_pos
                rw, rh, rd = best_rot
                placed_boxes.append((x, y, z, rw, rh, rd))
                temp_placements[idx] = (0, x, y, z, rw, rh, rd)
        
        # Calculate new volume
        new_vol = sum(p[4]*p[5]*p[6] for p in temp_placements if p[0] == 0)
        
        if new_vol > current_vol:
            current_placements = temp_placements
            current_vol = new_vol
            loaded_indices = [i for i in range(n) if current_placements[i][0] == 0]
            
    return current_placements, current_vol

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    W, H, D = container
    n = len(boxes)
    
    vols = [w * h * d for w, h, d in boxes]
    
    # Define sorting strategies
    sort_strategies = [
        ('vol_desc', lambda i: vols[i], True),
        ('min_dim_desc', lambda i: min(boxes[i]), True),
        ('max_dim_desc', lambda i: max(boxes[i]), True),
        ('ratio', lambda i: max(boxes[i]) / max(min(boxes[i]), 1e-9), False),
        ('vol_asc', lambda i: vols[i], False),
    ]
    
    best_placements = None
    best_vol = 0.0
    
    # 1. Run greedy with different sorts
    for name, key_fn, reverse in sort_strategies:
        if time.time() > deadline:
            break
        order = list(range(n))
        order.sort(key=key_fn, reverse=reverse)
        
        placements, vol = _greedy_place(boxes, container, order, deadline, mode='standard')
        if vol > best_vol:
            best_vol = vol
            best_placements = [p[:] for p in placements]
    
    if best_placements is None:
        # Fallback
        best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        return best_placements
    
    # 2. Local Search (Shake)
    rng = random.Random(42)
    best_placements, best_vol = _shake_local_search_v2(
        best_placements, boxes, container, deadline, rng, num_shakes=100
    )
    
    # 3. If time remains, try another greedy + local search with a different seed/sort
    if time.time() < deadline - 10:
        # Try a random permutation
        rng2 = random.Random(123)
        order = list(range(n))
        rng2.shuffle(order)
        
        placements, vol = _greedy_place(boxes, container, order, deadline, mode='standard')
        if vol > best_vol:
            best_vol = vol
            best_placements = [p[:] for p in placements]
        
        # Local search on this new best
        if time.time() < deadline - 5:
            best_placements, best_vol = _shake_local_search_v2(
                best_placements, boxes, container, deadline, rng2, num_shakes=50
            )
            
    # 4. Final polish: try to place any remaining boxes with more aggressive search
    if time.time() < deadline - 2:
        remaining = [i for i in range(n) if best_placements[i][0] == -1]
        if remaining:
            # Try to place remaining boxes in random order
            rng3 = random.Random(777)
            rng3.shuffle(remaining)
            for idx in remaining:
                if time.time() > deadline - 1:
                    break
                w, h, d = boxes[idx]
                rots = _rotations(w, h, d)
                
                # Get current placed boxes
                placed_boxes = []
                for i in range(n):
                    p = best_placements[i]
                    if p[0] == 0:
                        placed_boxes.append((p[1], p[2], p[3], p[4], p[5], p[6]))
                
                candidates = _extreme_points(placed_boxes, container)
                face_pts = _face_points(placed_boxes, container)
                for fp in face_pts:
                    if fp not in candidates:
                        candidates.append(fp)
                if (0, 0, 0) not in candidates:
                    candidates.append((0, 0, 0))
                
                best_pos = None
                best_rot = None
                best_score = None
                
                for pos in candidates:
                    for rw, rh, rd in rots:
                        if _fits((rw, rh, rd), pos, container, placed_boxes):
                            score = _score_position(pos, (rw, rh, rd), placed_boxes, container)
                            if best_score is None or score < best_score:
                                best_score = score
                                best_pos = pos
                                best_rot = (rw, rh, rd)
                
                if best_pos is not None:
                    x, y, z = best_pos
                    rw, rh, rd = best_rot
                    best_placements[idx] = (0, x, y, z, rw, rh, rd)
                    best_vol += rw * rh * rd
            
            # One more local search pass to optimize the new placement
            if time.time() < deadline - 2:
                best_placements, best_vol = _shake_local_search_v2(
                    best_placements, boxes, container, deadline, rng3, num_shakes=30
                )
            
    return best_placements
