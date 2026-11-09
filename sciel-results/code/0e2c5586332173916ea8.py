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

def _fits(box_dims, pos, container):
    W, H, D = container
    x, y, z = pos
    w, h, d = box_dims
    return x + w <= W and y + h <= H and z + d <= D

def _no_overlap(box_dims, pos, placed):
    x, y, z = pos
    w, h, d = box_dims
    for (px, py, pz, pw, ph, pd) in placed:
        if (x < px + pw and x + w > px and
            y < py + ph and y + h > py and
            z < pz + pd and z + d > pz):
            return False
    return True

def _extreme_points(placed, container):
    W, H, D = container
    candidates = set()
    candidates.add((0, 0, 0))
    for (px, py, pz, pw, ph, pd) in placed:
        points = [
            (px + pw, py, pz),
            (px, py + ph, pz),
            (px, py, pz + pd),
            (px + pw, py + ph, pz),
            (px, py + ph, pz + pd),
            (px + pw, py, pz + pd),
            (px + pw, py + ph, pz + pd),
        ]
        for (cx, cy, cz) in points:
            if cx <= W and cy <= H and cz <= D:
                candidates.add((cx, cy, cz))
    return list(candidates)

def _grid_check(grid, x, y, z, w, h, d):
    """Check if a box fits in the grid at position (x,y,z) with dims (w,h,d)."""
    if x < 0 or y < 0 or z < 0:
        return False
    if x + w >= grid.shape[0] or y + h >= grid.shape[1] or z + d >= grid.shape[2]:
        return False
    return not np.any(grid[x:x+w, y:y+h, z:z+d] == 1)

def _grid_place(grid, x, y, z, w, h, d, val=1):
    grid[x:x+w, y:y+h, z:z+d] = val

def _construct_solution(boxes, container, order_idx, scoring_func, deadline, rng=None):
    """
    Construct a solution by placing boxes in the given order using extreme points.
    Returns: (placements_list, volume_filled)
    placements_list: list of tuples (bin, x, y, z, w, h, d)
    """
    W, H, D = container
    placed = []
    placements = [None] * len(boxes)
    
    for orig_idx in order_idx:
        if time.time() > deadline:
            break
        w, h, d = boxes[orig_idx]
        rots = _rotations(w, h, d)
        candidates = _extreme_points(placed, container)
        
        best_pos = None
        best_dim = None
        best_score = None
        
        for (cx, cy, cz) in candidates:
            for (rw, rh, rd) in rots:
                if not _fits((rw, rh, rd), (cx, cy, cz), container):
                    continue
                if not _no_overlap((rw, rh, rd), (cx, cy, cz), placed):
                    continue
                
                score = scoring_func((cx, cy, cz), (rw, rh, rd))
                if best_score is None or score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
                    best_dim = (rw, rh, rd)
        
        if best_pos is not None:
            placed.append((best_pos[0], best_pos[1], best_pos[2],
                          best_dim[0], best_dim[1], best_dim[2]))
            placements[orig_idx] = (0, best_pos[0], best_pos[1], best_pos[2],
                                   best_dim[0], best_dim[1], best_dim[2])
        else:
            placements[orig_idx] = (-1, 0, 0, 0, w, h, d)
    
    for i in range(len(boxes)):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
    
    volume_filled = sum(p[4] * p[5] * p[6] for p in placements if p[0] == 0)
    return placements, volume_filled

def _local_search(placements, boxes, container, deadline, rng):
    """
    Iterated Local Search:
    1. Try to move every placed box to a better position (lower z, then y, then x).
    2. Try to place every unplaced box.
    3. Repeat until no improvement or time limit.
    """
    W, H, D = container
    n = len(boxes)
    box_rots = [_rotations(b[0], b[1], b[2]) for b in boxes]
    
    # Build grid
    grid = np.zeros((W, H, D), dtype=np.int8)
    for p in placements:
        if p[0] == 0:
            _, x, y, z, w, h, d = p
            _grid_place(grid, x, y, z, w, h, d, 1)
            
    current_vol = sum(p[4] * p[5] * p[6] for p in placements if p[0] == 0)
    best_placements = placements[:]
    best_vol = current_vol
    
    placed_indices = [i for i, p in enumerate(placements) if p[0] == 0]
    unplaced_indices = [i for i, p in enumerate(placements) if p[0] == -1]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Pass 1: Optimize positions of placed boxes
        # We iterate in reverse order of z (top to bottom) to allow shifting down? 
        # Actually, shifting down might block others. 
        # Standard ILS: pick a random box, try all positions, pick best.
        # For speed, we do a deterministic "best fit" sweep for all placed boxes.
        
        for i in placed_indices[:]: # Copy list as it might change? No, placed set is stable in this pass
            if time.time() > deadline: break
            if placements[i][0] != 0: continue
            
            old_x, old_y, old_z = placements[i][1], placements[i][2], placements[i][3]
            old_w, old_h, old_d = placements[i][4], placements[i][5], placements[i][6]
            
            # Remove from grid
            _grid_place(grid, old_x, old_y, old_z, old_w, old_h, old_d, 0)
            
            # Find best position
            best_pos = None
            best_dim = None
            best_score = None
            
            # Candidate points: Extreme points of remaining boxes + old position?
            # To be fast, we check a subset of extreme points.
            # Re-calculating extreme points for every box is O(N^2).
            # Instead, we can just check the "current" extreme points of the state 
            # minus the box being moved? That's complex.
            # Simpler heuristic: Check the old position and a few nearby/valid spots.
            # Or, use the global extreme points of the *current* grid state (which includes the hole).
            
            # Let's use the extreme points of the *other* placed boxes.
            other_placed = [placements[j] for j in range(n) if j != i and placements[j][0] == 0]
            # This is expensive. 
            # Alternative: Just try to shift the box down/left/back as much as possible?
            # "Gravity" heuristic.
            
            # Let's try a "Best Fit" among a limited set of candidates:
            # 1. The old position (to keep it if no better)
            # 2. (0,0,0)
            # 3. Extreme points of the *original* set of placed boxes (approximation)
            
            # Actually, a very effective and fast move is:
            # Try to move the box to the "lowest" valid position using a simple scan
            # or just re-run the EAP selection for this single box against the current grid.
            
            # Let's generate candidates from the current grid's empty space boundaries.
            # For speed, we'll just try the old position and a few random valid positions?
            # No, let's do a proper "Best Fit" using the extreme points of the *current* configuration 
            # excluding the box i.
            
            # Optimization: Pre-compute extreme points for the *whole* set once per iteration?
            # No, the set changes.
            
            # Let's use a simpler, robust local move:
            # Try to slide the box to the "front" (min x), then "left" (min y), then "bottom" (min z).
            
            # 1. Slide in X
            min_x = old_x
            while min_x > 0 and _grid_check(grid, min_x - 1, old_y, old_z, old_w, old_h, old_d):
                min_x -= 1
            
            # 2. Slide in Y
            min_y = old_y
            while min_y > 0 and _grid_check(grid, min_x, min_y - 1, old_z, old_w, old_h, old_d):
                min_y -= 1
            
            # 3. Slide in Z
            min_z = old_z
            while min_z > 0 and _grid_check(grid, min_x, min_y, min_z - 1, old_w, old_h, old_d):
                min_z -= 1
            
            # Check if we found a better spot (lower z, then y, then x)
            # Score: (z, y, x)
            old_score = (old_z, old_y, old_x)
            new_score = (min_z, min_y, min_x)
            
            if new_score < old_score:
                # Place at new spot
                _grid_place(grid, min_x, min_y, min_z, old_w, old_h, old_d, 1)
                placements[i] = (0, min_x, min_y, min_z, old_w, old_h, old_d)
                improved = True
            else:
                # Restore at old spot
                _grid_place(grid, old_x, old_y, old_z, old_w, old_h, old_d, 1)
            
            # Also try rotating?
            # Try all rotations at the "best" position found (min_x, min_y, min_z)
            # And also at (0,0,0) and old position?
            # Let's try rotations at the "gravity" position (min_x, min_y, min_z)
            # And at the old position.
            
            best_rot_pos = None
            best_rot_dim = None
            best_rot_score = None
            
            # Candidate positions for rotation check:
            # 1. The gravity position (min_x, min_y, min_z)
            # 2. The old position (old_x, old_y, old_z)
            # 3. (0,0,0)
            
            check_positions = [(min_x, min_y, min_z), (old_x, old_y, old_z), (0,0,0)]
            # Add a few random valid positions?
            # To keep it fast, we stick to these 3.
            
            for cx, cy, cz in check_positions:
                for rw, rh, rd in box_rots[i]:
                    if _grid_check(grid, cx, cy, cz, rw, rh, rd):
                        score = (cz, cy, cx)
                        if best_rot_score is None or score < best_rot_score:
                            best_rot_score = score
                            best_rot_pos = (cx, cy, cz)
                            best_rot_dim = (rw, rh, rd)
            
            if best_rot_pos:
                # Compare with current placement
                cur_x, cur_y, cur_z = placements[i][1], placements[i][2], placements[i][3]
                cur_w, cur_h, cur_d = placements[i][4], placements[i][5], placements[i][6]
                cur_score = (cur_z, cur_y, cur_x)
                
                if best_rot_score < cur_score:
                    # Remove current
                    _grid_place(grid, cur_x, cur_y, cur_z, cur_w, cur_h, cur_d, 0)
                    # Place new
                    _grid_place(grid, best_rot_pos[0], best_rot_pos[1], best_rot_pos[2], 
                                best_rot_dim[0], best_rot_dim[1], best_rot_dim[2], 1)
                    placements[i] = (0, best_rot_pos[0], best_rot_pos[1], best_rot_pos[2],
                                    best_rot_dim[0], best_rot_dim[1], best_rot_dim[2])
                    improved = True
                    # Note: We don't update current_vol here because volume is invariant for a placed box
                    # unless it was unplaced. But we are iterating over placed_indices.
        
        # Pass 2: Try to place unplaced boxes
        for i in unplaced_indices[:]:
            if time.time() > deadline: break
            if placements[i][0] == 0: continue # It might have been placed by a previous move? No, we only move placed boxes.
            
            # Try to place this box
            w, h, d = boxes[i]
            best_pos = None
            best_dim = None
            best_score = None
            
            # Candidates: Extreme points of currently placed boxes
            placed_boxes_now = [p for p in placements if p[0] == 0]
            candidates = _extreme_points(placed_boxes_now, container)
            
            # Also add (0,0,0)
            candidates.append((0,0,0))
            
            for cx, cy, cz in candidates:
                for rw, rh, rd in box_rots[i]:
                    if _grid_check(grid, cx, cy, cz, rw, rh, rd):
                        score = (cz, cy, cx)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_pos = (cx, cy, cz)
                            best_dim = (rw, rh, rd)
            
            if best_pos:
                _grid_place(grid, best_pos[0], best_pos[1], best_pos[2], 
                            best_dim[0], best_dim[1], best_dim[2], 1)
                placements[i] = (0, best_pos[0], best_pos[1], best_pos[2],
                                best_dim[0], best_dim[1], best_dim[2])
                current_vol += best_dim[0] * best_dim[1] * best_dim[2]
                improved = True
                # Update indices
                unplaced_indices.remove(i)
                placed_indices.append(i)
        
        # Update best
        if current_vol > best_vol:
            best_vol = current_vol
            best_placements = placements[:]
    
    return best_placements, best_vol

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 140 # Safety margin
    
    W, H, D = container
    n = len(boxes)
    volumes = [b[0]*b[1]*b[2] for b in boxes]
    
    best_overall_placements = None
    best_overall_vol = -1
    
    # Define different heuristics for construction
    # 1. Volume Descending
    order_vol = sorted(range(n), key=lambda i: volumes[i], reverse=True)
    # 2. Volume Ascending
    order_vol_asc = sorted(range(n), key=lambda i: volumes[i])
    # 3. Aspect Ratio (Max dim) Descending
    order_aspect = sorted(range(n), key=lambda i: max(boxes[i]), reverse=True)
    # 4. Random
    rng = random.Random(42)
    order_rand = list(range(n))
    rng.shuffle(order_rand)
    
    # Scoring functions
    # Standard: Bottom, Left, Far
    def score_blf(pos, dim):
        return (pos[2], pos[1], pos[0])
    
    # Compactness: Minimize max dimension?
    def score_compact(pos, dim):
        # Prefer positions that are closer to the corner?
        return (pos[2], pos[1], pos[0]) # Same for now, but could be tuned
    
    # List of (order, scoring_func, name)
    strategies = [
        (order_vol, score_blf, "Vol_Desc"),
        (order_vol_asc, score_blf, "Vol_Asc"),
        (order_aspect, score_blf, "Aspect_Desc"),
        (order_rand, score_blf, "Random"),
    ]
    
    # Allocate time per strategy
    # We have ~140s. 
    # Construction is fast. Local Search is the bottleneck.
    # Let's do 4 strategies.
    # Each gets ~30s.
    
    for idx, (order, sc_func, name) in enumerate(strategies):
        if time.time() > deadline:
            break
            
        # Reserve time for local search
        ls_deadline = deadline - 5 # Leave 5s for final checks
        
        # 1. Construct
        placements, vol = _construct_solution(boxes, container, order, sc_func, ls_deadline, rng)
        
        # 2. Local Search
        improved_placements, improved_vol = _local_search(placements, boxes, container, ls_deadline, rng)
        
        if improved_vol > best_overall_vol:
            best_overall_vol = improved_vol
            best_overall_placements = improved_placements[:]
            
        # Optional: If we have time, do a second pass of local search on the best found so far?
        # No, _local_search already iterates until convergence or time limit.
        
    # If no placements found (shouldn't happen), return empty
    if best_overall_placements is None:
        return [(-1, 0, 0, 0, b[0], b[1], b[2]) for b in boxes]
        
    return best_overall_placements
