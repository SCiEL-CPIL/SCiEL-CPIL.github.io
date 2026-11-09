import time
import itertools

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

def _find_best_position(box_dims, placed, container):
    """
    Find the best position for a box using Extreme Points Approach.
    Returns (x, y, z, w, h, d) or None.
    """
    W, H, D = container
    candidates = _extreme_points(placed, container)
    
    best_pos = None
    best_dim = None
    best_score = None
    
    for (cx, cy, cz) in candidates:
        for (rw, rh, rd) in _rotations(*box_dims):
            if not _fits((rw, rh, rd), (cx, cy, cz), container):
                continue
            if not _no_overlap((rw, rh, rd), (cx, cy, cz), placed):
                continue
            # Scoring: minimize z, then y, then x (pack from bottom-left-front)
            score = (cz, cy, cx)
            if best_score is None or score < best_score:
                best_score = score
                best_pos = (cx, cy, cz)
                best_dim = (rw, rh, rd)
    
    if best_pos is not None:
        return (best_pos[0], best_pos[1], best_pos[2], best_dim[0], best_dim[1], best_dim[2])
    return None

def _constructive_solve(boxes, container, order_idx, deadline):
    """
    Greedy constructive phase.
    Returns:
    - placements: list of (bin, x, y, z, w, h, d)
    - placed_indices: list of indices of boxes that were placed
    """
    W, H, D = container
    n = len(boxes)
    placed_list = [] # List of (x, y, z, w, h, d) for placed boxes
    placements = [None] * n
    placed_indices = []
    
    for idx in order_idx:
        if time.time() > deadline:
            break
        
        w, h, d = boxes[idx]
        res = _find_best_position((w, h, d), placed_list, container)
        
        if res is not None:
            x, y, z, rw, rh, rd = res
            placed_list.append((x, y, z, rw, rh, rd))
            placements[idx] = (0, x, y, z, rw, rh, rd)
            placed_indices.append(idx)
        else:
            placements[idx] = (-1, 0, 0, 0, w, h, d)
            
    # Fill remaining None slots (if deadline hit mid-loop)
    for i in range(n):
        if placements[i] is None:
            w, h, d = boxes[i]
            placements[i] = (-1, 0, 0, 0, w, h, d)
            
    return placements, placed_indices, placed_list

def _local_search(boxes, container, placements, placed_indices, placed_list, deadline):
    """
    Local search phase: Try to swap placed boxes with unplaced ones.
    Returns improved placements and volume.
    """
    W, H, D = container
    n = len(boxes)
    
    # Identify unplaced boxes
    unplaced_indices = [i for i in range(n) if placements[i][0] == -1]
    
    # Sort unplaced by volume descending (try to fit big ones first)
    unplaced_indices.sort(key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    # Sort placed by volume ascending (try to remove small ones first to make room)
    # But we need to be careful: removing a small box might allow a large one to fit.
    # We iterate through placed boxes in some order.
    
    improved = True
    iterations = 0
    max_iterations = 500 # Safety cap
    
    while improved and time.time() < deadline and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # We need to re-evaluate unplaced list each iteration? 
        # Or just keep a static list and check if they are still unplaced?
        # Let's keep a dynamic set of unplaced indices.
        current_unplaced = [i for i in range(n) if placements[i][0] == -1]
        if not current_unplaced:
            break
            
        # Sort current unplaced by volume desc
        current_unplaced.sort(key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
        
        # We will iterate through placed boxes. 
        # To avoid O(N^2) overhead, we can just iterate through the current placed_list indices.
        # However, the indices in placed_list change as we swap.
        # It is safer to rebuild placed_indices and placed_list if we modify them, 
        # or just track the list of placed tuples and their original indices.
        
        # Let's maintain a list of (orig_idx, x, y, z, w, h, d) for placed boxes.
        # Actually, the input `placed_list` is just geometry. We need to know which box is which.
        # Let's create a parallel structure: placed_info = [(orig_idx, x, y, z, w, h, d), ...]
        
        # Re-initialize placed_info from current state to ensure consistency
        placed_info = []
        for i in range(n):
            if placements[i][0] == 0:
                _, x, y, z, w, h, d = placements[i]
                placed_info.append((i, x, y, z, w, h, d))
        
        # Try to swap each placed box with each unplaced box?
        # That's expensive. 
        # Strategy: For each unplaced box (biggest first), try to find a placed box 
        # (smallest first?) such that removing the placed box allows the unplaced box to fit.
        
        # To be efficient:
        # 1. Sort placed_info by volume ascending (easiest to remove).
        placed_info.sort(key=lambda item: item[4]*item[5]*item[6])
        
        for unplaced_idx in current_unplaced:
            if time.time() > deadline:
                break
            if placements[unplaced_idx][0] != -1:
                continue # Already placed in a previous swap
            
            u_w, u_h, u_d = boxes[unplaced_idx]
            u_vol = u_w * u_h * u_d
            
            # Try removing each placed box
            found_swap = False
            new_placed_info = []
            
            # We need to try removing each placed box one by one.
            # If we remove box P, can U fit?
            # If yes, do we gain volume? (u_vol > p_vol)
            # If yes, perform swap.
            
            # Optimization: Only try removing boxes with volume < u_vol
            # And maybe only a few candidates to save time.
            
            for k in range(len(placed_info)):
                p_idx, px, py, pz, pw, ph, pd = placed_info[k]
                p_vol = pw * ph * pd
                
                if p_vol >= u_vol:
                    continue # No gain
                
                # Construct list of placed boxes without p_idx
                # We can't just slice because we need to check overlap for U against the rest.
                # To avoid O(N^2) per swap attempt, we can precompute or just do it.
                # N=60, so N^2 is 3600. 500 iterations * 60 unplaced * 60 placed = 1.8M checks.
                # Each check is O(N) for overlap. Total ~100M ops. Might be tight in 120s.
                
                # Let's optimize: 
                # Instead of rebuilding the list every time, we can just check if U fits 
                # in the space created by removing P, assuming other boxes stay fixed.
                # But other boxes might have been placed *because* of P's presence? No, 
                # they are just there. Removing P creates a hole.
                # We need to check if U fits in the container without overlapping any 
                # OTHER placed boxes.
                
                # Check if U fits at some position in the container, 
                # given that all other placed boxes are still there.
                
                # To speed this up, we can just call _find_best_position 
                # with a modified placed list.
                
                # Constructing the modified placed list is O(N).
                # _find_best_position is O(N * Candidates * Rots).
                # Candidates ~ N. Rots ~ 6.
                # So ~ N^2 * 6 per check.
                
                # Let's try a limited number of placed boxes to remove (e.g., first 10 smallest).
                if k > 20: 
                    break
                
                # Build placed list without p_idx
                temp_placed = [item[1:] for item in placed_info if item[0] != p_idx]
                
                res = _find_best_position((u_w, u_h, u_d), temp_placed, container)
                
                if res is not None:
                    # Swap successful
                    _, nx, ny, nz, nw, nh, nd = res
                    
                    # Update placements
                    placements[p_idx] = (-1, 0, 0, 0, pw, ph, pd)
                    placements[unplaced_idx] = (0, nx, ny, nz, nw, nh, nd)
                    
                    # Update placed_info for next iterations? 
                    # We break out of the unplaced loop and restart the while loop 
                    # to refresh state.
                    improved = True
                    found_swap = True
                    break
            
            if found_swap:
                break # Restart while loop with new state
    
    return placements

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 115 # Safety margin
    
    W, H, D = container
    n = len(boxes)
    
    # 1. Constructive Phase
    # Use a few different ordering strategies for the initial greedy fill
    best_placements = None
    best_volume = -1
    
    orders = []
    # Volume descending
    orders.append(sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True))
    # Max dimension descending
    orders.append(sorted(range(n), key=lambda i: max(boxes[i]), reverse=True))
    # Surface area descending
    orders.append(sorted(range(n), key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True))
    
    for order in orders:
        if time.time() > deadline:
            break
        
        placements, placed_indices, placed_list = _constructive_solve(boxes, container, order, deadline)
        
        # Calculate volume
        vol = sum(p[4]*p[5]*p[6] for p in placements if p[0] == 0)
        
        if vol > best_volume:
            best_volume = vol
            best_placements = placements
            
            # 2. Local Search Phase (only if we have time)
            if time.time() < deadline - 5:
                improved_placements = _local_search(boxes, container, placements, placed_indices, placed_list, deadline)
                improved_vol = sum(p[4]*p[5]*p[6] for p in improved_placements if p[0] == 0)
                
                if improved_vol > best_volume:
                    best_volume = improved_vol
                    best_placements = improved_placements

    # Fallback if nothing placed
    if best_placements is None:
        best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    return best_placements
