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
    # 150s limit, keep margin for safety
    deadline = start_time + 140
    
    W, H, D = container
    container_volume = W * H * D
    n = len(boxes)
    
    # Pre-compute rotations for each box
    box_rots = [_rotations(b[0], b[1], b[2]) for b in boxes]
    box_vols = [b[0]*b[1]*b[2] for b in boxes]
    
    # If container is too large for a full grid (memory safety), 
    # fall back to a simpler heuristic or return empty? 
    # 100^3 = 1,000,000 ints = 8MB. 500^3 = 125M = 1GB. 
    # Assuming standard sizes (<= 200), we are safe.
    # If dimensions are huge, we might crash. Let's add a guard.
    if W * H * D > 10_000_000: 
        # Fallback: simple greedy packing without grid (very slow but safe memory-wise)
        # For now, just return empty to avoid crash, or a very basic heuristic.
        # Given the task constraints, we assume manageable sizes.
        pass

    grid = np.zeros((W, H, D), dtype=np.int8)
    
    def _grid_check(x, y, z, w, h, d):
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        # Vectorized check: is any cell in the sub-volume occupied?
        # Using sum or any on the slice
        return not np.any(grid[x:x+w, y:y+h, z:z+d] == 1)

    def _grid_place(x, y, z, w, h, d, val):
        grid[x:x+w, y:y+h, z:z+d] = val

    def _constructive_heuristic(order):
        """
        Places boxes in the given order using a 'first fit' strategy 
        scanning from bottom-left-back to top-right-front.
        This is fast and provides a decent baseline.
        """
        grid.fill(0)
        placements = [None] * n
        current_vol = 0
        
        for idx in order:
            w, h, d = boxes[idx]
            placed = False
            # Try all rotations
            for rw, rh, rd in box_rots[idx]:
                # Scan positions
                # Optimization: only scan where it might fit
                # Standard scan
                for x in range(W - rw + 1):
                    if placed: break
                    for y in range(H - rh + 1):
                        if placed: break
                        for z in range(D - rd + 1):
                            if _grid_check(x, y, z, rw, rh, rd):
                                _grid_place(x, y, z, rw, rh, rd, 1)
                                placements[idx] = (0, x, y, z, rw, rh, rd)
                                current_vol += rw * rh * rd
                                placed = True
                                break
            if not placed:
                placements[idx] = (-1, 0, 0, 0, w, h, d)
                
        return placements, current_vol

    def _local_search(placements, current_vol, time_limit):
        """
        Perform local search using 'Remove-Reinsert' and 'Rotate' moves.
        Uses the grid for fast collision checks.
        """
        # Identify placed and unplaced
        placed_indices = [i for i, p in enumerate(placements) if p[0] == 0]
        unplaced_indices = [i for i, p in enumerate(placements) if p[0] == -1]
        
        best_placements = placements[:]
        best_vol = current_vol
        
        # Sync grid with initial placements
        grid.fill(0)
        for p in placements:
            if p[0] == 0:
                _, x, y, z, w, h, d = p
                _grid_place(x, y, z, w, h, d, 1)
        
        rng = random.Random(42)
        no_improve_count = 0
        max_no_improve = 500 # Restart if stuck too long
        
        while time.time() < time_limit:
            if no_improve_count > max_no_improve:
                # Random restart of the search state? 
                # Or just break and let the outer loop handle it?
                # For now, just perturb heavily or break.
                # Let's try to force a perturbation: remove 10 random boxes
                if placed_indices:
                    for _ in range(min(10, len(placed_indices))):
                        i = rng.choice(placed_indices)
                        _, x, y, z, w, h, d = placements[i]
                        _grid_place(x, y, z, w, h, d, 0)
                        placements[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
                        placed_indices.remove(i)
                        unplaced_indices.append(i)
                        current_vol -= boxes[i][0]*boxes[i][1]*boxes[i][2]
                    no_improve_count = 0
                    continue

            move_type = rng.random()
            
            if move_type < 0.6 and placed_indices:
                # Move 1: Remove a random placed box and try to re-insert it elsewhere
                i = rng.choice(placed_indices)
                old_p = placements[i]
                old_x, old_y, old_z = old_p[1], old_p[2], old_p[3]
                old_w, old_h, old_d = old_p[4], old_p[5], old_p[6]
                
                # Remove from grid
                _grid_place(old_x, old_y, old_z, old_w, old_h, old_d, 0)
                placed_indices.remove(i)
                unplaced_indices.append(i)
                
                # Try to place back
                success = False
                new_pos = None
                new_dim = None
                
                # Try original orientation first, then random
                rots_to_try = [ (old_w, old_h, old_d) ] + [r for r in box_rots[i] if r != (old_w, old_h, old_d)]
                rng.shuffle(rots_to_try)
                
                for rw, rh, rd in rots_to_try:
                    # Sample random positions
                    # Bias towards "good" spots? Random is okay for diversity
                    for _ in range(20): 
                        rx = rng.randint(0, max(0, W - rw))
                        ry = rng.randint(0, max(0, H - rh))
                        rz = rng.randint(0, max(0, D - rd))
                        
                        if _grid_check(rx, ry, rz, rw, rh, rd):
                            _grid_place(rx, ry, rz, rw, rh, rd, 1)
                            placements[i] = (0, rx, ry, rz, rw, rh, rd)
                            unplaced_indices.remove(i)
                            placed_indices.append(i)
                            success = True
                            break
                    if success:
                        break
                
                if success:
                    # Volume is unchanged (same box, same volume)
                    # But we might have found a better spot that allows others?
                    # In this simple LS, volume only changes if we place an unplaced box.
                    # Moving a placed box doesn't change total volume.
                    # So this move is only useful if it enables future moves.
                    # We accept it to explore the space.
                    pass 
                else:
                    # Failed to place, so it's now unplaced. Volume drops.
                    current_vol -= boxes[i][0]*boxes[i][1]*boxes[i][2]
                    # Revert? Or accept the worse state?
                    # Standard LS usually accepts if it leads to improvement later.
                    # But we need to track current_vol correctly.
                    # If we can't place it, we are worse off. 
                    # Let's revert this move immediately to keep state valid and not degrade too fast,
                    # OR accept it with low probability (SA style).
                    # Given the "Hill Climbing" nature of the previous code's failure,
                    # let's implement a strict improvement rule for simplicity and stability,
                    # but allow "neutral" moves (volume unchanged) to escape local optima.
                    
                    # If success was False, volume dropped. We should revert.
                    # Restore old position
                    _grid_place(old_x, old_y, old_z, old_w, old_h, old_d, 1)
                    placements[i] = (0, old_x, old_y, old_z, old_w, old_h, old_d)
                    unplaced_indices.remove(i)
                    placed_indices.append(i)
                    # No volume change
                    no_improve_count += 1
                    continue

            elif move_type < 0.9 and unplaced_indices:
                # Move 2: Try to place an unplaced box
                i = rng.choice(unplaced_indices)
                success = False
                for rw, rh, rd in box_rots[i]:
                    for _ in range(20):
                        rx = rng.randint(0, max(0, W - rw))
                        ry = rng.randint(0, max(0, H - rh))
                        rz = rng.randint(0, max(0, D - rd))
                        
                        if _grid_check(rx, ry, rz, rw, rh, rd):
                            _grid_place(rx, ry, rz, rw, rh, rd, 1)
                            placements[i] = (0, rx, ry, rz, rw, rh, rd)
                            unplaced_indices.remove(i)
                            placed_indices.append(i)
                            current_vol += boxes[i][0]*boxes[i][1]*boxes[i][2]
                            success = True
                            break
                    if success:
                        break
                
                if success:
                    no_improve_count = 0
                    if current_vol > best_vol:
                        best_vol = current_vol
                        best_placements = placements[:]
                else:
                    no_improve_count += 1
            else:
                # Move 3: Rotate a placed box in place (if fits)
                if placed_indices:
                    i = rng.choice(placed_indices)
                    _, x, y, z, w, h, d = placements[i]
                    # Try other rotations at the same position
                    for rw, rh, rd in box_rots[i]:
                        if (rw, rh, rd) == (w, h, d):
                            continue
                        if _grid_check(x, y, z, rw, rh, rd):
                            # It fits! Update
                            _grid_place(x, y, z, w, h, d, 0)
                            _grid_place(x, y, z, rw, rh, rd, 1)
                            placements[i] = (0, x, y, z, rw, rh, rd)
                            # Volume unchanged
                            break
                    no_improve_count += 1 # Neutral move
            
            # Check for improvement
            # Note: Only "Place Unplaced" moves increase volume.
            # "Move Placed" and "Rotate" are neutral or negative (if failed, reverted).
            # So we only update best_vol when we place a new box.

        return best_placements, best_vol

    # Multi-Start Strategy
    # Generate diverse initial solutions using different sorting heuristics
    # 1. Volume Descending
    # 2. Surface Area Descending (large boxes first)
    # 3. Min Dimension Descending
    # 4. Random Shuffle (with seed)
    
    heuristics = [
        sorted(range(n), key=lambda i: box_vols[i], reverse=True),
        sorted(range(n), key=lambda i: 2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2]), reverse=True),
        sorted(range(n), key=lambda i: min(boxes[i]), reverse=True),
        list(range(n)) # Identity, then shuffled below
    ]
    
    # Add a few random shuffles
    rng_init = random.Random(123)
    for _ in range(3):
        idxs = list(range(n))
        rng_init.shuffle(idxs)
        heuristics.append(idxs)
    
    best_overall_placements = None
    best_overall_vol = 0
    
    # Allocate time to each start
    num_starts = len(heuristics)
    time_per_start = (deadline - time.time()) / num_starts
    # Ensure minimum time per start
    if time_per_start < 2.0:
        # If too little time, do fewer starts
        num_starts = max(1, int((deadline - time.time()) / 5.0))
        heuristics = heuristics[:num_starts]
        time_per_start = (deadline - time.time()) / num_starts

    for h_idx, order in enumerate(heuristics):
        if time.time() > deadline - 5: # Safety margin
            break
            
        # 1. Construct
        placements, vol = _constructive_heuristic(order)
        
        if vol > best_overall_vol:
            best_overall_vol = vol
            best_overall_placements = placements[:]
            
        # 2. Local Search
        # Give remaining time of this slot to LS
        ls_deadline = time.time() + time_per_start * 0.7 # Reserve 30% for next start
        if ls_deadline > deadline - 5:
            ls_deadline = deadline - 5
            
        ls_placements, ls_vol = _local_search(placements, vol, ls_deadline)
        
        if ls_vol > best_overall_vol:
            best_overall_vol = ls_vol
            best_overall_placements = ls_placements

    # If no boxes were placed (shouldn't happen), return empty
    if best_overall_placements is None:
        best_overall_placements = [(-1, 0, 0, 0, b[0], b[1], b[2]) for b in boxes]
        
    return best_overall_placements
