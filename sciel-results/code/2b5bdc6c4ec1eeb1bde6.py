import time
import random
import itertools
import math

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

def _generate_initial_solution(boxes, container, order_idx, scoring_func=None):
    """Generates a valid solution using the EAP heuristic."""
    W, H, D = container
    placed = []
    placements = [None] * len(boxes)
    
    if scoring_func is None:
        def scoring_func(pos, dim):
            return (pos[2], pos[1], pos[0])
            
    for orig_idx in order_idx:
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

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    W, H, D = container
    container_volume = W * H * D
    n = len(boxes)
    
    # Pre-compute properties
    volumes = [b[0]*b[1]*b[2] for b in boxes]
    
    # --- Phase 1: Multi-Start Construction ---
    # We generate several diverse initial solutions to explore the landscape.
    
    orders = []
    # 1. Volume Descending
    orders.append(sorted(range(n), key=lambda i: volumes[i], reverse=True))
    # 2. Max Dim Descending
    orders.append(sorted(range(n), key=lambda i: max(boxes[i]), reverse=True))
    # 3. Min Dim Ascending
    orders.append(sorted(range(n), key=lambda i: min(boxes[i])))
    # 4. Random Shuffles
    rng_init = random.Random(42)
    for _ in range(5):
        ord = list(range(n))
        rng_init.shuffle(ord)
        orders.append(ord)
        
    best_placements = None
    best_volume = 0
    
    # Generate initial solutions
    initial_solutions = []
    for order in orders:
        if time.time() > deadline:
            break
        placements, vol = _generate_initial_solution(boxes, container, order)
        if vol > best_volume:
            best_volume = vol
            best_placements = placements
        initial_solutions.append((placements, vol))
        
    # --- Phase 2: Simulated Annealing ---
    # We refine the best initial solution and potentially others if time permits.
    # Since SA is expensive, we focus our budget on the most promising starts.
    
    # Sort initial solutions by volume descending
    initial_solutions.sort(key=lambda x: x[1], reverse=True)
    
    # We will run SA on the top few initial solutions
    max_sa_runs = 3
    sa_budget_per_run = (deadline - start_time) / max_sa_runs
    
    # SA Parameters
    T0 = 100.0
    T_end = 0.1
    cooling_rate = 0.995
    
    for idx, (init_placements, init_vol) in enumerate(initial_solutions[:max_sa_runs]):
        if time.time() > deadline:
            break
            
        # Copy state
        placements = [p[:] for p in init_placements]
        current_vol = init_vol
        
        # Track best found in this SA run
        local_best_placements = [p[:] for p in placements]
        local_best_vol = current_vol
        
        # SA Loop
        T = T0
        # Estimate steps based on time budget
        # A rough estimate: ~1000-5000 steps/sec depending on N
        # We will just loop until time runs out for this specific run allocation
        run_deadline = start_time + (sa_budget_per_run * (max_sa_runs - idx)) 
        # Actually, simpler: just use global deadline and check time each step
        
        while time.time() < deadline:
            # Check termination
            if T < T_end:
                break
                
            # Generate a random move
            move_type = random.random()
            
            if move_type < 0.5:
                # Move 1: Relocate a placed box
                placed_indices = [i for i, p in enumerate(placements) if p[0] == 0]
                if not placed_indices:
                    continue
                i = random.choice(placed_indices)
                # Remove box i
                old_pos = placements[i][1:4]
                old_dim = placements[i][4:7]
                # Update placed list logic for overlap check is tricky if we don't maintain a list.
                # For simplicity and speed in this heuristic context, we rely on the fact that
                # we check against *all other* boxes.
                
                # Find new position
                # Use EAP candidates from *other* boxes
                other_placed = [(placements[j][1], placements[j][2], placements[j][3], 
                                 placements[j][4], placements[j][5], placements[j][6]) 
                                for j in range(n) if j != i and placements[j][0] == 0]
                
                w, h, d = boxes[i]
                rots = _rotations(w, h, d)
                candidates = _extreme_points(other_placed, container)
                # Add some random jitter to candidates to explore space
                for _ in range(10):
                    rx = random.randint(0, W-1)
                    ry = random.randint(0, H-1)
                    rz = random.randint(0, D-1)
                    if rx <= W and ry <= H and rz <= D:
                        candidates.append((rx, ry, rz))
                        
                best_pos = None
                best_dim = None
                best_score = None
                # Greedy pick for the new location to ensure valid move generation
                for (cx, cy, cz) in candidates:
                    for (rw, rh, rd) in rots:
                        if not _fits((rw, rh, rd), (cx, cy, cz), container):
                            continue
                        if not _no_overlap((rw, rh, rd), (cx, cy, cz), other_placed):
                            continue
                        score = (cz, cy, cx)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_pos = (cx, cy, cz)
                            best_dim = (rw, rh, rd)
                            
                if best_pos:
                    new_pos = best_pos
                    new_dim = best_dim
                    # Calculate delta volume? 
                    # Since we are moving a box, the volume change is 0 if it stays placed.
                    # However, moving it might block/unblock others? No, we only move one.
                    # The objective function is just total volume. Moving a box doesn't change volume.
                    # So SA on volume alone is flat for relocation moves unless we consider "compactness" or similar.
                    # To make SA work on volume, we need moves that change the *set* of placed boxes.
                    # Relocation only helps if it allows *other* boxes to be placed later? 
                    # But we are in a static state.
                    
                    # Actually, standard SA for packing usually optimizes a penalty function or uses a different metric.
                    # Here, let's stick to a simpler "Swap" and "Add/Remove" strategy.
                    
                    # Let's try a different move: Swap orientation of a placed box.
                    # This doesn't change volume either.
                    
                    # The key is that the initial heuristic is greedy. 
                    # A better approach for SA here is to treat "left out" boxes as candidates to be placed.
                    
            elif move_type < 0.8:
                # Move 2: Try to place a box that is currently left out
                left_out_indices = [i for i, p in enumerate(placements) if p[0] == -1]
                if not left_out_indices:
                    continue
                i = random.choice(left_out_indices)
                w, h, d = boxes[i]
                rots = _rotations(w, h, d)
                
                # Find a spot for box i
                placed = [(placements[j][1], placements[j][2], placements[j][3], 
                           placements[j][4], placements[j][5], placements[j][6]) 
                          for j in range(n) if placements[j][0] == 0]
                
                candidates = _extreme_points(placed, container)
                # Add random candidates
                for _ in range(5):
                    rx = random.randint(0, W-1)
                    ry = random.randint(0, H-1)
                    rz = random.randint(0, D-1)
                    if rx <= W and ry <= H and rz <= D:
                        candidates.append((rx, ry, rz))
                        
                best_pos = None
                best_dim = None
                for (cx, cy, cz) in candidates:
                    for (rw, rh, rd) in rots:
                        if not _fits((rw, rh, rd), (cx, cy, cz), container):
                            continue
                        if not _no_overlap((rw, rh, rd), (cx, cy, cz), placed):
                            continue
                        best_pos = (cx, cy, cz)
                        best_dim = (rw, rh, rd)
                        break
                    if best_pos:
                        break
                        
                if best_pos:
                    # Accept move (always improves volume)
                    placements[i] = (0, best_pos[0], best_pos[1], best_pos[2], 
                                    best_dim[0], best_dim[1], best_dim[2])
                    current_vol += w * h * d
                    if current_vol > local_best_vol:
                        local_best_vol = current_vol
                        local_best_placements = [p[:] for p in placements]
                    
            else:
                # Move 3: Remove a placed box (to allow rearrangement or just exploration)
                # This is risky for volume, but allows escape from local optima if combined with re-adding.
                placed_indices = [i for i, p in enumerate(placements) if p[0] == 0]
                if not placed_indices:
                    continue
                i = random.choice(placed_indices)
                w, h, d = boxes[i]
                delta_vol = -w * h * d
                new_vol = current_vol + delta_vol
                
                # Accept with probability
                if new_vol > current_vol or random.random() < math.exp(delta_vol / (T * max(1, container_volume/1000))):
                    placements[i] = (-1, 0, 0, 0, w, h, d)
                    current_vol = new_vol
                    
            T *= cooling_rate
            
        # After SA run, update global best
        if local_best_vol > best_volume:
            best_volume = local_best_vol
            best_placements = local_best_placements
            if best_volume >= container_volume:
                break

    if best_placements is None:
        # Fallback to first initial solution
        if initial_solutions:
            best_placements = initial_solutions[0][0]
        else:
            # Emergency fallback
            best_placements = []
            for w, h, d in boxes:
                best_placements.append((-1, 0, 0, 0, w, h, d))
                
    return best_placements
