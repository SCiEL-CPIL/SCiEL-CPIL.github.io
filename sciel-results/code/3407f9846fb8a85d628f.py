import time
import itertools
import random
import math

# EVOLVE-BLOCK-START

def load(boxes, container):
    W, H, D = container
    W, H, D = int(W), int(H), int(D)
    n = len(boxes)
    deadline = time.time() + 115.0  # Safety margin before 120s limit

    # Pre-compute unique orientations for each box
    def get_orientations(w, h, d):
        opts = set()
        for perm in itertools.permutations((int(w), int(h), int(d))):
            opts.add(perm)
        return list(opts)

    orient_cache = [get_orientations(*boxes[i]) for i in range(n)]

    # Fast Best-Fit placement for a given state (order of indices, orientations list)
    # Returns (placements, total_volume)
    def evaluate(state):
        """
        state: tuple of (box_indices, orientations)
        box_indices: list of box indices in the order they are processed
        orientations: list of orientation indices (index into orient_cache[box_idx])
        """
        placed = []  # list of (idx, x, y, z, w, h, d)
        total_vol = 0
        # We use a simple "First Fit" with a few candidate positions to keep it fast.
        # Since we are doing SA, we need this to be very fast.
        # O(N^2) is acceptable for N=60 if the constant is low.
        
        # To speed up overlap checks, we could use a grid, but for 60 boxes,
        # direct check is likely fine if we limit candidate positions.
        
        for i in range(n):
            idx = state[0][i]
            o_idx = state[1][i]
            bw, bh, bd = orient_cache[idx][o_idx]
            
            # Candidate positions:
            # 1. (0,0,0)
            # 2. Right of each placed box
            # 3. Above each placed box
            # 4. Back of each placed box
            # 5. Combinations (Top-Right, Top-Back, etc.)
            
            candidates = set()
            candidates.add((0, 0, 0))
            
            for (_, px, py, pz, pw, ph, pd) in placed:
                candidates.add((px + pw, py, pz))
                candidates.add((px, py + ph, pz))
                candidates.add((px, py, pz + pd))
                candidates.add((px + pw, py + ph, pz))
                candidates.add((px + pw, py, pz + pd))
                candidates.add((px, py + ph, pz + pd))
                candidates.add((px + pw, py + ph, pz + pd))
            
            best_pos = None
            best_score = None
            
            for (cx, cy, cz) in candidates:
                if cx < 0 or cy < 0 or cz < 0:
                    continue
                if cx + bw > W or cy + bh > H or cz + bd > D:
                    continue
                
                # Check overlap
                overlap = False
                for (_, ox, oy, oz, ow, oh, od) in placed:
                    if cx < ox + ow and cx + bw > ox and cy < oy + oh and cy + bh > oy and cz < oz + od and cz + bd > oz:
                        overlap = True
                        break
                
                if not overlap:
                    # Score: Minimize z, then y, then x (pack tight to bottom/corner)
                    # We can also add a penalty for "wasted space" above the box, but keep it simple.
                    score = (cz, cy, cx)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pos = (cx, cy, cz)
            
            if best_pos is not None:
                bx, by, bz = best_pos
                placed.append((idx, bx, by, bz, bw, bh, bd))
                total_vol += bw * bh * bd
            # If no position found, box is left out (implicitly, as it's not in 'placed')
            
        # Build result array
        result = [None] * n
        for p in placed:
            idx, x, y, z, w, h, d = p
            result[idx] = (0, x, y, z, w, h, d)
        for i in range(n):
            if result[i] is None:
                result[i] = (-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2])
        
        return result, total_vol

    # Initialize SA
    # Initial state: Sort by volume descending, pick best orientation greedily? 
    # No, SA needs a random-ish or heuristic start. Let's start with volume desc order.
    # Orientation: start with 0 (original) or best fit? Let's start with 0 for simplicity, SA will fix it.
    
    # Heuristic Start: Volume Desc
    init_order = sorted(range(n), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    init_orients = [0] * n  # All start with orientation 0
    
    current_state = (init_order, init_orients)
    current_result, current_vol = evaluate(current_state)
    
    best_state = current_state
    best_result = current_result
    best_vol = current_vol
    
    # SA Parameters
    T = 100.0  # Initial Temperature
    T_min = 0.1
    cooling = 0.99  # Cooling rate
    # We want to run as many iterations as possible until deadline
    # Estimate time per iteration:
    # evaluate() is O(N^2) ~ 3600 ops. Python overhead is high.
    # Let's measure or just run until time.
    
    t_start = time.time()
    
    # Pre-compute box volumes for potential tie-breaking or checks
    box_vols = [boxes[i][0]*boxes[i][1]*boxes[i][2] for i in range(n)]
    
    # To make SA effective, we need to allow swaps in order and changes in orientation.
    # We'll randomly pick a move:
    # 1. Swap two boxes in order (i, j)
    # 2. Change orientation of one box
    
    while time.time() < deadline:
        # Pick a move
        move_type = random.random()
        
        if move_type < 0.7:
            # Swap two positions in the order
            i, j = random.sample(range(n), 2)
            new_order = current_state[0][:]
            new_order[i], new_order[j] = new_order[j], new_order[i]
            new_orients = current_state[1]
            new_state = (new_order, new_orients)
        else:
            # Change orientation of a random box
            i = random.randint(0, n-1)
            idx = current_state[0][i]
            num_orients = len(orient_cache[idx])
            if num_orients <= 1:
                continue # Can't change orientation
            
            new_orients = current_state[1][:]
            # Pick a different orientation
            current_o = new_orients[i]
            new_o = random.choice([x for x in range(num_orients) if x != current_o])
            new_orients[i] = new_o
            new_state = (current_state[0], new_orients)
        
        new_result, new_vol = evaluate(new_state)
        delta = new_vol - current_vol
        
        if delta > 0:
            # Better, accept
            current_state = new_state
            current_result = new_result
            current_vol = new_vol
            
            if new_vol > best_vol:
                best_vol = new_vol
                best_state = new_state
                best_result = new_result
        else:
            # Worse, accept with probability exp(delta / T)
            if T > T_min:
                # Avoid overflow in exp
                if delta / T > -50: 
                    prob = math.exp(delta / T)
                else:
                    prob = 0.0
                if random.random() < prob:
                    current_state = new_state
                    current_result = new_result
                    current_vol = new_vol
        
        T *= cooling
        if T < T_min:
            # Restart with new temperature? Or just stop? 
            # Restarting helps escape local minima.
            T = 50.0 

    return best_result

# EVOLVE-BLOCK-END
