import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401
    num_dirs = q + 1 # 50 directions: v=(0,1) and v=(1,m) for m in F_49
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 570.0

    # --- Field Arithmetic Setup ---
    # F_49 = F_7[alpha] with alpha^2 = 3
    k_const = 3
    
    # Map (a, b) -> index a*p + b
    idx_to_elem = [(i // p, i % p) for i in range(q)]
    
    # Precompute multiplication table for F_q
    mul_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        ai, bi = idx_to_elem[i]
        for j in range(q):
            aj, bj = idx_to_elem[j]
            aa = (ai * aj + bi * bj * k_const) % p
            bb = (ai * bj + bi * aj) % p
            mul_table[i, j] = aa * p + bb
            
    # Precompute addition table
    add_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        ai, bi = idx_to_elem[i]
        for j in range(q):
            aj, bj = idx_to_elem[j]
            aa = (ai + aj) % p
            bb = (bi + bj) % p
            add_table[i, j] = aa * p + bb

    # --- Direction and Line Precomputation ---
    dir_dx = np.zeros(num_dirs, dtype=np.int32)
    dir_dy = np.zeros(num_dirs, dtype=np.int32)
    
    # v_vertical: dx=0 (idx 0), dy=1 (idx 7)
    dir_dx[0] = 0
    dir_dy[0] = 7
    
    # v_slant_m: dx=1 (idx 7), dy=m (idx m)
    for i in range(1, num_dirs):
        m = i - 1
        dir_dx[i] = 7
        dir_dy[i] = m

    zx_arr = np.array([z // q for z in range(total_points)], dtype=np.int32)
    zy_arr = np.array([z % q for z in range(total_points)], dtype=np.int32)
    t_vals = np.arange(q, dtype=np.int32)
    
    # Precompute line_members: line_members[x, v, t] = index of point x + t*v
    # We only need the points OTHER than x for the "blocked" logic.
    # line_members[x, v, 0] is x itself.
    # So line_members[x, v, 1:] are the other points.
    
    line_members = np.zeros((total_points, num_dirs, q), dtype=np.int32)
    
    for v_idx in range(num_dirs):
        dx = dir_dx[v_idx]
        dy = dir_dy[v_idx]
        
        tdx = mul_table[t_vals, dx]
        tdy = mul_table[t_vals, dy]
        
        nx = add_table[zx_arr[:, None], tdx[None, :]]
        ny = add_table[zy_arr[:, None], tdy[None, :]]
        
        line_members[:, v_idx, :] = nx * q + ny

    # Precompute affected pairs for fast updates
    # When z is added to E, it increments blocked[x, v] for all x such that z is on line(x, v) \ {x}.
    # This is equivalent to x being on line(z, v) \ {z}.
    # So affected_x[z] and affected_v[z] contain the x's and v's to update.
    
    affected_x = []
    affected_v = []
    
    for z in range(total_points):
        xs = []
        vs = []
        for v in range(num_dirs):
            # line_members[z, v, 1:] are the points on the line through z in direction v, excluding z
            other_pts = line_members[z, v, 1:]
            xs.extend(other_pts)
            vs.extend([v] * len(other_pts))
            
        affected_x.append(np.array(xs, dtype=np.int32))
        affected_v.append(np.array(vs, dtype=np.int32))

    # --- Local Search (Simulated Annealing) ---
    
    # State:
    # in_E: bool array (total_points)
    # blocked[x, v]: number of points in E on line(x, v) excluding x.
    # free_count[x]: number of v such that blocked[x, v] == 0.
    # x is valid if free_count[x] > 0.
    # We maintain the invariant that all x are valid (free_count[x] > 0) in the current state?
    # Actually, for SA, we might allow invalid states but penalize them, or restrict moves to valid states.
    # Restricting to valid states is safer and easier to reason about.
    # If we add z, we must ensure no x becomes invalid.
    # If we remove z, it always helps validity (increases free counts).
    
    # Let's use a greedy initialization first to get a good starting point.
    
    def greedy_init(rng):
        in_E = np.zeros(total_points, dtype=bool)
        blocked = np.zeros((total_points, num_dirs), dtype=np.int32)
        free_count = np.full(total_points, num_dirs, dtype=np.int32)
        
        order = list(range(total_points))
        rng.shuffle(order)
        
        size = 0
        for z in order:
            if in_E[z]:
                continue
            
            ax = affected_x[z]
            av = affected_v[z]
            
            # Apply update
            # blocked[ax, av] += 1
            # But we need to update free_count correctly.
            # For each (x, v) in affected:
            #   if blocked[x, v] was 0, it becomes 1, so free_count[x] decreases by 1.
            #   else, free_count[x] unchanged.
            
            # Vectorized approach:
            # old_blocked = blocked[ax, av]
            # was_zero = (old_blocked == 0)
            # new_blocked = old_blocked + 1
            # blocked[ax, av] = new_blocked
            
            # For free_count:
            # We need to decrement free_count[x] for each x where was_zero is true.
            # But we must check if free_count[x] becomes 0. If so, this move is invalid.
            
            # To check validity efficiently:
            # We only care about x's where free_count[x] == 1 and the update would make it 0.
            # i.e., x has exactly one free direction v_free, and (x, v_free) is in the affected set.
            
            # Let's find unique x in ax.
            # This is expensive in Python loops.
            
            # Alternative:
            # Just do the update, check min(free_count), and revert if min <= 0.
            # Update:
            old_blocked = blocked[ax, av]
            was_zero = (old_blocked == 0)
            
            # Count how many times each x appears in ax where was_zero is true.
            # This is a bincount operation.
            # unique_x, counts = np.unique(ax[was_zero], return_counts=True)
            # free_count[unique_x] -= counts
            
            # If any free_count[unique_x] <= 0, revert.
            
            # Let's try this.
            if np.any(was_zero):
                unique_x, counts = np.unique(ax[was_zero], return_counts=True)
                # Check if any will become 0
                new_free = free_count[unique_x] - counts
                if np.any(new_free <= 0):
                    continue # Invalid move
                
                # Apply update
                blocked[ax, av] = old_blocked + 1
                free_count[unique_x] = new_free
                in_E[z] = True
                size += 1
            else:
                # No free directions were blocked, so it's always valid
                blocked[ax, av] = old_blocked + 1
                in_E[z] = True
                size += 1
        
        return in_E, blocked, free_count, size

    def remove_point(z, in_E, blocked, free_count):
        ax = affected_x[z]
        av = affected_v[z]
        
        old_blocked = blocked[ax, av]
        was_one = (old_blocked == 1)
        
        if np.any(was_one):
            unique_x, counts = np.unique(ax[was_one], return_counts=True)
            free_count[unique_x] += counts
        
        blocked[ax, av] = old_blocked - 1
        in_E[z] = False
        return 1 # size decrease

    def add_point(z, in_E, blocked, free_count):
        ax = affected_x[z]
        av = affected_v[z]
        
        old_blocked = blocked[ax, av]
        was_zero = (old_blocked == 0)
        
        if np.any(was_zero):
            unique_x, counts = np.unique(ax[was_zero], return_counts=True)
            new_free = free_count[unique_x] - counts
            if np.any(new_free <= 0):
                return False # Invalid
        
        blocked[ax, av] = old_blocked + 1
        if np.any(was_zero):
            free_count[unique_x] = new_free
        in_E[z] = True
        return True

    # Run Greedy Init
    rng = random.Random(42)
    best_in_E, best_blocked, best_free_count, best_size = greedy_init(rng)
    
    # Simulated Annealing
    current_in_E = best_in_E.copy()
    current_blocked = best_blocked.copy()
    current_free_count = best_free_count.copy()
    current_size = best_size
    
    T = 1.0
    cooling = 0.999
    
    # Precompute random order for moves? Or just pick random z.
    # We want to maximize |E|.
    # Moves:
    # 1. Try to add a random point not in E.
    # 2. Try to remove a random point in E (to escape local optima, though this decreases size).
    #    Actually, since we want to maximize size, we mostly want to add.
    #    But sometimes we need to remove one point to add two others.
    #    So we allow removals with some probability.
    
    num_steps = 100000
    step = 0
    
    while step < num_steps and time.time() < deadline:
        if time.time() > deadline - 10:
            break
            
        # Pick random point z
        z = rng.randint(0, total_points - 1)
        
        if current_in_E[z]:
            # Remove z
            # This always improves validity (or keeps it same), but decreases size.
            # In SA, we accept if it allows future additions.
            # For now, let's just do it to explore.
            remove_point(z, current_in_E, current_blocked, current_free_count)
            current_size -= 1
        else:
            # Add z
            if add_point(z, current_in_E, current_blocked, current_free_count):
                current_size += 1
            else:
                # Move invalid. In SA, we might still accept it if T is high?
                # But our state representation doesn't easily handle invalid states (free_count < 0).
                # So we reject invalid moves.
                pass
        
        if current_size > best_size:
            best_size = current_size
            best_in_E = current_in_E.copy()
            best_blocked = current_blocked.copy()
            best_free_count = current_free_count.copy()
            
        T *= cooling
        step += 1

    # Build N
    points_list = []
    for pidx in range(total_points):
        if not best_in_E[pidx]:
            xi = pidx // q
            yi = pidx % q
            xa, xb = idx_to_elem[xi]
            ya, yb = idx_to_elem[yi]
            points_list.append([[xa, xb], [ya, yb]])

    if not points_list:
        # Fallback
        for pidx in range(total_points):
            xi = pidx // q
            yi = pidx % q
            xa, xb = idx_to_elem[xi]
            ya, yb = idx_to_elem[yi]
            points_list.append([[xa, xb], [ya, yb]])

    return np.array(points_list, dtype=np.int64)
