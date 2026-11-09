import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401
    num_dirs = q + 1  # 50 directions: v=(0,1) and v=(1,m) for m in F_q

    # Deadline guard
    deadline = time.time() + 570.0

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
    # Directions: index 0 is (0,1), index 1..49 are (1, m) where m is index 0..48
    dir_dx = np.zeros(num_dirs, dtype=np.int32)
    dir_dy = np.zeros(num_dirs, dtype=np.int32)
    
    # Field element 0 is index 0. Field element 1 is index 7 (since 1 = 1 + 0*alpha).
    dir_dx[0] = 0
    dir_dy[0] = 7
    
    for i in range(1, num_dirs):
        m = i - 1
        dir_dx[i] = 7
        dir_dy[i] = m

    # Precompute line_members: shape (total_points, num_dirs, q)
    # line_members[x, v, t] = index of point x + t*v
    zx_arr = np.array([z // q for z in range(total_points)], dtype=np.int32)
    zy_arr = np.array([z % q for z in range(total_points)], dtype=np.int32)
    t_vals = np.arange(q, dtype=np.int32)
    
    line_members = np.zeros((total_points, num_dirs, q), dtype=np.int32)
    
    for v_idx in range(num_dirs):
        dx = dir_dx[v_idx]
        dy = dir_dy[v_idx]
        
        tdx = mul_table[t_vals, dx]
        tdy = mul_table[t_vals, dy]
        
        nx = add_table[zx_arr[:, None], tdx[None, :]]
        ny = add_table[zy_arr[:, None], tdy[None, :]]
        
        line_members[:, v_idx, :] = nx * q + ny

    # --- Strategy: Construct N from a Unital, then Prune ---
    # A unital in AG(2, q) is a set U of q^2 + 1 points such that every line contains
    # exactly 1 or q points of U.
    # If N = U, then for any x in U:
    # - Lines with q points of U: contain q-1 other points of U. Valid.
    # - Lines with 1 point of U: contain 0 other points of U. Valid.
    # So U is a valid Nikodym set.
    # Size of U is q^2 + 1 = 2402? No, AG(2, q) has q^2 points.
    # Wait, unital in PG(2, q) has q^2+1 points. In AG(2, q), it's a subset of size q^2+1?
    # No, AG(2, q) has q^2 points. A unital in AG(2, q) usually refers to the affine part of a unital in PG(2, q).
    # Standard unital in PG(2, q) has q^2+1 points. If one point is at infinity, the affine part has q^2 points?
    # Let's use a simpler construction: A "circle" or "conic" like structure?
    
    # Actually, a known construction for Nikodym sets in AG(2, q) is the set of points (x, y) such that y = x^2?
    # No, that's a graph of a function, size q.
    # If N is the graph of y = f(x), size q.
    # For x in N: line with slope m passes through x. Other points (x+t, f(x)+tm).
    # We need f(x+t) = f(x) + tm for all t? No, that's impossible for non-linear f.
    # The condition is: for each x in N, there is AT LEAST ONE line through x with all other points in N.
    # If N is the graph of y=x^2, for x=(a, a^2), the tangent line has slope 2a (in char != 2).
    # The tangent line intersects the parabola at only one point (with multiplicity 2).
    # So the other points on the tangent line are NOT in N.
    # So the graph of x^2 is NOT a Nikodym set.
    
    # What if N is a line? Size q.
    # For x in N, the line N itself contains q-1 other points of N. Valid.
    # For x not in N, does there exist a line through x with all other points in N?
    # No, because any line through x not parallel to N intersects N at 1 point.
    # The line parallel to N through x has 0 points in N.
    # So x not in N is NOT covered.
    # So N must be large enough to cover points outside N.
    
    # Let's try the "unital" idea more carefully.
    # In AG(2, q), a set U of size q^2+1 is not possible.
    # Max size is q^2.
    # A "hyperoval" in PG(2, q) for q even has q+2 points.
    
    # Let's try a different construction:
    # N = { (x, y) : y = x^3 }?
    
    # Let's go back to the greedy approach but start with a good initial set.
    # Start with N = all points (size 2401). Valid.
    # Then try to remove points.
    # This is the "pruning" strategy mentioned in the knowledge.
    
    # State: in_N (bool array)
    # For each x, count[x, v] = number of points in N on line(x, v) excluding x.
    # Initially, count[x, v] = 48 for all x, v.
    # When we remove z from N, we decrement count[x, v] for all x, v such that z is on line(x, v) excluding x.
    # This is equivalent to x being on line(z, v) excluding z.
    # x is valid if min(count[x, :]) > 0.
    # We can remove z if for all x such that z is on line(x, v) excluding x, min(count[x, :]) remains > 0 after decrement.
    # i.e., if for all such x, min(count[x, :]) > 1 OR (min(count[x, :]) == 1 AND the direction with count 1 is NOT the one we are decrementing).
    
    # Let's precompute affected_pairs[z] as (x, v) pairs.
    affected_x = []
    affected_v = []
    
    for z in range(total_points):
        xs = []
        vs = []
        for v in range(num_dirs):
            other_pts = line_members[z, v, 1:]
            xs.extend(other_pts)
            vs.extend([v] * len(other_pts))
        affected_x.append(np.array(xs, dtype=np.int32))
        affected_v.append(np.array(vs, dtype=np.int32))

    # Search: Greedy Pruning
    # Start with N = all points.
    # Try to remove points in a random order.
    # If removing z keeps all x valid, remove it.
    
    best_N = np.ones(total_points, dtype=bool)
    best_size = total_points
    
    num_restarts = 20
    rng = random.Random(123)
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        in_N = np.ones(total_points, dtype=bool)
        count = np.full((total_points, num_dirs), 48, dtype=np.int32)
        
        # Random order
        order = list(range(total_points))
        rng.shuffle(order)
        
        size = total_points
        
        for z in order:
            if time.time() > deadline:
                break
            if not in_N[z]:
                continue
            
            ax = affected_x[z]
            av = affected_v[z]
            
            # Check if removal is safe
            # For each unique x in ax, check if min(count[x, :]) > 1 or (min==1 and the direction is not in av[x]).
            # Actually, we decrement count[x, v] for (x, v) in (ax, av).
            # x becomes invalid if min(count[x, :]) becomes 0.
            # This happens if count[x, v] was 1 and we decrement it.
            
            # Let's just apply, check, and revert if bad.
            count[ax, av] -= 1
            in_N[z] = False
            size -= 1
            
            unique_affected = np.unique(ax)
            min_count = np.min(count[unique_affected, :], axis=1)
            
            if np.any(min_count < 0):
                count[ax, av] += 1
                in_N[z] = True
                size += 1
            else:
                pass # Keep z removed
        
        if size < best_size:
            best_size = size
            best_N = in_N.copy()
            
        # If best_size is small, we can stop early?
        # Or continue to find even smaller?
        
    # Build N
    points_list = []
    for pidx in range(total_points):
        if best_N[pidx]:
            xi = pidx // q
            yi = pidx % q
            xa, xb = idx_to_elem[xi]
            ya, yb = idx_to_elem[yi]
            points_list.append([[xa, xb], [ya, yb]])

    if not points_list:
        # Fallback to full space
        for pidx in range(total_points):
            xi = pidx // q
            yi = pidx % q
            xa, xb = idx_to_elem[xi]
            ya, yb = idx_to_elem[yi]
            points_list.append([[xa, xb], [ya, yb]])

    return np.array(points_list, dtype=np.int64)
