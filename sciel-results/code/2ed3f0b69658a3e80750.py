# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 by maximizing the complement set S.
    Uses a greedy algorithm with bitmask tracking of available directions.
    """
    p = 7
    q = p * p # 49
    n_pts = q * q # 2401
    n_dirs = q + 1 # 50 directions (49 slopes + 1 vertical)
    
    # Field F_49 elements as (a, b) tuples
    # Precompute addition and multiplication tables for speed? 
    # Or just implement functions.
    
    def add_f49(u, v):
        return ((u[0] + v[0]) % p, (u[1] + v[1]) % p)
        
    def mul_f49(u, v):
        # (a1 + b1*a) * (a2 + b2*a) = (a1a2 + 3b1b2) + (a1b2 + a2b1)a
        a1, b1 = u
        a2, b2 = v
        a_res = (a1 * a2 + 3 * b1 * b2) % p
        b_res = (a1 * b2 + a2 * b1) % p
        return (a_res, b_res)
        
    def sub_f49(u, v):
        return ((u[0] - v[0]) % p, (u[1] - v[1]) % p)

    # Generate all points in F_49^2
    # Point index i corresponds to (x, y)
    # x = (xa, xb), y = (ya, yb)
    # i = ((xa * p + xb) * q + (ya * p + yb))
    
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    points.append(((xa, xb), (ya, yb)))
                    
    # Precompute lines
    # lines[i][v] = list of indices of points on line through i with direction v
    # v = 0..48 corresponds to slope m = index v in F_49
    # v = 49 corresponds to vertical direction (0, 1)
    
    # Map F_49 element to index
    def f49_to_idx(z):
        return z[0] * p + z[1]
        
    def idx_to_f49(i):
        return (i // p, i % p)
        
    # Directions
    # Slopes: all elements of F_49
    slopes = []
    for i in range(q):
        slopes.append(idx_to_f49(i))
        
    # Precompute line points
    # This takes O(n_pts * n_dirs * q) which is 2401 * 50 * 49 ~ 5.9M operations. Acceptable.
    lines = []
    for i in range(n_pts):
        x, y = points[i]
        line_set = []
        for v_idx in range(n_dirs):
            if v_idx < q:
                m = slopes[v_idx]
                # Direction (1, m)
                # Points: (x + t, y + t*m) for t in F_49
                pts_on_line = []
                for t_idx in range(q):
                    t = idx_to_f49(t_idx)
                    tx = add_f49(x, (1, 0)) # x + t? No, t is the parameter.
                    # x_new = x + t
                    # y_new = y + t*m
                    tx = add_f49(x, t)
                    tm = mul_f49(t, m)
                    ty = add_f49(y, tm)
                    # Map (tx, ty) to index
                    tx_idx = f49_to_idx(tx)
                    ty_idx = f49_to_idx(ty)
                    pt_idx = tx_idx * q + ty_idx
                    pts_on_line.append(pt_idx)
            else:
                # Vertical direction (0, 1)
                # Points: (x, y + t) for t in F_49
                pts_on_line = []
                for t_idx in range(q):
                    t = idx_to_f49(t_idx)
                    tx = x
                    ty = add_f49(y, t)
                    tx_idx = f49_to_idx(tx)
                    ty_idx = f49_to_idx(ty)
                    pt_idx = tx_idx * q + ty_idx
                    pts_on_line.append(pt_idx)
            line_set.append(pts_on_line)
        lines.append(line_set)
        
    # Greedy search
    best_complement_size = 0
    best_S = None
    
    start_time = time.time()
    timeout = 600 # seconds
    
    # We will try a few random restarts
    # In each restart, we shuffle the order in which we consider adding points
    # Actually, we consider adding points in an order that minimizes bad_count.
    # To avoid getting stuck, we can randomize ties.
    
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() - start_time > timeout - 10:
            break
            
        V_x = [1 << n_dirs for _ in range(n_pts)] # Bitmask of available directions
        crit = [[False] * n_dirs for _ in range(n_pts)] # crit[x][v] is True if v is the only available direction for x
        bad_count = [0] * n_pts
        
        S = set()
        
        # Randomize initial order? Or just pick min bad_count.
        # If we always pick min bad_count, it's deterministic.
        # Let's add some randomness by shuffling the list of candidates if there are ties.
        
        active_points = list(range(n_pts))
        random.shuffle(active_points)
        
        # We need a priority queue or just scan for min bad_count.
        # Scanning 2401 points is fast.
        
        while True:
            if time.time() - start_time > timeout - 5:
                break
                
            # Find point with min bad_count that is not in S
            min_bad = float('inf')
            candidate = -1
            candidates_with_min = []
            
            for y in active_points:
                if y in S:
                    continue
                if bad_count[y] < min_bad:
                    min_bad = bad_count[y]
                    candidates_with_min = [y]
                elif bad_count[y] == min_bad:
                    candidates_with_min.append(y)
                    
            if min_bad > 0:
                break # No safe point to add
                
            # Pick randomly from candidates_with_min to break symmetry
            candidate = random.choice(candidates_with_min)
            
            # Add candidate to S
            S.add(candidate)
            
            # Update V_x and crit and bad_count
            # For each direction v:
            for v in range(n_dirs):
                # Get points on line through candidate with direction v, excluding candidate
                # lines[candidate][v] contains all 49 points
                # We need to exclude candidate
                for x in lines[candidate][v]:
                    if x == candidate:
                        continue
                    # Block direction v for x
                    if (V_x[x] >> v) & 1:
                        V_x[x] &= ~(1 << v)
                        # Check if x became critical or invalid
                        # Count bits
                        # Using bin().count() is slow. Use lookup or __builtin_popcount equivalent?
                        # Python 3.10+ has int.bit_count()
                        cnt = V_x[x].bit_count()
                        if cnt == 0:
                            # This should not happen if bad_count was 0
                            # But if it does, we have a problem.
                            pass
                        elif cnt == 1:
                            # Find the remaining direction
                            # Iterate to find set bit
                            for v_rem in range(n_dirs):
                                if (V_x[x] >> v_rem) & 1:
                                    if not crit[x][v_rem]:
                                        crit[x][v_rem] = True
                                        # Update bad_count for all y on Line(x, v_rem) \ {x}
                                        for y in lines[x][v_rem]:
                                            if y != x:
                                                bad_count[y] += 1
                                    break
                                    # Only one bit is set, so break
                        else:
                            # cnt >= 2, no new criticality
                            pass
                            
        # End of this restart
        if len(S) > best_complement_size:
            best_complement_size = len(S)
            best_S = S.copy()
            
    # Construct N from best_S
    N = []
    for i in range(n_pts):
        if i not in best_S:
            x, y = points[i]
            N.append([list(x), list(y)])
            
    if not N:
        # Fallback to seed program logic if something went wrong
        # But best_S should at least be empty, so N is all points.
        pass
        
    return np.array(N, dtype=np.int64)

# EVOLVE-BLOCK-END
