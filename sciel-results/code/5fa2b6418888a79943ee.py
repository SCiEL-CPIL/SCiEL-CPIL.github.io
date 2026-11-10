import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 by maximizing the complement S.
    Condition: For every x in S, there exists a direction d such that 
    the line through x in direction d contains no other points from S.
    """
    start_time = time.time()
    deadline = 590.0  # Safety margin

    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3

    # Precompute field operations
    def idx(a, b):
        return a + b * p

    def a_of(i):
        return i % p

    def b_of(i):
        return i // p

    # Multiplication table
    mul_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        ai, bi = a_of(i), b_of(i)
        for j in range(q):
            aj, bj = a_of(j), b_of(j)
            res_a = (ai * aj + bi * bj * k) % p
            res_b = (ai * bj + bi * aj) % p
            mul_table[i, j] = idx(res_a, res_b)

    # Addition table
    add_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        ai, bi = a_of(i), b_of(i)
        for j in range(q):
            aj, bj = a_of(j), b_of(j)
            res_a = (ai + aj) % p
            res_b = (bi + bj) % p
            add_table[i, j] = idx(res_a, res_b)

    # Negation table
    neg_table = np.zeros(q, dtype=np.int32)
    for i in range(q):
        ai, bi = a_of(i), b_of(i)
        neg_a = (-ai) % p
        neg_b = (-bi) % p
        neg_table[i] = idx(neg_a, neg_b)

    total_points = q * q  # 2401
    num_dirs = 50  # 49 directions (1, m) + 1 direction (0, 1)

    # Precompute line membership: 
    # line_dir[d, x] is the set of points on the line through x in direction d.
    # Instead of storing sets, we can precompute for each direction d and point x,
    # the list of other points on the line.
    # But for the greedy algorithm, we need to update blocked directions for all other points.
    # It's better to precompute, for each pair (x, y) with x != y, the direction d 
    # such that y is on the line through x in direction d.
    # There is exactly one such direction for each pair (x, y).
    # Let's call this dir[x, y].
    
    # dir[x, y] = d if y is on the line through x in direction d.
    # Note: x and y are indices 0..2400.
    # We can compute this on the fly or precompute a 2401x2401 table.
    # 2401^2 = ~5.7 million entries. Fits in memory (int32).
    
    dir_table = np.zeros((total_points, total_points), dtype=np.int32)
    # Initialize to -1
    dir_table.fill(-1)
    
    for x_idx in range(total_points):
        x = x_idx // q
        y = x_idx % q
        for d in range(num_dirs):
            # Compute the 48 other points on the line
            if d < 49:
                m = d
                for t in range(1, q):
                    nx = add_table[x, t]
                    ny = add_table[y, mul_table[t, m]]
                    n_idx = nx * q + ny
                    dir_table[x_idx, n_idx] = d
                    dir_table[n_idx, x_idx] = d # Symmetric? 
                    # The direction from n to x is the same line, but the "direction vector" 
                    # definition in the problem is specific.
                    # The problem says: line {x + t*v : t in F_49}.
                    # If y = x + t*v, then x = y + (-t)*v.
                    # So the direction vector v is the same.
                    # So yes, symmetric in terms of the line set.
            else:
                # Direction (0, 1)
                for t in range(1, q):
                    nx = x
                    ny = add_table[y, t]
                    n_idx = nx * q + ny
                    dir_table[x_idx, n_idx] = d
                    dir_table[n_idx, x_idx] = d

    # Greedy construction of S
    # blocked[x] is a boolean array of size num_dirs.
    # blocked[x, d] is True if there is some y in S such that y is on line_d(x).
    
    best_S_size = 0
    best_S = set()
    
    # Try multiple random orderings
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() - start_time > deadline - 10:
            break
            
        np.random.seed(r)
        order = np.random.permutation(total_points)
        
        S = set()
        blocked = np.zeros((total_points, num_dirs), dtype=bool)
        
        for x_idx in order:
            if time.time() - start_time > deadline - 5:
                break
                
            # Check if x can be added
            # It can be added if there is at least one direction d where blocked[x_idx, d] is False.
            if not np.all(blocked[x_idx]):
                S.add(x_idx)
                # Update blocked for all other points y
                # For each y, if dir_table[y, x_idx] is valid, then that direction is blocked for y.
                # We can iterate over all y? That's 2401 iterations per addition.
                # Total additions ~ 200. 200 * 2400 = 480,000 ops. Very fast.
                for y_idx in range(total_points):
                    if y_idx == x_idx:
                        continue
                    d = dir_table[y_idx, x_idx]
                    if d != -1:
                        blocked[y_idx, d] = True
        
        if len(S) > best_S_size:
            best_S_size = len(S)
            best_S = set(S)
            
    # Construct N from best_S
    N_mask = np.ones(total_points, dtype=bool)
    for s in best_S:
        N_mask[s] = False
        
    points = []
    for i in range(total_points):
        if i not in best_S:
            x = i // q
            y = i % q
            xa, xb = a_of(x), b_of(x)
            ya, yb = a_of(y), b_of(y)
            points.append([[xa, xb], [ya, yb]])
            
    result = np.array(points, dtype=np.int64)
    return result
