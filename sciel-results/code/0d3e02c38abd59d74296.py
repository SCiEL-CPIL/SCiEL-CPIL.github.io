import numpy as np
import time
import random
import heapq

_P = 7
_Q = 49
_Q2 = 2401
_DEADLINE = time.time() + 600.0
_MARGIN = 10.0

def nikodym_set() -> np.ndarray:
    p = _P
    q = _Q
    q2 = _Q2
    
    # Precompute field operations
    mul_table = np.zeros((q, q), dtype=np.int32)
    add_table = np.zeros((q, q), dtype=np.int32)
    sub_table = np.zeros((q, q), dtype=np.int32)
    inv_table = np.zeros(q, dtype=np.int32)
    
    for i in range(q):
        for j in range(q):
            a1, b1 = i // p, i % p
            a2, b2 = j // p, j % p
            ra = (a1 * a2 + 3 * b1 * b2) % p
            rb = (a1 * b2 + a2 * b1) % p
            mul_table[i, j] = ra * p + rb
            
            ra = (a1 + a2) % p
            rb = (b1 + b2) % p
            add_table[i, j] = ra * p + rb
            
            ra = (a1 - a2) % p
            rb = (b1 - b2) % p
            sub_table[i, j] = ra * p + rb
    
    for i in range(1, q):
        a, b = i // p, i % p
        denom = (a * a - 3 * b * b) % p
        denom_inv = pow(denom, 5, 7)
        ia = (a * denom_inv) % p
        ib = (-b * denom_inv) % p
        inv_table[i] = ia * p + ib
    inv_table[0] = 0
    
    # Build all lines
    # Non-vertical: m in 0..48, c in 0..48 -> line_id = m*49 + c (total 2401)
    # Vertical: x0 in 0..48 -> line_id = 2401 + x0 (total 49)
    # Total: 2450 lines
    point_lines = [[] for _ in range(q2)]
    line_points = [set() for _ in range(2450)]
    
    for xi in range(q):
        for yi in range(q):
            pi = xi * q + yi
            vl = 2401 + xi
            point_lines[pi].append(vl)
            line_points[vl].add(pi)
            
            for m in range(q):
                mc = sub_table[yi, mul_table[m, xi]]
                nl = m * q + mc
                point_lines[pi].append(nl)
                line_points[nl].add(pi)
    
    line_points_list = [list(s) for s in line_points]
    
    # Precompute neighbors for faster damage calculation
    # For each point p, and each line L containing p, the other points on L
    # We need to know which witnesses are affected.
    
    def init_empty():
        return np.zeros(q2, dtype=bool)
        
    def init_parabola():
        in_C = np.zeros(q2, dtype=bool)
        for xa in range(p):
            for xb in range(p):
                ya = (xa * xa + 3 * xb * xb) % p
                yb = (2 * xa * xb) % p
                x_idx = xa * p + xb
                y_idx = ya * p + yb
                pi = x_idx * q + y_idx
                in_C[pi] = True
        return in_C

    def compute_witness(in_C):
        """
        Returns a list of sets, W[i] contains line IDs that are valid witnesses for point i.
        A line L is a witness for i if i in L and (L \ {i}) is disjoint from C.
        """
        W = [set() for _ in range(q2)]
        for i in range(q2):
            for lid in point_lines[i]:
                lp = line_points_list[lid]
                valid = True
                for q_idx in lp:
                    if q_idx != i and in_C[q_idx]:
                        valid = False
                        break
                if valid:
                    W[i].add(lid)
        return W

    def is_valid(in_C, W):
        for i in range(q2):
            if len(W[i]) == 0:
                return False
        return True

    def greedy_heap(in_C, W, rng):
        """
        Greedy addition using a heap based on 'damage'.
        Damage of adding p: number of points x for which adding p destroys their only witness.
        """
        in_C = in_C.copy()
        W = [set(s) for s in W]
        wcount = np.array([len(s) for s in W], dtype=np.int32)
        
        if not is_valid(in_C, W):
            return in_C, W, wcount, 0
            
        candidates = [i for i in range(q2) if not in_C[i]]
        added = 0
        
        # Heap of (damage, random_tie_breaker, point_idx)
        # We want min damage. If damage is 0, it's safe.
        
        while candidates and time.time() < _DEADLINE - _MARGIN:
            # Calculate damage for a sample of candidates or all?
            # For q2=2401, calculating damage for all is O(2401 * 49 * 49) ~ 5.7M ops.
            # This is fast enough to do every step? Maybe not.
            # Let's sample 200 candidates.
            
            sample_size = min(len(candidates), 200)
            sample = rng.sample(candidates, sample_size)
            
            best_p = -1
            best_damage = float('inf')
            best_tie = float('inf')
            
            for p_idx in sample:
                if in_C[p_idx]:
                    continue
                
                damage = 0
                # For each line containing p_idx
                for lid in point_lines[p_idx]:
                    # For each other point x on this line
                    for x_idx in line_points_list[lid]:
                        if x_idx == p_idx:
                            continue
                        # If this line is a witness for x_idx and it's the ONLY witness
                        if lid in W[x_idx] and wcount[x_idx] == 1:
                            damage += 1
                            
                tie = rng.random()
                if damage < best_damage or (damage == best_damage and tie < best_tie):
                    best_damage = damage
                    best_p = p_idx
                    best_tie = tie
            
            if best_p == -1 or best_damage > 0:
                # If we can't find a safe point, break
                break
                
            # Add best_p
            in_C[best_p] = True
            added += 1
            candidates.remove(best_p)
            
            # Update W and wcount
            for lid in point_lines[best_p]:
                for x_idx in line_points_list[lid]:
                    if x_idx == best_p:
                        continue
                    if lid in W[x_idx]:
                        W[x_idx].discard(lid)
                        wcount[x_idx] -= 1
        
        return in_C, W, wcount, added

    def local_search_swap(in_C, W, wcount, rng, max_iters=5000):
        """
        Try to swap a point in C with a point not in C to increase size or maintain size but improve structure.
        Actually, we just want to increase size. If stuck, try removing a point to unblock, then re-add.
        This is complex. Let's stick to greedy restarts.
        """
        pass

    best_in_C = None
    best_size = 0
    
    rng = random.Random(42)
    
    # Initializers
    initializers = [
        init_empty,
        init_parabola,
        lambda: init_empty(), # Just empty, greedy will build it
    ]
    
    for trial in range(100):
        if time.time() > _DEADLINE - _MARGIN:
            break
        
        # Vary the seed for random tie-breaking
        rng.seed(42 + trial)
        
        in_C = initializers[trial % len(initializers)]()
        
        W = compute_witness(in_C)
        if not is_valid(in_C, W):
            continue
            
        wcount = np.array([len(s) for s in W], dtype=np.int32)
        
        in_C, W, wcount, added = greedy_heap(in_C, W, rng)
        
        cur_size = int(np.sum(in_C))
        if cur_size > best_size:
            best_size = cur_size
            best_in_C = in_C.copy()
            
    # If no valid solution found, fall back to empty set (N = all points)
    if best_in_C is None:
        best_in_C = init_empty()
    
    # Construct N = complement of C
    in_N = ~best_in_C
    
    N = []
    for i in range(q2):
        if in_N[i]:
            x = i // q
            y = i % q
            xa, xb = x // p, x % p
            ya, yb = y // p, y % p
            N.append([[xa, xb], [ya, yb]])
    
    return np.array(N, dtype=np.int64)
