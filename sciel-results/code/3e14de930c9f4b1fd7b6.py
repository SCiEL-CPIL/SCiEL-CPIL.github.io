import numpy as np
import time
import random

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
    
    # Build lines
    # Line ID: m*49 + c for non-vertical, 2401 + x0 for vertical
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
    
    # Initial C = Parabola
    in_C = np.zeros(q2, dtype=bool)
    for xa in range(p):
        for xb in range(p):
            ya = (xa * xa + 3 * xb * xb) % p
            yb = (2 * xa * xb) % p
            x_idx = xa * p + xb
            y_idx = ya * p + yb
            pi = x_idx * q + y_idx
            in_C[pi] = True
            
    # Witness lines: W[x] = set of line IDs
    W = [set() for _ in range(q2)]
    for i in range(q2):
        for lid in point_lines[i]:
            lp = line_points[lid]
            valid = True
            for q_idx in lp:
                if q_idx != i and in_C[q_idx]:
                    valid = False
                    break
            if valid:
                W[i].add(lid)
                
    best_in_C = in_C.copy()
    best_size = np.sum(in_C)
    
    rng = random.Random(42)
    
    # Greedy addition to C
    candidates = [i for i in range(q2) if not in_C[i]]
    
    for trial in range(50):
        if time.time() > _DEADLINE - _MARGIN:
            break
            
        in_C = np.zeros(q2, dtype=bool)
        for xa in range(p):
            for xb in range(p):
                ya = (xa * xa + 3 * xb * xb) % p
                yb = (2 * xa * xb) % p
                x_idx = xa * p + xb
                y_idx = ya * p + yb
                pi = x_idx * q + y_idx
                in_C[pi] = True
                
        W = [set() for _ in range(q2)]
        for i in range(q2):
            for lid in point_lines[i]:
                lp = line_points[lid]
                valid = True
                for q_idx in lp:
                    if q_idx != i and in_C[q_idx]:
                        valid = False
                        break
                if valid:
                    W[i].add(lid)
                    
        candidates = [i for i in range(q2) if not in_C[i]]
        rng.shuffle(candidates)
        
        for p_idx in candidates:
            if time.time() > _DEADLINE - _MARGIN:
                break
                
            if in_C[p_idx]:
                continue
                
            # Check if adding p_idx to C is safe
            # A point x loses witness lines that contain p_idx (and p_idx != x)
            # If any x has no other witness lines, it's unsafe.
            
            safe = True
            # We need to check all x.
            # Optimization: Only x that have a witness line containing p_idx are affected.
            # Precompute affected_x? 
            # affected_x = {x | exists L in W[x] such that p_idx in L and p_idx != x}
            
            # For efficiency, iterate through lines containing p_idx
            affected = set()
            for lid in point_lines[p_idx]:
                for x_idx in line_points[lid]:
                    if x_idx != p_idx:
                        affected.add(x_idx)
                        
            for x_idx in affected:
                # Check if x_idx has a witness line NOT containing p_idx
                has_other = False
                for lid in W[x_idx]:
                    if p_idx not in line_points[lid]:
                        has_other = True
                        break
                if not has_other:
                    safe = False
                    break
                    
            if safe:
                in_C[p_idx] = True
                # Update W
                for x_idx in affected:
                    W[x_idx] = {lid for lid in W[x_idx] if p_idx not in line_points[lid]}
        
        if np.sum(in_C) > best_size:
            best_size = np.sum(in_C)
            best_in_C = in_C.copy()
            
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
