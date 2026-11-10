import numpy as np
import time
import random

_P = 7
_Q = 49
_Q2 = 2401
_DEADLINE = time.time() + 600.0
_MARGIN = 15.0

def nikodym_set() -> np.ndarray:
    p = _P
    q = _Q
    q2 = _Q2
    
    # Field F_49 = F_7[alpha], alpha^2 = 3
    # Element index = a * 7 + b for a + b*alpha
    # Point index = x_idx * 49 + y_idx
    
    # Precompute field multiplication table
    mul_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        for j in range(q):
            a1, b1 = i // p, i % p
            a2, b2 = j // p, j % p
            ra = (a1 * a2 + 3 * b1 * b2) % p
            rb = (a1 * b2 + a2 * b1) % p
            mul_table[i, j] = ra * p + rb
    
    # Precompute field addition table
    add_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        for j in range(q):
            a1, b1 = i // p, i % p
            a2, b2 = j // p, j % p
            ra = (a1 + a2) % p
            rb = (b1 + b2) % p
            add_table[i, j] = ra * p + rb
    
    # Precompute field subtraction table
    sub_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        for j in range(q):
            a1, b1 = i // p, i % p
            a2, b2 = j // p, j % p
            ra = (a1 - a2) % p
            rb = (b1 - b2) % p
            sub_table[i, j] = ra * p + rb
    
    # Precompute field inverse table
    inv_table = np.zeros(q, dtype=np.int32)
    for i in range(1, q):
        a, b = i // p, i % p
        denom = (a * a - 3 * b * b) % p
        denom_inv = pow(denom, 5, 7)
        ia = (a * denom_inv) % p
        ib = (-b * denom_inv) % p
        inv_table[i] = ia * p + ib
    inv_table[0] = 0
    
    # Build all lines
    # Line IDs:
    # Non-vertical: m in 0..48, c in 0..48 -> line_id = m*49 + c (total 2401)
    # Vertical: x0 in 0..48 -> line_id = 2401 + x0 (total 49)
    # Total: 2450 lines
    
    point_lines = [[] for _ in range(q2)]
    line_points = [set() for _ in range(2450)]
    
    for xi in range(q):
        for yi in range(q):
            pi = xi * q + yi
            # Vertical line: x = xi
            vl = 2401 + xi
            point_lines[pi].append(vl)
            line_points[vl].add(pi)
            
            # Non-vertical lines: for each m, c = yi - m*xi
            for m in range(q):
                mc = sub_table[yi, mul_table[m, xi]]
                nl = m * q + mc
                point_lines[pi].append(nl)
                line_points[nl].add(pi)
    
    # Start with parabola removed
    in_N = np.ones(q2, dtype=bool)
    
    for xa in range(p):
        for xb in range(p):
            ya = (xa * xa + 3 * xb * xb) % p
            yb = (2 * xa * xb) % p
            x_idx = xa * p + xb
            y_idx = ya * p + yb
            pi = x_idx * q + y_idx
            in_N[pi] = False
    
    # Compute initial witness lines
    witness_lines = [set() for _ in range(q2)]
    for i in range(q2):
        for lid in point_lines[i]:
            lp = line_points[lid]
            valid = True
            for q_idx in lp:
                if q_idx != i and not in_N[q_idx]:
                    valid = False
                    break
            if valid:
                witness_lines[i].add(lid)
    
    # Precompute line_through for efficiency
    # line_through[x, p] = line ID through points x and p (x != p)
    # This is a 2401x2401 table, which is about 5.7M entries. Too large.
    # Instead, compute on the fly.
    
    def get_line_through(x_idx, p_idx):
        """Get the line ID through points x_idx and p_idx (x_idx != p_idx)."""
        x1 = x_idx // q
        y1 = x_idx % q
        x2 = p_idx // q
        y2 = p_idx % q
        
        if x1 == x2:
            return 2401 + x1
        
        dx = sub_table[x2, x1]
        dy = sub_table[y2, y1]
        dx_inv = inv_table[dx]
        m = mul_table[dy, dx_inv]
        c = sub_table[y1, mul_table[m, x1]]
        return m * q + c
    
    # Greedy removal with multiple passes
    rng = random.Random(42)
    
    # Try multiple random orders
    best_in_N = in_N.copy()
    best_removed = 49
    
    for trial in range(20):
        in_N = np.ones(q2, dtype=bool)
        
        # Remove parabola
        for xa in range(p):
            for xb in range(p):
                ya = (xa * xa + 3 * xb * xb) % p
                yb = (2 * xa * xb) % p
                x_idx = xa * p + xb
                y_idx = ya * p + yb
                pi = x_idx * q + y_idx
                in_N[pi] = False
        
        # Recompute witness lines
        witness_lines = [set() for _ in range(q2)]
        for i in range(q2):
            for lid in point_lines[i]:
                lp = line_points[lid]
                valid = True
                for q_idx in lp:
                    if q_idx != i and not in_N[q_idx]:
                        valid = False
                        break
                if valid:
                    witness_lines[i].add(lid)
        
        removed_count = 49
        candidates = [i for i in range(q2) if in_N[i]]
        rng.shuffle(candidates)
        
        # Multiple passes
        for pass_num in range(5):
            if time.time() > _DEADLINE - _MARGIN:
                break
            
            progress = False
            for p_idx in candidates:
                if time.time() > _DEADLINE - _MARGIN:
                    break
                if not in_N[p_idx]:
                    continue
                
                # Check if removing p_idx is safe
                safe = True
                for lid in point_lines[p_idx]:
                    for x_idx in line_points[lid]:
                        if x_idx == p_idx:
                            continue
                        if lid in witness_lines[x_idx] and len(witness_lines[x_idx]) == 1:
                            safe = False
                            break
                    if not safe:
                        break
                
                if safe:
                    # Remove p_idx
                    in_N[p_idx] = False
                    removed_count += 1
                    progress = True
                    # Update witness lines
                    for lid in point_lines[p_idx]:
                        for x_idx in line_points[lid]:
                            if x_idx == p_idx:
                                continue
                            if lid in witness_lines[x_idx]:
                                witness_lines[x_idx].discard(lid)
            
            if not progress:
                break
        
        if removed_count > best_removed:
            best_removed = removed_count
            best_in_N = in_N.copy()
        
        if time.time() > _DEADLINE - _MARGIN:
            break
    
    # Use the best result found
    in_N = best_in_N
    
    # Build result
    N = []
    for i in range(q2):
        if in_N[i]:
            x = i // q
            y = i % q
            xa, xb = x // p, x % p
            ya, yb = y // p, y % p
            N.append([[xa, xb], [ya, yb]])
    
    return np.array(N, dtype=np.int64)
