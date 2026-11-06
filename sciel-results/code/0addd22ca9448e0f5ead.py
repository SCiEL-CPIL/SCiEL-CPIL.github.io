import math
import time

def construct():
    n_max = 256
    
    def is_perfect_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    # 1. Identify self-conflicting nodes (2*x is a square)
    self_conflict = [False] * (n_max + 1)
    for i in range(1, n_max + 1):
        if is_perfect_square(2 * i):
            self_conflict[i] = True

    # 2. Build conflict masks
    conflict_mask = [0] * (n_max + 1)
    for i in range(1, n_max + 1):
        mask = 0
        for j in range(1, n_max + 1):
            if i != j and is_perfect_square(i + j):
                mask |= (1 << j)
        conflict_mask[i] = mask

    valid_nodes = [i for i in range(1, n_max + 1) if not self_conflict[i]]
    valid_mask = 0
    for v in valid_nodes:
        valid_mask |= (1 << v)

    def popcount(x):
        return bin(x).count('1')

    # 3. Greedy initialization for a good lower bound
    def greedy_mis():
        remaining = valid_mask
        selected = 0
        while remaining:
            best_v = None
            min_deg = 1000
            temp = remaining
            while temp:
                lsb = temp & -temp
                u = lsb.bit_length() - 1
                deg = popcount(conflict_mask[u] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    best_v = u
                temp ^= lsb
            
            if best_v is None:
                break
            selected |= (1 << best_v)
            remaining = remaining & ~(1 << best_v) & ~conflict_mask[best_v]
        return selected

    init_mask = greedy_mis()
    best_size = popcount(init_mask)
    best_mask = init_mask

    start_time = time.time()
    TIME_LIMIT = 55.0

    def coloring_bound(P):
        remaining = P
        colors = 0
        while remaining:
            color_class = 0
            temp = remaining
            while temp:
                lsb = temp & -temp
                u = lsb.bit_length() - 1
                color_class |= lsb
                temp = temp & ~lsb & ~conflict_mask[u]
            remaining = remaining & ~color_class
            colors += 1
        return colors

    def branch_and_bound(P, S):
        nonlocal best_size, best_mask
        
        if time.time() - start_time > TIME_LIMIT:
            return

        if P == 0:
            if popcount(S) > best_size:
                best_size = popcount(S)
                best_mask = S
            return
        
        # Simple bound
        if popcount(S) + popcount(P) <= best_size:
            return
            
        # Coloring bound
        ub = coloring_bound(P)
        if popcount(S) + ub <= best_size:
            return

        # Select node v from P with max degree in P
        v = None
        max_deg = -1
        temp = P
        while temp:
            lsb = temp & -temp
            u = lsb.bit_length() - 1
            deg = popcount(conflict_mask[u] & P)
            if deg > max_deg:
                max_deg = deg
                v = u
            temp ^= lsb
            
        if v is None:
            return
            
        v_bit = 1 << v
        
        # Branch 1: Include v
        new_S = S | v_bit
        new_P = P & ~v_bit & ~conflict_mask[v]
        branch_and_bound(new_P, new_S)
        
        # Branch 2: Exclude v
        new_P = P & ~v_bit
        branch_and_bound(new_P, S)

    branch_and_bound(valid_mask, 0)
    
    # Convert best_mask to list
    result = []
    for i in range(1, n_max + 1):
        if best_mask & (1 << i):
            result.append(i)
            
    # Final verification
    def is_valid(s):
        for i in range(len(s)):
            for j in range(i, len(s)):
                if is_perfect_square(s[i] + s[j]):
                    return False
        return True

    if not is_valid(result):
        # Fallback to simple greedy
        result = []
        for i in range(1, n_max + 1):
            if self_conflict[i]: continue
            ok = True
            for j in result:
                if is_perfect_square(i + j):
                    ok = False
                    break
            if ok:
                result.append(i)
                
    return sorted(result)
