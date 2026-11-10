import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    TIMEOUT = 85.0  # Safety margin for 90s limit
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute LCM
    def lcm(a, b):
        return a * b // math.gcd(a, b)
        
    # Find all bad triples
    bad_triples = []
    for a, b, c in combinations(numbers, 3):
        if lcm(a, b) == lcm(a, c) == lcm(b, c):
            bad_triples.append((a, b, c))
            
    # Build conflict masks
    # conflict_mask[i][j] is a bitmask of k such that {i, j, k} is a bad triple
    conflict_mask = [[0] * N for _ in range(N)]
    
    for a, b, c in bad_triples:
        ia, ib, ic = a - 1, b - 1, c - 1
        conflict_mask[ia][ib] |= (1 << ic)
        conflict_mask[ib][ia] |= (1 << ic)
        
        conflict_mask[ia][ic] |= (1 << ib)
        conflict_mask[ic][ia] |= (1 << ib)
        
        conflict_mask[ib][ic] |= (1 << ia)
        conflict_mask[ic][ib] |= (1 << ia)
        
    # Greedy initial solution
    def greedy_solution():
        selected = []
        selected_mask = 0
        for x in numbers:
            ix = x - 1
            # Check if x conflicts with any pair in selected
            # For all u in selected, (conflict_mask[ix][u] & selected_mask) must be 0
            valid = True
            temp_mask = selected_mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                u_idx = lsb.bit_length() - 1
                if (conflict_mask[ix][u_idx] & selected_mask):
                    valid = False
                    break
                temp_mask ^= lsb
                
            if valid:
                selected.append(x)
                selected_mask |= (1 << ix)
        return selected, selected_mask, len(selected)
        
    best_set, best_mask, best_size = greedy_solution()
    
    import sys
    sys.setrecursionlimit(10000)
    
    def popcount(x):
        return bin(x).count('1')
        
    def solve(remaining_mask, current_mask, current_size):
        nonlocal best_size, best_set, best_mask
        nonlocal TIMEOUT, start_time
        
        if time.time() - start_time > TIMEOUT:
            return
            
        if remaining_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
                best_set = [i + 1 for i in range(N) if (best_mask >> i) & 1]
            return
            
        # Pruning
        upper_bound = current_size + popcount(remaining_mask)
        if upper_bound <= best_size:
            return
            
        # Choose a vertex to branch on (smallest index in remaining)
        lsb = remaining_mask & -remaining_mask
        v_idx = lsb.bit_length() - 1
        
        # Check compatibility of v with current_mask
        # Calculate union of conflicts of v with all u in current_mask
        conflict_with_v = 0
        temp_current = current_mask
        while temp_current:
            lsb_u = temp_current & -temp_current
            u_idx = lsb_u.bit_length() - 1
            conflict_with_v |= conflict_mask[v_idx][u_idx]
            temp_current ^= lsb_u
            
        # If any conflict is in current_mask, v is incompatible
        if (conflict_with_v & current_mask) == 0:
            # Branch 1: Include v
            new_remaining = remaining_mask & ~lsb
            new_remaining &= ~conflict_with_v
            
            solve(new_remaining, current_mask | lsb, current_size + 1)
            
        # Branch 2: Exclude v
        solve(remaining_mask & ~lsb, current_mask, current_size)
        
    # Initial call
    initial_remaining = (1 << N) - 1
    
    solve(initial_remaining, 0, 0)
    
    if not best_set:
        return list(range(1, N+1))
        
    return best_set
