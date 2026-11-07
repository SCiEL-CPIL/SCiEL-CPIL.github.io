import time
import random
import math

NMAX = 256
RANGE = list(range(2, NMAX + 1))
N = len(RANGE)
val_to_idx = {v: i for i, v in enumerate(RANGE)}
idx_to_val = RANGE

# Precompute valid_c_mask[a_idx][b_idx] = bitmask of c in [2..256] such that a | b*c
valid_c_mask = [[0] * N for _ in range(N)]
for i in range(N):
    a = idx_to_val[i]
    for j in range(N):
        b = idx_to_val[j]
        mask = 0
        g = math.gcd(a, b)
        req = a // g
        for c_val in range(2, NMAX + 1):
            if c_val % req == 0:
                mask |= (1 << val_to_idx[c_val])
        valid_c_mask[i][j] = mask

def get_degrees(S_mask):
    """Compute degree of each vertex: number of pairs (b,c) in S\{a} such that a|b*c."""
    degrees = [0] * N
    a_list = [i for i in range(N) if (S_mask >> i) & 1]
    for i in a_list:
        deg = 0
        for j in a_list:
            if j == i:
                continue
            count_mask = S_mask & valid_c_mask[i][j] & ~(1 << i) & ~(1 << j)
            deg += count_mask.bit_count()
        degrees[i] = deg
    return degrees

def can_add_to_set(x, S_mask):
    """Check if x can be added to the set represented by S_mask without creating conflicts."""
    x_idx = val_to_idx[x]
    S_list = [i for i in range(N) if (S_mask >> i) & 1]
    
    # Check 1: x | b*c for any b,c in S
    for b_idx in S_list:
        b = idx_to_val[b_idx]
        g = math.gcd(x, b)
        req = x // g
        # Need c in S, c != b, such that c % req == 0
        # Use valid_c_mask in reverse: we need to check if there exists c in S such that x|b*c
        # This is equivalent to: (x/gcd(x,b)) | c
        # We can use a precomputed approach or just check
        for c_idx in S_list:
            if c_idx == b_idx:
                continue
            c = idx_to_val[c_idx]
            if c % req == 0:
                return False
    
    # Check 2: b | x*c for any b,c in S
    for b_idx in S_list:
        b = idx_to_val[b_idx]
        g = math.gcd(b, x)
        req = b // g
        for c_idx in S_list:
            if c_idx == b_idx:
                continue
            c = idx_to_val[c_idx]
            if c % req == 0:
                return False
    
    # Check 3: c | x*b is same as Check 2 with b and c swapped, already covered
    return True

def greedy_removal(deadline, tie_breaker='largest', seed=42):
    S_mask = (1 << N) - 1
    degrees = get_degrees(S_mask)
    rng = random.Random(seed)
    
    while time.time() < deadline:
        max_deg = -1
        best_idx = -1
        for i in range(N):
            if not (S_mask >> i) & 1:
                continue
            deg = degrees[i]
            if deg > max_deg:
                max_deg = deg
                best_idx = i
            elif deg == max_deg:
                if best_idx == -1:
                    best_idx = i
                else:
                    val_i = idx_to_val[i]
                    val_best = idx_to_val[best_idx]
                    if tie_breaker == 'largest':
                        if val_i > val_best:
                            best_idx = i
                    elif tie_breaker == 'smallest':
                        if val_i < val_best:
                            best_idx = i
                    else:
                        if rng.random() < 0.5:
                            best_idx = i
        
        if best_idx == -1 or max_deg == 0:
            break
        
        S_mask &= ~(1 << best_idx)
        degrees = get_degrees(S_mask)
    
    res = [idx_to_val[i] for i in range(N) if (S_mask >> i) & 1]
    return res

def local_search(S, deadline, seed=42):
    """Improve S by trying to add removed elements or swap."""
    S_set = set(S)
    S_mask = 0
    for v in S:
        S_mask |= (1 << val_to_idx[v])
    
    best_S = S[:]
    best_size = len(S)
    rng = random.Random(seed)
    
    all_elements = set(RANGE)
    removed = list(all_elements - S_set)
    
    # Try to add elements
    max_rounds = 200
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        
        improved = False
        rng.shuffle(removed)
        
        for x in removed:
            if time.time() > deadline:
                break
            
            # Check if x can be added directly
            if can_add_to_set(x, S_mask):
                S_set.add(x)
                S_mask |= (1 << val_to_idx[x])
                removed.remove(x)
                if len(S_set) > best_size:
                    best_S = list(S_set)
                    best_size = len(S_set)
                improved = True
                break
        
        if not improved:
            # Try swaps: for each x in removed, try swapping with a low-degree element in S
            found_swap = False
            rng.shuffle(removed)
            for x in removed:
                if time.time() > deadline:
                    break
                
                x_idx = val_to_idx[x]
                S_list = [v for v in S_set]
                
                # Try swapping x with each y in S (prioritize low-degree)
                # Compute degrees for quick sorting
                degrees = get_degrees(S_mask)
                S_sorted = sorted(S_list, key=lambda v: degrees[val_to_idx[v]])
                
                for y in S_sorted:
                    if time.time() > deadline:
                        break
                    
                    y_idx = val_to_idx[y]
                    # Try S - {y} + {x}
                    S_temp_mask = S_mask & ~(1 << y_idx)
                    
                    if can_add_to_set(x, S_temp_mask):
                        S_set.remove(y)
                        S_set.add(x)
                        S_mask = S_temp_mask | (1 << x_idx)
                        removed.remove(x)
                        removed.append(y)
                        found_swap = True
                        break
                
                if found_swap:
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_S

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best = []
    best_size = 0
    
    # Run greedy removal with different tie breakers and seeds
    configs = [
        ('largest', 42), ('smallest', 42), ('random', 42),
        ('largest', 123), ('smallest', 123), ('random', 123),
        ('largest', 999), ('smallest', 999),
    ]
    
    for tb, seed in configs:
        if time.time() > deadline - 15:
            break
        c = greedy_removal(deadline - 15, tb, seed)
        if len(c) > best_size:
            best = c
            best_size = len(c)
    
    # Local search on best
    if time.time() < deadline - 5:
        c = local_search(best, deadline - 5, seed=42)
        if len(c) > best_size:
            best = c
            best_size = len(c)
    
    # Additional local search rounds with different seeds
    if time.time() < deadline - 2:
        c = local_search(best, deadline - 2, seed=777)
        if len(c) > best_size:
            best = c
            best_size = len(c)
    
    # Final verification
    res = list(best)
    n = len(res)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for k in range(n):
                if i == k or j == k:
                    continue
                if (res[j] * res[k]) % res[i] == 0:
                    return [256]
    return res
