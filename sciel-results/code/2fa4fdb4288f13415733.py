import time
import random
import math

NMAX = 256
RANGE = list(range(2, NMAX + 1))
N = len(RANGE)
val_to_idx = {v: i for i, v in enumerate(RANGE)}
idx_to_val = RANGE

# Precompute: for each pair (a_idx, b_idx), the bitmask of c values where a | b*c
# This lets us quickly count conflicts
valid_c_mask = [[0] * N for _ in range(N)]
for i in range(N):
    a = idx_to_val[i]
    for j in range(N):
        b = idx_to_val[j]
        mask = 0
        g = math.gcd(a, b)
        req = a // g
        if req == 1:
            # a | b*c for all c
            mask = (1 << N) - 1
        else:
            for c_val in range(2, NMAX + 1):
                if c_val % req == 0:
                    mask |= (1 << val_to_idx[c_val])
        valid_c_mask[i][j] = mask

def get_conflict_degrees(S_mask):
    """
    For each vertex a in S, compute the number of distinct pairs (b,c) in S\{a}
    with b != c such that a | b*c.
    We count each unordered pair {b,c} once for each a that conflicts.
    """
    degrees = [0] * N
    a_list = [i for i in range(N) if (S_mask >> i) & 1]
    
    for i in a_list:
        deg = 0
        for j in a_list:
            if j == i:
                continue
            # Count c in S, c != i, c != j, such that a | b*c
            count_mask = S_mask & valid_c_mask[i][j] & ~(1 << i) & ~(1 << j)
            deg += count_mask.bit_count()
        # Each pair (b,c) is counted twice: once as (b,c) and once as (c,b)
        # So divide by 2 to get number of unordered pairs
        degrees[i] = deg // 2
    return degrees

def get_weighted_degrees(S_mask):
    """
    Weighted degree: for each a, sum over conflicting pairs (b,c) of 1/min(quotient_b, quotient_c)
    where quotient_b = b / gcd(a,b) and quotient_c = c / gcd(a,c).
    Smaller quotients mean tighter constraints, so weight them more.
    """
    degrees = [0.0] * N
    a_list = [i for i in range(N) if (S_mask >> i) & 1]
    
    for i in a_list:
        a = idx_to_val[i]
        deg = 0.0
        for j in a_list:
            if j == i:
                continue
            b = idx_to_val[j]
            g_ab = math.gcd(a, b)
            req_b = a // g_ab
            # For each c in S, c != i, c != j, where a | b*c
            count_mask = S_mask & valid_c_mask[i][j] & ~(1 << i) & ~(1 << j)
            for k in a_list:
                if k == i or k == j:
                    continue
                if (count_mask >> k) & 1:
                    c = idx_to_val[k]
                    g_ac = math.gcd(a, c)
                    req_c = a // g_ac
                    # Weight: tighter constraints (smaller req) get higher weight
                    weight = 1.0 / max(1, min(req_b, req_c))
                    deg += weight
        degrees[i] = deg / 2.0  # Each pair counted twice
    return degrees

def can_add_to_set(x, S_mask):
    """Check if x can be added to the set represented by S_mask without creating conflicts."""
    x_idx = val_to_idx[x]
    S_list = [i for i in range(N) if (S_mask >> i) & 1]
    
    # Check 1: x | b*c for any b,c in S, b != c
    for b_idx in S_list:
        b = idx_to_val[b_idx]
        g = math.gcd(x, b)
        req = x // g
        if req == 1:
            # x | b*c for all c, so need at least one c != b in S
            if len(S_list) > 1:
                return False
        else:
            for c_idx in S_list:
                if c_idx == b_idx:
                    continue
                c = idx_to_val[c_idx]
                if c % req == 0:
                    return False
    
    # Check 2: b | x*c for any b,c in S, b != c
    for b_idx in S_list:
        b = idx_to_val[b_idx]
        g = math.gcd(b, x)
        req = b // g
        if req == 1:
            if len(S_list) > 1:
                return False
        else:
            for c_idx in S_list:
                if c_idx == b_idx:
                    continue
                c = idx_to_val[c_idx]
                if c % req == 0:
                    return False
    
    return True

def greedy_removal(deadline, tie_breaker='largest', seed=42, use_weighted=False):
    S_mask = (1 << N) - 1
    if use_weighted:
        degrees = get_weighted_degrees(S_mask)
    else:
        degrees = get_conflict_degrees(S_mask)
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
        if use_weighted:
            degrees = get_weighted_degrees(S_mask)
        else:
            degrees = get_conflict_degrees(S_mask)
    
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
    
    max_rounds = 300
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        
        improved = False
        rng.shuffle(removed)
        
        # Try to add elements
        for x in removed:
            if time.time() > deadline:
                break
            
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
            # Try 1-opt swaps
            found_swap = False
            rng.shuffle(removed)
            for x in removed:
                if time.time() > deadline:
                    break
                
                x_idx = val_to_idx[x]
                degrees = get_conflict_degrees(S_mask)
                S_sorted = sorted(list(S_set), key=lambda v: degrees[val_to_idx[v]])
                
                for y in S_sorted:
                    if time.time() > deadline:
                        break
                    
                    y_idx = val_to_idx[y]
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
            # Try 2-opt swaps: remove two, add two
            found_2swap = False
            if len(removed) >= 2 and len(S_set) >= 2:
                rng.shuffle(removed)
                rng.shuffle(list(S_set))
                S_list = list(S_set)
                
                for idx1 in range(min(len(removed), 20)):
                    if time.time() > deadline:
                        break
                    x1 = removed[idx1]
                    for idx2 in range(idx1 + 1, min(len(removed), 20)):
                        if time.time() > deadline:
                            break
                        x2 = removed[idx2]
                        
                        for y1_idx in range(min(len(S_list), 15)):
                            if time.time() > deadline:
                                break
                            y1 = S_list[y1_idx]
                            for y2_idx in range(y1_idx + 1, min(len(S_list), 15)):
                                if time.time() > deadline:
                                    break
                                y2 = S_list[y2_idx]
                                
                                if y1 == y2 or x1 == x2:
                                    continue
                                
                                y1_idx_val = val_to_idx[y1]
                                y2_idx_val = val_to_idx[y2]
                                x1_idx_val = val_to_idx[x1]
                                x2_idx_val = val_to_idx[x2]
                                
                                S_temp_mask = S_mask & ~(1 << y1_idx_val) & ~(1 << y2_idx_val)
                                
                                # Check if both x1 and x2 can be added
                                temp_S = S_set.copy()
                                temp_S.discard(y1)
                                temp_S.discard(y2)
                                temp_mask = S_mask & ~(1 << y1_idx_val) & ~(1 << y2_idx_val)
                                
                                if can_add_to_set(x1, temp_mask):
                                    temp_mask |= (1 << x1_idx_val)
                                    if can_add_to_set(x2, temp_mask):
                                        # Valid 2-opt swap
                                        S_set.discard(y1)
                                        S_set.discard(y2)
                                        S_set.add(x1)
                                        S_set.add(x2)
                                        S_mask = temp_mask | (1 << x2_idx_val)
                                        removed.remove(x1)
                                        removed.remove(x2)
                                        removed.append(y1)
                                        removed.append(y2)
                                        found_2swap = True
                                        break
                            if found_2swap:
                                break
                        if found_2swap:
                            break
                    if found_2swap:
                        break
                
                if found_2swap:
                    improved = True
        
        if not improved:
            break
    
    return best_S

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best = []
    best_size = 0
    
    # Run greedy removal with different strategies
    configs = [
        ('largest', 42, False), ('smallest', 42, False), ('random', 42, False),
        ('largest', 123, False), ('smallest', 123, False), ('random', 123, False),
        ('largest', 999, False), ('smallest', 999, False),
        ('largest', 42, True), ('smallest', 42, True), ('random', 42, True),
        ('largest', 777, True), ('smallest', 777, True),
    ]
    
    for tb, seed, use_weighted in configs:
        if time.time() > deadline - 20:
            break
        c = greedy_removal(deadline - 20, tb, seed, use_weighted)
        if len(c) > best_size:
            best = c
            best_size = len(c)
    
    # Local search on best
    if time.time() < deadline - 10:
        c = local_search(best, deadline - 10, seed=42)
        if len(c) > best_size:
            best = c
            best_size = len(c)
    
    # Additional local search rounds with different seeds
    if time.time() < deadline - 5:
        c = local_search(best, deadline - 5, seed=777)
        if len(c) > best_size:
            best = c
            best_size = len(c)
    
    if time.time() < deadline - 2:
        c = local_search(best, deadline - 2, seed=1234)
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
