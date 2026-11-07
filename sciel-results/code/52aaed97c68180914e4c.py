import time
import sys
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 87.0  # Safety margin: 87s out of 90s limit
    
    N = 256
    candidates = list(range(2, N + 1))
    num_candidates = len(candidates)
    
    # Precompute prime factorization
    factors = [None] * num_candidates
    for i, v in enumerate(candidates):
        x = v
        f = {}
        d = 2
        while d * d <= x:
            while x % d == 0:
                f[d] = f.get(d, 0) + 1
                x //= d
            d += 1
        if x > 1:
            f[x] = f.get(x, 0) + 1
        factors[i] = f

    # Precompute conflict matrix
    # conflict[i][j] = True if candidates[i] divides candidates[j] * candidates[k] for some k?
    # No, the constraint is for triples. 
    # Let's precompute a matrix "divides_prod" where divides_prod[i][j][k] is True if i | j*k.
    # This is 255^3 ~ 16M booleans. Memory: 16M bytes if bitset, or 16M * 1 byte in Python list of lists of lists? 
    # Better: Precompute for each pair (j, k), the set of i that divide j*k.
    # Or, for each i, the set of pairs (j, k) such that i | j*k.
    
    # Precompute conflict using bitmasks for speed
    # conflict_mask[i] is a bitmask where bit (j*num_candidates + k) is set if i | candidates[j]*candidates[k]
    # Since num_candidates=255, we need 255*255=65025 bits = ~8KB per mask, total ~2MB. Fine.
    
    # Actually, let's use a simpler approach: precompute a 2D boolean array for each i
    # conflict[i][j][k] = True if i | j*k, but that's 255^3 booleans which is too much in Python.
    
    # Better: For each i, store a set of (j,k) pairs where j<k and i|j*k.
    # But for fast lookup, let's precompute for each i a 2D list conflict_2d[i][j] = set of k such that i|j*k
    
    # Even better: precompute a flat list and use numpy for speed
    import numpy as np
    
    # Precompute divisibility: div[a][b] = True if a divides b
    # a ranges over candidates (2..256), b ranges over products (up to 256*255=65280)
    # But products can be large. Instead, for each pair (j,k), compute prod and check which i divide it.
    
    # Let's use a different approach: precompute for each value v in [2,256], which values w in [2,256] 
    # have the property that v | w*x for some x? No, that's not right.
    
    # Simplest fast approach: precompute a 3D structure using lists of sets
    # conflict[i] = set of (j,k) with j<k such that candidates[i] | candidates[j]*candidates[k]
    
    conflict = [set() for _ in range(num_candidates)]
    
    # For each pair (j, k), find all i such that candidates[i] | candidates[j]*candidates[k]
    for j in range(num_candidates):
        for k in range(j + 1, num_candidates):
            prod = candidates[j] * candidates[k]
            for i in range(num_candidates):
                if prod % candidates[i] == 0:
                    conflict[i].add((j, k))
    
    # For fast can_add, precompute a lookup: for each i, a 2D boolean grid conflict_grid[i][j][k]
    # But 255^3 is too much. Instead, use the set-based approach but optimize can_add.
    
    # Precompute a faster structure: for each i, conflict_adj[i] is a list of length num_candidates
    # where conflict_adj[i][j] is a bitmask (as int) of k values such that (j,k) in conflict[i]
    # Since k < 255, we can use Python's arbitrary precision integers as bitmasks.
    
    conflict_adj = [0] * num_candidates  # Will be list of lists of ints
    conflict_adj = [[] for _ in range(num_candidates)]
    for i in range(num_candidates):
        conflict_adj[i] = [0] * num_candidates  # Each is a bitmask of k values
    
    for i in range(num_candidates):
        for (j, k) in conflict[i]:
            # Set bit k in conflict_adj[i][j]
            conflict_adj[i][j] |= (1 << k)
            # Also set bit j in conflict_adj[i][k] for symmetry? No, (j,k) and (k,j) are different
            # But in our check, we always use j<k, so we only need one direction.
            # However, when checking if (a,b) is in conflict[i], we need to check both (a,b) and (b,a)
            # Let's store both to simplify lookup.
            conflict_adj[i][k] |= (1 << j)
    
    def can_add(idx_new, S_indices):
        """Check if idx_new can be added to S_indices without violating constraints."""
        n = len(S_indices)
        # For each pair (i, j) in S, check three conditions:
        # 1. idx_new | i*j: check if (i,j) in conflict[idx_new]
        # 2. i | idx_new*j: check if (idx_new,j) in conflict[i]
        # 3. j | idx_new*i: check if (idx_new,i) in conflict[j]
        
        for a_idx in range(n):
            a = S_indices[a_idx]
            # Check condition 2 and 3 for all b in S where b > a to avoid double counting? 
            # No, we need all pairs. Let's iterate all pairs.
            for b_idx in range(a_idx + 1, n):
                b = S_indices[b_idx]
                
                # Check 1: idx_new | a*b
                # (a, b) should not be in conflict[idx_new]
                if a < b:
                    if conflict_adj[idx_new][a] & (1 << b):
                        return False
                else:
                    if conflict_adj[idx_new][b] & (1 << a):
                        return False
                
                # Check 2: a | idx_new*b
                # (idx_new, b) should not be in conflict[a]
                if idx_new < b:
                    if conflict_adj[a][idx_new] & (1 << b):
                        return False
                else:
                    if conflict_adj[a][b] & (1 << idx_new):
                        return False
                
                # Check 3: b | idx_new*a
                # (idx_new, a) should not be in conflict[b]
                if idx_new < a:
                    if conflict_adj[b][idx_new] & (1 << a):
                        return False
                else:
                    if conflict_adj[b][a] & (1 << idx_new):
                        return False
        return True

    def is_valid_set(S_indices):
        """Verify that no triple in S_indices violates the constraint."""
        n = len(S_indices)
        for i_idx in range(n):
            i = S_indices[i_idx]
            for j_idx in range(i_idx + 1, n):
                j = S_indices[j_idx]
                for k_idx in range(j_idx + 1, n):
                    k = S_indices[k_idx]
                    # Check i | j*k
                    if j < k:
                        if conflict_adj[i][j] & (1 << k):
                            return False
                    else:
                        if conflict_adj[i][k] & (1 << j):
                            return False
        return True

    best_S = []
    best_size = 0
    
    # Heuristic ordering
    sorted_indices_by_risk = sorted(range(num_candidates), key=lambda i: len(conflict[i]))
    sorted_indices_desc = sorted(range(num_candidates), key=lambda i: candidates[i], reverse=True)
    
    def greedy_construct(order):
        S = []
        for idx in order:
            if can_add(idx, S):
                S.append(idx)
        return S

    def local_search(S_indices, max_time=None):
        if max_time is None:
            max_time = deadline - 3.0
            
        S_set = set(S_indices)
        not_in_S = [i for i in range(num_candidates) if i not in S_set]
        
        improved = True
        stall_count = 0
        while improved and time.time() < max_time:
            improved = False
            stall_count += 1
            
            # 1. Try Additions
            shuffled_not_in_S = not_in_S[:]
            random.shuffle(shuffled_not_in_S)
            for idx in shuffled_not_in_S:
                if can_add(idx, S_indices):
                    S_indices.append(idx)
                    S_set.add(idx)
                    not_in_S.remove(idx)
                    improved = True
                    stall_count = 0
                    break
            
            if improved:
                continue
            
            # 2. Try Single Swaps - try more systematically
            if S_indices and not_in_S:
                # Try all elements in S against random elements in not_in_S
                for i in S_indices:
                    if time.time() >= max_time:
                        break
                    for j in not_in_S:
                        if time.time() >= max_time:
                            break
                        S_minus_i = [x for x in S_indices if x != i]
                        if can_add(j, S_minus_i):
                            S_indices = S_minus_i + [j]
                            S_set.remove(i)
                            S_set.add(j)
                            not_in_S.remove(j)
                            not_in_S.append(i)
                            improved = True
                            stall_count = 0
                            break
                    if improved:
                        break
            
            if improved:
                continue

            # 3. Try Double Swaps
            if len(S_indices) >= 2 and len(not_in_S) >= 2:
                # Try random double swaps
                for attempt in range(30):
                    if time.time() >= max_time:
                        break
                    i, k = random.sample(S_indices, 2)
                    j, l = random.sample(not_in_S, 2)
                    
                    S_minus_ik = [x for x in S_indices if x != i and x != k]
                    
                    if can_add(j, S_minus_ik):
                        S_minus_ik_plus_j = S_minus_ik + [j]
                        if can_add(l, S_minus_ik_plus_j):
                            S_indices = S_minus_ik + [j, l]
                            S_set.remove(i)
                            S_set.remove(k)
                            S_set.add(j)
                            S_set.add(l)
                            
                            not_in_S = [x for x in not_in_S if x != j and x != l]
                            not_in_S.append(i)
                            not_in_S.append(k)
                            
                            improved = True
                            stall_count = 0
                            break
        
        return S_indices

    # Run multiple restarts
    rng = random.Random(42)
    num_restarts = 300  # More restarts with faster can_add
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        if r % 5 == 0:
            order = sorted_indices_by_risk[:]
            k = min(40, len(order))
            rng.shuffle(order[:k])
        elif r % 5 == 1:
            order = sorted_indices_desc[:]
            k = min(40, len(order))
            rng.shuffle(order[:k])
        elif r % 5 == 2:
            order = list(range(num_candidates))
            rng.shuffle(order)
        elif r % 5 == 3:
            # Mix of risk and value
            order = sorted(range(num_candidates), key=lambda i: (len(conflict[i]), -candidates[i]))
            k = min(40, len(order))
            rng.shuffle(order[:k])
        else:
            # Random with slight bias toward low-conflict elements
            weights = [1.0 / (1 + len(conflict[i])) for i in range(num_candidates)]
            order = list(range(num_candidates))
            # Weighted shuffle
            selected = []
            remaining = order[:]
            while remaining:
                total_w = sum(weights[i] for i in remaining)
                r_val = rng.random() * total_w
                cum = 0
                for idx, i in enumerate(remaining):
                    cum += weights[i]
                    if cum >= r_val:
                        selected.append(i)
                        remaining.pop(idx)
                        break
            order = selected
            
        S = greedy_construct(order)
        
        time_left = deadline - time.time()
        if time_left < 5:
            break
        search_time = max(0.5, time_left * 0.4)
        S = local_search(S, max_time=start_time + search_time)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S[:]
            # If we've found a large set, do more focused search
            if best_size >= 20:
                # Try to improve further with more local search
                extra_time = min(5.0, deadline - time.time() - 5)
                if extra_time > 1:
                    S = local_search(S, max_time=start_time + extra_time)
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = S[:]

    if not best_S or not is_valid_set(best_S):
        fallback = []
        for i in sorted_indices_desc:
            if can_add(i, fallback):
                fallback.append(i)
        return [candidates[i] for i in fallback]
        
    result = [candidates[i] for i in best_S]
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))
