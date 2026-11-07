import time
import sys
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin: 85s out of 90s limit
    
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
    
    # Let's use a faster check: 
    # Precompute a bitmask for each i? No, 255 bits is large but manageable.
    # Actually, let's just precompute the full 3D conflict in a compact way.
    # conflict[i] is a set of (j, k) indices such that i | j*k.
    # Since we need to check "for all triples", we can check if any pair in the set S (excluding i) is in conflict[i].
    
    # Optimization: 
    # For a fixed set S, adding i is valid if for all j, k in S, j != k, i does not divide j*k.
    # This is equivalent to: S does not contain any pair (j, k) from conflict[i].
    
    # Let's precompute conflict[i] as a list of pairs (j, k) with j < k.
    conflict = [set() for _ in range(num_candidates)]
    
    # To speed this up, we can iterate j, k and find which i divide j*k.
    # For each pair (j, k), let prod = candidates[j] * candidates[k].
    # Find all i in [0, num_candidates) such that candidates[i] divides prod.
    # candidates[i] ranges from 2 to 256.
    # So we can iterate d from 2 to min(256, prod), if prod % d == 0, then i = index of d.
    
    # This is 255^2 / 2 pairs ~ 32k pairs.
    # For each pair, we iterate d up to 256.
    # Total ops: 32k * 256 ~ 8M. This is fast.
    
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    
    for j in range(num_candidates):
        for k in range(j + 1, num_candidates):
            prod = candidates[j] * candidates[k]
            # Find all divisors of prod in [2, 256]
            # We can just check all candidates
            for i in range(num_candidates):
                if prod % candidates[i] == 0:
                    conflict[i].add((j, k))
                    # Note: The constraint is symmetric in j and k for the product, 
                    # but the triple (i, j, k) is distinct.
                    # The condition is "a does not divide b*c".
                    # So if i divides j*k, then (i, j, k) is a forbidden triple.
                    # Since j and k are distinct and i is distinct from them (in the triple check),
                    # we just need to ensure no such triple exists in S.
                    
    # Wait, the precomputation above adds (j, k) to conflict[i] if i | j*k.
    # When checking if we can add i to S, we need to check if there exists j, k in S such that (j, k) in conflict[i].
    # Also, we need to check if i creates conflicts as j or k?
    # The condition is: for ALL distinct a, b, c in S, a does not divide b*c.
    # This is symmetric in the sense that if we have S, we check all permutations.
    # If we add i to S, we need to check:
    # 1. i does not divide j*k for any j, k in S. (Checked by conflict[i])
    # 2. j does not divide i*k for any j, k in S. (This means i*k is a product that j divides. 
    #    So (i, k) is a pair in conflict[j]?)
    #    Let's rephrase: j | i*k. This is equivalent to (i, k) being a "bad pair" for j.
    #    So we need to check if (i, k) is in conflict[j] for any k in S.
    # 3. k does not divide i*j for any j, k in S.
    #    This is equivalent to (i, j) being in conflict[k] for any j in S.
    
    # So, can_add(i, S) is:
    # For all j, k in S, j != k:
    #   - (j, k) not in conflict[i]  (i | j*k)
    #   - (i, k) not in conflict[j]  (j | i*k)
    #   - (i, j) not in conflict[k]  (k | i*j)
    
    # This is O(|S|^2) per check.
    
    def can_add(idx_new, S_indices):
        n = len(S_indices)
        # We need to check pairs in S
        for i_idx in range(n):
            i = S_indices[i_idx]
            for j_idx in range(i_idx + 1, n):
                j = S_indices[j_idx]
                # Check i | new*j  -> (new, j) in conflict[i]
                if (idx_new, j) in conflict[i] or (j, idx_new) in conflict[i]:
                    return False
                # Check j | new*i  -> (new, i) in conflict[j]
                if (idx_new, i) in conflict[j] or (i, idx_new) in conflict[j]:
                    return False
                # Check new | i*j  -> (i, j) in conflict[new]
                if (i, j) in conflict[idx_new] or (j, i) in conflict[idx_new]:
                    return False
        return True

    def is_valid_set(S_indices):
        n = len(S_indices)
        for i_idx in range(n):
            i = S_indices[i_idx]
            for j_idx in range(i_idx + 1, n):
                j = S_indices[j_idx]
                for k_idx in range(j_idx + 1, n):
                    k = S_indices[k_idx]
                    # Check i | j*k
                    if (j, k) in conflict[i] or (k, j) in conflict[i]:
                        return False
        return True

    best_S = []
    best_size = 0
    
    # Heuristic ordering
    # Sort by "risk" = len(conflict[i]) ascending?
    # Lower conflict count means it's safer to add.
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
            max_time = deadline - 5.0
            
        S_set = set(S_indices)
        not_in_S = [i for i in range(num_candidates) if i not in S_set]
        
        improved = True
        while improved and time.time() < max_time:
            improved = False
            
            # 1. Try Additions
            random.shuffle(not_in_S)
            for idx in not_in_S:
                if can_add(idx, S_indices):
                    S_indices.append(idx)
                    S_set.add(idx)
                    not_in_S.remove(idx)
                    improved = True
                    break
            
            if improved:
                continue
                
            # 2. Try Single Swaps
            if S_indices and not_in_S:
                attempts = 0
                max_attempts = 100
                while attempts < max_attempts and time.time() < max_time:
                    i = random.choice(S_indices)
                    j = random.choice(not_in_S)
                    
                    S_minus_i = [x for x in S_indices if x != i]
                    if can_add(j, S_minus_i):
                        S_indices = S_minus_i + [j]
                        S_set.remove(i)
                        S_set.add(j)
                        not_in_S.remove(j)
                        not_in_S.append(i)
                        improved = True
                        break
                    attempts += 1
            
            if improved:
                continue

            # 3. Try Double Swaps
            if len(S_indices) >= 2 and len(not_in_S) >= 2:
                attempts = 0
                max_attempts = 50
                while attempts < max_attempts and time.time() < max_time:
                    i, k = random.sample(S_indices, 2)
                    j, l = random.sample(not_in_S, 2)
                    
                    if len({i, k, j, l}) < 4:
                        attempts += 1
                        continue

                    S_minus_ik = [x for x in S_indices if x != i and x != k]
                    
                    if not can_add(j, S_minus_ik):
                        attempts += 1
                        continue
                    
                    S_minus_ik_plus_j = S_minus_ik + [j]
                    if not can_add(l, S_minus_ik_plus_j):
                        attempts += 1
                        continue
                    
                    S_indices = S_minus_ik + [j, l]
                    S_set.remove(i)
                    S_set.remove(k)
                    S_set.add(j)
                    S_set.add(l)
                    
                    not_in_S = [x for x in not_in_S if x != j and x != l]
                    not_in_S.append(i)
                    not_in_S.append(k)
                    
                    improved = True
                    break
                    attempts += 1
        
        return S_indices

    # Run multiple restarts
    rng = random.Random(42)
    num_restarts = 200
    
    for r in range(num_restarts):
        if time.time() > deadline - 15:
            break
            
        if r % 4 == 0:
            order = sorted_indices_by_risk[:]
            k = min(50, len(order))
            rng.shuffle(order[:k])
        elif r % 4 == 1:
            order = sorted_indices_desc[:]
            k = min(50, len(order))
            rng.shuffle(order[:k])
        elif r % 4 == 2:
            order = list(range(num_candidates))
            rng.shuffle(order)
        else:
            # Mix of risk and value
            order = sorted(range(num_candidates), key=lambda i: (len(conflict[i]), -candidates[i]))
            k = min(50, len(order))
            rng.shuffle(order[:k])
            
        S = greedy_construct(order)
        
        time_left = deadline - time.time()
        search_time = max(1.0, time_left / 2.0)
        S = local_search(S, max_time=start_time + search_time)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S[:]

    if not is_valid_set(best_S):
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
