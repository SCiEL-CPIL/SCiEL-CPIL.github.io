import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit

    N = 256
    candidates = list(range(2, N + 1))
    num_candidates = len(candidates)

    # Precompute prime factorization for each candidate
    # factors[i] is a dict {prime: exponent}
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

    # Precompute the "divides product" relation for all triples
    # div_triple[i][j][k] = True if candidates[i] divides candidates[j] * candidates[k]
    # We only need this for distinct i, j, k
    # To save memory and time, we'll use a 3D boolean array or a set of violating triples
    # Let's use a set of frozensets or just a 3D list of bools
    # 255^3 ~ 16.5 million booleans. That's about 16.5 MB. Acceptable.
    # But computing it takes O(N^3 * factor_check). Factor check is fast.
    
    # Actually, let's precompute a matrix div[i][j] = factors of candidates[i] that are not covered by candidates[j]
    # No, let's just precompute the full 3D array.
    # div3[i][j][k] = 1 if candidates[i] | candidates[j] * candidates[k]
    
    # Optimization: We can store this as a list of bitmasks or just a 3D list.
    # Let's try a 3D list of booleans.
    div3 = [[[False] * num_candidates for _ in range(num_candidates)] for __ in range(num_candidates)]
    
    for i in range(num_candidates):
        fa = factors[i]
        for j in range(num_candidates):
            fb = factors[j]
            for k in range(num_candidates):
                if i == j or i == k or j == k:
                    continue
                fc = factors[k]
                valid = True
                for p, exp_a in fa.items():
                    exp_bc = fb.get(p, 0) + fc.get(p, 0)
                    if exp_a > exp_bc:
                        valid = False
                        break
                div3[i][j][k] = valid

    # Helper to check if adding idx_new to a set S (list of indices) is valid
    def can_add(idx_new, S_indices):
        n = len(S_indices)
        for i_idx in range(n):
            i = S_indices[i_idx]
            for j_idx in range(i_idx + 1, n):
                j = S_indices[j_idx]
                if div3[idx_new][i][j] or div3[i][idx_new][j] or div3[j][i][idx_new]:
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
                    if div3[i][j][k] or div3[j][i][k] or div3[k][i][j]:
                        return False
        return True

    best_S = []
    best_size = 0

    # Strategy:
    # 1. Compute "danger" score for each candidate: how many pairs (b,c) does it divide?
    #    This helps in greedy ordering.
    danger = [0] * num_candidates
    for i in range(num_candidates):
        for j in range(num_candidates):
            for k in range(j + 1, num_candidates):
                if div3[i][j][k]:
                    danger[i] += 1

    sorted_indices = sorted(range(num_candidates), key=lambda i: danger[i])

    def greedy_construct(order):
        S = []
        for idx in order:
            if can_add(idx, S):
                S.append(idx)
        return S

    def local_search(S_indices):
        S_set = set(S_indices)
        not_in_S = [i for i in range(num_candidates) if i not in S_set]
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Try additions
            for idx in sorted(not_in_S, key=lambda i: danger[i]):
                if can_add(idx, S_indices):
                    S_indices.append(idx)
                    S_set.add(idx)
                    not_in_S.remove(idx)
                    improved = True
                    break
            
            if improved:
                continue
                
            # Try swaps
            if S_indices and not_in_S:
                for _ in range(20): # Limited swap attempts
                    if time.time() > deadline - 5: break
                    i = random.choice(S_indices)
                    j = random.choice(not_in_S)
                    
                    # Check if swap is valid
                    S_minus_i = [x for x in S_indices if x != i]
                    if can_add(j, S_minus_i):
                        # Perform swap
                        S_indices = S_minus_i + [j]
                        S_set.remove(i)
                        S_set.add(j)
                        not_in_S.remove(j)
                        not_in_S.append(i)
                        improved = True
                        break
        
        return S_indices

    # Run multiple restarts with different random orders
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Create a random order, but biased towards safe elements
        if r % 2 == 0:
            order = sorted_indices[:]
        else:
            order = sorted_indices[:]
            random.shuffle(order)
            
        S = greedy_construct(order)
        S = local_search(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S[:]
            
    # Convert indices back to values
    result = [candidates[i] for i in best_S]
    
    # Final validation check
    if not is_valid_set(best_S):
        # Fallback to a known valid set (e.g., large primes)
        fallback = []
        for i in range(num_candidates):
            if candidates[i] > 128:
                if can_add(i, fallback):
                    fallback.append(i)
        return [candidates[i] for i in fallback]
        
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))
