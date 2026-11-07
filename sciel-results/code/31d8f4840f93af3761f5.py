import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin: 58s out of 60s limit
    
    N = 256
    candidates = list(range(2, N + 1))
    num_candidates = len(candidates)
    
    # Map value to index for quick lookup
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    
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

    # Helper: check if a | b*c using factorizations
    # Returns True if a divides b*c
    def divides_product(idx_a, idx_b, idx_c):
        fa = factors[idx_a]
        fb = factors[idx_b]
        fc = factors[idx_c]
        
        # Check for each prime in fa
        for p, exp_a in fa.items():
            exp_bc = fb.get(p, 0) + fc.get(p, 0)
            if exp_a > exp_bc:
                return False
        return True

    # can_add(idx_new, S_indices): 
    # S_indices is a list of indices of elements currently in S.
    # Checks if adding idx_new creates any violation with any pair in S.
    def can_add(idx_new, S_indices):
        n = len(S_indices)
        for i_idx in range(n):
            i = S_indices[i_idx]
            for j_idx in range(i_idx + 1, n):
                j = S_indices[j_idx]
                # Check the three permutations involving idx_new
                if divides_product(idx_new, i, j):
                    return False
                if divides_product(i, idx_new, j):
                    return False
                if divides_product(j, i, idx_new):
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
                    if divides_product(i, j, k):
                        return False
        return True

    best_S = []
    best_size = 0

    # Heuristic: Precompute "danger" or "safety"
    # A number x is "safe" if it divides few pairs (b*c).
    # We want to add safe numbers first to build a large core.
    # However, the constraint is symmetric. 
    # Let's use a simpler heuristic: sort by value descending.
    # Large numbers are less likely to be divisors of smaller products.
    # But small numbers are less likely to be products of two other numbers? 
    # Actually, if a is small, a | b*c is easy. So small 'a' is bad.
    # So we should prioritize LARGE numbers as 'a'.
    # But if we have large numbers, can we add small numbers?
    # If S is all large, and we add small s.
    # We need L_i | s * L_j.
    # If L_i is large and s is small, this is hard. So large numbers restrict small numbers.
    # This suggests a mixed approach is best.
    
    # Let's use a dynamic greedy:
    # At each step, pick the element that has the fewest conflicts with the current set?
    # Or just try all elements and pick the one that allows the most future additions?
    # Too slow.
    
    # Let's stick to:
    # 1. Sort by "number of prime factors" ascending? (Primes are safest).
    # 2. Sort by value descending.
    
    # Let's calculate a static "risk" score for each candidate.
    # Risk(x) = number of pairs (b, c) in candidates such that x | b*c.
    # Lower risk is better to add early? 
    # Actually, if x has high risk, it conflicts with many pairs. 
    # If we add it, we might block many others.
    # So we should add LOW RISK elements first.
    
    risk = [0] * num_candidates
    # Optimization: Only compute for a subset or use a fast approximation?
    # 255^3 is ~16M ops. In Python, this might take a few seconds. 
    # Let's do it once at the start if time permits, otherwise use a proxy.
    # Proxy: Number of divisors in [2, 256].
    # More divisors -> more likely to divide b*c?
    # Actually, if x has many divisors, it is more likely that x | b*c?
    # No, if x is prime, x|b*c iff x|b or x|c.
    # If x is composite, x|b*c is "harder" to satisfy? 
    # Wait. 
    # If x=6, b=2, c=3. 6|6. Yes.
    # If x=5, b=2, c=3. 5|6. No.
    # So composites are "easier" to divide products? 
    # Let's just use value descending as a primary sort, and randomize slightly.
    
    sorted_indices_desc = sorted(range(num_candidates), key=lambda i: candidates[i], reverse=True)
    sorted_indices_asc = sorted(range(num_candidates), key=lambda i: candidates[i])
    
    def greedy_construct(order):
        S = []
        # To speed up, we can check candidates in the given order
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
            # Shuffle not_in_S to avoid bias
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
            # Remove i, Add j
            if S_indices and not_in_S:
                # Try a limited number of random swaps to avoid getting stuck in a local loop
                attempts = 0
                max_attempts = 50
                while attempts < max_attempts and time.time() < max_time:
                    i = random.choice(S_indices)
                    j = random.choice(not_in_S)
                    if i == j: continue
                    
                    # Check if swap is valid: (S - {i}) U {j}
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
                    attempts += 1
            
            if improved:
                continue

            # 3. Try Double Swaps (Remove i, k; Add j, l)
            # This is more expensive but can escape deeper local optima
            if len(S_indices) >= 2 and len(not_in_S) >= 2:
                attempts = 0
                max_attempts = 20
                while attempts < max_attempts and time.time() < max_time:
                    # Pick two to remove
                    i, k = random.sample(S_indices, 2)
                    # Pick two to add
                    j, l = random.sample(not_in_S, 2)
                    
                    # Ensure distinct
                    if len({i, k, j, l}) < 4:
                        attempts += 1
                        continue

                    S_minus_ik = [x for x in S_indices if x != i and x != k]
                    
                    # Check if j can be added to S_minus_ik
                    if not can_add(j, S_minus_ik):
                        attempts += 1
                        continue
                    
                    # Check if l can be added to S_minus_ik U {j}
                    S_minus_ik_plus_j = S_minus_ik + [j]
                    if not can_add(l, S_minus_ik_plus_j):
                        attempts += 1
                        continue
                    
                    # Perform swap
                    S_indices = S_minus_ik + [j, l]
                    S_set.remove(i)
                    S_set.remove(k)
                    S_set.add(j)
                    S_set.add(l)
                    
                    # Update not_in_S
                    # Remove j, l from not_in_S
                    # Add i, k to not_in_S
                    # This is O(N) to update list, acceptable for small N
                    not_in_S = [x for x in not_in_S if x != j and x != l]
                    not_in_S.append(i)
                    not_in_S.append(k)
                    
                    improved = True
                    break
                    attempts += 1
        
        return S_indices

    # Run multiple restarts
    num_restarts = 100
    rng = random.Random(42) # Fixed seed for reproducibility in logic, but we vary order
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Create a randomized order
        # Mix of descending value and random
        if r % 3 == 0:
            order = sorted_indices_desc[:]
            # Perturb top part
            k = min(50, len(order))
            rng.shuffle(order[:k])
        elif r % 3 == 1:
            order = sorted_indices_asc[:]
            k = min(50, len(order))
            rng.shuffle(order[:k])
        else:
            order = list(range(num_candidates))
            rng.shuffle(order)
            
        S = greedy_construct(order)
        
        # Local search with time limit
        # Allocate remaining time / 2 to local search
        time_left = deadline - time.time()
        search_time = max(1.0, time_left / 2.0)
        S = local_search(S, max_time=start_time + search_time)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S[:]
            # If we found a very large set, we might want to try to improve it further
            # But let's stick to the loop for now.

    # Final validation check
    if not is_valid_set(best_S):
        # Fallback to a known valid set (e.g., large primes)
        fallback = []
        for i in range(num_candidates):
            if candidates[i] > 128:
                if can_add(i, fallback):
                    fallback.append(i)
        return [candidates[i] for i in fallback]
        
    result = [candidates[i] for i in best_S]
    
    # Ensure types are correct
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))
