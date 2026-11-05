import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    n = 255
    target = 255
    
    # Precompute prime factors as bitmasks for numbers 1..255
    # Primes up to 255: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251
    # There are 54 primes. We can use an integer bitmask.
    
    primes = []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
                
    prime_index = {p: i for i, p in enumerate(primes)}
    
    factors = [0] * (n + 1)
    for i in range(2, n + 1):
        x = i
        mask = 0
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                mask |= (1 << prime_index[p])
                while x % p == 0:
                    x //= p
        if x > 1:
            mask |= (1 << prime_index[x])
        factors[i] = mask
        
    # Candidates: must share a factor with 255
    # 255 = 3 * 5 * 17
    # mask_255 = factors[255]
    
    candidates = [x for x in range(1, n + 1) if (factors[x] & factors[target]) != 0]
    
    def greedy_add(current_set_mask, current_set_list, remaining_pool):
        """
        current_set_mask: bitmask of elements in set? No, bitmask of prime factors? 
        We need to check compatibility of new candidate c with ALL x in current_set.
        Compatibility: (factors[c] & factors[x]) != 0 for all x.
        
        To speed up, we can precompute for each candidate c, a bitmask of candidates it is compatible with?
        Or just iterate.
        """
        current_set = set(current_set_list)
        remaining = set(remaining_pool)
        
        # Optimization: Precompute compatibility for remaining candidates against current set?
        # Since current_set changes, this is hard to precompute globally.
        # But we can filter remaining pool first.
        
        while remaining:
            if time.time() > deadline:
                break
                
            best_candidate = None
            best_score = -1
            
            # Filter remaining to only those compatible with current_set
            compatible_cands = []
            for c in remaining:
                is_compatible = True
                for x in current_set:
                    if (factors[c] & factors[x]) == 0:
                        is_compatible = False
                        break
                if is_compatible:
                    compatible_cands.append(c)
                    
            if not compatible_cands:
                break
                
            # Score: number of other compatible_cands that are compatible with this cand
            # We can approximate by counting how many primes it shares with others?
            # Or just count compatible pairs within compatible_cands.
            
            # To save time, just pick the one with max degree in the subgraph of compatible_cands
            # Degree of c in compatible_cands = sum(1 for o in compatible_cands if (factors[c] & factors[o]) != 0)
            
            # If |compatible_cands| is small, this is fast.
            
            for c in compatible_cands:
                score = 0
                fc = factors[c]
                for o in compatible_cands:
                    if o == c:
                        continue
                    if (fc & factors[o]) != 0:
                        score += 1
                        
                if score > best_score:
                    best_score = score
                    best_candidate = c
                    
            if best_candidate is None:
                break
                
            current_set.add(best_candidate)
            remaining.remove(best_candidate)
            
        return list(current_set)
        
    # Initial Greedy
    current_set = [target]
    remaining_pool = [c for c in candidates if c != target]
    
    # We need to pass a list to greedy_add, but it modifies a set internally.
    # Let's rewrite greedy_add to be cleaner.
    
    def run_greedy(start_set, pool):
        current_set = set(start_set)
        remaining = set(pool)
        
        while remaining:
            if time.time() > deadline:
                break
                
            compatible_cands = []
            for c in remaining:
                is_compatible = True
                for x in current_set:
                    if (factors[c] & factors[x]) == 0:
                        is_compatible = False
                        break
                if is_compatible:
                    compatible_cands.append(c)
                    
            if not compatible_cands:
                break
                
            best_candidate = None
            best_score = -1
            
            # Heuristic: Maximize connections within compatible_cands
            for c in compatible_cands:
                score = 0
                fc = factors[c]
                for o in compatible_cands:
                    if o != c and (fc & factors[o]) != 0:
                        score += 1
                        
                if score > best_score:
                    best_score = score
                    best_candidate = c
                    
            if best_candidate is None:
                break
                
            current_set.add(best_candidate)
            remaining.remove(best_candidate)
            
        return list(current_set)
        
    # Initial Construction
    S = run_greedy([target], [c for c in candidates if c != target])
    
    # Local Search: Remove and Rebuild
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
            
        for y in S:
            if y == target:
                continue
                
            S_trial = [x for x in S if x != y]
            pool_trial = [c for c in candidates if c not in S_trial]
            
            # Run greedy on the trial set
            S_new = run_greedy(S_trial, pool_trial)
            
            if len(S_new) > len(S):
                S = S_new
                improved = True
                break # Restart the for loop with the new S
                
    return S
