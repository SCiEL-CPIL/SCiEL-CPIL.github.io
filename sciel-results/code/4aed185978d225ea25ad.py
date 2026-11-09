import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    MAX_VAL = 10000
    MAX_SUM = 2 * MAX_VAL
    
    # 1. Sieve for Smallest Prime Factor (SPF) up to 20000
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(math.isqrt(MAX_SUM)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def is_smooth(n, primes_set):
        if n == 1:
            return True
        while n > 1:
            p = spf[n]
            if p not in primes_set:
                return False
            while n % p == 0:
                n //= p
        return True

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    
    best_result = None
    best_prime_count = float('inf')

    # Try increasing sets of primes
    for k in range(1, len(primes) + 1):
        if time.time() > deadline - 10:
            break
            
        current_primes = frozenset(primes[:k])
        
        # Pre-compute smoothness for all sums
        smooth_flags = [0] * (MAX_SUM + 1)
        for s in range(2, MAX_SUM + 1):
            if is_smooth(s, current_primes):
                smooth_flags[s] = 1

        # Build compatibility masks
        # masks[i] is a bitset where bit j is 1 if i+j is smooth and i != j
        masks = [0] * (MAX_VAL + 1)
        
        if time.time() > deadline - 5:
            break

        for i in range(1, MAX_VAL + 1):
            m = 0
            # Iterate j
            # Optimization: local variable for smooth_flags
            sf = smooth_flags
            for j in range(1, MAX_VAL + 1):
                if i != j and sf[i + j]:
                    m |= (1 << j)
            masks[i] = m
            
        if time.time() > deadline - 5:
            break

        # Find clique of size 12
        # Backtracking with heuristic
        
        def find_clique():
            # Returns list of 12 indices or None
            # candidates: bitmask
            # depth: current clique size
            # current_clique: list
            
            # We use a stack for backtracking to avoid recursion limits (though 12 is fine)
            # But Python recursion is slower. Let's use recursion with a limit.
            
            # To avoid trying all 10000 nodes at each step, we order them by degree.
            
            def backtrack(candidates_mask, depth, current_clique):
                if depth == 12:
                    return current_clique[:]
                
                if candidates_mask.bit_count() < 12 - depth:
                    return None
                
                # Get candidates as a list, sorted by degree in candidates
                # This is expensive: O(N log N) per node?
                # N=10000. 12 levels.
                # Maybe just pick top K candidates?
                
                # Let's try a simple approach:
                # Iterate through set bits in candidates_mask.
                # Pick the one with highest degree.
                
                best_node = -1
                best_deg = -1
                temp_mask = candidates_mask
                # To speed up, we can limit the search for the "best" node
                # or just pick the first one?
                # Let's pick the one with highest degree.
                
                # Optimization: If the graph is dense, the first node might be fine.
                # Let's try a few random nodes if the first few fail?
                
                # Deterministic: Highest degree.
                # To avoid O(N) scan every time, we can pre-sort?
                # No, degrees change as candidates shrink.
                
                # Let's just scan. 10000 scans * 12 depth = 120k scans.
                # Each scan involves bit_count.
                # bit_count on 10000-bit int is very fast (C implementation).
                
                while temp_mask:
                    lsb = temp_mask & -temp_mask
                    node = lsb.bit_length() - 1
                    deg = (masks[node] & candidates_mask).bit_count()
                    if deg > best_deg:
                        best_deg = deg
                        best_node = node
                    temp_mask ^= lsb
                
                if best_node == -1:
                    return None
                
                # Try best_node
                new_clique = current_clique + [best_node]
                new_candidates = candidates_mask & masks[best_node]
                new_candidates &= ~(1 << best_node)
                
                res = backtrack(new_candidates, depth + 1, new_clique)
                if res:
                    return res
                
                # Backtrack: Try next best node
                # This requires us to have a list of nodes sorted by degree.
                # Let's generate that list once per call.
                
                # Refactor: Generate sorted list of candidates by degree
                # This is O(N log N). 12 * 10000 log 10000 ~ 12 * 140k = 1.7M ops. Fast.
                
                return None # Placeholder, will replace with proper backtracking

        # Proper Backtracking Implementation
        def solve():
            initial_candidates = 0
            for i in range(1, MAX_VAL + 1):
                initial_candidates |= (1 << i)
                
            # State: (candidates_mask, depth, current_clique)
            # We use a stack to implement backtracking manually to control time
            # Or just recursion.
            
            def rec(candidates_mask, depth, current_clique):
                if depth == 12:
                    return current_clique[:]
                
                if candidates_mask.bit_count() < 12 - depth:
                    return None
                
                # Get sorted candidates by degree
                # Create list of (degree, node)
                cands = []
                temp_mask = candidates_mask
                while temp_mask:
                    lsb = temp_mask & -temp_mask
                    node = lsb.bit_length() - 1
                    deg = (masks[node] & candidates_mask).bit_count()
                    cands.append((deg, node))
                    temp_mask ^= lsb
                    
                # Sort by degree descending
                cands.sort(reverse=True)
                
                # Try top candidates
                # Limit to top 20 to avoid explosion if greedy fails
                for deg, node in cands[:20]:
                    new_clique = current_clique + [node]
                    new_candidates = candidates_mask & masks[node]
                    new_candidates &= ~(1 << node)
                    
                    res = rec(new_candidates, depth + 1, new_clique)
                    if res:
                        return res
                        
                return None

            return rec(initial_candidates, 0, [])

        solution = solve()
        
        if solution:
            best_result = solution
            best_prime_count = k
            break

    if best_result:
        return best_result
    
    # Fallback
    return list(range(1, 13))
# EVOLVE-BLOCK-END
