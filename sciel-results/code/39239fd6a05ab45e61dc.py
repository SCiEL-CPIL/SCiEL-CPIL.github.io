import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    MAX_VAL = 10000
    N = 12
    LIMIT = 2 * MAX_VAL
    
    # 1. SPF Table
    spf = list(range(LIMIT + 1))
    for i in range(2, int(LIMIT**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, LIMIT + 1, i):
                if spf[j] == j:
                    spf[j] = i
                    
    def get_prime_factors(n):
        factors = set()
        while n > 1:
            factors.add(spf[n])
            n //= spf[n]
        return factors

    def is_p_smooth(n, prime_set):
        if n <= 1:
            return True
        while n > 1:
            p = spf[n]
            if p not in prime_set:
                return False
            while n % p == 0:
                n //= p
        return True

    def count_primes(arr):
        union = set()
        for i in range(N):
            for j in range(i+1, N):
                union |= get_prime_factors(arr[i] + arr[j])
        return len(union)

    best_result = None
    best_score = float('inf')
    
    def try_update(arr, score):
        nonlocal best_result, best_score
        if score < best_score:
            best_score = score
            best_result = arr[:]
            return True
        return False

    # Strategy 1: Multiples of smooth base M
    # If all a_i are multiples of M, then a_i + a_j is a multiple of M.
    # Prime factors of M are in the union.
    # We want a_i + a_j to have no other prime factors.
    # This implies (a_i + a_j) / M must be S-smooth, where S is the set of prime factors of M.
    # Let a_i = M * b_i. Then M * b_i + M * b_j = M * (b_i + b_j).
    # So we need b_i + b_j to be S-smooth.
    # This reduces the problem to finding a clique of size N in the graph of b in [1, MAX_VAL//M]
    # where edges connect b, c if b+c is S-smooth.
    
    # Generate candidate bases M
    # We want M to be composed of small primes to keep the smooth set dense.
    # Also M should be large enough to reduce the range of b, but not too large to allow N distinct multiples.
    
    # Let's try bases that are products of subsets of {2, 3, 5, 7, 11}
    primes_to_consider = [2, 3, 5, 7, 11]
    
    # Precompute is_smooth for various prime sets?
    # It's better to do it on the fly for the specific M.
    
    # List of M to try
    # We want M to be such that MAX_VAL // M >= N.
    # So M <= MAX_VAL // N = 833.
    
    # Generate all 5-smooth numbers <= 833? Or just products of small primes?
    # Let's iterate over all M in 1..833 and check if M is "smooth" (only small prime factors).
    # Actually, any M works, but the "smoothness" of the sums depends on the prime factors of M.
    # If M has a large prime factor p, then p is in the union.
    # We want to minimize the union.
    # So M should be composed of the smallest primes.
    
    # Let's just try all M that are products of {2,3,5,7,11} and <= 833.
    bases = set()
    def gen_bases(current, max_val, primes, idx):
        if idx == len(primes):
            bases.add(current)
            return
        p = primes[idx]
        val = current
        while val <= max_val:
            gen_bases(val, max_val, primes, idx + 1)
            val *= p
            if val == 0: break # overflow protection
            
    gen_bases(1, 833, primes_to_consider, 0)
    bases = sorted(list(bases), reverse=True) # Try larger bases first? Smaller range of b?
    # Actually, smaller M means larger range of b, harder to find clique.
    # Larger M means smaller range of b, easier to find clique.
    # But larger M means more prime factors in M?
    # No, M is fixed. The union size is |primes(M) U primes(b_i+b_j)|.
    # If b_i+b_j is S-smooth, then primes(b_i+b_j) subset S.
    # So union is S.
    # So we want to minimize |S|.
    # S is the set of prime factors of M.
    # So we should try M with fewest prime factors first.
    # But M must be large enough to make the clique search easy?
    # Let's sort by number of prime factors, then by value.
    
    def count_pf(n):
        c = 0
        while n > 1:
            p = spf[n]
            c += 1
            while n % p == 0:
                n //= p
        return c
        
    bases = sorted(bases, key=lambda m: (count_pf(m), -m))
    
    for M in bases:
        if time.time() > deadline - 10:
            break
            
        if M < N:
            continue
            
        max_b = MAX_VAL // M
        if max_b < N:
            continue
            
        # Determine S (prime factors of M)
        S = get_prime_factors(M)
        
        # If |S| >= best_score, skip (we can't do better)
        if len(S) >= best_score:
            continue
            
        # Build graph for b in [1, max_b]
        # Edge (b, c) if b+c is S-smooth.
        # Note: b and c must be distinct.
        
        # Precompute is_smooth for sums up to 2*max_b
        limit_b = 2 * max_b
        is_smooth_b = [False] * (limit_b + 1)
        for s in range(2, limit_b + 1):
            if is_p_smooth(s, S):
                is_smooth_b[s] = True
                
        # Build adjacency list
        # neighbors[b] = list of c
        neighbors = [[] for _ in range(max_b + 1)]
        for b in range(1, max_b + 1):
            for c in range(b+1, max_b + 1):
                if b + c <= limit_b and is_smooth_b[b + c]:
                    neighbors[b].append(c)
                    neighbors[c].append(b)
                    
        # Find clique of size N
        # Use greedy/backtracking or random restarts
        # Since max_b is small (<= 10000/M), this might be fast.
        # If M=1, max_b=10000. Too big for full clique search.
        # But we sorted M by pf count. M=1 has 0 pf. S=empty.
        # If S=empty, b+c must be 1-smooth (i.e., 1). Impossible for b,c >= 1.
        # So M=1 fails immediately.
        
        found_clique = None
        
        # Randomized Greedy with Backtracking
        # Or just try to find a clique of size N.
        # N=12.
        
        # Heuristic: Start with a node with high degree.
        # Then pick neighbors with high degree within the remaining candidates.
        
        # Let's try a few random starts.
        num_restarts = 200
        if max_b > 1000:
            num_restarts = 50 # Less time for larger graphs
        
        for _ in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            # Pick a random start node with degree >= N-1
            candidates_start = [b for b in range(1, max_b + 1) if len(neighbors[b]) >= N - 1]
            if not candidates_start:
                break
                
            start = random.choice(candidates_start)
            clique = [start]
            # Candidates must be neighbors of all in clique
            # Start with neighbors of start
            current_candidates = set(neighbors[start])
            
            for _ in range(N - 1):
                if not current_candidates:
                    break
                    
                # Pick a node from current_candidates
                # Prefer nodes with high degree within current_candidates
                # Score: number of neighbors in current_candidates
                best_node = -1
                best_score_node = -1
                # To speed up, sample a few?
                # For now, check all if small, else sample.
                if len(current_candidates) < 100:
                    for node in current_candidates:
                        score = sum(1 for nb in neighbors[node] if nb in current_candidates)
                        if score > best_score_node:
                            best_score_node = score
                            best_node = node
                else:
                    # Sample 10 nodes
                    sample = random.sample(list(current_candidates), 10)
                    for node in sample:
                        score = sum(1 for nb in neighbors[node] if nb in current_candidates)
                        if score > best_score_node:
                            best_score_node = score
                            best_node = node
                
                if best_node == -1:
                    break
                    
                clique.append(best_node)
                # Update candidates: must be neighbor to best_node AND in current_candidates
                new_candidates = set(neighbors[best_node])
                current_candidates &= new_candidates
                # Remove nodes already in clique (they are not in neighbors as no self loops, but just in case)
                # Actually, neighbors don't include self.
                
            if len(clique) == N:
                found_clique = clique
                break
                
        if found_clique:
            # Convert to a_i
            result = [M * b for b in found_clique]
            # Verify
            valid = True
            primes_used = set()
            for i in range(N):
                for j in range(i+1, N):
                    s = result[i] + result[j]
                    pf = get_prime_factors(s)
                    if not pf.issubset(S):
                        valid = False
                        break
                    primes_used |= pf
                if not valid:
                    break
                    
            if valid:
                score = len(primes_used)
                if try_update(result, score):
                    # Found a good solution.
                    # Can we do better?
                    # Only with smaller |S|.
                    # We are iterating M by |S|, so if we found one with |S|=k,
                    # any subsequent M will have |S| >= k.
                    # So we can return immediately?
                    # Yes, because we want to minimize distinct primes.
                    return result

    # Strategy 2: Local Search Fallback
    if best_result is None:
        # Initialize with a simple set
        best_result = list(range(1, 13))
        best_score = count_primes(best_result)
        
    current = best_result[:]
    current_score = best_score
    
    # Local search
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(N):
            old = current[i]
            # Try swapping with a random number
            for _ in range(100):
                new_val = random.randint(1, MAX_VAL)
                if new_val in current:
                    continue
                current[i] = new_val
                new_score = count_primes(current)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if try_update(current, current_score):
                        pass
                    break
                current[i] = old
            if improved:
                break
                
    return best_result
