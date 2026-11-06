import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    # Precompute Smallest Prime Factor (SPF) up to 20000
    # Max sum is 10000 + 9999 = 19999
    MAX_SUM = 20000
    spf = list(range(MAX_SUM + 1))
    for i in range(2, isqrt(MAX_SUM) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def get_prime_factors(n):
        """Return set of prime factors of n."""
        if n <= 1:
            return set()
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return factors
    
    # Precompute prime factor sets for all numbers up to MAX_SUM
    # This is O(MAX_SUM log MAX_SUM)
    pf_cache = [set() for _ in range(MAX_SUM + 1)]
    for i in range(2, MAX_SUM + 1):
        pf_cache[i] = get_prime_factors(i)
    
    def is_smooth(n, prime_set):
        """Check if all prime factors of n are in prime_set."""
        if n <= 1:
            return True
        return pf_cache[n].issubset(prime_set)
    
    # Candidate prime sets to try, in order of size
    # We want to minimize the number of distinct primes.
    # The primes in the sums will be a superset of the primes used to generate the candidates
    # if we use the LCM strategy.
    
    candidate_prime_sets = [
        {2, 3},
        {2, 3, 5},
        {2, 3, 5, 7},
        {2, 3, 5, 7, 11},
        {2, 3, 5, 7, 11, 13},
        {2, 3, 5, 7, 11, 13, 17},
    ]
    
    target_size = 12
    
    for prime_set in candidate_prime_sets:
        if time.time() > deadline:
            break
            
        # Calculate LCM of the prime set
        lcm_val = 1
        for p in prime_set:
            lcm_val *= p
            
        # Max value for k such that lcm_val * k <= 10000
        max_k = 10000 // lcm_val
        
        # Generate all k <= max_k such that prime factors of k are in prime_set
        # These are the "smooth" multipliers
        candidates = []
        for k in range(1, max_k + 1):
            if is_smooth(k, prime_set):
                candidates.append(k)
        
        if len(candidates) < target_size:
            continue
        
        # We need to find target_size distinct k's such that for any pair k_i, k_j:
        # prime factors of (lcm_val * k_i + lcm_val * k_j) are in prime_set
        # lcm_val * (k_i + k_j)
        # Since lcm_val is composed of primes in prime_set, 
        # the prime factors of the sum are prime_set U prime_factors(k_i + k_j).
        # So we need prime_factors(k_i + k_j) to be a subset of prime_set.
        
        n_cands = len(candidates)
        
        # Build adjacency list
        # adj[i] is a list of indices j > i such that candidates[i] + candidates[j] is smooth
        adj = [[] for _ in range(n_cands)]
        
        for i in range(n_cands):
            ki = candidates[i]
            for j in range(i + 1, n_cands):
                kj = candidates[j]
                s = ki + kj
                if is_smooth(s, prime_set):
                    adj[i].append(j)
        
        # Backtracking to find a clique of size target_size
        # We use a simple recursive search with pruning
        # Order candidates by degree to improve pruning? 
        # The candidates are already sorted by value.
        
        found = False
        clique = []
        
        def backtrack(start_idx, current_clique):
            nonlocal found
            if time.time() > deadline:
                return
            if len(current_clique) == target_size:
                found = True
                clique = current_clique[:]
                return
            if found:
                return
                
            remaining_needed = target_size - len(current_clique)
            # Pruning: if not enough nodes left in the range [start_idx, n_cands), fail
            if n_cands - start_idx < remaining_needed:
                return
            
            # Iterate through candidates starting from start_idx
            for i in range(start_idx, n_cands):
                # Pruning: if i is not compatible with all in current_clique, skip
                # Note: current_clique contains indices into 'candidates'
                # We need to check if i is connected to all nodes in current_clique
                
                # Optimization: 
                # The nodes in current_clique are < i (since we iterate i increasing and add to clique)
                # So we just need to check if i is in adj[node] for all node in current_clique?
                # No, adj is directional i < j.
                # If current_clique has nodes c1, c2... all < i.
                # We need edge (c, i) for all c in current_clique.
                # Since c < i, we check if i is in adj[c].
                
                compatible = True
                for c in current_clique:
                    # Check if i is in adj[c]
                    # Since adj[c] is sorted (as j iterates increasing), we can use binary search or linear scan
                    # Given small sizes, linear scan or a set is fine.
                    # Let's precompute adjacency sets for faster lookup
                    pass
                
                # Let's use a set for adjacency for O(1) lookup
                # Rebuild adj as sets?
                # adj_set[i] = set of j such that i is connected to j.
                
                if not compatible:
                    continue
                
                current_clique.append(i)
                backtrack(i + 1, current_clique)
                current_clique.pop()
                
                if found:
                    return
        
        # Convert adj lists to sets for O(1) lookup in backtracking
        # Actually, for small n_cands, lists are fine, but sets are safer for "is i in adj[c]"
        adj_sets = [set(lst) for lst in adj]
        
        # Modify backtrack to use adj_sets
        def backtrack_v2(start_idx, current_clique):
            nonlocal found
            if time.time() > deadline:
                return
            if len(current_clique) == target_size:
                found = True
                clique = current_clique[:]
                return
            if found:
                return
                
            remaining_needed = target_size - len(current_clique)
            if n_cands - start_idx < remaining_needed:
                return
            
            for i in range(start_idx, n_cands):
                compatible = True
                for c in current_clique:
                    if i not in adj_sets[c]:
                        compatible = False
                        break
                
                if compatible:
                    current_clique.append(i)
                    backtrack_v2(i + 1, current_clique)
                    current_clique.pop()
                    if found:
                        return
        
        backtrack_v2(0, [])
        
        if found:
            # Construct the final numbers
            result = [lcm_val * candidates[idx] for idx in clique]
            # Verify distinctness and limit
            # They are distinct because candidates are distinct
            # They are <= 10000 by construction
            return result
            
    # Fallback if no clique found in reasonable time or with these prime sets
    # Return a valid set of 12 distinct integers
    # The seed program returns list(range(1, 13))
    # We should try to return something better than random if possible, but range(1,13) is safe.
    # Let's try to return 12 multiples of 2 to at least get 2 as a factor?
    # But we might have already tried {2,3}.
    # Just return the seed fallback.
    return list(range(1, 13))

if __name__ == "__main__":
    res = construct()
    print(res)
