import time
import math
import sys

_START = time.time()
_HARD_DEADLINE = _START + 80.0

def _init_spf(limit):
    spf = list(range(limit + 1))
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

SPF_LIMIT = 20000
_spf = _init_spf(SPF_LIMIT)

def _prime_factors(num):
    if num <= 1:
        return frozenset()
    out = set()
    x = num
    while x > 1:
        p = _spf[x]
        out.add(p)
        while x % p == 0:
            x //= p
    return frozenset(out)

def _is_smooth(num, allowed_primes):
    if num <= 1:
        return True
    for p in allowed_primes:
        while num % p == 0:
            num //= p
    return num == 1

def _find_smooth_clique(n_target, primes, max_k, time_limit):
    """
    Find n_target distinct integers k_i in 1..max_k such that 
    k_i + k_j is P-smooth for all i != j.
    """
    # Precompute smoothness for sums up to 2*max_k
    max_sum = 2 * max_k
    smooth_sums = [False] * (max_sum + 1)
    for s in range(2, max_sum + 1):
        if _is_smooth(s, primes):
            smooth_sums[s] = True
    
    # Precompute compatibility
    # compat[i] is a set of j such that i+j is smooth
    # We use 1-based indexing for k values to match 1..max_k
    # But for speed, let's use 0-based indices for the list of candidates
    candidates = list(range(1, max_k + 1))
    N = len(candidates)
    
    # Map value to index? Or just use values directly?
    # Values are 1..max_k.
    # compat_val[v] = set of u such that u+v is smooth
    compat_val = [set() for _ in range(max_k + 1)]
    for v in range(1, max_k + 1):
        for u in range(1, max_k + 1):
            if u != v and smooth_sums[u + v]:
                compat_val[v].add(u)
                
    # Sort candidates by degree (number of compatible partners) descending
    # High degree nodes are more likely to be part of a large clique
    degrees = [len(compat_val[v]) for v in candidates]
    indices = list(range(len(candidates)))
    indices.sort(key=lambda i: degrees[i], reverse=True)
    sorted_candidates = [candidates[i] for i in indices]
    
    # Backtracking search
    def backtrack(selected, available):
        if len(selected) == n_target:
            return selected[:]
        
        if time.time() > time_limit:
            return None
            
        # Heuristic: pick the first available candidate that has the most connections 
        # to the rest of the available candidates?
        # Or just iterate through available.
        # To prune, we can check if any candidate in 'available' is compatible with all in 'selected'
        # 'available' is already filtered to be compatible with all in 'selected'
        
        for i in range(len(available)):
            v = available[i]
            
            # Check if adding v allows us to complete the clique
            # The new available set will be the intersection of current available (excluding v)
            # and compat_val[v]
            
            new_available = []
            for j in range(i + 1, len(available)):
                u = available[j]
                if u in compat_val[v]:
                    new_available.append(u)
            
            # If we can't reach n_target, prune
            if len(selected) + 1 + len(new_available) < n_target:
                continue
                
            result = backtrack(selected + [v], new_available)
            if result:
                return result
                
        return None

    # Start with all candidates
    # We need to be careful with the initial 'available' set
    # It should be all candidates.
    # But 'backtrack' assumes 'available' is sorted and we pick from it.
    # The logic above iterates 'available' and builds 'new_available' from the tail.
    # This requires 'available' to be sorted in a way that preserves the "future" candidates.
    # Since we just need *a* clique, any order works, but we want to try promising ones first.
    # The sorted_candidates order is good for the first pick.
    # For subsequent picks, we filter.
    
    # However, the standard clique finding approach:
    # P = set of candidates
    # X = set of selected
    # While P is not empty:
    #   Pick v from P
    #   If v is compatible with all in X:
    #      Add v to X
    #      P = P intersect N(v)
    #   Else:
    #      Remove v from P
      
    # Let's implement a simpler recursive search on the sorted list.
    
    def search(selected, candidates_list):
        if len(selected) == n_target:
            return selected[:]
        if time.time() > time_limit:
            return None
        if not candidates_list:
            return None
            
        # Try to add candidates from the list
        # We can prune if len(selected) + len(candidates_list) < n_target
        if len(selected) + len(candidates_list) < n_target:
            return None
            
        for i in range(len(candidates_list)):
            v = candidates_list[i]
            
            # v is compatible with all in selected (guaranteed by how candidates_list is formed)
            
            # New candidates must be compatible with v
            # And must be from candidates_list[i+1:] to avoid duplicates and backtracking
            new_candidates = []
            for j in range(i + 1, len(candidates_list)):
                u = candidates_list[j]
                if u in compat_val[v]:
                    new_candidates.append(u)
            
            # Prune
            if len(selected) + 1 + len(new_candidates) < n_target:
                continue
                
            res = search(selected + [v], new_candidates)
            if res:
                return res
                
        return None

    # Initial call
    # We pass the sorted candidates.
    # Note: The search function assumes candidates_list contains only nodes compatible with all in 'selected'.
    # Initially selected is empty, so all nodes are compatible.
    return search([], sorted_candidates)

def construct():
    deadline = time.time() + 80.0
    n = 12
    max_val = 10000
    
    prime_sets = [
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11]
    ]
    
    for primes in prime_sets:
        if time.time() > deadline - 5.0:
            break
            
        # Try to find a small set of k_i
        # Start with small max_k, increase if not found
        for max_k in [100, 500, 2000]:
            time_limit = deadline - 2.0
            clique = _find_smooth_clique(n, primes, max_k, time_limit)
            if clique:
                # Scale the clique to fit within max_val
                max_k_val = max(clique)
                if max_k_val == 0:
                    continue
                # Find max multiplier M such that M * max_k_val <= max_val
                # M must be integer >= 1
                # Also, we just need the result to be distinct and <= max_val
                # The prime factors of sums will be subset of primes
                # So the score is len(primes)
                
                # We can choose M=1 if max_k_val <= max_val
                # Or larger M to "spread" them out? No, M just scales.
                # The values are M * k_i.
                # We need M * max_k_val <= max_val.
                M = max_val // max_k_val
                if M < 1:
                    M = 1 # Should not happen if max_k_val <= max_val
                    # If max_k_val > max_val, we can't use this clique directly with M=1.
                    # But we searched k_i in 1..max_k.
                    # If max_k > max_val, we should have restricted k_i to max_val.
                    # In our loop, max_k is 100, 500, 2000. All <= 10000.
                    # So M will be >= 1.
                    
                result = [M * k for k in clique]
                
                # Verify distinctness
                if len(set(result)) == n and all(x <= max_val for x in result):
                    return result
                else:
                    # Fallback to unscaled if scaling caused issues (shouldn't)
                    if max_k_val <= max_val:
                         return clique
                    else:
                         continue

    # Fallback
    # Return a set with minimal known primes
    # {2,3,5,7} is a safe bet for small numbers
    result = [1, 2, 3, 5, 6, 9, 10, 15, 18, 25, 27, 30]
    return result
