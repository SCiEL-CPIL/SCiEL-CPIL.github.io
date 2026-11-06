import time
import itertools
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 60  # 60s limit, keep margin
    
    # Range of interest
    N = 256
    
    # Precompute reciprocals
    recips = {i: 1.0 / i for i in range(2, N + 1)}
    
    # Base set: all numbers in [129, 256]
    # We know any subset of [129, 256] is valid internally.
    # We will define a function to compute the valid large set given a set of small numbers.
    
    def get_large_set(small_set):
        """
        Returns the list of y in [129, 256] such that lcm(y, s) > 256 for all s in small_set.
        """
        large_candidates = list(range(129, N + 1))
        if not small_set:
            return large_candidates
        
        valid_large = []
        for y in large_candidates:
            is_valid = True
            for s in small_set:
                # lcm(s, y) > 256
                g = gcd(s, y)
                lcm_val = (s * y) // g
                if lcm_val <= 256:
                    is_valid = False
                    break
            if is_valid:
                valid_large.append(y)
        return valid_large

    def calculate_score(small_set, large_set):
        score = sum(recips[s] for s in small_set)
        score += sum(recips[y] for y in large_set)
        return score

    best_score = 0.0
    best_set = set()
    
    # Case 1: No small numbers
    large_set = get_large_set([])
    score = calculate_score([], large_set)
    if score > best_score:
        best_score = score
        best_set = set(large_set)
        
    # Case 2: Single small number
    for s in range(2, 129):
        if time.time() > deadline:
            break
        small_set = [s]
        large_set = get_large_set(small_set)
        score = calculate_score(small_set, large_set)
        if score > best_score:
            best_score = score
            best_set = set(small_set) | set(large_set)
            
    # Case 3: Pairs of small numbers
    # Only consider pairs where lcm(s1, s2) > 256
    smalls = list(range(2, 129))
    # Optimization: precompute valid pairs
    valid_pairs = []
    for i in range(len(smalls)):
        s1 = smalls[i]
        for j in range(i + 1, len(smalls)):
            s2 = smalls[j]
            g = gcd(s1, s2)
            lcm_val = (s1 * s2) // g
            if lcm_val > 256:
                valid_pairs.append((s1, s2))
                
    for s1, s2 in valid_pairs:
        if time.time() > deadline:
            break
        small_set = [s1, s2]
        large_set = get_large_set(small_set)
        score = calculate_score(small_set, large_set)
        if score > best_score:
            best_score = score
            best_set = set(small_set) | set(large_set)
            
    # Case 4: Triples of small numbers
    # This might be too slow if we iterate all triples.
    # But we can filter triples from valid pairs.
    # A triple (s1, s2, s3) is valid if all pairs are valid.
    # We can iterate over valid pairs (s1, s2) and find s3 such that (s1, s3) and (s2, s3) are valid.
    
    # To keep it fast, we might skip triples or limit the search.
    # Given that adding small numbers reduces the large set significantly,
    # and 1/s is diminishing, triples might not improve over pairs.
    # But let's try a limited search for triples if time permits.
    
    # Build a graph of compatible small numbers
    # compat[s] = set of s' such that lcm(s, s') > 256
    compat = {s: set() for s in smalls}
    for s1, s2 in valid_pairs:
        compat[s1].add(s2)
        compat[s2].add(s1)
        
    # Iterate over pairs and extend to triples
    triples_checked = 0
    for s1, s2 in valid_pairs:
        if time.time() > deadline:
            break
        # Find s3 compatible with both s1 and s2
        # s3 must be in compat[s1] and compat[s2]
        common = compat[s1].intersection(compat[s2])
        for s3 in common:
            if s3 <= s2: # Avoid duplicates and ordering
                continue
            small_set = [s1, s2, s3]
            large_set = get_large_set(small_set)
            score = calculate_score(small_set, large_set)
            if score > best_score:
                best_score = score
                best_set = set(small_set) | set(large_set)
            triples_checked += 1
            # Limit triples to avoid timeout
            if triples_checked > 1000:
                break
        if triples_checked > 1000:
            break
            
    # Ensure non-empty
    if not best_set:
        # Fallback: just pick 256
        return [256]
        
    return list(best_set)

# EVOLVE-BLOCK-END
