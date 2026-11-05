import math
import time

def construct():
    n = 255
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    # Prime factors of 255: 3, 5, 17
    # Every element in the set must share a factor with 255, so must be divisible by 3, 5, or 17.
    
    # Candidate pool: numbers in [1,255] divisible by 3, 5, or 17
    candidates = set()
    for i in range(1, n + 1):
        if i % 3 == 0 or i % 5 == 0 or i % 17 == 0:
            candidates.add(i)
    
    # 255 must be in the set
    # Start with the baseline: all multiples of 3 (size 85)
    baseline = list(range(3, n + 1, 3))
    baseline_set = set(baseline)
    
    best_set = set(baseline)
    best_size = len(best_set)
    
    # Try greedy heuristic: start with 255, add candidates one by one
    # that maintain the pairwise gcd > 1 property
    # Use multiple strategies and keep the best
    
    def is_pairwise_noncoprime(s):
        lst = list(s)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if math.gcd(lst[i], lst[j]) == 1:
                    return False
        return True
    
    def greedy_from(start_elements, candidate_list):
        """Greedy: start with start_elements, add candidates that connect to all current elements."""
        current = set(start_elements)
        remaining = list(candidate_list)
        
        while remaining:
            if time.time() > deadline:
                break
            # Find candidates that have gcd > 1 with ALL elements in current
            valid = []
            for c in remaining:
                if c in current:
                    continue
                ok = True
                for e in current:
                    if math.gcd(c, e) == 1:
                        ok = False
                        break
                if ok:
                    valid.append(c)
            if not valid:
                break
            # Pick the one with the most connections (sum of gcd values or count of shared factors)
            # Use count of other candidates it connects to as heuristic
            best_c = max(valid, key=lambda c: sum(1 for v in valid if v != c and math.gcd(c, v) > 1))
            current.add(best_c)
            remaining.remove(best_c)
        
        return current
    
    # Strategy 1: Start with just 255, greedily add
    # Sort candidates by how many other candidates they connect to (descending)
    cand_list = [c for c in candidates if c != 255]
    
    # Precompute connection counts for ordering
    def connection_count(c, pool):
        count = 0
        for other in pool:
            if other != c and math.gcd(c, other) > 1:
                count += 1
        return count
    
    if time.time() < deadline:
        # Strategy 1: Greedy from {255}
        s1 = greedy_from([255], cand_list)
        if len(s1) > best_size:
            best_set = s1
            best_size = len(best_set)
    
    # Strategy 2: Start with multiples of 3, try to add more
    if time.time() < deadline:
        # From the multiples of 3 set, try to add non-multiples of 3
        mult3 = set(range(3, n + 1, 3))
        non_mult3 = [c for c in candidates if c % 3 != 0]
        # Try adding each non-mult3 one at a time (removing conflicting elements if needed)
        # Simpler: try greedy addition
        s2 = greedy_from(list(mult3), non_mult3)
        if len(s2) > best_size:
            best_set = s2
            best_size = len(best_set)
    
    # Strategy 3: Try different starting points
    # Start with {255, 6} (6 = 2*3, connects to many)
    if time.time() < deadline:
        s3 = greedy_from([255, 6], [c for c in cand_list if c != 6])
        if len(s3) > best_size:
            best_set = s3
            best_size = len(best_set)
    
    # Strategy 4: Start with {255, 10} (10 = 2*5)
    if time.time() < deadline:
        s4 = greedy_from([255, 10], [c for c in cand_list if c != 10])
        if len(s4) > best_size:
            best_set = s4
            best_size = len(best_set)
    
    # Strategy 5: Start with {255, 15} (15 = 3*5, highly connected)
    if time.time() < deadline:
        s5 = greedy_from([255, 15], [c for c in cand_list if c != 15])
        if len(s5) > best_size:
            best_set = s5
            best_size = len(best_set)
    
    # Strategy 6: Start with {255, 30} (30 = 2*3*5)
    if time.time() < deadline:
        s6 = greedy_from([255, 30], [c for c in cand_list if c != 30])
        if len(s6) > best_size:
            best_set = s6
            best_size = len(best_set)
    
    # Strategy 7: Try starting with multiples of 5
    if time.time() < deadline:
        mult5 = set(range(5, n + 1, 5))
        non_mult5 = [c for c in candidates if c % 5 != 0]
        s7 = greedy_from(list(mult5), non_mult5)
        if len(s7) > best_size:
            best_set = s7
            best_size = len(best_set)
    
    # Strategy 8: Try a more aggressive greedy with different tie-breaking
    if time.time() < deadline:
        # Sort candidates by number of prime factors (more factors = more connections)
        def num_prime_factors(x):
            count = 0
            d = 2
            tmp = x
            while d * d <= tmp:
                if tmp % d == 0:
                    count += 1
                    while tmp % d == 0:
                        tmp //= d
                d += 1
            if tmp > 1:
                count += 1
            return count
        
        sorted_cands = sorted(cand_list, key=num_prime_factors, reverse=True)
        s8 = greedy_from([255], sorted_cands)
        if len(s8) > best_size:
            best_set = s8
            best_size = len(best_set)
    
    # Ensure 255 is in the result
    if 255 not in best_set:
        best_set.add(255)
    
    return list(best_set)
