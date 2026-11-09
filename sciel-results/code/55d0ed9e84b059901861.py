"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0

    def time_exceeded():
        return time.time() - start_time > deadline - 2.0

    def get_prime_factors(n):
        """Return set of prime factors of n."""
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors

    def count_distinct_primes(arr):
        primes = set()
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                primes |= get_prime_factors(arr[i] + arr[j])
        return len(primes)

    def is_smooth(n, prime_set):
        if n <= 1:
            return True
        for p in prime_set:
            while n % p == 0:
                n //= p
        return n == 1

    # Precompute small primes for smoothness testing
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    # Strategy: For each candidate prime set (sorted by size), find 12 numbers
    # such that all pairwise sums are smooth over that set.
    # Use a smarter greedy: try to build a set where elements are close together
    # (so sums are in a limited range) and all sums avoid certain primes.
    
    # Key structural insight: If we pick numbers all congruent to r mod m for some
    # modulus m, then all pairwise sums are congruent to 2r mod m. This constrains
    # which primes can divide the sums.
    
    # Try: pick 12 numbers from an arithmetic progression or near-AP structure.
    # For prime set {2}: all sums must be powers of 2. Sums range from 1+2=3 to 9999+10000=19999.
    # Powers of 2 in that range: 4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384
    # We need 12 distinct numbers a<b<c<... such that all pairwise sums are powers of 2.
    # This is very restrictive. Let's check if {2} alone works.
    
    # For {2,3}: sums must be of form 2^a * 3^b. More flexibility.
    
    # Better structural approach: Use the fact that if all numbers are multiples of L,
    # then all sums are multiples of L. So prime factors of L are "wasted" but guaranteed.
    # Instead, try to make all sums have only a few prime factors by choosing numbers
    # in a structured way.
    
    # New structural idea: Pick numbers such that a_i + a_j is always in a small set
    # of "good" values (smooth numbers). Use graph coloring / clique finding.
    
    # Precompute: for each prime set, which pairs (x,y) with x<y have x+y smooth?
    # Then find a clique of size 12 in the compatibility graph.
    
    # But the graph is huge (10000 nodes). Instead, use a layered approach:
    # 1. For each prime set, generate candidate numbers (those that can participate
    #    in many smooth-sum pairs).
    # 2. Use greedy + local search to find 12 compatible numbers.
    
    prime_set_candidates = [
        (1, {2}),
        (1, {3}),
        (1, {5}),
        (1, {7}),
        (2, {2, 3}),
        (2, {2, 5}),
        (2, {3, 5}),
        (2, {3, 7}),
        (2, {5, 7}),
        (3, {2, 3, 5}),
        (3, {2, 3, 7}),
        (3, {2, 5, 7}),
        (3, {3, 5, 7}),
        (4, {2, 3, 5, 7}),
    ]
    
    best_result = None
    best_count = float('inf')
    
    for expected_primes, ps in prime_set_candidates:
        if time_exceeded():
            break
        
        # Find 12 numbers where all pairwise sums are smooth over ps
        # Use a different structural approach: 
        # Group numbers by their residue classes and exploit structure.
        
        # Approach: Try to find numbers in a small range first (easier to verify),
        # then expand. Use backtracking with pruning.
        
        # Precompute smooth sums: for each possible sum value s, check if smooth
        # Cache this for speed.
        smooth_cache = {}
        
        def is_smooth_cached(n):
            if n in smooth_cache:
                return smooth_cache[n]
            result = is_smooth(n, ps)
            smooth_cache[n] = result
            return result
        
        # Try to build a set of 12 numbers using backtracking
        # Start with small numbers and try to extend
        found = None
        
        # Try multiple starting points / strategies
        for start_offset in range(0, 100, 5):
            if time_exceeded():
                break
            
            # Try to find 12 numbers in range [start_offset+1, start_offset+200]
            # that are all pairwise smooth-sum compatible
            low = start_offset + 1
            high = min(start_offset + 300, 10000)
            
            # Greedy with backtracking
            def backtrack(current_set, start_idx):
                if len(current_set) == 12:
                    return list(current_set)
                if time_exceeded():
                    return None
                for i in range(start_idx, high + 1):
                    x = i
                    valid = True
                    for y in current_set:
                        if not is_smooth_cached(x + y):
                            valid = False
                            break
                    if valid:
                        current_set.append(x)
                        result = backtrack(current_set, i + 1)
                        if result is not None:
                            return result
                        current_set.pop()
                return None
            
            # Start with a few initial candidates to bootstrap
            # Try starting with the smallest compatible numbers
            initial_candidates = []
            for x in range(low, min(low + 50, high + 1)):
                # Check if x can be part of a large smooth-sum set
                # (heuristic: count how many other small numbers it's compatible with)
                compat_count = 0
                for y in range(low, min(low + 50, high + 1)):
                    if x != y and is_smooth_cached(x + y):
                        compat_count += 1
                if compat_count >= 10:
                    initial_candidates.append(x)
            
            if len(initial_candidates) >= 3:
                # Use the most compatible numbers as starting points
                initial_candidates.sort(key=lambda x: -sum(1 for y in initial_candidates if y != x and is_smooth_cached(x + y)))
                for start_x in initial_candidates[:3]:
                    result = backtrack([start_x], start_x + 1)
                    if result is not None:
                        found = result
                        break
            if found is not None:
                break
        
        if found is not None and len(found) == 12:
            if len(set(found)) == 12 and all(1 <= x <= 10000 for x in found):
                cnt = count_distinct_primes(found)
                if cnt < best_count:
                    best_count = cnt
                    best_result = found
                if best_count <= expected_primes:
                    break
    
    if best_result is not None:
        return best_result
    
    # Fallback: try the original greedy approach with more prime sets
    for ps in [{2}, {2,3}, {3}, {3,5}, {2,3,5}, {5}, {5,7}, {2,3,5,7}, {7}, {2,5}]:
        if time_exceeded():
            break
        candidates = []
        for x in range(1, 10001):
            if time_exceeded():
                break
            valid = True
            for y in candidates:
                if not is_smooth(x + y, ps):
                    valid = False
                    break
            if valid:
                candidates.append(x)
            if len(candidates) >= 12:
                break
        if len(candidates) >= 12:
            result = candidates[:12]
            if len(set(result)) == 12 and all(1 <= x <= 10000 for x in result):
                cnt = count_distinct_primes(result)
                if cnt < best_count:
                    best_count = cnt
                    best_result = result
                if best_count <= len(ps):
                    break
    
    if best_result is not None:
        return best_result
    
    # Final fallback
    return list(range(1, 13))
# EVOLVE-BLOCK-END
