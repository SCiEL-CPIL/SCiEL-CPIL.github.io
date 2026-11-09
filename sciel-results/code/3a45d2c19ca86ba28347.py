"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from math import gcd
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 88.0

    def time_exceeded():
        return time.time() - start_time > deadline - 2.0

    # Precompute prime factors for all possible sums (2..20000)
    MAX_SUM = 20000
    pf = [set() for _ in range(MAX_SUM + 1)]
    for p in range(2, MAX_SUM + 1):
        if not pf[p]:  # p is prime
            for m in range(p, MAX_SUM + 1, p):
                pf[m].add(p)

    def count_distinct_primes(arr):
        """Count distinct prime factors across all pairwise sums."""
        primes = set()
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                primes |= pf[arr[i] + arr[j]]
        return len(primes)

    def get_primes_for(arr):
        """Return the set of primes for a given array."""
        primes = set()
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                primes |= pf[arr[i] + arr[j]]
        return primes

    def is_smooth(n, prime_set):
        """Check if n has only prime factors in prime_set."""
        if n <= 1:
            return True
        for p in prime_set:
            while n % p == 0:
                n //= p
        return n == 1

    # Strategy 1: Powers of 2 (structural) - baseline
    powers_of_2 = [2**i for i in range(12)]
    assert all(x <= 10000 for x in powers_of_2)
    assert len(set(powers_of_2)) == 12

    best_result = list(powers_of_2)
    best_count = count_distinct_primes(best_result)

    # Strategy 2: CSP approach - find clique of size 12 where all pairwise sums are 7-smooth
    # This guarantees at most 4 distinct primes (2, 3, 5, 7)
    
    # Precompute which sums are 7-smooth
    smooth_7 = [False] * (MAX_SUM + 1)
    for s in range(2, MAX_SUM + 1):
        n = s
        for p in [2, 3, 5, 7]:
            while n % p == 0:
                n //= p
        smooth_7[s] = (n == 1)

    # Build compatibility: for each number x (1..10000), find all y > x such that x+y is 7-smooth
    # We'll use a smarter approach: group by residues mod 2,3,5,7 to prune
    
    # For 7-smooth sums, x+y must be of form 2^a * 3^b * 5^c * 7^d
    # Let's precompute all 7-smooth numbers up to 20000
    smooth_numbers = set()
    for a in range(15):  # 2^a <= 20000
        v2 = 2**a
        if v2 > MAX_SUM:
            break
        for b in range(10):  # 3^b <= 20000
            v3 = v2 * (3**b)
            if v3 > MAX_SUM:
                break
            for c in range(7):  # 5^c <= 20000
                v5 = v3 * (5**c)
                if v5 > MAX_SUM:
                    break
                for d in range(5):  # 7^d <= 20000
                    v7 = v5 * (7**d)
                    if v7 > MAX_SUM:
                        break
                    if v7 >= 2:
                        smooth_numbers.add(v7)

    # For backtracking, we need to find 12 distinct numbers from 1..10000
    # such that all pairwise sums are in smooth_numbers
    
    # Key insight: if all numbers are even, sums are even -> need 2-smooth part
    # If all numbers are multiples of 2, sums are multiples of 2
    # Let's try: all numbers are powers of 2 times small odd factors
    
    # Better approach: use the structure that x+y = 2^a*3^b*5^c*7^d
    # If we pick all x as powers of 2: x=2^i, y=2^j, sum=2^i+2^j=2^min(i,j)*(1+2^|i-j|)
    # 1+2^k is odd, and for k>=1, 1+2^k can have various prime factors
    # This is why powers of 2 give more than 4 primes
    
    # New strategy: pick numbers that are all multiples of a large power of 2
    # e.g., x = 2^k * m where m is small
    # Then x+y = 2^k*(m1+m2), and m1+m2 needs to be 7-smooth
    # If k is large enough, 2^k handles the power of 2, and m1+m2 <= 2*max_m
    
    # Let's try: x_i = 2^10 * i for i in some set, but need m1+m2 to be 7-smooth
    # Actually, let's use the CSP backtracking directly
    
    # Build adjacency: for each x, which y (y > x) have x+y 7-smooth
    # To make this tractable, restrict to a smaller universe first
    
    # Try universe: numbers of form 2^a * 3^b * 5^c * 7^d <= 10000 (7-smooth numbers themselves)
    # If x and y are both 7-smooth, is x+y 7-smooth? Not necessarily.
    # But let's try a different structure.
    
    # Key: if all x are congruent to 0 mod 2^k for large k, then x+y is 0 mod 2^k
    # Let's try x_i = 2^10 * a_i where a_i are small distinct integers
    # Then x_i + x_j = 2^10 * (a_i + a_j)
    # Need a_i + a_j to be 7-smooth for all i != j
    # a_i + a_j <= 2 * max_a, so if max_a is small, we can check
    
    # Let's find 12 distinct small positive integers a_1,...,a_12 such that all a_i+a_j are 7-smooth
    # Then x_i = 2^10 * a_i <= 10000 means a_i <= 10000/1024 = 9.76, so a_i <= 9
    # But we need 12 distinct values from 1..9, impossible.
    
    # Try 2^8 = 256: a_i <= 10000/256 = 39.06, so a_i <= 39
    # Need 12 distinct a_i in 1..39 with all pairwise sums 7-smooth
    # Let's use backtracking for this smaller problem
    
    def find_smooth_clique(max_val, size, smooth_set):
        """Find 'size' distinct integers from 1..max_val such that all pairwise sums are in smooth_set."""
        # Backtracking with forward checking
        candidates = list(range(1, max_val + 1))
        
        def backtrack(selected, remaining_candidates):
            if time_exceeded():
                return None
            if len(selected) == size:
                return list(selected)
            
            # Try each candidate
            for i, c in enumerate(remaining_candidates):
                # Check if c is compatible with all selected
                compatible = True
                for s in selected:
                    if (s + c) not in smooth_set:
                        compatible = False
                        break
                if not compatible:
                    continue
                
                # Forward checking: for each remaining candidate after i, check if it's still compatible with all selected + c
                new_remaining = remaining_candidates[i+1:]
                new_selected = selected + [c]
                
                # Prune: check if there are enough compatible candidates left
                if len(new_remaining) < size - len(new_selected):
                    continue
                
                result = backtrack(new_selected, new_remaining)
                if result is not None:
                    return result
            return None
        
        return backtrack([], candidates)
    
    # Try with 2^8 scaling: a_i in 1..39
    smooth_set_39 = set()
    for s in range(2, 80):  # max sum is 39+38=77
        if s in smooth_numbers:
            smooth_set_39.add(s)
    
    # For backtracking efficiency, precompute compatibility
    compat = {}
    for a in range(1, 40):
        for b in range(a + 1, 40):
            if (a + b) in smooth_numbers:
                if a not in compat:
                    compat[a] = set()
                compat[a].add(b)
                if b not in compat:
                    compat[b] = set()
                compat[b].add(a)
    
    # Use a more efficient backtracking with degree ordering
    def find_clique_efficient(size=12):
        """Find clique of given size in the compatibility graph on 1..39."""
        # Get all nodes with their neighbors
        nodes = list(range(1, 40))
        
        # Order by degree (descending) for better pruning
        def degree(node):
            return len(compat.get(node, set()))
        
        # Backtracking
        def backtrack(selected, available):
            if time_exceeded():
                return None
            if len(selected) == size:
                return list(selected)
            
            # Forward checking: filter available to only those compatible with all selected
            filtered = []
            for v in available:
                ok = True
                for s in selected:
                    if v not in compat.get(s, set()):
                        ok = False
                        break
                if ok:
                    filtered.append(v)
            
            if len(filtered) < size - len(selected):
                return None
            
            # Try each candidate in filtered, ordered by degree
            for v in filtered:
                new_selected = selected + [v]
                new_available = [u for u in filtered if u > v]  # maintain ordering to avoid duplicates
                
                result = backtrack(new_selected, new_available)
                if result is not None:
                    return result
            return None
        
        return backtrack([], nodes)
    
    a_clique = find_clique_efficient(12)
    
    if a_clique is not None and len(a_clique) == 12:
        # Scale by 2^8
        scaled = [a * 256 for a in a_clique]
        if len(set(scaled)) == 12 and all(1 <= x <= 10000 for x in scaled):
            cnt = count_distinct_primes(scaled)
            if cnt < best_count:
                best_count = cnt
                best_result = list(scaled)
            if best_count <= 4:
                return best_result
    
    # If that didn't work, try 2^7 = 128 scaling: a_i in 1..78
    smooth_set_78 = set()
    for s in range(2, 157):  # max sum is 78+77=155
        if s in smooth_numbers:
            smooth_set_78.add(s)
    
    # Build compatibility for 1..78
    compat2 = {}
    for a in range(1, 79):
        for b in range(a + 1, 79):
            if (a + b) in smooth_numbers:
                if a not in compat2:
                    compat2[a] = set()
                compat2[a].add(b)
                if b not in compat2:
                    compat2[b] = set()
                compat2[b].add(a)
    
    def find_clique_78(size=12):
        nodes = list(range(1, 79))
        
        def backtrack(selected, available):
            if time_exceeded():
                return None
            if len(selected) == size:
                return list(selected)
            
            filtered = []
            for v in available:
                ok = True
                for s in selected:
                    if v not in compat2.get(s, set()):
                        ok = False
                        break
                if ok:
                    filtered.append(v)
            
            if len(filtered) < size - len(selected):
                return None
            
            for v in filtered:
                new_selected = selected + [v]
                new_available = [u for u in filtered if u > v]
                result = backtrack(new_selected, new_available)
                if result is not None:
                    return result
            return None
        
        return backtrack([], nodes)
    
    a_clique2 = find_clique_78(12)
    
    if a_clique2 is not None and len(a_clique2) == 12:
        scaled = [a * 128 for a in a_clique2]
        if len(set(scaled)) == 12 and all(1 <= x <= 10000 for x in scaled):
            cnt = count_distinct_primes(scaled)
            if cnt < best_count:
                best_count = cnt
                best_result = list(scaled)
            if best_count <= 4:
                return best_result

    # Strategy 3: Structured sets (fallbacks)
    odds = list(range(1, 24, 2))[:12]
    cnt = count_distinct_primes(odds)
    if cnt < best_count:
        best_count = cnt
        best_result = list(odds)

    mult6 = [6*(i+1) for i in range(12)]
    cnt = count_distinct_primes(mult6)
    if cnt < best_count:
        best_count = cnt
        best_result = list(mult6)

    close = list(range(100, 112))
    cnt = count_distinct_primes(close)
    if cnt < best_count:
        best_count = cnt
        best_result = list(close)

    small = list(range(1, 13))
    cnt = count_distinct_primes(small)
    if cnt < best_count:
        best_count = cnt
        best_result = list(small)

    return best_result
# EVOLVE-BLOCK-END
