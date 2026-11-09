"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START

def _smallest_prime_factor(n, limit):
    """Return smallest prime factor of n, or None if n is 1 or no factor <= limit."""
    if n <= 1:
        return None
    if n % 2 == 0:
        return 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            return f
        f += 2
    return n  # n is prime

def _factorize(n, prime_set):
    """Factor n using primes in prime_set. Return set of prime factors found, or None if not fully factorable."""
    result = set()
    for p in prime_set:
        if n % p == 0:
            result.add(p)
            while n % p == 0:
                n //= p
    if n > 1:
        return None
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s out of 90s budget
    
    # We want to find 12 distinct positive integers <= 10000
    # such that the union of prime factors of all pairwise sums is minimized.
    
    # Strategy: Use a small set of primes P, and find 12 numbers such that
    # all pairwise sums are P-smooth (i.e., have no prime factors outside P).
    
    # Try increasing sets of primes: {2,3}, {2,3,5}, {2,3,5,7}, {2,3,5,7,11}, etc.
    
    # For a given prime set P, we need to find 12 distinct numbers a1,...,a12 in [1,10000]
    # such that for all i != j, ai + aj is P-smooth.
    
    # Approach: 
    # 1. Generate all P-smooth numbers up to 20000 (max possible sum is 10000+9999=19999)
    # 2. For each pair (x, y) where x+y is P-smooth, we can have both x and y in our set
    # 3. This forms a graph where vertices are numbers 1..10000, and edges connect x,y if x+y is P-smooth
    # 4. Find a clique of size 12 in this graph
    
    # Better approach: Work with residues or just directly search.
    
    # Let's try a direct approach:
    # For small prime sets, the P-smooth numbers are relatively sparse.
    # Generate all P-smooth numbers up to 20000.
    # For each pair (a, b) with a < b and a+b P-smooth, both a and b could be in our set.
    # We need 12 numbers all pairwise connected.
    
    # Actually, let me think differently. If all pairwise sums must be P-smooth,
    # then the numbers themselves don't need to be P-smooth, just their sums.
    
    # Let's use a greedy/backtracking approach:
    # 1. For a candidate prime set P, generate all P-smooth numbers up to 20000.
    # 2. Build a compatibility graph: node i (1..10000) connects to node j if i+j is P-smooth.
    # 3. Find a 12-clique.
    
    # This graph has 10000 nodes. Building it explicitly is expensive.
    # Instead, use a different approach:
    
    # Key insight: If we fix one number a, then all other numbers b must satisfy
    # a+b is P-smooth, i.e., b is in {s - a : s is P-smooth, 1 <= s-a <= 10000, s-a != a}.
    # So for each candidate a, the compatible set is determined.
    
    # Let's try: for small prime sets, enumerate candidate a values, compute their
    # compatible sets, and look for intersections of size >= 11.
    
    prime_sets_to_try = [
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13],
    ]
    
    for prime_set in prime_sets_to_try:
        if time.time() > deadline:
            break
        ps = set(prime_set)
        max_sum = 19999  # max is 10000 + 9999
        
        # Generate all P-smooth numbers up to max_sum
        smooth = set()
        def gen_smooth(n, primes):
            if n > max_sum:
                return
            smooth.add(n)
            for p in primes:
                if n * p <= max_sum:
                    gen_smooth(n * p, primes)
        
        gen_smooth(1, sorted(ps))
        
        # For each a in 1..10000, find compatible b's
        # compatible[a] = set of b != a such that a+b is in smooth
        # But this is too memory-intensive for 10000 nodes.
        
        # Alternative: For each a, the set of valid b's is {s - a for s in smooth if 1 <= s-a <= 10000 and s-a != a}
        # Let's just check: can we find 12 numbers?
        
        # Simpler approach: pick a random starting point and do local search.
        # Or: enumerate small a values and check if their compatible sets have large intersections.
        
        # Let's try: for a in a subset of 1..10000, compute the set of valid partners.
        # If two numbers a, b are both valid and a+b is smooth, they're compatible.
        # We need 12 mutually compatible numbers.
        
        # Let's use a different strategy: 
        # Pick numbers from a structured set. E.g., all multiples of some base,
        # or numbers of the form k*m + r for fixed m, r.
        
        # Actually, let's just do a direct search with the smooth set.
        # For efficiency, only consider a values where many smooth numbers are close.
        
        # Let me try: for each a in 1..10000 (sampled), compute valid_b set.
        # Then check if there's a 12-clique in the induced subgraph.
        
        # This is still hard. Let me try a simpler heuristic:
        # Start with a random set of 12 numbers, and iteratively replace the "worst"
        # number to reduce the prime factor count.
        
        pass
    
    # Fallback: local search approach
    # Start with a random set of 12 numbers, minimize the number of distinct prime factors
    # of all pairwise sums via local search (swap-based).
    
    random.seed(42)
    
    def count_primes(arr):
        """Count distinct prime factors of all pairwise sums."""
        primes = set()
        n = len(arr)
        for i in range(n):
            for j in range(i+1, n):
                s = arr[i] + arr[j]
                # Factor s
                temp = s
                d = 2
                while d * d <= temp:
                    if temp % d == 0:
                        primes.add(d)
                        while temp % d == 0:
                            temp //= d
                    d += 1
                if temp > 1:
                    primes.add(temp)
        return primes
    
    def objective(arr):
        return len(count_primes(arr))
    
    # Initial solution: try structured sets
    # Set 1: small numbers
    init1 = list(range(1, 13))
    # Set 2: multiples of 2
    init2 = [2*i for i in range(1, 13)]
    # Set 3: random
    init3 = random.sample(range(1, 10001), 12)
    
    best_arr = None
    best_score = float('inf')
    
    for init in [init1, init2, init3]:
        arr = list(init)
        score = objective(arr)
        if score < best_score:
            best_score = score
            best_arr = list(arr)
    
    # Local search: try swaps
    improved = True
    while improved and time.time() < deadline:
        improved = False
        arr = list(best_arr)
        current_score = objective(arr)
        
        # Try replacing each element with a random alternative
        for i in range(12):
            if time.time() > deadline:
                break
            for _ in range(200):  # 200 random candidates per position
                new_val = random.randint(1, 10000)
                if new_val in arr and arr.index(new_val) != i:
                    continue
                old_val = arr[i]
                arr[i] = new_val
                new_score = objective(arr)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    best_arr = list(arr)
                    best_score = new_score
                else:
                    arr[i] = old_val
            if time.time() > deadline:
                break
    
    # Also try the structured approach: find numbers where sums are smooth
    # Try with prime set {2, 3, 5}
    if time.time() < deadline:
        # Generate 5-smooth numbers up to 20000
        smooth5 = set()
        for i in range(0, 15):  # 2^i
            for j in range(0, 10):  # 3^j
                for k in range(0, 8):  # 5^k
                    v = (2**i) * (3**j) * (5**k)
                    if v <= 20000:
                        smooth5.add(v)
        
        # For each a in 1..10000, valid partners are {s-a for s in smooth5 if 1<=s-a<=10000}
        # Find a set of 12 numbers all pairwise with smooth sums.
        # Use greedy: pick a, then pick b compatible with a, then c compatible with a,b, etc.
        
        best_struct = None
        best_struct_score = float('inf')
        
        # Try different starting points
        for start_a in range(1, min(100, 10001)):
            if time.time() > deadline:
                break
            # Find all b compatible with start_a
            compat_a = set()
            for s in smooth5:
                b = s - start_a
                if 1 <= b <= 10000 and b != start_a:
                    compat_a.add(b)
            
            if len(compat_a) < 11:
                continue
            
            # Greedily build a set
            current_set = {start_a}
            candidates = list(compat_a)
            random.shuffle(candidates)
            
            for b in candidates:
                if len(current_set) >= 12:
                    break
                # Check if b is compatible with all in current_set
                compatible = True
                for c in current_set:
                    if (b + c) not in smooth5:
                        compatible = False
                        break
                if compatible:
                    current_set.add(b)
            
            if len(current_set) == 12:
                arr = list(current_set)
                score = objective(arr)
                if score < best_struct_score:
                    best_struct_score = score
                    best_struct = list(arr)
        
        if best_struct is not None:
            if best_struct_score < best_score:
                best_score = best_struct_score
                best_arr = best_struct
    
    # Final check
    if best_arr is None:
        best_arr = list(range(1, 13))
    
    # Ensure distinct and within bounds
    best_arr = sorted(set(best_arr))
    while len(best_arr) < 12:
        best_arr.append(random.randint(1, 10000))
    best_arr = best_arr[:12]
    
    return best_arr

# EVOLVE-BLOCK-END
