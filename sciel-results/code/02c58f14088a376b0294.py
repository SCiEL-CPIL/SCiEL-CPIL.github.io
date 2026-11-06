"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

# Precompute prime factors for all possible sums (1 to 20000)
_MAX_SUM = 20000
_prime_factors_cache = [None] * (_MAX_SUM + 1)

def _init_factors():
    """Precompute prime factors for all numbers up to _MAX_SUM."""
    spf = list(range(_MAX_SUM + 1))
    for i in range(2, int(_MAX_SUM**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, _MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    for n in range(2, _MAX_SUM + 1):
        factors = set()
        x = n
        while x > 1:
            p = spf[x]
            factors.add(p)
            while x % p == 0:
                x //= p
        _prime_factors_cache[n] = factors
    
    _prime_factors_cache[1] = set()

_init_factors()

def _get_factors(n):
    return _prime_factors_cache[n]

def _generate_smooth(primes, limit):
    """Generate all numbers <= limit whose prime factors are in primes."""
    smooth = set([1])
    for p in primes:
        new_smooth = set()
        for s in smooth:
            val = s
            while val * p <= limit:
                val *= p
                new_smooth.add(val)
        smooth.update(new_smooth)
    return sorted(smooth)

def _find_clique_for_prime_set(prime_set, time_limit=5.0):
    """
    Finds a clique of size 12 in the graph where vertices are numbers 1..10000
    and edges exist if sum of endpoints has prime factors subset of prime_set.
    Uses optimized backtracking.
    """
    target_size = 12
    max_val = 10000
    start_time = time.time()
    
    smooth_numbers = _generate_smooth(sorted(prime_set), _MAX_SUM)
    smooth_set = set(smooth_numbers)
    
    # To speed up, we can precompute for each smooth number s, the pairs (a, b) such that a+b=s?
    # No, we just need to check if x+y is in smooth_set.
    
    # Backtracking
    # We enforce x1 < x2 < ... < x12
    # x1 can be small. Let's try x1 from 1 to 100.
    # If no solution found, we can expand, but likely a solution exists with small x1.
    
    def backtrack(clique, min_val):
        if time.time() - start_time > time_limit:
            return None
        
        if len(clique) == target_size:
            return clique
        
        last = clique[-1] if clique else 0
        
        # Generate candidates for next element y
        # y > last
        # y <= max_val
        # For all x in clique, x+y must be in smooth_set
        
        candidates = []
        if clique:
            # Intersection of valid y for each x in clique
            # For a specific x, y must be s-x for some s in smooth_set.
            # y > last => s > x + last
            # y <= max_val => s <= x + max_val
            
            # Start with all possible y from the last element's constraint
            # This is the strongest constraint usually?
            # Let's collect all potential y from the last element
            potential_y = set()
            for s in smooth_numbers:
                y = s - last
                if y > last and y <= max_val:
                    potential_y.add(y)
            
            # Filter potential_y by other elements in clique
            for y in potential_y:
                valid = True
                for x in clique[:-1]:
                    if (x + y) not in smooth_set:
                        valid = False
                        break
                if valid:
                    candidates.append(y)
        else:
            # First element
            # We can just try a range of small numbers
            # To avoid symmetry issues and speed up, we fix x1 to be small
            # If a solution exists, we can likely find one with x1 < 100
            candidates = list(range(1, 101))
        
        # Sort candidates
        # Heuristic: smaller y first?
        candidates.sort()
        
        for y in candidates:
            res = backtrack(clique + [y], y + 1)
            if res:
                return res
        
        return None

    # Try to find a clique
    # We iterate x1 manually to allow restarting or different strategies if needed
    # But the backtrack function above handles x1 selection in the 'else' branch.
    # However, the 'else' branch limits x1 to 1..100.
    # If that fails, we might need to expand.
    
    res = backtrack([], 1)
    if res:
        return res
    
    # If not found with x1 < 100, try a few more?
    # Given the time limit, we should be careful.
    # Let's try x1 from 101 to 200 if time permits.
    if time.time() - start_time < time_limit - 1.0:
        # Modify backtrack to accept a specific first element?
        # Or just call backtrack with a modified range?
        # Easier: just try specific first elements
        for x1 in range(101, 201):
            if time.time() - start_time > time_limit:
                break
            res = backtrack([x1], x1 + 1)
            if res:
                return res
                
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Try small prime sets
    prime_sets_to_try = [
        {2, 3},
        {2, 5},
        {2, 3, 5},
        {2, 3, 7},
        {2, 5, 7},
        {2, 3, 5, 7},
        {2, 3, 5, 11},
    ]
    
    best_arr = None
    best_score = float('inf')
    
    for ps in prime_sets_to_try:
        if time.time() > deadline:
            break
            
        # Try to find a clique for this prime set
        # Allocate a portion of the remaining time
        remaining = deadline - time.time()
        time_alloc = min(10.0, remaining * 0.5) # Don't spend all time on one set
        
        clique = _find_clique_for_prime_set(ps, time_limit=time_alloc)
        
        if clique:
            # Calculate actual score
            primes_in_sums = set()
            for i in range(len(clique)):
                for j in range(i+1, len(clique)):
                    s = clique[i] + clique[j]
                    primes_in_sums.update(_get_factors(s))
            
            score = len(primes_in_sums)
            if score < best_score:
                best_score = score
                best_arr = clique
            # If score is 2, it's very good. 
            # Theoretical minimum is likely 2 or 3.
            if score <= 2:
                break
                
    if best_arr is None:
        # Fallback to a known good set or random
        # A set of even numbers ensures 2 is a factor.
        # If we pick numbers such that sums/2 are 3-smooth, we get {2,3}.
        # Let's try a simple heuristic fallback if search fails.
        # Use a set of numbers that are multiples of a large power of 2?
        # Or just return a random valid set.
        # To ensure a good score, let's try to construct a set with factors {2,3} manually if search fails.
        # Example: 2, 4, 6, 10, 14, 18, 26, 38, 54, 78, 114, 170
        # Sums:
        # 2+4=6 (2,3)
        # 2+6=8 (2)
        # 2+10=12 (2,3)
        # 4+6=10 (2,5) -> 5 appears.
        
        # Let's just use a random set of 12 distinct integers <= 10000.
        # This will have a high score, but it's valid.
        random.seed(42)
        best_arr = random.sample(range(1, 10001), 12)
        
    return best_arr
# EVOLVE-BLOCK-END
