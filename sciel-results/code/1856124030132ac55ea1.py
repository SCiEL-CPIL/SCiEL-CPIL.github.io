"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from collections import defaultdict

P = {'n': 210, 'maximum_window': 10000, 'maximum_size': 100}

def _is_perfect_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _factorize(n):
    """Return dict of prime -> exponent for n."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def _squarefree_kernel(n):
    """Return the squarefree part of n (product of primes with odd exponent)."""
    factors = _factorize(n)
    kernel = 1
    for p, e in factors.items():
        if e % 2 == 1:
            kernel *= p
    return kernel

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    n = P['n']  # 210 = 2 * 3 * 5 * 7
    max_val = 10210
    max_size = P['maximum_size']  # 100
    
    # Factorize 210: 2^1 * 3^1 * 5^1 * 7^1
    # We need product of selected numbers to have odd exponents for 2,3,5,7 (to make total even)
    # and even exponents for all other primes.
    # Equivalent: the squarefree kernel of (210 * product) must be 1.
    # i.e., squarefree_kernel(product) must equal squarefree_kernel(210) = 2*3*5*7 = 210
    
    target_kernel = 210  # 2*3*5*7
    
    # Precompute squarefree kernels for all numbers in range
    # Numbers must be > 210 and <= 10210
    start = n + 1  # 211
    end = max_val  # 10210
    
    # Greedy approach: try to build a set where the XOR (multiplicative) of kernels = target_kernel
    # Since we want kernel(product) = target_kernel, and kernel(a*b) = kernel(kernel(a)*kernel(b))
    # Actually kernel(a*b) = kernel(a) * kernel(b) / gcd-related... 
    # Better: track the parity vector directly.
    
    # Let's use a different approach: track the product's prime parity as a frozenset of primes with odd exponent.
    # Target: the set of primes with odd exponent in (210 * product) must be empty.
    # 210 has primes {2,3,5,7} with odd exponent.
    # So product must have odd exponents for exactly {2,3,5,7} (to cancel) and even for all others.
    # i.e., the "odd-exponent prime set" of the product must be exactly {2,3,5,7}.
    
    # Strategy: Greedily pick small numbers. For each candidate, compute its odd-prime-set.
    # We want the symmetric difference (XOR) of all selected odd-prime-sets to equal {2,3,5,7}.
    
    # Precompute odd-prime-set for each number
    odd_sets = {}
    for val in range(start, end + 1):
        factors = _factorize(val)
        odd_set = frozenset(p for p, e in factors.items() if e % 2 == 1)
        odd_sets[val] = odd_set
    
    # Greedy: try to find a small set. Start with the smallest numbers and build up.
    # Use a priority approach: try increasing max value.
    
    best_result = None
    best_raw = float('inf')
    
    # Approach: For each candidate max value M (from start upward), try to find a subset of {start, ..., M}
    # whose XOR of odd-sets equals target_set = {2,3,5,7}, using at most max_size elements.
    # This is a linear algebra problem over GF(2) if we represent each prime as a bit.
    
    # Collect all distinct primes up to 10210
    all_primes = set()
    for val in range(start, end + 1):
        all_primes.update(odd_sets[val])
    
    # Also include primes from 210
    all_primes.update({2, 3, 5, 7})
    
    prime_list = sorted(all_primes)
    prime_to_idx = {p: i for i, p in enumerate(prime_list)}
    num_primes = len(prime_list)
    
    target_set = frozenset({2, 3, 5, 7})
    target_vec = [1 if p in target_set else 0 for p in prime_list]
    
    # For a given M, we have vectors for numbers start..M.
    # We want a subset whose XOR = target_vec.
    # This is solvable via Gaussian elimination over GF(2).
    
    # But we also want to MINIMIZE the number of elements (or just find any valid set).
    # Actually the task says 1..100 distinct integers, so any valid set of size 1-100 works.
    # We want to minimize M (the max value), so we try M = start, start+1, ...
    
    # For efficiency, let's build up incrementally.
    
    # Start with M = start, add one number at a time, and maintain a basis.
    
    # Let's try: for each M from start to end, check if target_vec is in the span of vectors for {start..M}.
    # If yes, extract a solution.
    
    # Use incremental Gaussian elimination.
    
    # We'll maintain a list of (value, vector) and do elimination.
    
    # For speed, let's try a simpler greedy first:
    # Pick numbers in order, tracking current XOR. When we can complete, stop.
    
    # Actually, let's just do: try M from start upward. For each M, do Gaussian elimination on the
    # vectors of {start, ..., M} to see if target is reachable, and if so, extract a minimal-ish solution.
    
    # Given time constraints, let's try a few approaches:
    
    # Approach 1: Greedy with backtracking on small sets
    # Try sets of size 1, 2, 3, ... up to some limit.
    
    # Size 1: need a number x such that odd_set(x) = target_set = {2,3,5,7}
    # That means x = 2^a * 3^b * 5^c * 7^d * (square), with a,b,c,d all odd.
    # Smallest such: 2*3*5*7 = 210, but must be > 210. Next: 2^3*3*5*7 = 1680, or 2*3^3*5*7 = 1890, etc.
    # Or 2*3*5*7*4 = 840 (square factor 4). 840 > 210. Check: 840 = 2^3 * 3 * 5 * 7. Odd primes: {2,3,5,7}. Yes!
    
    # So [840] might work! Let's verify: 210 * 840 = 210 * 840.
    # 210 = 2*3*5*7, 840 = 2^3*3*5*7. Product = 2^4 * 3^2 * 5^2 * 7^2 = (2^2*3*5*7)^2 = 420^2. Perfect square!
    
    # But wait, we need to check if there's a smaller one. The constraint is > 210.
    # Smallest number with odd-prime-set {2,3,5,7}: must be of form 2^(2a+1) * 3^(2b+1) * 5^(2c+1) * 7^(2d+1) * k^2
    # where a,b,c,d >= 0 and k is any integer.
    # Minimum: a=b=c=d=0, k=1: 2*3*5*7 = 210. But must be > 210.
    # Next smallest: multiply by a square. 210 * 4 = 840. Or 210 * 1 (not > 210).
    # Or change exponents: 2^3*3*5*7 = 1680, 2*3^3*5*7 = 1890, 2*3*5^3*7 = 3150, 2*3*5*7^3 = 5460.
    # Or 210 * k^2 where k^2 > 1: k=2 gives 840, k=3 gives 1890, etc.
    # So 840 is the smallest valid single number.
    
    # But maybe a set of 2+ numbers with smaller max value exists?
    # E.g., two numbers x, y where odd_set(x) XOR odd_set(y) = {2,3,5,7}.
    # We want min(max(x,y)) as small as possible.
    
    # Let's check: can we do it with max value < 840?
    # Numbers from 211 to 839.
    # We need two numbers whose odd-sets XOR to {2,3,5,7}.
    
    # This is getting complex. Let me just implement the search properly.
    
    # Given the deadline, let's do a focused search:
    
    # 1. Check if single number works (we found 840, but let's verify and find minimum).
    # 2. Try pairs, triples, etc. with increasing max value.
    
    # Actually, the simplest valid answer is [840] with raw value 840-210 = 630.
    # But we can likely do better with multiple numbers.
    
    # Let me implement a proper search.
    
    # For the Gaussian elimination approach over GF(2):
    
    def solve_gaussian(vectors, target, num_cols):
        """Solve Ax = b over GF(2). vectors is list of (index, bitvec). Returns solution indices or None."""
        # Convert to list of lists for manipulation
        n = len(vectors)
        if n == 0:
            return None
        # Build augmented matrix
        # Each row: [bitvec..., rhs]
        # We want to find subset of rows that XOR to target.
        
        # Use standard Gaussian elimination
        # rows[i] = (original_index, list_of_bits)
        rows = []
        for i, (idx, vec) in enumerate(vectors):
            rows.append([idx] + list(vec))
        
        # Add target as the RHS we want to achieve
        # Actually, we want to find x such that sum x_i * v_i = target
        # This is a system: for each column j, sum_i x_i * v_i[j] = target[j]
        
        # Let me redo: we have n equations (one per prime) and n unknowns (one per number).
        # But n can be large. Let's use the standard approach:
        # Find a basis, then check if target is in the span.
        
        # Build basis
        basis = {}  # pivot_col -> row
        row_map = {}  # pivot_col -> (original_index, bitvec)
        
        for i, (idx, vec) in enumerate(vectors):
            v = list(vec)
            # Try to eliminate
            for pcol in sorted(basis.keys(), reverse=True):
                if v[pcol]:
                    bv = list(basis[pcol])
                    for j in range(num_cols):
                        v[j] ^= bv[j]
            # Find new pivot
            pivot = -1
            for j in range(num_cols):
                if v[j]:
                    pivot = j
                    break
            if pivot >= 0:
                basis[pivot] = v
                row_map[pivot] = (idx, list(vec))
        
        # Check if target is in span
        t = list(target)
        for pcol in sorted(basis.keys(), reverse=True):
            if t[pcol]:
                bv = list(basis[pcol])
                for j in range(num_cols):
                    t[j] ^= bv[j]
        
        if any(t):
            return None  # not in span
        
        # Back-track to find solution
        # This is complex. For now, let's just return that it's solvable.
        # For actual solution extraction, we need to track the combinations.
        
        return 'solvable'
    
    # This is getting too complex for the time budget. Let me use a simpler approach:
    
    # Since we know [840] works with raw=630, let's try to find something better.
    # Try all pairs (x, y) with x <= y, both in [211, M] for increasing M,
    # checking if odd_set(x) XOR odd_set(y) == target_set.
    
    # And triples, etc.
    
    # For pairs: O(M^2) which for M up to a few thousand is fine.
    
    target_frozen = frozenset({2, 3, 5, 7})
    
    # Precompute odd_sets for range
    odd_sets_list = {}
    for val in range(start, min(end, 5000) + 1):  # limit to 5000 for now
        factors = _factorize(val)
        odd_set = frozenset(p for p, e in factors.items() if e % 2 == 1)
        odd_sets_list[val] = odd_set
    
    # Try single numbers
    for val in range(start, min(end, 5000) + 1):
        if odd_sets_list[val] == target_frozen:
            # Verify
            prod = n * val
            if _is_perfect_square(prod):
                return [val]
    
    # Try pairs
    vals = list(odd_sets_list.keys())
    for i in range(len(vals)):
        if time.time() > deadline - 5:
            break
        x = vals[i]
        sx = odd_sets_list[x]
        for j in range(i + 1, len(vals)):
            y = vals[j]
            sy = odd_sets_list[y]
            # XOR of two sets
            if sx ^ sy == target_frozen:
                # Verify
                prod = n * x * y
                if _is_perfect_square(prod):
                    return [x, y]
    
    # Try triples
    for i in range(len(vals)):
        if time.time() > deadline - 5:
            break
        x = vals[i]
        sx = odd_sets_list[x]
        for j in range(i + 1, len(vals)):
            y = vals[j]
            sy = odd_sets_list[y]
            sxy = sx ^ sy
            for k in range(j + 1, len(vals)):
                z = vals[k]
                sz = odd_sets_list[z]
                if sxy ^ sz == target_frozen:
                    prod = n * x * y * z
                    if _is_perfect_square(prod):
                        return [x, y, z]
    
    # Fallback: return the known valid answer
    return [840]
# EVOLVE-BLOCK-END
