"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def _is_perfect_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _factorize(n, primes):
    """Return dict of prime -> exponent for n, using precomputed primes."""
    factors = {}
    for p in primes:
        if p * p > n:
            break
        while n % p == 0:
            factors[p] = factors.get(p, 0) + 1
            n //= p
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def _squarefree_kernel(n, primes):
    """Return the squarefree kernel of n as a frozenset of primes with odd exponent."""
    factors = _factorize(n, primes)
    kernel = frozenset(p for p, e in factors.items() if e % 2 == 1)
    return kernel

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin
    
    N = 210
    MAX_VAL = 10210
    MAX_SIZE = 100
    
    # Generate primes up to MAX_VAL
    sieve = [True] * (MAX_VAL + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(MAX_VAL)) + 1):
        if sieve[i]:
            for j in range(i*i, MAX_VAL + 1, i):
                sieve[j] = False
    primes = [i for i in range(2, MAX_VAL + 1) if sieve[i]]
    
    # The target kernel: we need product * 210 to be a perfect square
    # 210 = 2*3*5*7, so product must have odd exponents for {2,3,5,7}
    target_kernel = frozenset([2, 3, 5, 7])
    
    # Precompute squarefree kernels for all numbers in (210, MAX_VAL]
    kernels = {}
    for x in range(N + 1, MAX_VAL + 1):
        kernels[x] = _squarefree_kernel(x, primes)
    
    # For each candidate max M, try to find a valid subset
    # Strategy: greedy with backtracking on kernel state
    # We want to minimize M, so iterate M from 211 upward
    
    best_result = None
    best_M = MAX_VAL + 1
    
    for M in range(N + 1, MAX_VAL + 1):
        if time.time() > deadline:
            break
        
        # Available numbers: (210, M]
        # We need a subset S of these such that XOR of their kernels == target_kernel
        # and |S| <= 100, all distinct
        
        # Greedy approach: try to build a set whose combined kernel equals target
        # Use a simple search: start with empty set, add numbers that move us closer
        
        # More efficient: use the fact that we can always add pairs that cancel
        # Key insight: if we include a number x, we can also include x*k^2 for any k
        # such that x*k^2 <= M, which has the same kernel as x
        
        # Simple greedy: try to find a small set
        # Start with target kernel, try to "cancel" it by selecting numbers
        
        # Approach: BFS/DFS on kernel states, but limit depth
        # Since we just need ANY valid set, try greedy:
        
        # Try: select numbers one by one, tracking current kernel
        # We want current_kernel XOR target_kernel to be achievable by remaining numbers
        
        # Simpler: just try to find if target_kernel is in the span of available kernels
        # For small M, the number of available numbers is small
        
        # Greedy: repeatedly pick a number whose kernel reduces the "distance" to target
        # Distance = size of symmetric difference (Hammming distance on kernel sets)
        
        remaining = list(range(N + 1, M + 1))
        current_kernel = frozenset()
        selected = []
        
        # Try greedy with multiple restarts
        found = False
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            cur_kernel = frozenset()
            sel = []
            used = set()
            
            # Greedily pick numbers that reduce distance to target
            for _ in range(MAX_SIZE):
                best_x = None
                best_dist = None
                for x in remaining:
                    if x in used:
                        continue
                    new_kernel = cur_kernel ^ kernels[x]
                    dist = len(new_kernel ^ target_kernel)
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        best_x = x
                
                if best_x is None:
                    break
                
                cur_kernel = cur_kernel ^ kernels[best_x]
                sel.append(best_x)
                used.add(best_x)
                
                if cur_kernel == target_kernel:
                    found = True
                    break
            
            if found:
                if len(sel) >= 1 and len(sel) <= MAX_SIZE:
                    # Verify
                    prod = N
                    for s in sel:
                        prod *= s
                    if _is_perfect_square(prod):
                        best_result = sel
                        best_M = M
                        break
        
        if found and best_result is not None:
            break
    
    if best_result is None:
        # Fallback: use the seed solution
        best_result = [4 * N]
    
    # Ensure return type is list of Python ints
    return [int(x) for x in best_result]
# EVOLVE-BLOCK-END
