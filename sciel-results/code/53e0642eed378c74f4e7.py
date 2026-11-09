"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import heapq
from functools import lru_cache

def _is_perfect_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _square_free_kernel(n, primes):
    """Return the square-free kernel of n with respect to given primes."""
    kernel = 1
    for p in primes:
        cnt = 0
        while n % p == 0:
            n //= p
            cnt += 1
        if cnt % 2 == 1:
            kernel *= p
    return kernel

def construct():
    deadline = time.time() + 55  # safety margin before 60s kill
    n_base = 210
    max_val = 10210
    min_val = n_base + 1  # strictly larger than 210
    
    # The primes that matter for the parity condition with 210 = 2*3*5*7
    # We need product * 210 to be a perfect square.
    # 210 = 2^1 * 3^1 * 5^1 * 7^1
    # So for each prime p, the total exponent in (product * 210) must be even.
    # For p in {2,3,5,7}: exponent in product must be odd (1 + odd = even)
    # For other primes: exponent in product must be even.
    
    # Key primes to track: 2, 3, 5, 7 (from 210) and any others that appear
    # Let's use a state based on the square-free part of the product relative to 210.
    # State: a number whose prime factorization has each prime to power 0 or 1,
    # representing the "parity vector" of exponents.
    
    # Target state: the product must have odd exponent for 2,3,5,7 and even for all others.
    # So the "deficit" state we need to achieve is: 2*3*5*7 = 210 (in terms of parity).
    # Actually, let's think of it as: we need the square-free kernel of (product) to be 210
    # (since 210 * 210 = 44100 is a perfect square, and more generally product * 210 is a
    # perfect square iff the square-free part of product equals the square-free part of 210).
    
    # Square-free part of 210 is 210 itself (it's square-free).
    # So we need: square_free_kernel(product) == 210.
    
    # BFS/Dijkstra: state = square-free kernel of current product.
    # Start: kernel = 1 (empty product)
    # Goal: kernel = 210
    # Edge: multiply by a number x in (210, max_val], new kernel = kernel * square_free_kernel(x) mod squares
    # Cost: minimize max(x used so far)
    
    # Since we want to minimize the maximum value, we can do a binary search on the max value,
    # or use Dijkstra where the "distance" is the max value used.
    
    # Better approach: for each candidate max value M, check if there exists a subset of
    # numbers in (210, M] whose product has square-free kernel 210.
    # We can do BFS on the kernel state using only numbers <= M.
    
    # The state space: square-free kernels are products of distinct primes.
    # Primes up to 10210: there are ~1269 primes. The state space could be huge.
    # But in practice, the numbers we use will have limited prime factors.
    
    # Alternative: since we only need a few numbers, let's try small combinations.
    
    # First, check single numbers: x * 210 is a perfect square.
    # x * 210 = k^2 => x = k^2 / 210. For x to be integer, 210 must divide k^2.
    # 210 = 2*3*5*7, so k must be divisible by 2*3*5*7 = 210.
    # k = 210 * m, x = (210*m)^2 / 210 = 210 * m^2.
    # x > 210 => m >= 1. x <= 10210 => m^2 <= 48.6 => m <= 6.
    # m=1: x=210 (not > 210)
    # m=2: x=840
    # m=3: x=1890
    # m=4: x=3360
    # m=5: x=5250
    # m=6: x=7560
    # So single-element solutions: 840, 1890, 3360, 5250, 7560.
    # Best: 840, giving max-210 = 630.
    
    best_solution = [840]
    best_raw = 840 - 210  # 630
    
    # Now try to find solutions with max < 840, i.e., all elements in (210, 840).
    # Use BFS on the square-free kernel state.
    
    # For numbers in (210, 840), what primes can appear?
    # Let's collect all numbers in (210, 840) and their square-free kernels.
    
    # To keep state space manageable, let's limit to numbers with small prime factors.
    # Actually, let's just do BFS with a state limit.
    
    # State: square-free kernel (an integer). We track the set of numbers used.
    # To avoid exponential blowup, we'll use Dijkstra minimizing max element.
    
    # Let's precompute square-free kernels for numbers in (210, 840).
    # For efficiency, factor each number.
    
    def factorize(n):
        """Return list of (prime, exponent) for n."""
        factors = []
        d = 2
        while d * d <= n:
            if n % d == 0:
                cnt = 0
                while n % d == 0:
                    n //= d
                    cnt += 1
                factors.append((d, cnt))
            d += 1 if d == 2 else 2
        if n > 1:
            factors.append((n, 1))
        return factors
    
    def sf_kernel(n):
        """Square-free kernel of n."""
        k = 1
        for p, e in factorize(n):
            if e % 2 == 1:
                k *= p
        return k
    
    # Precompute for range (210, 840)
    candidates = list(range(211, 840))
    kernels = {}
    for x in candidates:
        kernels[x] = sf_kernel(x)
    
    # BFS: state = (kernel, frozenset_of_used) is too expensive.
    # Instead, Dijkstra where distance = max element used, state = kernel.
    # We want to reach kernel = 210.
    
    # dist[kernel] = minimum max-element to achieve this kernel
    # parent[kernel] = (prev_kernel, element)
    
    INF = float('inf')
    dist = {1: 0}
    parent = {}  # kernel -> (prev_kernel, element)
    
    # Priority queue: (max_element_used, kernel)
    pq = [(0, 1)]
    visited = set()
    
    target_kernel = 210  # we need product's sf kernel to be 210
    
    found = False
    while pq and time.time() < deadline:
        d, state = heapq.heappop(pq)
        if state in visited:
            continue
        visited.add(state)
        
        if state == target_kernel:
            # Reconstruct path
            path = []
            cur = state
            while cur != 1:
                prev, elem = parent[cur]
                path.append(elem)
                cur = prev
            path.reverse()
            if d < best_raw + 210:  # d is the max element
                best_solution = path
                best_raw = d - 210
            found = True
            break
        
        # Try adding each candidate
        for x in candidates:
            if x > d:  # can't improve if x > current best max
                continue
            new_d = max(d, x)
            if new_d >= 840:  # we're looking for < 840
                continue
            new_kernel = state * kernels[x]
            # Reduce: if a prime appears twice, it cancels
            # Actually, kernel multiplication: sf(a*b) = sf(a) * sf(b) / sf(gcd(sf(a),sf(b))^2)
            # Simpler: factorize new_kernel and take sf of it
            # But this is expensive. Let's use a different approach.
            
            # Actually, the square-free kernel of a product is the XOR (symmetric difference)
            # of the prime sets. Let's represent kernel as a set of primes.
            pass
    
    # The above is getting complex. Let me try a simpler approach:
    # For small sets (1-4 elements), brute force with pruning.
    
    # Actually, let's just verify 840 is optimal by checking if any 2-element solution
    # exists with both elements < 840.
    
    # For 2 elements a, b in (210, 840): a*b*210 must be a perfect square.
    # a*b*210 = k^2
    # So a*b = k^2/210. For this to be integer, 210 | k^2, so 210 | k (since 210 is square-free).
    # k = 210*m, a*b = 210*m^2.
    # a, b in (210, 840), so a*b in (210^2, 840^2) = (44100, 705600).
    # 210*m^2 in (44100, 705600) => m^2 in (210, 3360) => m in (14.5, 58) => m in 15..58.
    # For each m, a*b = 210*m^2, and a, b in (210, 840), a != b.
    
    found_better = False
    for m in range(15, 59):
        prod = 210 * m * m
        # Find factor pairs (a, b) with a < b, a > 210, b < 840
        a_max = min(839, prod // 211)  # a <= prod/211 since b >= 211
        for a in range(211, min(a_max, 840)):
            if prod % a == 0:
                b = prod // a
                if 211 <= b <= 839 and a != b:
                    max_val_found = max(a, b)
                    if max_val_found < 840:
                        best_solution = [a, b]
                        best_raw = max_val_found - 210
                        found_better = True
                        break
        if found_better and best_raw < 630:
            break
    
    if not found_better:
        best_solution = [840]
        best_raw = 630
    
    # Ensure we return a list of Python ints
    return [int(x) for x in best_solution]
# EVOLVE-BLOCK-END
