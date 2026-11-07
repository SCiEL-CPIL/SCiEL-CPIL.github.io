"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
from itertools import combinations

# EVOLVE-BLOCK-START

def _squarefree_kernel(n):
    """Return frozenset of primes with odd exponent in factorization of n."""
    if n <= 1:
        return frozenset()
    result = frozenset()
    f = n
    d = 2
    while d * d <= f:
        if f % d == 0:
            exp = 0
            while f % d == 0:
                f //= d
                exp += 1
            if exp % 2 == 1:
                result = result | frozenset([d])
        d += 1 if d == 2 else 2
    if f > 1:
        result = result | frozenset([f])
    return result

def _target_kernel():
    """Squarefree kernel of 210 = {2,3,5,7}."""
    return frozenset([2, 3, 5, 7])

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s limit
    target = _target_kernel()
    
    # Precompute squarefree kernels for numbers 211 to some limit
    # We want to minimize max(selected) - 210, so try small windows first
    
    # Strategy: For each window size W (max - 210 = W), check if there's a subset
    # of {211, ..., 210+W} whose kernel XOR equals target.
    
    # Precompute kernels up to a reasonable limit
    max_search = 10210
    kernels = {}
    for i in range(211, max_search + 1):
        if time.time() > deadline - 5.0:
            break
        kernels[i] = _squarefree_kernel(i)
    
    # Try increasing window sizes
    # For small windows, use exhaustive search; for larger, use greedy
    best_result = None
    best_val = float('inf')
    
    # First, check single element: need kernel(n) == target
    # n * 210 = square means kernel(n) XOR kernel(210) = empty, so kernel(n) == kernel(210)
    for n in range(211, max_search + 1):
        if time.time() > deadline - 5.0:
            break
        if n not in kernels:
            break
        if kernels[n] == target:
            val = n - 210
            if val < best_val:
                best_val = val
                best_result = [n]
            break  # first one found is smallest
    
    # Try 2 elements: kernel(a) XOR kernel(b) == target
    # i.e., kernel(b) == kernel(a) XOR target
    if best_val > 50 or best_result is None:  # only if single didn't give great answer
        found = False
        for a in range(211, max_search + 1):
            if time.time() > deadline - 5.0:
                break
            if a not in kernels:
                break
            ka = kernels[a]
            # Need kb = ka XOR target
            # Compute XOR of two frozensets
            kb = (ka | target) - (ka & target)
            # Find smallest b > 210, b != a, with kernel(b) == kb
            for b in range(211, max_search + 1):
                if b == a:
                    continue
                if b not in kernels:
                    break
                if kernels[b] == kb:
                    val = max(a, b) - 210
                    if val < best_val:
                        best_val = val
                        best_result = [a, b]
                    found = True
                    break
            # Optimization: if a is already large enough that any b >= 211 gives worse result
            if a - 210 >= best_val and best_result is not None:
                break
    
    # Try 3 elements for potentially better results
    if best_val > 30 or best_result is None:
        # For 3 elements: ka XOR kb XOR kc == target
        # This is more complex; use a map from kernel to smallest number
        # Build mapping: for each kernel, store the smallest number with that kernel
        kernel_to_smallest = {}
        for n in range(211, max_search + 1):
            if time.time() > deadline - 5.0:
                break
            if n not in kernels:
                break
            k = kernels[n]
            if k not in kernel_to_smallest:
                kernel_to_smallest[k] = n
            elif n < kernel_to_smallest[k]:
                kernel_to_smallest[k] = n
        
        # For each pair (a, b), need c with kernel(c) = ka XOR kb XOR target
        # Try to find solutions with small max
        for a in range(211, min(max_search + 1, 211 + best_val)):
            if time.time() > deadline - 3.0:
                break
            if a not in kernels:
                break
            ka = kernels[a]
            for b in range(a + 1, min(max_search + 1, 211 + best_val)):
                if time.time() > deadline - 3.0:
                    break
                if b not in kernels:
                    break
                kb = kernels[b]
                # Need kc = ka XOR kb XOR target
                # Compute step by step
                tmp = (ka | target) - (ka & target)  # ka XOR target
                kc = (tmp | kb) - (tmp & kb)  # (ka XOR target) XOR kb
                if kc in kernel_to_smallest:
                    c = kernel_to_smallest[kc]
                    if c != a and c != b and c > 210:
                        val = max(a, b, c) - 210
                        if val < best_val:
                            best_val = val
                            best_result = [a, b, c]
    
    if best_result is not None:
        return list(best_result)
    
    # Fallback: use the known solution
    return [840]

# EVOLVE-BLOCK-END
