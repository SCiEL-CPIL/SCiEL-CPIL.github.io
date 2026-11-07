"""Rewritten program using squarefree-kernel pairing to minimize max(selected)."""
import math
import time
from collections import defaultdict
import bisect

def _sf_kernel_sieve(n_max):
    """Compute squarefree kernel for all numbers up to n_max using sieve."""
    sf = list(range(n_max + 1))  # sf[i] will be squarefree kernel of i
    for i in range(2, int(math.isqrt(n_max)) + 1):
        if sf[i] == i:  # i is prime
            i2 = i * i
            for j in range(i2, n_max + 1, i2):
                sf[j] //= i2
    return sf

def _is_perfect_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def construct():
    start = time.time()
    deadline = start + 85.0

    n = 210
    max_val = 10210
    max_size = 100

    # Precompute squarefree kernels using sieve
    sf_arr = _sf_kernel_sieve(max_val)

    # Build kernel groups: kernel -> sorted list of numbers with that kernel
    kernel_groups = defaultdict(list)
    for x in range(n + 1, max_val + 1):
        kernel_groups[sf_arr[x]].append(x)

    for k in kernel_groups:
        kernel_groups[k].sort()

    # Also build a set of all kernels present for quick lookup
    all_kernels = set(kernel_groups.keys())

    best_result = None
    best_max = max_val + 1

    # Size 1: find x > 210 such that x * 210 is a perfect square
    # x = 210 * k^2
    size1_candidates = []
    for k in range(1, 100):
        x = 210 * k * k
        if x > 210 and x <= max_val:
            size1_candidates.append(x)
        elif x > max_val:
            break
    if size1_candidates:
        best_max = min(size1_candidates)
        best_result = [best_max]

    # Size 2: find (a, b) with 210 < a < b <= max_val, a*b*210 is a perfect square
    # For given b, need a with sf(a) = sf(sf(b) * 210)
    # Precompute for each b: required_a_kernel = sf(sf(b) * 210)
    # But sf(b)*210 might exceed max_val, so we need sf of that product.
    # sf(a*b*210) = 1 iff sf(a) * sf(b) * sf(210) is a perfect square
    # Since all are squarefree: the symmetric difference of prime sets must be empty.
    # Equivalently: sf(a) * sf(b) * 210 has all exponents even.
    # For squarefree A, B, T: A*B*T is square iff for each prime, it appears in 0 or 2 of {A,B,T}.
    # This means: A XOR B XOR T = 1 (identity), i.e., A XOR B = T.
    # So sf(a) = sf(b) XOR 210 (in the squarefree group).
    # XOR of two squarefree numbers: product / gcd^2
    
    def sf_xor(a, b):
        """XOR of two squarefree numbers in the multiplicative group."""
        g = math.gcd(a, b)
        return a * b // (g * g)

    # For size 2: sf(a) = sf(b) XOR 210
    found_size2 = False
    for b in range(211, max_val + 1):
        if time.time() > deadline:
            break
        if best_result is not None and b >= best_max:
            break
        
        required_a_kernel = sf_xor(sf_arr[b], 210)
        
        if required_a_kernel in kernel_groups:
            lst = kernel_groups[required_a_kernel]
            # Find if there's an a in lst with 210 < a < b
            idx = bisect.bisect_left(lst, b)
            if idx > 0:
                a = lst[idx - 1]
                if a > 210:
                    # Verify
                    prod_check = a * b * 210
                    if _is_perfect_square(prod_check):
                        best_result = [a, b]
                        best_max = b
                        found_size2 = True
                        break

    # Size 3: find (a, b, M) with 210 < a < b < M, a*b*M*210 is a perfect square
    # sf(a) XOR sf(b) XOR sf(M) XOR 210 = 1
    # sf(a) XOR sf(b) = sf(M) XOR 210
    # For each M, target = sf(M) XOR 210
    # For each a in (210, M), needed_b = sf(a) XOR target
    # Check if needed_b is a kernel with an element in (a, M)
    
    for M in range(211, best_max):
        if time.time() > deadline:
            break
        if best_result is not None and M >= best_max:
            break
        
        target = sf_xor(sf_arr[M], 210)
        
        # For efficiency, iterate a from 211 to M-1
        # For each a, compute needed_b_kernel = sf(a) XOR target
        # Check if kernel_groups has needed_b_kernel with element in (a, M)
        found = False
        for a in range(211, M):
            if time.time() > deadline:
                break
            needed_b_kernel = sf_xor(sf_arr[a], target)
            
            if needed_b_kernel in kernel_groups:
                lst = kernel_groups[needed_b_kernel]
                # Find element in (a, M)
                idx_lo = bisect.bisect_right(lst, a)
                if idx_lo < len(lst) and lst[idx_lo] < M:
                    b = lst[idx_lo]
                    # Verify
                    prod_check = a * b * M * 210
                    if _is_perfect_square(prod_check):
                        best_result = [a, b, M]
                        best_max = M
                        found = True
                        break
        if found:
            break

    # Size 4: try to find 4-element solutions with max < best_max
    # sf(a) XOR sf(b) XOR sf(c) XOR sf(M) XOR 210 = 1
    # sf(a) XOR sf(b) XOR sf(c) = sf(M) XOR 210
    # For each M, target = sf(M) XOR 210
    # For each pair (a,b) in (210, M), needed_c = sf(a) XOR sf(b) XOR target
    # Check if needed_c is in kernel_groups with element in (b, M)
    
    for M in range(211, best_max):
        if time.time() > deadline:
            break
        if best_result is not None and M >= best_max:
            break
        
        target = sf_xor(sf_arr[M], 210)
        
        found = False
        for a in range(211, M):
            if time.time() > deadline:
                break
            for b in range(a + 1, M):
                if time.time() > deadline:
                    break
                needed_c_kernel = sf_xor(sf_xor(sf_arr[a], sf_arr[b]), target)
                
                if needed_c_kernel in kernel_groups:
                    lst = kernel_groups[needed_c_kernel]
                    idx_lo = bisect.bisect_right(lst, b)
                    if idx_lo < len(lst) and lst[idx_lo] < M:
                        c = lst[idx_lo]
                        # Verify
                        prod_check = a * b * c * M * 210
                        if _is_perfect_square(prod_check):
                            best_result = [a, b, c, M]
                            best_max = M
                            found = True
                            break
            if found:
                break
        if found:
            break

    # Verify the best result
    if best_result is not None:
        prod = 210
        for x in best_result:
            prod *= x
        if not _is_perfect_square(prod):
            best_result = [840]
            best_max = 840

    if best_result is None:
        best_result = [840]
        best_max = 840

    return [int(x) for x in best_result]
