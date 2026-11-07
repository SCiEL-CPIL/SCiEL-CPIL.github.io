import math
import time

def _is_perfect_square(n):
    if n < 0:
        return False
    x = math.isqrt(n)
    return x * x == n

def _squarefree_kernel(n):
    f = n
    result = 1
    d = 2
    while d * d <= f:
        if f % d == 0:
            exp = 0
            while f % d == 0:
                f //= d
                exp += 1
            if exp % 2 == 1:
                result *= d
        d += 1 if d == 2 else 2
    if f > 1:
        result *= f
    return result

def construct():
    N = 210
    MAX_VAL = 10210
    MAX_SIZE = 100
    
    start_time = time.time()
    deadline = start_time + 50.0  # Leave margin for safety
    
    # Target: kernel of 210 = 2*3*5*7
    target_ker = _squarefree_kernel(N)
    
    # Precompute kernels for all numbers from 211 to MAX_VAL
    # and map to bitmasks over primes
    # First, collect all primes up to MAX_VAL
    # Actually, we only need primes that appear in kernels of numbers in (210, MAX_VAL]
    
    # Precompute kernels
    kernels = {}
    for m in range(211, MAX_VAL + 1):
        kernels[m] = _squarefree_kernel(m)
    
    # Collect all primes that appear in any kernel
    all_primes = set()
    for m in range(211, MAX_VAL + 1):
        k = kernels[m]
        d = 2
        while d * d <= k:
            if k % d == 0:
                all_primes.add(d)
                while k % d == 0:
                    k //= d
            d += 1 if d == 2 else 2
        if k > 1:
            all_primes.add(k)
    
    # Also add primes from target
    tk = target_ker
    d = 2
    while d * d <= tk:
        if tk % d == 0:
            all_primes.add(d)
            while tk % d == 0:
                tk //= d
        d += 1 if d == 2 else 2
    if tk > 1:
        all_primes.add(tk)
    
    prime_list = sorted(all_primes)
    prime_to_idx = {p: i for i, p in enumerate(prime_list)}
    num_primes = len(prime_list)
    
    def kernel_to_mask(ker):
        mask = 0
        f = ker
        d = 2
        while d * d <= f:
            if f % d == 0:
                exp = 0
                while f % d == 0:
                    f //= d
                    exp += 1
                if exp % 2 == 1:
                    if d in prime_to_idx:
                        mask |= (1 << prime_to_idx[d])
            d += 1 if d == 2 else 2
        if f > 1:
            if f in prime_to_idx:
                mask |= (1 << prime_to_idx[f])
        return mask
    
    target_mask = kernel_to_mask(target_ker)
    
    # Precompute masks for all numbers
    masks = {}
    for m in range(211, MAX_VAL + 1):
        masks[m] = kernel_to_mask(kernels[m])
    
    # Gaussian elimination over GF(2)
    # We iterate M from 211 upward. At each step, we add the mask of M to the basis.
    # We check if target_mask is in the span.
    # To find the actual subset, we track representations.
    
    # basis: dict pivot_bit -> (mask, rep)
    # rep is a bitmask where bit i is set if number (211+i) is in the subset
    # But we need to be careful about the size limit (100)
    
    basis = {}
    
    def find_solution(target_m, basis):
        current_rep = 0
        current_mask = target_m
        for p in sorted(basis.keys(), reverse=True):
            if current_mask & (1 << p):
                b_mask, b_rep = basis[p]
                current_mask ^= b_mask
                current_rep ^= b_rep
        if current_mask == 0:
            return current_rep
        return None
    
    # Iterate M from 211 to MAX_VAL
    # For each M, add its mask to the basis, then check if target is solvable
    # If solvable, extract the solution and check size <= 100
    # Return the first valid solution (smallest M)
    
    best_solution = None
    best_max = MAX_VAL + 1
    
    for M in range(211, MAX_VAL + 1):
        if time.time() > deadline:
            break
            
        mask = masks[M]
        rep = (1 << (M - 211))
        
        # Try to add to basis
        current_mask = mask
        current_rep = rep
        
        for p in sorted(basis.keys(), reverse=True):
            if current_mask & (1 << p):
                b_mask, b_rep = basis[p]
                current_mask ^= b_mask
                current_rep ^= b_rep
        
        if current_mask == 0:
            # Linearly dependent, skip
            # But we can still check if target is solvable (it might have been solvable before)
            sol_rep = find_solution(target_mask, basis)
            if sol_rep is not None:
                count = sol_rep.bit_count()
                if 0 < count <= MAX_SIZE:
                    selected = []
                    for i in range(M - 211 + 1):
                        if (sol_rep >> i) & 1:
                            selected.append(211 + i)
                    # Validate
                    prod = 1
                    for x in selected:
                        prod *= x
                    if _is_perfect_square(prod * N):
                        if M < best_max:
                            best_max = M
                            best_solution = selected
                        break
            continue
        
        # Add to basis
        pivot = current_mask.bit_length() - 1
        basis[pivot] = (current_mask, current_rep)
        
        # Eliminate this pivot from other basis elements
        for p in list(basis.keys()):
            if p != pivot:
                if basis[p][0] & (1 << pivot):
                    b_mask, b_rep = basis[p]
                    basis[p] = (b_mask ^ current_mask, b_rep ^ current_rep)
        
        # Check if target is solvable
        sol_rep = find_solution(target_mask, basis)
        if sol_rep is not None:
            count = sol_rep.bit_count()
            if 0 < count <= MAX_SIZE:
                selected = []
                for i in range(M - 211 + 1):
                    if (sol_rep >> i) & 1:
                        selected.append(211 + i)
                # Validate
                prod = 1
                for x in selected:
                    prod *= x
                if _is_perfect_square(prod * N):
                    if M < best_max:
                        best_max = M
                        best_solution = selected
                    break
    
    if best_solution is not None:
        return list(best_solution)
    
    # Fallback: return the pair (224, 240)
    return [224, 240]
