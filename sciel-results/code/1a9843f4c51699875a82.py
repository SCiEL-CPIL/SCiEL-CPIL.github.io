import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin before 60s kill
    
    MAX_VAL = 10210
    MIN_VAL = 211
    N = 210
    MAX_SIZE = 100
    
    # 1. Sieve for Smallest Prime Factor (SPF)
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Identify 7-smooth numbers and their kernels
    # Primes of interest: 2, 3, 5, 7
    # Kernel vector indices: 0:2, 1:3, 2:5, 3:7
    target_mask = 0b1111 # 2*3*5*7 = 210
    
    candidates = [] # List of (value, mask)
    
    for M in range(MIN_VAL, MAX_VAL + 1):
        if time.time() > deadline:
            return [840]
            
        x = M
        mask = 0
        is_smooth = True
        # Check if M is 7-smooth
        # We can just factorize using spf
        while x > 1:
            p = spf[x]
            if p > 7:
                is_smooth = False
                break
            exp = 0
            while x % p == 0:
                x //= p
                exp += 1
            if exp % 2 == 1:
                if p == 2: mask ^= 1
                elif p == 3: mask ^= 2
                elif p == 5: mask ^= 4
                elif p == 7: mask ^= 8
        if is_smooth:
            candidates.append((M, mask))
            
    # 3. Gaussian Elimination to find minimal max
    # We process candidates in increasing order of M
    # basis: dict pivot_bit -> (mask, rep_mask)
    # rep_mask tracks which candidates (indices in candidates list) are used
    
    basis = {}
    
    def find_solution(target_m):
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

    for idx, (M, mask) in enumerate(candidates):
        if time.time() > deadline:
            return [840]
            
        rep = (1 << idx)
        
        current_mask = mask
        current_rep = rep
        
        for p in sorted(basis.keys(), reverse=True):
            if current_mask & (1 << p):
                b_mask, b_rep = basis[p]
                current_mask ^= b_mask
                current_rep ^= b_rep
        
        if current_mask == 0:
            # Dependent
            sol_rep = find_solution(target_mask)
            if sol_rep is not None:
                count = sol_rep.bit_count()
                if 0 < count <= MAX_SIZE:
                    selected = []
                    for i in range(idx + 1):
                        if (sol_rep >> i) & 1:
                            selected.append(candidates[i][0])
                    prod = 1
                    for x in selected:
                        prod *= x
                    if isqrt(prod * N) ** 2 == prod * N:
                        return selected
            continue
        
        pivot = current_mask.bit_length() - 1
        basis[pivot] = (current_mask, current_rep)
        
        # RREF
        for p in list(basis.keys()):
            if p != pivot:
                if basis[p][0] & (1 << pivot):
                    b_mask, b_rep = basis[p]
                    basis[p] = (b_mask ^ current_mask, b_rep ^ current_rep)
        
        sol_rep = find_solution(target_mask)
        if sol_rep is not None:
            count = sol_rep.bit_count()
            if 0 < count <= MAX_SIZE:
                selected = []
                for i in range(idx + 1):
                    if (sol_rep >> i) & 1:
                        selected.append(candidates[i][0])
                prod = 1
                for x in selected:
                    prod *= x
                if isqrt(prod * N) ** 2 == prod * N:
                    return selected

    return [840]
