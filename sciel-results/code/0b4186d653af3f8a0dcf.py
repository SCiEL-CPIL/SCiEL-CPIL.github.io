import math
import time

# EVOLVE-BLOCK-START
N = 210
MAX_VAL = 10210
MAX_SIZE = 100
DEADLINE = 85.0

def get_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

primes = get_primes(MAX_VAL)
prime_to_idx = {p: i for i, p in enumerate(primes)}

def compute_squarefree_kernels(limit):
    sf = list(range(limit + 1))
    for p in primes:
        p2 = p * p
        if p2 > limit:
            break
        for j in range(p2, limit + 1, p2):
            sf[j] = sf[j // p2]
    return sf

sf_kernels = compute_squarefree_kernels(MAX_VAL)

def kernel_to_mask(kernel):
    mask = 0
    temp = kernel
    for p in primes:
        if p * p > temp:
            break
        if temp % p == 0:
            mask |= (1 << prime_to_idx[p])
            temp //= p
    if temp > 1:
        mask |= (1 << prime_to_idx[temp])
    return mask

def is_perfect_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    # Precompute vectors
    vecs = {}
    for i in range(N + 1, MAX_VAL + 1):
        k = sf_kernels[i]
        vecs[i] = kernel_to_mask(k)
    
    target_vec = kernel_to_mask(sf_kernels[N])
    
    # Build basis incrementally, tracking dependencies
    # basis: list of (pivot_bit, reduced_vec, dep_mask)
    # dep_mask is a bitset over indices 0..(M-211) representing which original vectors are used
    basis = []
    
    def add_to_basis(v, orig_mask):
        for pivot, bvec, bmask in basis:
            if (v >> pivot) & 1:
                v ^= bvec
                orig_mask ^= bmask
        if v != 0:
            pivot = (v & -v).bit_length() - 1
            basis.append((pivot, v, orig_mask))
        return orig_mask
    
    best_M = None
    for M in range(N + 1, MAX_VAL + 1):
        if time.time() > deadline:
            break
        
        v_M = vecs[M]
        orig_mask = 1 << (M - (N + 1))
        add_to_basis(v_M, orig_mask)
        
        # Check if target_vec is in the span
        t = target_vec
        for pivot, bvec, _ in basis:
            if (t >> pivot) & 1:
                t ^= bvec
        if t == 0:
            best_M = M
            break
    
    if best_M is None:
        return [840]
    
    # Solve for target_vec using the current basis
    t = target_vec
    sol_mask = 0
    for pivot, bvec, bmask in basis:
        if (t >> pivot) & 1:
            t ^= bvec
            sol_mask ^= bmask
    
    if t != 0:
        return [840]
    
    selected = []
    for i in range(N + 1, best_M + 1):
        if (sol_mask >> (i - (N + 1))) & 1:
            selected.append(i)
    
    if len(selected) > MAX_SIZE:
        return [840]
    
    product = N
    for x in selected:
        product *= x
        
    if is_perfect_square(product):
        return [int(x) for x in selected]
    else:
        return [840]

# EVOLVE-BLOCK-END
