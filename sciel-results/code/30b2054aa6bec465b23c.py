import time
import math
from collections import defaultdict

# EVOLVE-BLOCK-START
N = 210
MAX_VAL = 10210
MAX_SIZE = 100
DEADLINE = 88.0
SAFETY_MARGIN = 2.0

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

def squarefree_vector(n):
    """Return an integer bitmask representing the squarefree kernel of n."""
    temp = n
    mask = 0
    for p in primes:
        if p * p > temp:
            break
        if temp % p == 0:
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            if count % 2 == 1:
                mask |= (1 << prime_to_idx[p])
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
    deadline = start_time + DEADLINE - SAFETY_MARGIN

    # Precompute vectors for all candidates
    # We only need up to the point where we find a solution, 
    # but precomputing all is fast enough (~10k numbers)
    vecs = {}
    for i in range(N + 1, MAX_VAL + 1):
        vecs[i] = squarefree_vector(i)
    
    target_vec = squarefree_vector(N)
    
    # We iterate M from 211 upwards
    # We maintain a basis for the span of {v(211), ..., v(M)}
    # Basis is a list of (pivot_bit, vector)
    # We use the highest set bit as pivot for simplicity, or lowest.
    # Let's use lowest set bit for consistency with standard GF(2) elimination if we iterate bits.
    # Actually, any unique pivot works.
    
    basis = []
    # To allow backtracking/subset reconstruction, we might need to store the dependency.
    # But since M is small, we can just solve the system at the end.
    # For the check "is target in span", we just need the basis.
    
    # Function to reduce a vector by the basis
    def reduce_vec(v):
        # We want to reduce v using basis
        # To make it efficient, we can sort basis by pivot?
        # Or just iterate and reduce if pivot is present.
        # Since basis size is small, O(N) is fine.
        res = v
        for pivot, bvec in basis:
            if (res >> pivot) & 1:
                res ^= bvec
        return res

    # Function to add vector to basis
    def add_to_basis(v):
        # Reduce v by existing basis
        for pivot, bvec in basis:
            if (v >> pivot) & 1:
                v ^= bvec
        if v != 0:
            # Find pivot (lowest set bit)
            pivot = (v & -v).bit_length() - 1
            basis.append((pivot, v))

    # Iterate M
    best_M = None
    for M in range(N + 1, MAX_VAL + 1):
        if time.time() > deadline:
            break
            
        v_M = vecs[M]
        add_to_basis(v_M)
        
        # Check if target is in span
        # Reduce target by basis
        t = target_vec
        for pivot, bvec in basis:
            if (t >> pivot) & 1:
                t ^= bvec
        if t == 0:
            best_M = M
            break

    if best_M is None:
        # Fallback: single number 840
        # Check if 840 is valid
        # sf(210 * 840) = sf(210 * 4 * 210) = sf(4 * 210^2) = 1
        # So {840} works.
        return [840]

    # Now we have best_M. We need to find the subset S of {211, ..., best_M}
    # such that sum(v(x)) = target_vec.
    # We can solve the linear system.
    # Variables: x_i for i in 211..best_M.
    # Matrix A: rows are primes? No, columns are numbers, rows are prime bits.
    # We have ~30 numbers.
    # We can just iterate all subsets? 2^30 is 1e9, too slow.
    # But we can use the basis construction to find the solution.
    # During add_to_basis, we can store the linear combination.
    
    # Let's restart the basis construction with dependency tracking.
    # dep[i] = mask of indices (relative to 211..M) used to form basis vector i?
    # Or simpler:
    # We have the vectors V = [v(211), ..., v(best_M)]
    # We want to find coefficients c in {0,1} such that sum(c_i * V_i) = target.
    
    # We can perform Gaussian elimination on the matrix of vectors, keeping track of the original indices.
    # Let's create a list of tuples (vector, index_mask)
    # index_mask is a bitset of which original numbers are used.
    
    vecs_list = []
    for i in range(N + 1, best_M + 1):
        vecs_list.append((vecs[i], 1 << (i - (N + 1))))
    
    # Perform Gaussian elimination to solve for target
    # We want to express target as a linear combination.
    # Initialize a list of (vector, mask)
    # We will reduce the target vector, and also maintain the "solution" mask.
    
    # Actually, it's easier to augment the matrix with the target column?
    # No, we just reduce the target vector using the basis, and track what we XORed.
    
    # Let's rebuild the basis with dependency.
    # basis_deps: list of (pivot, vector, mask)
    # mask represents the combination of original vectors that makes this basis vector.
    
    basis_deps = []
    
    def add_to_basis_dep(v, mask):
        for pivot, bvec, bmask in basis_deps:
            if (v >> pivot) & 1:
                v ^= bvec
                mask ^= bmask
        if v != 0:
            pivot = (v & -v).bit_length() - 1
            basis_deps.append((pivot, v, mask))
        return v, mask

    # Add all vectors to basis_deps
    for i in range(N + 1, best_M + 1):
        v = vecs[i]
        mask = 1 << (i - (N + 1))
        add_to_basis_dep(v, mask)
        
    # Now reduce target_vec using basis_deps
    t = target_vec
    sol_mask = 0
    for pivot, bvec, bmask in basis_deps:
        if (t >> pivot) & 1:
            t ^= bvec
            sol_mask ^= bmask
            
    if t != 0:
        # Should not happen if best_M was found correctly
        return [840]
        
    # sol_mask contains the indices of the selected numbers
    # Convert mask to list of integers
    selected = []
    for i in range(N + 1, best_M + 1):
        if (sol_mask >> (i - (N + 1))) & 1:
            selected.append(i)
            
    # Verify
    if len(selected) > MAX_SIZE:
        # This shouldn't happen for small M, but just in case
        # Try to find a smaller subset?
        # For now, fallback
        return [840]
        
    product = N
    for x in selected:
        product *= x
        
    if is_perfect_square(product):
        return [int(x) for x in selected]
    else:
        return [840]

# EVOLVE-BLOCK-END
