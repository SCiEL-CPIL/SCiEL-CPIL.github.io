"""Structural change: BFS over squarefree-kernel states to find min-max subset."""
import math
import time
from collections import deque


# ---------- Prime sieve ----------
def _sieve(n):
    if n < 2:
        return []
    is_prime = bytearray([1]) * (n + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start::step] = bytearray(len(range(start, n + 1, step)))
    return [i for i in range(2, n + 1) if is_prime[i]]


MAX_VAL = 10210
_PRIMES = _sieve(102)


def _squarefree_kernel(n):
    if n <= 1:
        return 1
    result = 1
    f = n
    for p in _PRIMES:
        if p * p > f:
            break
        if f % p == 0:
            exp = 0
            while f % p == 0:
                f //= p
                exp += 1
            if exp % 2 == 1:
                result *= p
    if f > 1:
        result *= f
    return result


_KER = [0] * (MAX_VAL + 1)
for i in range(211, MAX_VAL + 1):
    _KER[i] = _squarefree_kernel(i)


def _is_perfect_square(n):
    if n < 0:
        return False
    x = math.isqrt(n)
    return x * x == n


def _validate(result, N=210):
    if not result or len(result) > 100:
        return False
    if len(set(result)) != len(result):
        return False
    for x in result:
        if not isinstance(x, int) or isinstance(x, bool):
            return False
        if x <= 210 or x > MAX_VAL:
            return False
    prod = 1
    for x in result:
        prod *= x
    return _is_perfect_square(prod * N)


def construct():
    N = 210
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    # Precompute kernel for 210
    target_kernel = _squarefree_kernel(N)
    # target_kernel = 210 = 2 * 3 * 5 * 7

    # Map kernel -> list of numbers (sorted ascending) that have that kernel
    # Only include numbers in (210, MAX_VAL]
    ker_to_nums = {}
    for n in range(211, MAX_VAL + 1):
        k = _KER[n]
        if k not in ker_to_nums:
            ker_to_nums[k] = []
        ker_to_nums[k].append(n)

    # BFS from target_kernel to 1
    # State: (kernel, max_val_so_far, frozenset_of_numbers)
    # We want to minimize max_val_so_far.
    # Since we process by max_val, the first time we reach kernel 1 is optimal.
    
    # However, storing the full set of numbers in the state makes it expensive.
    # Instead, we can store the path or just reconstruct later.
    # But for BFS correctness with "minimize max", we need to be careful.
    # Actually, if we iterate max_val from 211 to MAX_VAL, and for each max_val
    # we check if there is a valid subset using only numbers <= max_val,
    # then the first max_val that works is the answer.
    # This is a monotonic property: if a solution exists with max <= M,
    # it also exists with max <= M+1 (same solution).
    
    # So we can do a binary search or linear search on max_val.
    # But linear search with a fast "does a solution exist with numbers <= M?" check is good.
    
    # The check "does a solution exist with numbers <= M?" is:
    # Can we reach kernel 1 from target_kernel using XOR operations
    # with kernels of numbers in [211, M]?
    # This is a reachability problem in the group of squarefree kernels under XOR.
    # We can use BFS on the kernel space, but the kernel space is large.
    
    # Alternative: For a fixed M, we have a set of available kernels K_M.
    # We want to know if 1 is in the XOR span of K_M (including the ability to use each at most once? No, we can use any subset).
    # Wait, the problem is: find a subset S of {211..M} such that XOR_{n in S} K(n) == K(210).
    # This is equivalent to: is K(210) in the linear span (over GF(2)) of the vectors K(n) for n in {211..M}?
    # But wait, the XOR operation on squarefree kernels corresponds to addition in the vector space
    # of exponents mod 2. So yes, it's a linear algebra problem over GF(2).
    
    # The "vectors" are the prime factorization exponents mod 2.
    # The primes involved are those up to MAX_VAL.
    # The dimension is the number of primes <= MAX_VAL, which is ~1200.
    # We can maintain a basis for the span of the kernels as we increase M.
    # When we add a new kernel K(M), we try to reduce it by the current basis.
    # If it reduces to 0, it's dependent. If not, we add it to the basis.
    # We want to know when K(210) can be formed by the basis.
    # Actually, we want to know when K(210) is in the span of the kernels of numbers <= M.
    
    # Let's implement this.
    # We need to represent kernels as bit vectors or just use the integer representation
    # and perform XOR-based Gaussian elimination.
    # Since kernels are integers, XOR is just ^.
    # We can maintain a list of basis vectors (kernels) that are linearly independent.
    # To check if a kernel k is in the span, we try to reduce it by the basis.
    # If it becomes 0, it is in the span.
    
    # We also need to reconstruct the actual numbers used.
    # So for each basis vector, we store which number(s) formed it?
    # No, a basis vector is a combination of numbers.
    # This is getting complicated.
    
    # Simpler approach:
    # Iterate M from 211 to MAX_VAL.
    # For each M, check if there is a subset of {211..M} with XOR == 210.
    # Use the linear basis method to check membership.
    # If found, we need to reconstruct the subset.
    # To reconstruct, we can store for each basis vector the actual set of numbers that form it.
    # But the set can be large.
    
    # Alternative reconstruction:
    # When we determine that K(210) is in the span, we can backtrack to find which numbers were used.
    # This requires storing the combination for each basis vector.
    
    # Let's try a different tactic:
    # Since the answer is likely small (raw < 100), we can just do a BFS on the state (kernel, frozenset)
    # but limit the size of the frozenset or the max value.
    # But the state space is huge.
    
    # Let's go back to the linear basis approach with reconstruction.
    # We maintain:
    # basis: list of (kernel, frozenset_of_numbers)
    # To check if k is in span:
    #   current = k
    #   used = frozenset()
    #   for (bk, bs) in basis:
    #       if current ^ bk < current:  # heuristic to pick pivot
    #           current ^= bk
    #           used = used | bs
    #   if current == 0:
    #       return used
    #   else:
    #       return None
    
    # We need to order the basis to make the reduction work.
    # Standard Gaussian elimination:
    # Maintain basis such that each basis vector has a unique leading bit (highest set bit).
    # When adding a new vector v:
    #   for each basis vector b:
    #       if v has the same leading bit as b:
    #           v ^= b
    #   if v != 0:
    #       add v to basis
    
    # For reconstruction, we store with each basis vector the set of original numbers that formed it.
    # When we add a new number n with kernel k_n:
    #   v = k_n
    #   s = {n}
    #   for each (b_k, b_s) in basis:
    #       if v and b_k have the same highest bit:
    #           v ^= b_k
    #           s = s ^ b_s  # symmetric difference
    #   if v != 0:
    #       add (v, s) to basis
    
    # Then to check if target_kernel is in span:
    #   v = target_kernel
    #   s = frozenset()
    #   for each (b_k, b_s) in basis (in order of decreasing highest bit):
    #       if v and b_k have the same highest bit:
    #           v ^= b_k
    #           s = s ^ b_s
    #   if v == 0:
    #       return list(s)
    #   else:
    #       return None
    
    # This should work. The size of the sets s might grow, but the number of basis vectors is at most ~1200.
    # And the sets are subsets of {211..M}, so they can be large. But M is small initially.
    
    # Let's implement this.
    
    def _highest_bit(n):
        if n == 0:
            return -1
        return n.bit_length() - 1

    # basis: list of (kernel, frozenset)
    # We sort basis by highest bit in decreasing order for efficient reduction.
    basis = []
    
    def _add_to_basis(k, s):
        """Add kernel k with set s to the basis. Return True if added, False if dependent."""
        v = k
        ss = s
        for i in range(len(basis)):
            bk, bs = basis[i]
            hb = _highest_bit(bk)
            if _highest_bit(v) == hb:
                v ^= bk
                ss = ss ^ bs
        if v != 0:
            # Insert into basis in sorted order by highest bit
            hb = _highest_bit(v)
            idx = 0
            while idx < len(basis) and _highest_bit(basis[idx][0]) > hb:
                idx += 1
            basis.insert(idx, (v, ss))
            return True
        return False

    def _check_in_span(k):
        """Check if kernel k is in the span of the basis. Return the set of numbers if so, else None."""
        v = k
        s = frozenset()
        for bk, bs in basis:
            hb = _highest_bit(bk)
            if _highest_bit(v) == hb:
                v ^= bk
                s = s ^ bs
        if v == 0:
            return s
        return None

    # Iterate M from 211 to MAX_VAL
    # For each M, add K(M) to the basis, then check if target_kernel is in the span.
    # If yes, we have a solution with max <= M.
    # We want the smallest M.
    
    # Note: The set s returned by _check_in_span might contain numbers > M?
    # No, because we only added numbers <= M to the basis.
    # But wait, the basis vectors are combinations of numbers added so far, all <= M.
    # So the resulting set will only contain numbers <= M.
    
    # However, we must ensure that the numbers in the set are distinct and in (210, M].
    # The symmetric difference operation ensures that each number appears at most once in the set.
    # And since we only add numbers in (210, M], the set will be a subset of that range.
    
    # Let's run the loop.
    
    for M in range(211, MAX_VAL + 1):
        if time.time() > deadline:
            break
        
        k_m = _KER[M]
        s_m = frozenset([M])
        _add_to_basis(k_m, s_m)
        
        # Check if target_kernel is in the span
        sol_set = _check_in_span(target_kernel)
        if sol_set is not None:
            # We have a solution
            sol_list = list(sol_set)
            if _validate(sol_list, N):
                return sol_list
    
    # Fallback: use the original 2-element search
    seen = {}
    for n in range(211, MAX_VAL + 1):
        ker = _KER[n]
        need = ker ^ target_kernel
        if need in seen:
            return [seen[need], n]
        if ker not in seen:
            seen[ker] = n
    
    # Final fallback
    return [840]


if __name__ == "__main__":
    result = construct()
    print(result)
    print("Raw value:", max(result) - 210)
    print("Valid:", _validate(result))
