"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from collections import deque

P = {'n': 210, 'maximum_window': 10210, 'maximum_size': 100}

def _squarefree_kernel(n):
    """Compute the squarefree kernel of n (product of primes with odd exponent)."""
    result = 1
    d = 2
    temp = n
    while d * d <= temp:
        if temp % d == 0:
            count = 0
            while temp % d == 0:
                temp //= d
                count += 1
            if count % 2 == 1:
                result *= d
        d += 1
    if temp > 1:
        result *= temp
    return result

def _factorize(n):
    """Return dict of prime: exponent for n."""
    factors = {}
    d = 2
    temp = n
    while d * d <= temp:
        if temp % d == 0:
            count = 0
            while temp % d == 0:
                temp //= d
                count += 1
            factors[d] = count
        d += 1
    if temp > 1:
        factors[temp] = factors.get(temp, 0) + 1
    return factors

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin within 90s limit
    
    n = P['n']  # 210
    max_val = P['maximum_window']  # 10210
    max_size = P['maximum_size']  # 100
    
    # Target kernel: squarefree kernel of n
    target_kernel = _squarefree_kernel(n)
    
    # We need to find a set of distinct integers in (210, M] such that
    # the XOR of their squarefree kernels equals target_kernel (so combined with n, all even)
    # Actually: kernel(n) * kernel(prod(selected)) must be a perfect square
    # This means kernel(n) XOR kernel(prod(selected)) = 1 (in multiplicative sense)
    # i.e., kernel(prod(selected)) = kernel(n) since kernel(n)^2 is a square
    # Wait: kernel(a*b) = kernel(kernel(a) * kernel(b)) 
    # For kernel(n * prod) = 1, we need kernel(n) * kernel(prod) to be a perfect square
    # Since both are squarefree, this means kernel(n) = kernel(prod)
    
    # So we need: XOR of squarefree kernels of selected numbers = target_kernel
    # (where XOR is multiplicative: the product of squarefree kernels, reduced to squarefree)
    
    # Actually, the squarefree kernel of a product is the XOR (symmetric difference) of the prime sets.
    # So we need the symmetric difference of prime sets of selected numbers to equal the prime set of target_kernel.
    
    # BFS approach: state = squarefree kernel (as integer), start = 1, target = target_kernel
    # Edges: multiply by a number a in (210, M], new_state = kernel(state * a)
    # But we need distinct numbers, so we track which numbers we've used.
    
    # Better approach: iterate M from 211 upward. For each M, check if there's a subset
    # of {211, ..., M} whose kernels XOR to target_kernel.
    # Use BFS where state is (kernel, frozenset_of_used) - too expensive.
    
    # Alternative: Since we just need ANY subset, and the kernel space is small,
    # we can do BFS on kernel states where each "step" picks one number.
    # But we need distinct numbers. However, if we pick the same kernel twice,
    # they cancel out, so we can allow revisiting kernels as long as we use different numbers.
    
    # Simpler: For a given M, collect all kernels of numbers in (210, M].
    # Then check if target_kernel can be formed as XOR of a subset of these kernels.
    # This is a subset XOR problem. With up to ~10000 numbers, this is hard.
    
    # Better: BFS where state is the current kernel value. We start at 1.
    # For each number a in (210, M], we can transition from kernel k to kernel(k * a).
    # We want to reach target_kernel. Use BFS to find if it's reachable.
    # Since we can use each number at most once, but BFS doesn't track usage...
    # However, if we find a path, the numbers in the path are distinct by construction
    # if we ensure we don't reuse. But standard BFS might reuse.
    
    # Key insight: If kernel(a) = kernel(b), then using a or b gives the same transition.
    # So we can deduplicate by kernel. For each distinct kernel value, we just need one number.
    # Then the problem becomes: can we reach target_kernel from 1 using at most one of each kernel value?
    # But multiple numbers can have the same kernel, and using two with the same kernel cancels.
    
    # Let me just do BFS on kernel states, allowing reuse (which is valid since if we use
    # two numbers with the same kernel, their product has kernel 1, so it's fine to include both
    # as long as they're distinct numbers). Actually no - if we use a and b with same kernel,
    # the combined kernel contribution is 1, so it's equivalent to not using either.
    # So we can restrict to using at most one number per kernel value.
    
    # For each M, gather distinct kernels of numbers in (210, M].
    # BFS from kernel 1, using each kernel at most once, to reach target_kernel.
    # State: (current_kernel, frozenset of used kernel values) - still expensive.
    
    # Alternative: Since the kernel space is small (squarefree numbers up to 10210),
    # and we just need reachability, we can do BFS where state is just the kernel,
    # and we allow using each kernel value multiple times (but this over-approximates).
    # If we can reach target_kernel, we need to verify we can do it with distinct numbers.
    
    # Let me try: for each M, get the set of available kernels. Do BFS on kernel states
    # where from state k, we can go to k * a for any a in available_kernels.
    # If we reach target_kernel, extract the path and verify the numbers are distinct.
    
    # Actually, since using kernel a twice gives identity, the BFS on kernel states
    # with unlimited use of each kernel is equivalent to the linear span over GF(2).
    # So we just need to check if target_kernel is in the span of the available kernels.
    
    # This is a linear algebra problem over GF(2)! Represent each squarefree kernel
    # as a bit vector of its prime factors, and check if target is in the span.
    
    # But we also need the actual numbers to be distinct and within range.
    # If the span check passes, we can extract a basis and get the actual numbers.
    
    # Let me implement this:
    
    # First, get all primes up to 10210
    def sieve(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    primes = sieve(max_val)
    prime_index = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    def kernel_to_bits(k):
        """Convert squarefree kernel to bit vector (as integer bitmask)."""
        bits = 0
        d = 2
        temp = k
        while d * d <= temp:
            if temp % d == 0:
                if d in prime_index:
                    bits |= (1 << prime_index[d])
                temp //= d
                while temp % d == 0:
                    temp //= d
            d += 1
        if temp > 1 and temp in prime_index:
            bits |= (1 << prime_index[temp])
        return bits
    
    target_bits = kernel_to_bits(target_kernel)
    
    # Precompute kernels for all numbers in (210, 10210]
    kernels = {}  # num -> kernel
    for a in range(211, max_val + 1):
        k = _squarefree_kernel(a)
        kernels[a] = k
    
    # For each M from 211 to max_val, check if target is in span of kernels[211..M]
    # Use Gaussian elimination over GF(2)
    
    best_result = None
    best_M = None
    
    # Maintain a basis for the span as we add kernels
    basis = []  # list of (bits, number)
    
    for M in range(211, max_val + 1):
        if time.time() > deadline:
            break
        
        k = kernels[M]
        bits = kernel_to_bits(k)
        
        # Try to add to basis
        b = bits
        for basis_bits, basis_num in basis:
            b = min(b, b ^ basis_bits)  # This isn't right for Gaussian elimination
        
        # Proper Gaussian elimination
        new_bits = bits
        for i, (bb, bn) in enumerate(basis):
            # Find highest set bit
            if new_bits == 0:
                break
            hb = bb.bit_length() - 1
            if (new_bits >> hb) & 1:
                new_bits ^= bb
        
        if new_bits != 0:
            # Add to basis
            hb = new_bits.bit_length() - 1
            # Insert in order
            inserted = False
            for i, (bb, bn) in enumerate(basis):
                if bb.bit_length() - 1 < hb:
                    continue
                else:
                    basis.insert(i, (new_bits, M))
                    inserted = True
                    break
            if not inserted:
                basis.append((new_bits, M))
        
        # Check if target is in span
        check = target_bits
        for bb, bn in basis:
            hb = bb.bit_length() - 1
            if (check >> hb) & 1:
                check ^= bb
        
        if check == 0:
            # Target is in span! Extract the subset.
            # Re-do Gaussian elimination to track which basis vectors are used
            subset_nums = []
            remaining = target_bits
            # We need to recompute the basis with tracking
            # For now, just record that M works and extract later
            best_M = M
            break
    
    if best_M is None:
        # Fallback: return [4*210] = [840], which gives max-210 = 630
        return [840]
    
    # Extract the actual subset by re-doing the elimination with tracking
    # Rebuild basis up to best_M
    basis_tracked = []  # list of (bits, number)
    for a in range(211, best_M + 1):
        k = kernels[a]
        bits = kernel_to_bits(k)
        new_bits = bits
        for bb, bn in basis_tracked:
            hb = bb.bit_length() - 1
            if (new_bits >> hb) & 1:
                new_bits ^= bb
        if new_bits != 0:
            hb = new_bits.bit_length() - 1
            inserted = False
            for i, (bb, bn) in enumerate(basis_tracked):
                if bb.bit_length() - 1 < hb:
                    continue
                else:
                    basis_tracked.insert(i, (new_bits, a))
                    inserted = True
                    break
            if not inserted:
                basis_tracked.append((new_bits, a))
    
    # Now extract subset
    subset_nums = []
    remaining = target_bits
    for bb, bn in reversed(basis_tracked):
        hb = bb.bit_length() - 1
        if (remaining >> hb) & 1:
            remaining ^= bb
            subset_nums.append(bn)
    
    if remaining != 0:
        # Shouldn't happen, but fallback
        return [840]
    
    # Verify: product of subset * 210 should be a perfect square
    prod = 210
    for x in subset_nums:
        prod *= x
    root = isqrt(prod)
    if root * root != prod:
        return [840]
    
    # Ensure distinct and within range
    if len(set(subset_nums)) != len(subset_nums):
        return [840]
    if any(x <= 210 or x > max_val for x in subset_nums):
        return [840]
    if len(subset_nums) > max_size:
        return [840]
    if len(subset_nums) < 1:
        return [840]
    
    return list(subset_nums)
# EVOLVE-BLOCK-END
