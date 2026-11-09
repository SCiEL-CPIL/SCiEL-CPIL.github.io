import math
import time
import heapq
from collections import defaultdict

P = {'n': 210, 'maximum_window': 10210, 'maximum_size': 100}

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    n = 210
    max_val = 10210
    max_size = 100

    # Sieve for primes up to max_val
    sieve = [True] * (max_val + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if sieve[i]:
            for j in range(i * i, max_val + 1, i):
                sieve[j] = False
    primes = [i for i in range(2, max_val + 1) if sieve[i]]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)

    # Target parity: 210 = 2*3*5*7
    target_primes = [2, 3, 5, 7]
    target_mask = 0
    for p in target_primes:
        if p in prime_to_idx:
            target_mask |= (1 << prime_to_idx[p])

    def get_parity_mask(x):
        """Return bitmask of prime indices with odd exponent in x."""
        mask = 0
        temp = x
        for i, p in enumerate(primes):
            if p * p > temp:
                break
            if temp % p == 0:
                cnt = 0
                while temp % p == 0:
                    temp //= p
                    cnt += 1
                if cnt % 2 == 1:
                    mask |= (1 << i)
        if temp > 1:
            if temp in prime_to_idx:
                mask |= (1 << prime_to_idx[temp])
        return mask

    # Precompute parities for numbers 211 to max_val
    parities = [0] * (max_val + 1)
    for x in range(211, max_val + 1):
        parities[x] = get_parity_mask(x)

    # Build parity_map: map from parity mask to sorted list of numbers
    parity_map = defaultdict(list)
    for x in range(211, max_val + 1):
        parity_map[parities[x]].append(x)

    # Dijkstra-based search:
    # State: (current_parity_mask, max_value_used)
    # We want to reach target_mask with minimal max_value_used
    # 
    # Key insight: Instead of tracking all selected numbers, we track:
    # - The current parity mask (XOR of all selected numbers' parities)
    # - The maximum number used so far
    # 
    # But this doesn't work directly because we need to ensure distinctness.
    # 
    # Better approach: Use BFS/Dijkstra where the state is just the parity mask,
    # and the "cost" is the minimum possible maximum value to achieve that mask.
    # 
    # For each parity mask, we want to know: what is the smallest M such that
    # there exists a subset of {211, ..., M} whose parity XOR equals that mask?
    #
    # This is equivalent to: find the minimal M such that target_mask is in the
    # span (over GF(2)) of the parity vectors of {211, ..., M}.
    #
    # We can use Gaussian elimination incrementally: add numbers one by one in
    # increasing order, maintaining a basis, and check when target_mask becomes
    # representable.

    # Incremental Gaussian elimination approach
    # Maintain a basis of parity vectors, adding numbers in order from 211 upward
    # At each step, check if target_mask is in the span

    # Build basis incrementally
    # basis[i] = a vector with leading 1 at position i (or 0 if no such vector)
    # We'll use a list where basis[i] represents a vector with the highest set bit at position i

    def try_add_to_basis(basis, vec):
        """Try to add vec to the basis. Returns True if vec is independent (was added)."""
        v = vec
        for i in range(num_primes - 1, -1, -1):
            if not (v & (1 << i)):
                continue
            if basis[i]:
                v ^= basis[i]
            else:
                basis[i] = v
                return True
        return False  # vec is in the span

    def is_in_span(basis, vec):
        """Check if vec is in the span of the basis."""
        v = vec
        for i in range(num_primes - 1, -1, -1):
            if not (v & (1 << i)):
                continue
            if basis[i]:
                v ^= basis[i]
            else:
                return False
        return True  # v reduced to 0, so it's in the span

    # Now search: add numbers from 211 upward, maintaining basis
    # When target_mask becomes in the span, we've found the minimal M
    # But we also need to reconstruct the actual subset

    # To reconstruct, we need to track which numbers were used to build the basis
    # and how to express target_mask as a combination

    # Alternative: Once we find the minimal M, do a second pass to extract the subset
    # using back-substitution on the basis

    # First, find minimal M
    basis = [0] * num_primes
    # Also track for each basis vector, which original number it came from
    # and the chain of reductions

    # For reconstruction, let's store the full reduction path
    # basis_info[i] = (vector, original_number, list of other numbers used in reduction)
    
    # Simpler: just find M first, then do a separate extraction

    minimal_M = None
    for M in range(211, max_val + 1):
        if time.time() > deadline:
            break
        vec = parities[M]
        if try_add_to_basis(basis, vec):
            # Check if target_mask is now in the span
            if is_in_span(basis, target_mask):
                minimal_M = M
                break

    if minimal_M is None:
        # Fallback
        best_result = [840]
        best_raw = 630
    else:
        # Now we need to extract a subset of {211, ..., minimal_M} whose parity XOR = target_mask
        # Use the basis to express target_mask as a combination of basis vectors
        # Each basis vector corresponds to some original number (or combination)
        
        # Rebuild basis with tracking
        basis = [0] * num_primes
        basis_source = [None] * num_primes  # basis_source[i] = set of numbers that form basis[i]
        
        for M in range(211, minimal_M + 1):
            if time.time() > deadline:
                break
            vec = parities[M]
            v = vec
            used_numbers = {M}
            for i in range(num_primes - 1, -1, -1):
                if not (v & (1 << i)):
                    continue
                if basis[i]:
                    v ^= basis[i]
                    used_numbers = used_numbers | basis_source[i]
                else:
                    basis[i] = v
                    basis_source[i] = used_numbers
                    break

        # Now express target_mask as combination of basis vectors
        v = target_mask
        solution_set = set()
        for i in range(num_primes - 1, -1, -1):
            if not (v & (1 << i)):
                continue
            if basis[i]:
                v ^= basis[i]
                solution_set = solution_set | basis_source[i]
            else:
                # Should not happen since we verified target_mask is in span
                break

        # Convert to list
        best_result = sorted(solution_set)
        best_raw = minimal_M - 210

    # Verify and fallback
    def verify(solution):
        if not isinstance(solution, (list, tuple)):
            return False
        if len(solution) < 1 or len(solution) > max_size:
            return False
        if len(set(solution)) != len(solution):
            return False
        for x in solution:
            if not isinstance(x, int) or isinstance(x, bool):
                return False
            if x <= n or x > max_val:
                return False
        prod = n
        for x in solution:
            prod *= x
        r = math.isqrt(prod)
        if r * r != prod:
            return False
        return True

    if not verify(best_result):
        # Fallback to known good solution
        best_result = [840]
        best_raw = 630

    # Return as list of Python ints
    return [int(x) for x in best_result]
