"""Improved: Incremental search for minimum max value with proper kernel matching."""
import math
import time


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
_PRIMES = _sieve(MAX_VAL)
_PRIME_INDEX = {p: i for i, p in enumerate(_PRIMES)}


def _bitmask_kernel(n):
    """Compute the bitmask of the squarefree kernel of n."""
    if n <= 1:
        return 0
    result = 0
    f = n
    for i, p in enumerate(_PRIMES):
        if p * p > f:
            break
        if f % p == 0:
            exp = 0
            while f % p == 0:
                f //= p
                exp += 1
            if exp % 2 == 1:
                result |= (1 << i)
    if f > 1:
        result |= (1 << _PRIME_INDEX[f])
    return result


_KER_MASK = [0] * (MAX_VAL + 1)
for i in range(211, MAX_VAL + 1):
    _KER_MASK[i] = _bitmask_kernel(i)

_TARGET_MASK = _bitmask_kernel(210)


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
    deadline = start_time + 85.0

    # Precompute: for each kernel value, store the list of numbers with that kernel
    # This will be built incrementally as we increase max_val

    # Strategy: For each candidate max M from 211 to MAX_VAL:
    #   Check if size 1 works: any number in [211, M] with kernel == target
    #   Check if size 2 works: two distinct numbers a,b in [211, M] with kernel(a)^kernel(b) == target
    #   Check if size 3 works: three distinct numbers a,b,c in [211, M] with kernel(a)^kernel(b)^kernel(c) == target
    #   If any works, return the solution (since M is increasing, first hit is optimal)

    # For efficiency, maintain:
    #   - set of kernels seen so far (for size 1 and 2)
    #   - for size 2: map from kernel to one number with that kernel
    #   - for size 3: we need to check if for some c in [211,M], 
    #     there exist a,b in [211,M], a!=b, a!=c, b!=c, with kernel(a)^kernel(b) == target^kernel(c)

    # Let's iterate M from 211 upward
    # At each M, add M to our data structures, then check for solutions

    # Data structures:
    kernel_to_num = {}  # kernel -> one number with that kernel (smallest? or any)
    kernel_count = {}   # kernel -> count of numbers with that kernel
    all_numbers = []    # list of numbers added so far

    for M in range(211, MAX_VAL + 1):
        if time.time() > deadline - 5.0:
            break

        ker_M = _KER_MASK[M]
        all_numbers.append(M)
        if ker_M in kernel_to_num:
            kernel_count[ker_M] = kernel_count.get(ker_M, 1) + 1
        else:
            kernel_to_num[ker_M] = M
            kernel_count[ker_M] = 1

        # Check size 1
        if ker_M == _TARGET_MASK:
            result = [M]
            if _validate(result, N):
                return result

        # Check size 2: need two numbers a, b in [211, M] with kernel(a) ^ kernel(b) == target
        # For each a in [211, M], check if (target ^ kernel(a)) exists among kernels of [211, M]
        # and the number with that kernel is not a (or there are at least 2 numbers with that kernel if a has that kernel)
        found_pair = None
        for a in all_numbers:
            ker_a = _KER_MASK[a]
            need = _TARGET_MASK ^ ker_a
            if need in kernel_to_num:
                b = kernel_to_num[need]
                if b != a:
                    found_pair = [a, b]
                    break
                else:
                    # b == a, so we need another number with kernel 'need'
                    # Check if count of 'need' >= 2
                    if kernel_count.get(need, 0) >= 2:
                        # Find another number with kernel 'need'
                        for c in all_numbers:
                            if c != a and _KER_MASK[c] == need:
                                found_pair = [a, c]
                                break
            if found_pair:
                break

        if found_pair:
            if _validate(found_pair, N):
                return found_pair

        # Check size 3: need three distinct a, b, c in [211, M] with kernel(a)^kernel(b)^kernel(c) == target
        # Fix c, then need a, b with kernel(a)^kernel(b) == target ^ kernel(c)
        # This is a size-2 subproblem for each c
        # To be efficient, for each c, check if the pair condition holds
        # We can reuse the pair-finding logic but for a modified target
        found_triple = None
        # Optimization: only check c values that were recently added or are likely to help
        # For now, check all c in all_numbers, but break early if found
        # This could be slow for large M, so we limit the search
        # If M is getting large, we might need a better approach
        if len(all_numbers) <= 500:
            for c in all_numbers:
                ker_c = _KER_MASK[c]
                target2 = _TARGET_MASK ^ ker_c
                # Find pair a, b != c with kernel(a)^kernel(b) == target2
                found_ab = None
                for a in all_numbers:
                    if a == c:
                        continue
                    ker_a = _KER_MASK[a]
                    need = target2 ^ ker_a
                    if need in kernel_to_num:
                        b = kernel_to_num[need]
                        if b != a and b != c:
                            found_ab = [a, b]
                            break
                        elif b == c and kernel_count.get(need, 0) >= 2:
                            # Find another b' != c with kernel 'need'
                            for b2 in all_numbers:
                                if b2 != a and b2 != c and _KER_MASK[b2] == need:
                                    found_ab = [a, b2]
                                    break
                    if found_ab:
                        break
                if found_ab:
                    found_triple = found_ab + [c]
                    break
        else:
            # For larger M, use a more efficient approach
            # Build a map of kernel -> list of numbers (or at least 2 numbers)
            # Then for each c, check if pair exists for target^kernel(c)
            # This is still O(n^2) in worst case but with better constants
            # Let's just check a subset of c values to save time
            # Check the last 100 c values
            for c in all_numbers[-100:]:
                ker_c = _KER_MASK[c]
                target2 = _TARGET_MASK ^ ker_c
                found_ab = None
                for a in all_numbers:
                    if a == c:
                        continue
                    ker_a = _KER_MASK[a]
                    need = target2 ^ ker_a
                    if need in kernel_to_num:
                        b = kernel_to_num[need]
                        if b != a and b != c:
                            found_ab = [a, b]
                            break
                        elif b == c and kernel_count.get(need, 0) >= 2:
                            for b2 in all_numbers:
                                if b2 != a and b2 != c and _KER_MASK[b2] == need:
                                    found_ab = [a, b2]
                                    break
                    if found_ab:
                        break
                if found_ab:
                    found_triple = found_ab + [c]
                    break

        if found_triple:
            if _validate(found_triple, N):
                return found_triple

    # Fallback: return a known valid solution
    # 210 = 2*3*5*7, so we need product * 2*3*5*7 to be a square
    # Simple solution: [211, 211*211] but 211*211 = 44521 > 10210
    # Try: [420, 420] not distinct
    # Try: find n such that n*210 is a square: n = 210*k^2, 210*k^2 <= 10210
    # k=1: 210, not > 210
    # k=2: 840, valid
    result = [840]
    if _validate(result, N):
        return result
    return [840]
