"""Improved: Systematic search using incremental basis for minimum max element."""
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


def _xor_indices(list1, list2):
    """Symmetric difference of two index lists."""
    s1 = set(list1)
    s2 = set(list2)
    return list(s1 ^ s2)


def _find_subset_via_basis(pairs, target):
    """
    Find a subset of pairs whose kernels XOR to target.
    pairs: list of (number, kernel)
    Returns list of numbers or None.
    """
    n = len(pairs)
    if n == 0:
        return [0] if target == 0 else None

    # Build basis incrementally with tracking
    # basis: list of (pivot_bit, kernel_value, indices_list)
    basis = []

    for idx, (num, ker) in enumerate(pairs):
        cur = ker
        cur_indices = [idx]
        # Reduce by existing basis vectors
        for i in range(len(basis)):
            pivot, b_ker, b_indices = basis[i]
            if cur & (1 << pivot):
                cur ^= b_ker
                cur_indices = _xor_indices(cur_indices, b_indices)
        if cur == 0:
            continue
        pivot = cur.bit_length() - 1
        # Insert maintaining order by pivot descending
        insert_pos = 0
        for i in range(len(basis)):
            if basis[i][0] > pivot:
                insert_pos = i + 1
            else:
                break
        basis.insert(insert_pos, (pivot, cur, cur_indices))

    # Check if target is in span
    t = target
    t_indices = []
    for pivot, b_ker, b_indices in basis:
        if t & (1 << pivot):
            t ^= b_ker
            t_indices = _xor_indices(t_indices, b_indices)
    if t != 0:
        return None
    if not t_indices:
        return None
    result = [pairs[i][0] for i in t_indices]
    # Verify
    xor_sum = 0
    for i in t_indices:
        xor_sum ^= pairs[i][1]
    if xor_sum != target:
        return None
    return result


def construct():
    N = 210
    start_time = time.time()
    deadline = start_time + 80.0

    best_result = None
    best_raw = float('inf')

    # Search M from 211 upward, find minimum max element
    # For each M, check if any subset of [211, M] works
    for M in range(211, min(351, MAX_VAL + 1)):
        if time.time() > deadline:
            break
        pairs = [(n, _KER_MASK[n]) for n in range(211, M + 1)]
        subset = _find_subset_via_basis(pairs, _TARGET_MASK)
        if subset is not None and len(subset) >= 1:
            if _validate(subset, N):
                raw = max(subset) - N
                if raw < best_raw:
                    best_raw = raw
                    best_result = list(subset)
                if best_raw == 1:
                    break

    # If not found in [211, 350], try larger range with optimized approach
    if best_result is None:
        # Group by kernel, keep smallest number per kernel
        # Then search on distinct kernels
        ker_to_min_num = {}
        for n in range(211, MAX_VAL + 1):
            k = _KER_MASK[n]
            if k not in ker_to_min_num:
                ker_to_min_num[k] = n

        # Now search for subset of distinct kernels that XOR to target
        # Use basis on the distinct kernels
        distinct_kers = list(ker_to_min_num.keys())
        pairs2 = [(ker_to_min_num[k], k) for k in distinct_kers]
        subset = _find_subset_via_basis(pairs2, _TARGET_MASK)
        if subset is not None and len(subset) >= 1:
            if _validate(subset, N):
                best_result = list(subset)
                best_raw = max(subset) - N

    if best_result is None:
        # Fallback: size 1
        for n in range(211, MAX_VAL + 1):
            if _KER_MASK[n] == _TARGET_MASK:
                best_result = [n]
                break

    if best_result is None:
        best_result = [840]

    if _validate(best_result, N):
        return list(best_result)
    return [840]

if __name__ == "__main__":
    construct()
