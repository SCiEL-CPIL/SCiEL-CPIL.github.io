"""Optimized: find small subset with squarefree kernel XOR == 210."""
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


def _find_2element(max_val):
    """Find two distinct numbers in (210, max_val] with kernel XOR == 210."""
    target = 210
    seen = {}
    for n in range(211, max_val + 1):
        ker = _KER[n]
        need = ker ^ target
        if need in seen:
            return [seen[need], n]
        if ker not in seen:
            seen[ker] = n
    return None


def _find_3element(max_val):
    """Find three distinct numbers in (210, max_val] with kernel XOR == 210."""
    target = 210
    ker_to_num = {}
    for n in range(211, max_val + 1):
        ker = _KER[n]
        if ker not in ker_to_num:
            ker_to_num[ker] = n

    for n1 in range(211, max_val + 1):
        k1 = _KER[n1]
        for n2 in range(n1 + 1, max_val + 1):
            k2 = _KER[n2]
            need = k1 ^ k2 ^ target
            n3 = ker_to_num.get(need)
            if n3 is not None and n3 != n1 and n3 != n2:
                return [n1, n2, n3]
    return None


def _find_4element(max_val):
    """Find four distinct numbers with kernel XOR == 210.
    Use meet-in-the-middle: split into two pairs."""
    target = 210
    mid = (211 + max_val) // 2
    pair_map = {}
    for n1 in range(211, mid + 1):
        k1 = _KER[n1]
        for n2 in range(n1 + 1, mid + 1):
            k2 = _KER[n2]
            px = k1 ^ k2
            if px not in pair_map:
                pair_map[px] = (n1, n2)

    for n3 in range(mid + 1, max_val + 1):
        k3 = _KER[n3]
        for n4 in range(n3 + 1, max_val + 1):
            k4 = _KER[n4]
            need = k3 ^ k4 ^ target
            pair = pair_map.get(need)
            if pair is not None:
                n1, n2 = pair
                if len({n1, n2, n3, n4}) == 4:
                    return [n1, n2, n3, n4]
    return None


def construct():
    N = 210
    start_time = time.time()
    deadline = start_time + 80.0

    best_result = None
    best_raw = float('inf')

    def try_update(result):
        nonlocal best_result, best_raw
        if result is None:
            return
        if not _validate(result, N):
            return
        raw = max(result) - N
        if raw < best_raw:
            best_raw = raw
            best_result = list(result)

    # 1. Check 1-element solutions in [211, 230]
    for n in range(211, 231):
        if _KER[n] == 210:
            best_result = [n]
            best_raw = n - N
            return list(best_result)

    # 2. Check 3-element solutions in [211, 230]
    ker_to_nums = {}
    for n in range(211, 231):
        k = _KER[n]
        if k not in ker_to_nums:
            ker_to_nums[k] = []
        ker_to_nums[k].append(n)

    found_triple = None
    for a in range(211, 231):
        ka = _KER[a]
        for b in range(a + 1, 231):
            kb = _KER[b]
            need_k = ka ^ kb ^ 210
            if need_k in ker_to_nums:
                for c in ker_to_nums[need_k]:
                    if c != a and c != b:
                        found_triple = [a, b, c]
                        break
            if found_triple:
                break
        if found_triple:
            break

    if found_triple:
        if _validate(found_triple, N):
            best_result = found_triple
            best_raw = max(found_triple) - N
            return list(best_result)

    # 3. Try 2-element first across all max_val
    r2 = _find_2element(MAX_VAL)
    try_update(r2)

    # 4. Try 3-element for increasing max_val
    max_val = 211
    while max_val <= MAX_VAL and time.time() < deadline:
        r3 = _find_3element(max_val)
        try_update(r3)

        if best_raw is not None and best_raw <= 3:
            break

        if max_val < 500:
            max_val += 1
        elif max_val < 1500:
            max_val += 20
        elif max_val < 4000:
            max_val += 200
        else:
            max_val += 2000

    # 5. If still nothing, try 4-element
    if best_result is None and time.time() < deadline:
        r4 = _find_4element(MAX_VAL)
        try_update(r4)

    # Single element check (full range)
    if best_result is None:
        for n in range(211, MAX_VAL + 1):
            if _KER[n] == 210:
                best_result = [n]
                break

    # Fallback
    if best_result is None:
        best_result = [840]

    if _validate(best_result, N):
        return list(best_result)
    return [840]
