"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial

TARGET = factorial(12)  # 479001600

PRIMES = [2, 3, 5, 7, 11]
EXPS = [10, 5, 2, 1, 1]


def _build_divisors(max_val):
    divisors = []
    n = len(PRIMES)

    def gen(idx, cur):
        if cur > max_val:
            return
        if idx == n:
            if cur > 12:
                divisors.append(cur)
            return
        p = PRIMES[idx]
        for e in range(EXPS[idx] + 1):
            gen(idx + 1, cur * (p ** e))

    gen(0, 1)
    divisors.sort()
    return divisors


def _prime_exponents(n):
    """Return list of exponents for PRIMES in n."""
    exps = []
    for p in PRIMES:
        e = 0
        while n % p == 0:
            e += 1
            n //= p
        exps.append(e)
    return exps


def construct():
    start = time.time()
    deadline = start + 85.0

    all_divs = _build_divisors(TARGET)
    n_divs = len(all_divs)

    # Precompute prime exponents for each divisor
    div_exp_list = [_prime_exponents(d) for d in all_divs]

    target_exp = EXPS[:]  # [10, 5, 2, 1, 1]

    for k in range(n_divs):
        if time.time() > deadline - 2.0:
            break

        bound = all_divs[k]
        # Check if prime exponents are sufficient
        avail_exp = [0] * 5
        for j in range(k + 1):
            for p_idx in range(5):
                avail_exp[p_idx] += div_exp_list[j][p_idx]
        
        if any(avail_exp[i] < target_exp[i] for i in range(5)):
            continue  # Not enough prime factors available

        divs_subset = all_divs[:k + 1]
        divs_desc = divs_subset[::-1]
        n = len(divs_desc)

        memo = {}

        def dfs(idx, tgt, rem):
            if time.time() > deadline - 1.0:
                return None
            if tgt == 1:
                return []
            if rem == 0 or idx >= n:
                return None
            
            key = (idx, tgt, rem)
            if key in memo:
                return memo[key]
            
            for i in range(idx, n):
                d = divs_desc[i]
                if d > tgt:
                    continue
                if tgt % d != 0:
                    continue
                
                sub_target = tgt // d
                if sub_target == 1:
                    memo[key] = [d]
                    return [d]
                
                sub = dfs(i + 1, sub_target, rem - 1)
                if sub is not None:
                    result = [d] + sub
                    memo[key] = result
                    return result
            
            memo[key] = None
            return None

        result = dfs(0, TARGET, 12)
        if result is not None:
            return result

    return [TARGET]
# EVOLVE-BLOCK-END
