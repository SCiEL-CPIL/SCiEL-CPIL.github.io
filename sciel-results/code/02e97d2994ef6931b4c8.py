"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _validity_check(S_list):
    """Check if sorted list S_list satisfies: for all distinct a,b,c, a does not divide b*c."""
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            for k in range(j + 1, n):
                c = S_list[k]
                if (b * c) % a == 0:
                    return False
    return True

P = {'n': 256}

def construct():
    deadline = time.time() + 70

    n_max = P['n']
    primes = [x for x in range(2, n_max + 1) if _is_prime(x)]
    S = sorted(primes)
    best = S[:]

    # Try adding each composite to the prime set
    composites = [x for x in range(2, n_max + 1) if not _is_prime(x)]
    for x in composites:
        if time.time() > deadline:
            break
        S_test = S + [x]
        S_test.sort()
        if _validity_check(S_test):
            S = S_test
            best = S[:]

    # Local search: try swapping one element out and another in
    all_nums = list(range(2, n_max + 1))
    max_rounds = 200
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        for x in all_nums:
            if x in S:
                continue
            for y in S:
                if time.time() > deadline:
                    break
                S_test = [z for z in S if z != y] + [x]
                S_test.sort()
                if _validity_check(S_test):
                    S = S_test
                    best = S[:]
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return [int(x) for x in sorted(best)]
# EVOLVE-BLOCK-END
