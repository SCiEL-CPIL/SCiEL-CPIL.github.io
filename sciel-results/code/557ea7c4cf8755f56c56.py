"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt

P = {'n': 256}

def _valid_set(S):
    """Check that for every three distinct members a,b,c, a does not divide b*c."""
    lst = list(S)
    n = len(lst)
    for i in range(n):
        a = lst[i]
        for j in range(n):
            if j == i:
                continue
            b = lst[j]
            for k in range(n):
                if k == i or k == j:
                    continue
                c = lst[k]
                if (b * c) % a == 0:
                    return False
    return True

def _fast_check_add(S, x):
    """Check if adding x to set S keeps validity. S is a set of ints not containing x."""
    lst = list(S)
    n = len(lst)
    for i in range(n):
        a = lst[i]
        # need: for all b,c distinct from each other and from a, a | b*c fails
        # i.e., there must NOT exist b,c in S, b!=c, such that a | b*c
        # (x is not in S yet, so b,c are from S)
        # But also we must check triples involving x as 'a' and as 'b'/'c'
        pass
    # Check x as 'a': for all b,c in S distinct, x | b*c must be false
    for i in range(n):
        b = lst[i]
        for j in range(n):
            if j == i:
                continue
            c = lst[j]
            if (b * c) % x == 0:
                return False
    # Check x as 'b' or 'c': for all a in S, and some other d in S, a | x*d must be false
    # i.e., there must NOT exist a,d in S, a!=d, such that a | x*d
    for i in range(n):
        a = lst[i]
        for j in range(n):
            if j == i:
                continue
            d = lst[j]
            if (x * d) % a == 0:
                return False
    return True

def construct():
    deadline = time.time() + 80.0
    n = P['n']
    rng = random.Random(12345)

    # Precompute primes in 2..n
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(n) + 1):
        if is_prime[i]:
            for j in range(i*i, n+1, i):
                is_prime[j] = False
    primes = [i for i in range(2, n+1) if is_prime[i]]

    # Start with all primes
    best = set(primes)
    best_size = len(best)

    # Precompute for each x in 2..n, its prime factors
    def prime_factors(x):
        facs = set()
        d = 2
        t = x
        while d * d <= t:
            while t % d == 0:
                facs.add(d)
                t //= d
            d += 1
        if t > 1:
            facs.add(t)
        return facs

    pf = {x: prime_factors(x) for x in range(2, n+1)}

    # Greedy improvement: try to add composites
    # For each composite, check if it can be added
    composites = [x for x in range(2, n+1) if not is_prime[x]]

    # Try adding each composite if valid
    for x in composites:
        if x in best:
            continue
        if _fast_check_add(best, x):
            best.add(x)
            best_size = len(best)

    # Now try swapping: remove a prime, add composites that become valid
    # Local search with swaps
    for iteration in range(200):
        if time.time() > deadline - 10:
            break
        improved = False
        # Try removing each prime and adding best composite
        for p in list(primes):
            if p not in best:
                continue
            # Try removing p and adding some composites
            trial = set(best)
            trial.remove(p)
            added_any = False
            for x in composites:
                if x in trial:
                    continue
                if x == p:
                    continue
                if _fast_check_add(trial, x):
                    trial.add(x)
                    added_any = True
            if len(trial) > len(best):
                best = trial
                best_size = len(best)
                improved = True
        if not improved:
            break

    # Randomized restarts / perturbations
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        trial = set(primes)
        # Randomly exclude some small primes
        small_primes = [p for p in primes if p <= 30]
        for p in small_primes:
            if rng.random() < 0.5:
                trial.add(p)
        # Add composites greedily
        for x in composites:
            if x in trial:
                continue
            if _fast_check_add(trial, x):
                trial.add(x)
        if len(trial) > best_size:
            best = trial
            best_size = len(trial)

    # Final verification
    if not _valid_set(best):
        # Fallback to primes
        best = set(primes)

    result = [int(x) for x in sorted(best)]
    return result
# EVOLVE-BLOCK-END
