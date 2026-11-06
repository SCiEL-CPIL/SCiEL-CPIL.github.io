"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    t0 = time.time()
    deadline = t0 + 80.0

    N = 255
    target = 255
    target_primes = set()
    tmp = target
    d = 2
    while d * d <= tmp:
        if tmp % d == 0:
            target_primes.add(d)
            while tmp % d == 0:
                tmp //= d
        d += 1
    if tmp > 1:
        target_primes.add(tmp)

    # Candidate: all multiples of the smallest prime factor of 255
    base_prime = min(target_primes)
    base_set = frozenset(x for x in range(1, N + 1) if x % base_prime == 0)
    best_size = len(base_set)
    best_set = base_set

    # Verify: try to see if removing some base elements and adding others helps
    # For each prime p in target_primes, p != base_prime:
    for p in target_primes:
        if p == base_prime:
            continue
        # Elements divisible by p but not by base_prime
        extra_candidates = [x for x in range(1, N + 1)
                           if x % p == 0 and x % base_prime != 0]
        for x in extra_candidates:
            # To add x, we must remove all base elements coprime to x
            x_primes = set()
            tmp2 = x
            d2 = 2
            while d2 * d2 <= tmp2:
                if tmp2 % d2 == 0:
                    x_primes.add(d2)
                    while tmp2 % d2 == 0:
                        tmp2 //= d2
                d2 += 1
            if tmp2 > 1:
                x_primes.add(tmp2)

            # Remove base elements that share no prime with x
            to_remove = []
            for y in base_set:
                if y == x:
                    continue
                # Check gcd(y, x)
                if math.gcd(y, x) == 1:
                    to_remove.append(y)
            # Also check: remaining base elements + x must be pairwise non-coprime
            # Since all remaining base elements share base_prime with each other,
            # and x shares p with... wait, x is not divisible by base_prime.
            # So remaining base elements must share a factor with x.
            # They share base_prime with each other, but need to share with x.
            # A remaining base element y must have gcd(y,x)>1.
            # Since y is divisible by base_prime and x is not, they share a factor
            # only if y has another prime factor in common with x.

            # Actually the condition is: for ALL pairs in the new set, gcd>1.
            # Pairs of base elements: all divisible by base_prime, so gcd >= base_prime > 1. OK.
            # Pairs involving x: x must share a factor with every remaining base element.
            # x is divisible by p (not base_prime). A base element y is divisible by base_prime.
            # gcd(x,y) > 1 iff x and y share a prime other than base_prime (since x isn't div by base_prime).
            # So y must have a prime factor in x_primes (other than base_prime, which x doesn't have).

            # Count how many base elements we'd need to remove
            removed_count = len(to_remove)
            new_size = best_size - removed_count + 1
            if new_size > best_size:
                best_size = new_size
                best_set = base_set - set(to_remove) | {x}

            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Also try: what if we use a different base prime?
    for p in target_primes:
        if p == base_prime:
            continue
        alt_set = frozenset(x for x in range(1, N + 1) if x % p == 0)
        if len(alt_set) > best_size:
            best_size = len(alt_set)
            best_set = alt_set

    return sorted(best_set)
# EVOLVE-BLOCK-END
