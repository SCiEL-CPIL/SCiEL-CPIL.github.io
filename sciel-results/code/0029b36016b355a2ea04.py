import math
import time
import random

def construct():
    deadline = time.time() + 85
    N = 256

    def sieve(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for p in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[p]:
                for m in range(p * p, limit + 1, p):
                    is_prime[m] = False
        return [i for i in range(2, limit + 1) if is_prime[i]]

    primes = sieve(N)

    def prime_factors(n):
        if n == 1:
            return frozenset()
        factors = set()
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                factors.add(p)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            factors.add(temp)
        return frozenset(factors)

    num_factors = {n: prime_factors(n) for n in range(1, N + 1)}

    def greedy():
        selected = set()
        used_primes = set()
        for n in range(N, 0, -1):
            if not (num_factors[n] & used_primes):
                selected.add(n)
                used_primes |= num_factors[n]
        return selected

    best_set = greedy()
    best_sum = sum(best_set)

    def local_search(current_set):
        nonlocal best_set, best_sum
        current_sum = sum(current_set)
        used_primes = set()
        for n in current_set:
            used_primes |= num_factors[n]

        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding
            for n in range(1, N + 1):
                if n not in current_set and not (num_factors[n] & used_primes):
                    current_set.add(n)
                    used_primes |= num_factors[n]
                    current_sum += n
                    improved = True
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_set = set(current_set)
            # Try 1-for-1 swaps
            for n_out in list(current_set):
                new_used = used_primes - num_factors[n_out]
                for n_in in range(1, N + 1):
                    if n_in in current_set:
                        continue
                    if not (num_factors[n_in] & new_used):
                        new_sum = current_sum - n_out + n_in
                        if new_sum > current_sum:
                            current_set.discard(n_out)
                            current_set.add(n_in)
                            used_primes = new_used | num_factors[n_in]
                            current_sum = new_sum
                            improved = True
                            if current_sum > best_sum:
                                best_sum = current_sum
                                best_set = set(current_set)
                            break

    local_search(best_set)

    random.seed(42)
    for _ in range(200):
        if time.time() > deadline - 5:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        selected = set()
        used_primes = set()
        for n in order:
            if not (num_factors[n] & used_primes):
                selected.add(n)
                used_primes |= num_factors[n]
        local_search(selected)

    return [int(x) for x in best_set]
