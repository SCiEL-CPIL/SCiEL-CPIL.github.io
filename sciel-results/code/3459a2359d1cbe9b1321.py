"""Stronger algorithm for pairwise coprime maximization with bitmask + local search."""
import math
import random
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    N = 512

    # Precompute prime factor bitmasks for all numbers 1..511
    is_prime = [True] * N
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N, i):
                is_prime[j] = False

    primes = [i for i in range(2, N) if is_prime[i]]
    prime_index = {p: i for i, p in enumerate(primes)}

    prime_factors_mask = [0] * N
    for a in range(2, N):
        x = a
        mask = 0
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                mask |= (1 << prime_index[p])
                while x % p == 0:
                    x //= p
        if x > 1:
            mask |= (1 << prime_index[x])
        prime_factors_mask[a] = mask
    prime_factors_mask[1] = 0

    def score(arr):
        return sum(1.0 / (N - a) for a in arr)

    # Greedy from largest to smallest
    def greedy_desc():
        chosen = []
        used_mask = 0
        for a in range(511, 0, -1):
            if (prime_factors_mask[a] & used_mask) == 0:
                chosen.append(a)
                used_mask |= prime_factors_mask[a]
        return chosen

    best = greedy_desc()
    best_score = score(best)

    # Local search: try to swap or add/remove elements
    def local_search(chosen, max_iters=500):
        chosen_set = set(chosen)
        current_score = sum(1.0 / (N - a) for a in chosen)
        used_mask = 0
        for a in chosen:
            used_mask |= prime_factors_mask[a]

        improved = True
        iters = 0
        while improved and iters < max_iters:
            if time.time() > deadline:
                break
            improved = False
            iters += 1

            # Try adding a new element
            for a in range(511, 0, -1):
                if a in chosen_set:
                    continue
                if (prime_factors_mask[a] & used_mask) == 0:
                    chosen.append(a)
                    chosen_set.add(a)
                    used_mask |= prime_factors_mask[a]
                    current_score += 1.0 / (N - a)
                    improved = True
                    break

            if improved:
                continue

            # Try 1-for-1 swap: remove one element, greedily add back best
            for i, a in enumerate(chosen):
                if time.time() > deadline:
                    break
                remaining = [x for x in chosen if x != a]
                rem_set = set(remaining)
                rem_mask = 0
                for x in remaining:
                    rem_mask |= prime_factors_mask[x]

                new_chosen = list(remaining)
                new_set = set(remaining)
                new_mask = rem_mask
                new_score = sum(1.0 / (N - x) for x in new_chosen)

                for c in range(511, 0, -1):
                    if c in new_set:
                        continue
                    if (prime_factors_mask[c] & new_mask) == 0:
                        new_chosen.append(c)
                        new_set.add(c)
                        new_mask |= prime_factors_mask[c]
                        new_score += 1.0 / (N - c)

                if new_score > current_score:
                    chosen = new_chosen
                    chosen_set = new_set
                    used_mask = new_mask
                    current_score = new_score
                    improved = True
                    break

            if improved:
                continue

            # Try 2-for-2 swap: remove two elements, greedily add back
            n = len(chosen)
            if n >= 2:
                # Sort by contribution (smaller first) to try removing low-value elements
                sorted_indices = sorted(range(n), key=lambda i: chosen[i])
                for i in range(min(n, 25)):
                    if time.time() > deadline:
                        break
                    for j in range(i+1, min(n, i+25)):
                        if time.time() > deadline:
                            break
                        a, b = chosen[sorted_indices[i]], chosen[sorted_indices[j]]
                        remaining = [x for x in chosen if x != a and x != b]
                        rem_set = set(remaining)
                        rem_mask = 0
                        for x in remaining:
                            rem_mask |= prime_factors_mask[x]

                        new_chosen = list(remaining)
                        new_set = set(remaining)
                        new_mask = rem_mask
                        new_score = sum(1.0 / (N - x) for x in new_chosen)

                        for c in range(511, 0, -1):
                            if c in new_set:
                                continue
                            if (prime_factors_mask[c] & new_mask) == 0:
                                new_chosen.append(c)
                                new_set.add(c)
                                new_mask |= prime_factors_mask[c]
                                new_score += 1.0 / (N - c)

                        if new_score > current_score:
                            chosen = new_chosen
                            chosen_set = new_set
                            used_mask = new_mask
                            current_score = new_score
                            improved = True
                            break
                    if improved:
                        break

        return chosen, current_score

    best, best_score = local_search(best)

    # More random restarts with local search
    for restart in range(150):
        if time.time() > deadline - 5:
            break

        rng_seed = 1000 + restart * 7919
        rng2 = random.Random(rng_seed)

        # Randomized greedy
        seed_candidates = list(range(511, 300, -1))
        rng2.shuffle(seed_candidates)

        chosen = []
        used_mask = 0
        for a in seed_candidates:
            if (prime_factors_mask[a] & used_mask) == 0:
                chosen.append(a)
                used_mask |= prime_factors_mask[a]
            if len(chosen) >= 3:
                break

        if not chosen:
            continue

        for a in range(511, 0, -1):
            if a in chosen:
                continue
            if (prime_factors_mask[a] & used_mask) == 0:
                chosen.append(a)
                used_mask |= prime_factors_mask[a]

        chosen, s = local_search(chosen, max_iters=100)
        if s > best_score:
            best_score = s
            best = chosen

    # One more pass: try to add any remaining coprime numbers
    best_set = set(best)
    used_mask = 0
    for a in best:
        used_mask |= prime_factors_mask[a]
    for a in range(511, 0, -1):
        if a not in best_set:
            if (prime_factors_mask[a] & used_mask) == 0:
                best.append(a)
                best_set.add(a)
                used_mask |= prime_factors_mask[a]

    return best

# EVOLVE-BLOCK-END
