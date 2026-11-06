"""Combined: prime-factor bitmask for fast coprime checks + greedy with multiple orderings + local search."""
import math
import random
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    N = 512

    # Precompute prime factor bitmasks for all numbers 1..511
    # This allows O(1) coprime check via bitwise AND
    prime_factors_mask = [0] * N

    # Find all primes < 512
    is_prime = [True] * N
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N, i):
                is_prime[j] = False

    primes = [i for i in range(2, N) if is_prime[i]]
    prime_index = {p: i for i, p in enumerate(primes)}

    # Compute bitmask for each number
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

    # Greedy from largest to smallest (baseline)
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

    # Precompute all numbers 1..511
    all_nums = list(range(1, 512))

    # Try different random orderings with greedy selection
    rng = random.Random(42)

    def greedy_with_order(order):
        """Greedy: iterate through order, add a if coprime with all chosen."""
        chosen = []
        used_mask = 0
        for a in order:
            if (prime_factors_mask[a] & used_mask) == 0:
                chosen.append(a)
                used_mask |= prime_factors_mask[a]
        return chosen

    # Try several random permutations (sampled, not full)
    # To be efficient, try orders biased toward large values
    # Use numpy for fast weighted sampling
    import numpy as np
    np_rng = np.random.RandomState(42)
    
    # Precompute weights once
    all_weights = np.array([1.0 / (N - a) for a in all_nums])
    
    for trial in range(500):
        if time.time() > deadline:
            break

        # Create a biased random order using numpy weighted sampling
        # Sample all indices without replacement with replacement probability proportional to weight
        order_idx = np_rng.choice(len(all_nums), size=len(all_nums), replace=False, p=all_weights / all_weights.sum())
        order = [all_nums[i] for i in order_idx]

        cand = greedy_with_order(order)
        s = score(cand)
        if s > best_score:
            best_score = s
            best = cand

    # Local search: try to swap or add/remove elements
    def local_search(chosen, max_iters=200):
        chosen_set = set(chosen)
        current_score = sum(1.0 / (N - a) for a in chosen)
        used_mask = 0
        for a in chosen:
            used_mask |= prime_factors_mask[a]

        # Precompute sorted list of all numbers by descending value (1/(N-a))
        all_by_value = list(range(511, 0, -1))

        improved = True
        iters = 0
        while improved and iters < max_iters:
            if time.time() > deadline:
                break
            improved = False
            iters += 1

            # Try adding a new element
            # Find candidates not in chosen that are coprime with all chosen
            for a in all_by_value:
                if a in chosen_set:
                    continue
                if (prime_factors_mask[a] & used_mask) == 0:
                    # Adding improves score (always positive)
                    chosen.append(a)
                    chosen_set.add(a)
                    used_mask |= prime_factors_mask[a]
                    current_score += 1.0 / (N - a)
                    improved = True
                    break

            if improved:
                continue

            # Try single-element swap: remove one element, greedily complete
            # Sort chosen by ascending contribution (remove low-value ones first)
            sorted_chosen = sorted(chosen, key=lambda x: 1.0 / (N - x))
            for a in sorted_chosen:
                if time.time() > deadline:
                    break
                # Try removing a
                remaining = [x for x in chosen if x != a]
                rem_mask = 0
                for x in remaining:
                    rem_mask |= prime_factors_mask[x]

                # Greedily add best available
                new_chosen = list(remaining)
                new_set = set(remaining)
                new_mask = rem_mask
                new_score = sum(1.0 / (N - x) for x in new_chosen)

                for c in all_by_value:
                    if c in new_set:
                        continue
                    if (prime_factors_mask[c] & new_mask) == 0:
                        new_chosen.append(c)
                        new_set.add(c)
                        new_mask |= prime_factors_mask[c]
                        new_score += 1.0 / (N - c)

                if new_score > current_score + 1e-15:
                    chosen = new_chosen
                    chosen_set = new_set
                    used_mask = new_mask
                    current_score = new_score
                    improved = True
                    break

            if improved:
                continue

            # Try two-element swap: remove two low-value elements, greedily complete
            if len(chosen) >= 3:
                sorted_chosen2 = sorted(chosen, key=lambda x: 1.0 / (N - x))
                for i in range(len(sorted_chosen2) - 1):
                    if time.time() > deadline:
                        break
                    a1, a2 = sorted_chosen2[i], sorted_chosen2[i+1]
                    remaining = [x for x in chosen if x != a1 and x != a2]
                    rem_mask = 0
                    for x in remaining:
                        rem_mask |= prime_factors_mask[x]

                    new_chosen = list(remaining)
                    new_set = set(remaining)
                    new_mask = rem_mask
                    new_score = sum(1.0 / (N - x) for x in new_chosen)

                    for c in all_by_value:
                        if c in new_set:
                            continue
                        if (prime_factors_mask[c] & new_mask) == 0:
                            new_chosen.append(c)
                            new_set.add(c)
                            new_mask |= prime_factors_mask[c]
                            new_score += 1.0 / (N - c)

                    if new_score > current_score + 1e-15:
                        chosen = new_chosen
                        chosen_set = new_set
                        used_mask = new_mask
                        current_score = new_score
                        improved = True
                        break

        return chosen, current_score

    best, best_score = local_search(best)

    # More random restarts with local search
    for restart in range(100):
        if time.time() > deadline - 5:
            break

        rng_seed = 1000 + restart * 7919
        rng2 = random.Random(rng_seed)

        # Randomized greedy with varied strategies
        strategy = restart % 3
        
        if strategy == 0:
            # Start with random high-value seeds
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
        elif strategy == 1:
            # Start with primes in high range (primes are always coprime)
            high_primes = [p for p in primes if p >= 300]
            rng2.shuffle(high_primes)
            chosen = []
            used_mask = 0
            for a in high_primes:
                if (prime_factors_mask[a] & used_mask) == 0:
                    chosen.append(a)
                    used_mask |= prime_factors_mask[a]
                if len(chosen) >= 3:
                    break
        else:
            # Start with 1 and two random high composites
            chosen = [1]
            used_mask = 0
            seed_candidates = list(range(511, 400, -1))
            rng2.shuffle(seed_candidates)
            for a in seed_candidates:
                if a == 1:
                    continue
                if (prime_factors_mask[a] & used_mask) == 0:
                    chosen.append(a)
                    used_mask |= prime_factors_mask[a]
                if len(chosen) >= 3:
                    break

        if not chosen:
            continue

        # Complete with greedy from largest
        for a in range(511, 0, -1):
            if a in chosen:
                continue
            if (prime_factors_mask[a] & used_mask) == 0:
                chosen.append(a)
                used_mask |= prime_factors_mask[a]

        # Local search with more iterations
        chosen, s = local_search(chosen, max_iters=150)
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
