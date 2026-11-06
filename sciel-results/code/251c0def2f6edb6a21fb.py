import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = 120.0
    safety_margin = 10.0

    N = 512
    max_a = N - 1

    # Precompute prime factors for each number 1..511
    def prime_factors(n):
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors

    pf = [set() for _ in range(max_a + 1)]
    for i in range(2, max_a + 1):
        pf[i] = prime_factors(i)

    # Precompute weights: weight[a] = 1/(512 - a)
    weights = [0.0] * (max_a + 1)
    for a in range(1, max_a + 1):
        weights[a] = 1.0 / (N - a)

    # Beam search with prime-factor-based coprime check
    beam_width = 80

    # State: (score, chosen_list, chosen_prime_set)
    # chosen_prime_set is the union of prime factors of all chosen elements
    beams = [(0.0, [], set())]

    for a in range(max_a, 0, -1):
        w = weights[a]
        a_factors = pf[a]
        new_beams = []

        for score, chosen, cprimes in beams:
            # Option 1: Skip a
            new_beams.append((score, chosen, cprimes))

            # Option 2: Include a if coprime to all chosen
            if not a_factors & cprimes:
                new_chosen = chosen + [a]
                new_cprimes = cprimes | a_factors
                new_beams.append((score + w, new_chosen, new_cprimes))

        # Prune to beam_width
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_width]

        # Time check
        if time.time() - start_time > deadline - safety_margin:
            break

    # Get best beam
    best_score, best_chosen, _ = max(beams, key=lambda x: x[0])

    # Greedy fill: try to add remaining elements that are coprime
    chosen_set = set(best_chosen)
    cprimes = set()
    for a in best_chosen:
        cprimes |= pf[a]

    improved = True
    while improved:
        improved = False
        for a in range(max_a, 0, -1):
            if a in chosen_set:
                continue
            if not pf[a] & cprimes:
                chosen_set.add(a)
                cprimes |= pf[a]
                improved = True
                if time.time() - start_time > deadline - safety_margin:
                    break
        if time.time() - start_time > deadline - safety_margin:
            break

    # Local search: 1-swap and drop-refill
    # Convert to list for easier manipulation
    result = list(chosen_set)

    def compute_score(lst):
        return sum(weights[a] for a in lst)

    def get_cprimes(lst):
        s = set()
        for a in lst:
            s |= pf[a]
        return s

    best_list = result
    best_sc = compute_score(best_list)

    for _iteration in range(50):
        if time.time() - start_time > deadline - safety_margin:
            break

        improved = False

        # 1-swap: try replacing one element with another
        best_cprimes = get_cprimes(best_list)
        best_set = set(best_list)

        for i, a_remove in enumerate(best_list):
            # Remove a_remove, see what we can add
            temp_cprimes = set()
            for a in best_list:
                if a != a_remove:
                    temp_cprimes |= pf[a]

            # Find best replacement
            best_add = None
            best_add_w = weights[a_remove]  # must beat this to improve

            for a_add in range(max_a, 0, -1):
                if a_add in best_set and a_add != a_remove:
                    continue
                if not pf[a_add] & temp_cprimes:
                    if weights[a_add] > best_add_w:
                        best_add_w = weights[a_add]
                        best_add = a_add
                    break  # Since we go from high a to low, first valid is best weight

            if best_add is not None:
                new_list = list(best_list)
                new_list[i] = best_add
                new_sc = best_sc - weights[a_remove] + weights[best_add]
                if new_sc > best_sc + 1e-15:
                    best_list = new_list
                    best_sc = new_sc
                    best_set = set(best_list)
                    improved = True

        # Drop-and-refill: remove one element, then greedily fill
        if not improved:
            for a_remove in best_list:
                temp_list = [a for a in best_list if a != a_remove]
                temp_cprimes = get_cprimes(temp_list)
                temp_set = set(temp_list)

                # Greedily add elements
                for a_add in range(max_a, 0, -1):
                    if a_add in temp_set:
                        continue
                    if not pf[a_add] & temp_cprimes:
                        temp_list.append(a_add)
                        temp_set.add(a_add)
                        temp_cprimes |= pf[a_add]

                temp_sc = compute_score(temp_list)
                if temp_sc > best_sc + 1e-15:
                    best_list = temp_list
                    best_sc = temp_sc
                    improved = True

        if not improved:
            break

    return best_list
