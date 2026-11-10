import time
import random
import numpy as np

def _primes_upto(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

_PRIMES = _primes_upto(1000)
_N = len(_PRIMES)  # 168
MAX_L = 100000
DEADLINE = 145.0


def _coverage_array(res, limit):
    """Compute coverage count for each x in 1..limit."""
    mark = np.zeros(limit + 1, dtype=np.int16)
    for pp, rr in zip(_PRIMES, res):
        rr = int(rr)
        if rr == 0:
            start = pp
        else:
            start = rr
            if start < 1:
                start += ((1 - start + pp - 1) // pp) * pp
        if start <= limit:
            mark[start::pp] += 1
    return mark


def _prefix_L(coverage, limit):
    """Largest L such that coverage[1..L] >= 2."""
    ok = coverage[1:limit + 1] >= 2
    if ok.all():
        return limit
    return int(np.argmax(~ok))


def _score(res, limit):
    cov = _coverage_array(res, limit)
    return _prefix_L(cov, limit)


def _initial_greedy(target_L, seed=0):
    """Greedy: assign residues to maximize coverage of 1..target_L."""
    rng = random.Random(seed)
    n = _N
    residues = [0] * n
    limit = min(target_L + 200, 5000)
    coverage = _coverage_array(residues, limit)

    for _pass in range(3):
        order = list(range(n))
        if _pass == 0:
            order.sort(key=lambda i: _PRIMES[i])
        elif _pass == 1:
            order.sort(key=lambda i: -_PRIMES[i])
        else:
            rng.shuffle(order)

        for i in order:
            p = _PRIMES[i]
            old_r = residues[i]

            old_start = old_r if old_r > 0 else p
            if old_start <= limit:
                coverage[old_start::p] -= 1

            best_r = old_r
            best_score = -1

            if p > 200:
                candidates = list(range(p))
                rng.shuffle(candidates)
                candidates = candidates[:min(50, p)]
            else:
                candidates = list(range(p))

            for r in candidates:
                new_start = r if r > 0 else p
                if new_start <= limit:
                    coverage[new_start::p] += 1
                    s = int(np.sum(coverage[1:limit + 1] >= 2))
                    coverage[new_start::p] -= 1
                else:
                    s = int(np.sum(coverage[1:limit + 1] >= 2))
                if s > best_score:
                    best_score = s
                    best_r = r

            residues[i] = best_r
            new_start = best_r if best_r > 0 else p
            if new_start <= limit:
                coverage[new_start::p] += 1

    return residues


def _bottleneck_ga(residues, limit, budget, seed=0):
    """
    Bottleneck-driven genetic algorithm:
    - Population of candidate solutions
    - Selection based on performance at the current failure point
    - Crossover only on the subset of primes covering the first integer with count < 2
    """
    rng = random.Random(seed)
    n = _N
    pop_size = 30
    t_start = time.time()

    # Initialize population
    pop = [list(residues)]
    for _ in range(pop_size - 1):
        r = list(residues)
        for i in range(n):
            if rng.random() < 0.3:
                r[i] = rng.randint(0, _PRIMES[i] - 1)
        pop.append(r)

    best_score = 0
    best_res = list(residues)

    # Evaluate initial population
    for i, ind in enumerate(pop):
        s = _score(ind, limit)
        if s > best_score:
            best_score = s
            best_res = list(ind)

    while time.time() - t_start < budget:
        # Evaluate all individuals
        scores = []
        for ind in pop:
            s = _score(ind, limit)
            scores.append(s)
            if s > best_score:
                best_score = s
                best_res = list(ind)

        # Find bottleneck position for the best individual
        cov = _coverage_array(best_res, limit)
        ok = cov[1:limit + 1] >= 2
        if not ok.all():
            bottleneck_pos = int(np.argmax(~ok)) + 1  # 1-indexed
        else:
            bottleneck_pos = limit + 1

        # Identify which primes cover the bottleneck position
        # A prime p with residue r covers x if x % p == r
        covering_primes = []
        for i, p in enumerate(_PRIMES):
            r = best_res[i]
            if r == 0:
                r_eff = p
            else:
                r_eff = r
            if bottleneck_pos % p == (r_eff % p):
                covering_primes.append(i)

        # If no primes cover the bottleneck, all primes are candidates
        if not covering_primes:
            covering_primes = list(range(n))

        # Crossover: only on covering primes
        new_pop = []
        for _ in range(pop_size // 2):
            # Select parents based on scores
            parent1_idx = rng.choices(range(pop_size), weights=[s + 1 for s in scores])[0]
            parent2_idx = rng.choices(range(pop_size), weights=[s + 1 for s in scores])[0]
            p1 = pop[parent1_idx]
            p2 = pop[parent2_idx]

            child = list(p1)
            # Crossover on covering primes only
            for i in covering_primes:
                if rng.random() < 0.5:
                    child[i] = p2[i]

            # Mutation
            for i in range(n):
                if rng.random() < 0.1:
                    child[i] = rng.randint(0, _PRIMES[i] - 1)

            new_pop.append(child)

        # Keep best individuals
        pop = new_pop + pop[:pop_size - len(new_pop)]
        # Ensure unique-ish
        pop = pop[:pop_size]

    return best_score, best_res


def _local_search(residues, limit, budget, seed=0):
    """Local search with incremental coverage updates."""
    rng = random.Random(seed)
    n = _N
    best_res = list(residues)
    coverage = _coverage_array(best_res, limit)
    best_score = _prefix_L(coverage, limit)

    t_start = time.time()
    no_improve = 0
    max_no_improve = 300

    while time.time() - t_start < budget and no_improve < max_no_improve:
        i = rng.randint(0, n - 1)
        p = _PRIMES[i]
        old_r = best_res[i]

        if p > 300:
            candidates = [rng.randint(0, p - 1) for _ in range(15)]
            if old_r not in candidates:
                candidates.append(old_r)
        else:
            candidates = list(range(p))

        best_delta = 0
        best_new_r = old_r

        old_start = old_r if old_r > 0 else p
        if old_start <= limit:
            coverage[old_start::p] -= 1

        for r in candidates:
            if r == old_r:
                continue
            new_start = r if r > 0 else p
            if new_start <= limit:
                coverage[new_start::p] += 1
                s = _prefix_L(coverage, limit)
                coverage[new_start::p] -= 1
            else:
                s = _prefix_L(coverage, limit)

            delta = s - best_score
            if delta > best_delta:
                best_delta = delta
                best_new_r = r

        if old_start <= limit:
            coverage[old_start::p] += 1

        if best_delta > 0:
            new_start = best_new_r if best_new_r > 0 else p
            if new_start <= limit:
                coverage[new_start::p] += 1
            best_res[i] = best_new_r
            best_score += best_delta
            no_improve = 0
        else:
            no_improve += 1

    return best_score, best_res


def construct():
    start_time = time.time()

    best_score = 84
    best_res = [((i + 2) // 2) % p for i, p in enumerate(_PRIMES)]

    # Try various target L values with multiple strategies
    L_candidates = [200, 400, 600, 800, 1000, 1200, 1500]

    for L in L_candidates:
        if time.time() >= DEADLINE - 10:
            break

        limit = min(L + 200, 5000)

        # Strategy 1: Multi-start local search
        for s in range(3):
            if time.time() >= DEADLINE - 5:
                break
            res = _initial_greedy(L, seed=2024 + L + s * 1000)
            score = _score(res, limit)
            if score > best_score:
                best_score = score
                best_res = list(res)

            if time.time() < DEADLINE - 10:
                lc_budget = min(2.0, max(0.5, (DEADLINE - time.time()) / 8))
                s2, r2 = _local_search(res, limit, budget=lc_budget, seed=2024 + L + s * 1000 + 1)
                if s2 > best_score:
                    best_score = s2
                    best_res = list(r2)

        # Strategy 2: Bottleneck-driven GA
        if time.time() < DEADLINE - 15:
            ga_budget = min(3.0, max(1.0, (DEADLINE - time.time()) / 6))
            s3, r3 = _bottleneck_ga(best_res, limit, budget=ga_budget, seed=2024 + L + 999)
            if s3 > best_score:
                best_score = s3
                best_res = list(r3)

        if best_score >= 1000:
            break

    # Final local search refinement
    if time.time() < DEADLINE - 5:
        final_limit = min(best_score + 500, 5000)
        lc_budget = max(1.0, DEADLINE - 10 - time.time())
        s4, r4 = _local_search(best_res, final_limit, budget=lc_budget, seed=99999)
        if s4 > best_score:
            best_score = s4
            best_res = list(r4)

    # Final verification with extended limit
    final_score = _score(best_res, min(best_score + 2000, MAX_L))
    if final_score > best_score:
        best_score = final_score

    return [int(r) for r in best_res]
