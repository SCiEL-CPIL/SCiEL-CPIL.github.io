"""Covering 1..L by prime residue classes with reciprocal budget 1."""
import time
import math
import random
from fractions import Fraction

# EVOLVE-BLOCK-START

_DEADLINE = None
_MAXN = 20000  # coverage array size (L capped at 1e5, but we optimize up to 20000)


def _primes_upto(n):
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]


def _compute_L(classes):
    """Largest L with every integer 1..L covered. Exact, capped at _MAXN."""
    if not classes:
        return 0
    covered = [0] * (_MAXN + 1)
    for p, r in classes:
        first = r if r >= 1 else p
        for n in range(first, _MAXN + 1, p):
            covered[n] += 1
    L = 0
    for n in range(1, _MAXN + 1):
        if covered[n] > 0:
            L = n
        else:
            break
    return L


def _first_gap(classes):
    """Smallest n>=1 not covered (or None if 1.._MAXN all covered). Early exit."""
    covered = [0] * (_MAXN + 1)
    for p, r in classes:
        first = r if r >= 1 else p
        for n in range(first, _MAXN + 1, p):
            covered[n] += 1
    for n in range(1, _MAXN + 1):
        if covered[n] == 0:
            return n
    return None


def _rebuild_count(primes, residues):
    """Coverage count array for a residue vector over a fixed prime list."""
    covered = [0] * (_MAXN + 1)
    for p, r in zip(primes, residues):
        first = r if r >= 1 else p
        for n in range(first, _MAXN + 1, p):
            covered[n] += 1
    return covered


def _gap_of_count(covered):
    for n in range(1, _MAXN + 1):
        if covered[n] == 0:
            return n
    return None


def _L_from_gap(gap):
    return _MAXN if gap is None else gap - 1


def _seed_classes(primes):
    seed_primes = [p for p in primes if p >= 13]
    return [[p, (i + 1) % p] for i, p in enumerate(seed_primes)]


def _seed_budget(primes):
    return sum(Fraction(1, p) for p in primes if p >= 13)


def _anneal(primes, residues, start_temp, min_temp, iterations, deadline, rng):
    """Simulated annealing over residue vector for a fixed prime set."""
    n = len(primes)
    covered = _rebuild_count(primes, residues)
    gap = _gap_of_count(covered)
    cur_L = _L_from_gap(gap)
    best_L = cur_L
    best_res = list(residues)
    temp = start_temp

    for it in range(iterations):
        if time.time() > deadline:
            break
        # pick a prime to perturb
        i = rng.randrange(n)
        old_r = residues[i]
        new_r = rng.randrange(primes[i])
        if new_r == old_r:
            continue
        # update coverage
        p = primes[i]
        for n in range(old_r if old_r >= 1 else p, _MAXN + 1, p):
            covered[n] -= 1
        for n in range(new_r if new_r >= 1 else p, _MAXN + 1, p):
            covered[n] += 1
        residues[i] = new_r
        new_gap = _gap_of_count(covered)
        new_L = _L_from_gap(new_gap)
        delta = new_L - cur_L
        if delta >= 0 or rng.random() < math.exp(delta / max(temp, 1e-9)):
            cur_L = new_L
            gap = new_gap
            if cur_L > best_L:
                best_L = cur_L
                best_res = list(residues)
        else:
            # revert
            for n in range(new_r if new_r >= 1 else p, _MAXN + 1, p):
                covered[n] -= 1
            for n in range(old_r if old_r >= 1 else p, _MAXN + 1, p):
                covered[n] += 1
            residues[i] = old_r
        temp = max(min_temp, temp * 0.9995)

    return best_L, best_res


def _greedy_polish(primes, residues, deadline):
    """Hill-climb: repeatedly set each residue to its best single-move value."""
    n = len(primes)
    covered = _rebuild_count(primes, residues)
    gap = _gap_of_count(covered)
    cur_L = _L_from_gap(gap)
    best_res = list(residues)
    best_L = cur_L
    improved = True
    passes = 0
    while improved and passes < 60:
        improved = False
        passes += 1
        for i in range(n):
            if time.time() > deadline:
                return best_L, best_res
            p = primes[i]
            old_r = residues[i]
            # current L for this prime's position
            base_gap = _gap_of_count(covered)
            base_L = _L_from_gap(base_gap)
            best_r = old_r
            best_L_i = base_L
            for r2 in range(p):
                if r2 == old_r:
                    continue
                for n in range(old_r if old_r >= 1 else p, _MAXN + 1, p):
                    covered[n] -= 1
                for n in range(r2 if r2 >= 1 else p, _MAXN + 1, p):
                    covered[n] += 1
                g = _gap_of_count(covered)
                L_i = _L_from_gap(g)
                if L_i > best_L_i:
                    best_L_i = L_i
                    best_r = r2
                # revert for next trial
                for n in range(r2 if r2 >= 1 else p, _MAXN + 1, p):
                    covered[n] -= 1
                for n in range(old_r if old_r >= 1 else p, _MAXN + 1, p):
                    covered[n] += 1
            if best_r != old_r:
                # commit
                for n in range(old_r if old_r >= 1 else p, _MAXN + 1, p):
                    covered[n] -= 1
                for n in range(best_r if best_r >= 1 else p, _MAXN + 1, p):
                    covered[n] += 1
                residues[i] = best_r
                improved = True
                new_gap = _gap_of_count(covered)
                new_L = _L_from_gap(new_gap)
                if new_L > best_L:
                    best_L = new_L
                    best_res = list(residues)
    return best_L, best_res


def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 140  # safety margin before 150s kill
    rng = random.Random(12345)

    primes_all = _primes_upto(1000)
    small_primes = [2, 3, 5, 7, 11]

    # Seed: all primes >= 13
    seed_primes = [p for p in primes_all if p >= 13]
    seed_budget = sum(Fraction(1, p) for p in seed_primes)
    residues = [(i + 1) % p for i, p in enumerate(seed_primes)]
    classes = [[p, r] for p, r in zip(seed_primes, residues)]
    best_L = _compute_L(classes)
    best_classes = list(classes)

    # Add small primes one by one if budget allows, with best residue
    current_primes = list(seed_primes)
    current_res = list(residues)
    current_budget = seed_budget
    for p in small_primes:
        if time.time() > _DEADLINE - 20:
            break
        cost = Fraction(1, p)
        if current_budget + cost > Fraction(1):
            continue
        # find best residue
        best_r = 0
        best_L_i = best_L
        for r in range(p):
            test = current_classes = [[q, rr] for q, rr in zip(current_primes, current_res)]
            test.append([p, r])
            L_i = _compute_L(test)
            if L_i > best_L_i:
                best_L_i = L_i
                best_r = r
        current_primes.append(p)
        current_res.append(best_r)
        current_budget += cost
        if best_L_i > best_L:
            best_L = best_L_i
            best_classes = [[q, rr] for q, rr in zip(current_primes, current_res)]

    # Simulated annealing on the full prime set (multiple restarts)
    if time.time() < _DEADLINE - 15:
        for restart in range(3):
            if time.time() > _DEADLINE - 10:
                break
            start_res = list(current_res) if restart == 0 else [rng.randrange(p) for p in current_primes]
            L_a, res_a = _anneal(current_primes, start_res,
                                 start_temp=8.0, min_temp=0.3,
                                 iterations=200000,
                                 deadline=_DEADLINE - 12, rng=rng)
            if L_a > best_L:
                best_L = L_a
                best_res = res_a
                best_classes = [[q, rr] for q, rr in zip(current_primes, best_res)]

    # Greedy polish to finish
    if time.time() < _DEADLINE - 5:
        polish_primes = list(current_primes)
        polish_res = [rr for q, rr in best_classes] if best_classes else list(current_res)
        # make sure lengths match
        if len(polish_res) != len(polish_primes):
            polish_res = list(current_res)
        L_p, res_p = _greedy_polish(polish_primes, polish_res, _DEADLINE - 3)
        if L_p > best_L:
            best_L = L_p
            best_classes = [[q, rr] for q, rr in zip(polish_primes, res_p)]

    # Validate budget and distinctness (safety)
    final_budget = sum(Fraction(1, p) for p, r in best_classes)
    if final_budget > Fraction(1):
        # fall back to seed
        best_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_primes)]
        best_L = _compute_L(best_classes)

    return best_classes

# EVOLVE-BLOCK-END
