"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np


def construct(q, n, radius):
    start = time.time()
    deadline = start + 118.0  # safety margin before 120s kill

    # ---- helpers ----
    def int_to_word(idx):
        w = [0] * n
        for i in range(n):
            w[i] = idx % q
            idx //= q
        return w

    def word_to_int(w):
        v = 0
        for i in range(n - 1, -1, -1):
            v = v * q + w[i]
        return v

    total = q ** n

    # Precompute each word's ball (itself + all distance-1 neighbours) as a set of indices.
    ball = []
    for i in range(total):
        w = int_to_word(i)
        s = {i}
        if radius >= 1:
            for pos in range(n):
                ov = w[pos]
                for nv in range(q):
                    if nv != ov:
                        w[pos] = nv
                        s.add(word_to_int(w))
                w[pos] = ov
        ball.append(s)

    def is_covered(cw):
        """True if every word is within radius of some codeword in cw (list of ints)."""
        cov = np.zeros(total, dtype=bool)
        for c in cw:
            cov[list(ball[c])] = True
        return bool(cov.all())

    # ---- seed: direct sum of ternary tetracode with all tails ----
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed = [word_to_int(h + list(tail))
            for h in tetracode
            for tail in itertools.product(range(q), repeat=n - 4)]
    seed = sorted(set(seed))
    best = list(seed)

    def try_add(x):
        """Add x if it reduces the size while keeping coverage."""
        nonlocal best
        cand = list(best) + [x]
        if is_covered(cand):
            best = cand
            return True
        return False

    # ---- greedy add (covers everything, usually larger than seed) ----
    uncovered = set(range(total))
    for c in seed:
        for nb in ball[c]:
            uncovered.discard(nb)
    cw = list(seed)
    while uncovered and time.time() < deadline:
        best_c, best_n = -1, -1
        for c in range(total):
            cnt = 0
            for nb in ball[c]:
                if nb in uncovered:
                    cnt += 1
            if cnt > best_n:
                best_n, best_c = cnt, c
        if best_c < 0:
            break
        cw.append(best_c)
        for nb in ball[best_c]:
            uncovered.discard(nb)

    # ---- pruning: drop any codeword that is redundant ----
    def prune(cw):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(cw)):
                if time.time() > deadline:
                    break
                others = cw[:i] + cw[i + 1:]
                if is_covered(others):
                    cw = others
                    changed = True
                    break
        return cw

    cw = prune(cw)
    if len(cw) < len(best):
        best = cw

    # ---- genetic local search: keep coverage, minimise count ----
    rng = random.Random(12345)

    def evaluate(cw):
        """Fitness: (size, uncovered_count). Smaller is better."""
        cov = np.zeros(total, dtype=bool)
        for c in cw:
            cov[list(ball[c])] = True
        unc = int((~cov).sum())
        return (len(cw), unc)

    def crossover(a, b):
        """Preserve local structure: keep the union of the two parents' spheres,
        then greedily drop redundant members (a structural-preserving reduction)."""
        union = list(dict.fromkeys(list(a) + list(b)))
        return prune(union)

    def mutate(cw):
        """Try a single swap (replace one codeword with a new one) or a single add;
        keep only if it does not break coverage and improves size."""
        if not cw:
            return cw
        r = rng.random()
        if r < 0.5:
            # swap
            i = rng.randrange(len(cw))
            x = rng.randrange(total)
            cand = cw[:i] + [x] + cw[i + 1:]
            return cand if is_covered(cand) else cw
        else:
            # add
            x = rng.randrange(total)
            cand = cw + [x]
            return cand if is_covered(cand) else cw

    population = [list(best)]
    for _ in range(1):
        population.append(prune(list(best) + [rng.randrange(total)]))
    population = [p for p in population if is_covered(p)]
    if not population:
        population = [list(best)]

    gen = 0
    while time.time() < deadline - 2.0:
        gen += 1
        population.sort(key=evaluate)
        # update best
        if len(population[0]) < len(best):
            best = list(population[0])

        # build next generation
        next_pop = []
        # elitism
        next_pop.append(list(population[0]))
        if len(population) > 1:
            next_pop.append(list(population[1]))

        # crossover + mutation
        while len(next_pop) < 8 and time.time() < deadline - 2.0:
            a = rng.choice(population[:max(2, len(population) // 2)])
            b = rng.choice(population[:max(2, len(population) // 2)])
            child = crossover(a, b)
            child = mutate(child)
            if is_covered(child):
                next_pop.append(child)
        # a couple of pure mutations
        while len(next_pop) < 10 and time.time() < deadline - 2.0:
            parent = rng.choice(population)
            child = mutate(parent)
            if is_covered(child):
                next_pop.append(child)

        population = next_pop

    # final prune of the best
    best = prune(best)
    if not is_covered(best):
        # fallback to the valid seed
        best = list(seed)

    return [int_to_word(c) for c in best]
# EVOLVE-BLOCK-END
