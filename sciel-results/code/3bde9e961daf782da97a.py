# EVOLVE-BLOCK-START
import time
import numpy as np


def _initial_code(q, n):
    # Ternary tetracode (9 words of length 4, perfect covering radius 1) direct-summed
    # with every word on the remaining n-4 coordinates -> 9 * q**(n-4) words (73 for 3,6,1).
    if n < 4:
        return [list(range(0, 0))]  # degenerate; should not occur for this task
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = []
    for h in tetracode:
        for tail in np.ndindex(*([q] * (n - 4))):
            code.append(h + list(tail))
    return code


def _neighbors(c, q, n, total, powers):
    # 1 (center) + q*n (radius-1) neighbor indices, deduped.
    idx = [c]
    for j in range(n):
        base = c - c[j] * powers[j]
        for v in range(q):
            if v != c[j]:
                idx.append(base + v * powers[j])
    return np.array(sorted(set(idx)), dtype=np.int64)


def _coverage(codewords, q, n, total, powers):
    if not codewords:
        return np.zeros(total, dtype=np.int32)
    nb = np.array([_neighbors(c, q, n, total, powers) for c in codewords], dtype=np.int64)
    cov = np.zeros(total, dtype=np.int32)
    for row in nb:
        cov[row] += 1
    return cov


def _is_valid(cov):
    return bool((cov >= 1).all())


def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 108.0  # safety margin under the 150s kill

    total = q ** n
    powers = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)

    init = _initial_code(q, n)
    init = sorted(set(tuple(c) for c in init))
    init = [list(c) for c in init]
    cov = _coverage(init, q, n, total, powers)
    best_code = list(init)
    best_cov = cov
    best_fitness = int(np.sum(cov.astype(np.int64) ** 2))

    # Precompute per-position powers for fast index arithmetic.
    pop = [list(c) for c in init]
    pop_set = set(tuple(c) for c in pop)

    rng = np.random.default_rng(0)

    def index_of(c):
        return int(sum(c[j] * powers[j] for j in range(n)))

    def try_remove(c, cov):
        # returns (new_cov, ok)
        ni = index_of(c)
        nb = _neighbors(c, q, n, total, powers)
        new_cov = cov - 1
        if (new_cov[nb] < 1).any():
            return None, False
        return new_cov, True

    def add_if_new(c, pop, pop_set):
        t = tuple(c)
        if t not in pop_set:
            pop.append(list(c))
            pop_set.add(t)
            return True
        return False

    def fitness(cov):
        return int(np.sum(cov.astype(np.int64) ** 2))

    # --- Iteration 1: greedy removal of redundant codewords (min coverage, tie by index) ---
    improved = True
    while improved and (time.time() < deadline - 15.0):
        improved = False
        order = np.argsort(cov, kind="stable")
        for i in order:
            if time.time() > deadline - 15.0:
                break
            ci = int(i)
            if ci >= len(pop):
                continue
            new_cov, ok = try_remove(pop[ci], cov)
            if ok:
                f = fitness(new_cov)
                if f < best_fitness:
                    best_fitness = f
                    best_code = [list(c) for c in pop]
                    best_cov = new_cov
                cov = new_cov
                del pop[ci]
                pop_set.discard(tuple(best_code[ci]))  # stale; rebuild below
                improved = True
        # rebuild pop_set safely
        pop_set = set(tuple(c) for c in pop)

    # Re-sync best if pop is smaller than best_code
    if len(pop) < len(best_code):
        best_code = [list(c) for c in pop]
        best_cov = cov
        best_fitness = fitness(cov)

    # --- Iteration 2: GA over the population, preserving validity ---
    def crossover(a, b):
        child = list(a)
        for j in range(n):
            if rng.random() < 0.5:
                child[j] = b[j]
        return child

    def mutate(c):
        child = list(c)
        for j in range(n):
            if rng.random() < 0.1:
                child[j] = int(rng.integers(q))
        return child

    if time.time() < deadline - 10.0:
        # Build a candidate pool from pop plus some mutated/crossovered variants.
        # We evaluate candidates against the current cov; accept only if validity holds
        # and fitness does not worsen.
        pop_size = len(pop)
        for _ in range(4000):
            if time.time() > deadline - 10.0:
                break
            if pop_size < 2:
                break
            a = pop[int(rng.integers(pop_size))]
            b = pop[int(rng.integers(pop_size))]
            child = crossover(a, b)
            if rng.random() < 0.3:
                child = mutate(child)
            t = tuple(child)
            if t in pop_set:
                continue
            # Tentatively add: check validity with cov + 1 on its neighbors.
            ni = index_of(child)
            nb = _neighbors(child, q, n, total, powers)
            new_cov = cov + 1
            # We only add if it lets us later remove something; here we test a
            # "swap": add child, then try to remove a now-redundant existing codeword.
            # Find a codeword whose removal is now possible and reduces fitness.
            best_swap = None
            # candidate removals: those with coverage == 1 in new_cov
            cand = np.where(new_cov == 1)[0]
            for ci in cand:
                ci = int(ci)
                if ci >= len(pop):
                    continue
                if tuple(pop[ci]) == t:
                    continue
                # remove pop[ci]
                rnb = _neighbors(pop[ci], q, n, total, powers)
                after = new_cov - 1
                if (after[rnb] < 1).any():
                    continue
                f = fitness(after)
                if f < best_fitness:
                    best_swap = (ci, after, f)
                    break
            if best_swap is not None:
                ci, after, f = best_swap
                best_fitness = f
                best_code = [list(c) for c in pop]
                best_code[ci] = child
                best_cov = after
                # commit to pop
                pop[ci] = child
                pop_set.discard(tuple(best_code[ci]))  # stale guard
                pop_set.add(t)
                cov = after

    # Final: ensure best_code is valid and distinct.
    # Rebuild distinctness.
    seen = set()
    dedup = []
    for c in best_code:
        t = tuple(c)
        if t not in seen:
            seen.add(t)
            dedup.append(list(c))
    best_code = dedup

    # If somehow best_code lost validity (shouldn't), fall back to init.
    final_cov = _coverage(best_code, q, n, total, powers)
    if not _is_valid(final_cov):
        best_code = init
        final_cov = cov
        if not _is_valid(final_cov):
            final_cov = _coverage(best_code, q, n, total, powers)

    return best_code
# EVOLVE-BLOCK-END
