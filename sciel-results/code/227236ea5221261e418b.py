import time
import random
import math
import numpy as np

_DEADLINE = 0.0
_BEST = None


def _valid(rows, k):
    seen = set()
    for r in rows:
        if r[0] != 0 or len(r) != k + 1:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0 or d in seen:
                    return False
                seen.add(d)
    return True


def _scope(rows):
    return max(a for r in rows for a in r[1:])


def _gaps_to_row(gaps):
    row = [0]
    x = 0
    for g in gaps:
        x += g
        row.append(x)
    return row


def _row_to_gaps(row):
    return [row[i + 1] - row[i] for i in range(len(row) - 1)]


def _row_diffs_bitset(row):
    """Return a Python int bitset of all positive differences in the row."""
    b = 0
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            d = row[j] - row[i]
            if d > 0:
                b |= (1 << d)
    return b


def _row_max(row):
    return row[-1]


def _pop_cost(pop_rows, W=20.0):
    """
    pop_rows: list of rows (each a list of ints).
    Returns (scope, collisions, total_cost).
    Uses bitset OR to count collisions efficiently.
    """
    combined = 0
    scope = 0
    for r in pop_rows:
        b = _row_diffs_bitset(r)
        # collisions contributed by this row = bits already set in combined
        coll = (combined & b).bit_count()
        combined |= b
        m = r[-1]
        if m > scope:
            scope = m
    # total collisions = sum over rows of (bits in this row that were already in prior rows)
    # but we need total = sum over all rows of (popcount(row_bits & (OR of all other rows))) / 2
    # Actually simpler: total_collisions = sum over each bit position of (count - 1) where count >= 1
    # Let's recompute properly:
    all_bits = 0
    counts = {}
    for r in pop_rows:
        b = _row_diffs_bitset(r)
        # iterate set bits
        x = b
        while x:
            lb = x & (-x)
            d = lb.bit_length() - 1
            counts[d] = counts.get(d, 0) + 1
            x ^= lb
    collisions = sum(max(0, c - 1) for c in counts.values())
    total_cost = scope + W * collisions
    return scope, collisions, total_cost


def _fast_greedy(k, n_rows, rng, spread=False):
    used = 0
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = 0
            for a in row:
                new |= (1 << (x - a))
            if (used & new) == 0:
                row.append(x)
                used |= new
                if spread:
                    x += max(1, (k - len(row) + 1) * 2 - 2)
        rows.append(row)
    return rows


def _ga_optimize(initial_rows, k, n_rows, deadline, rng, pop_size=40, W=20.0):
    """
    Genetic Algorithm operating on gap representations.
    Each individual = list of n_rows rows (each row = list of k+1 ints starting at 0).
    Fitness = -(scope + W * collisions). We maximize fitness.
    """
    n_gaps = k  # each row has k gaps

    # Encode initial population
    pop = [list(r) for r in initial_rows]
    # Add diverse individuals
    while len(pop) < pop_size:
        # Random perturbation of a random existing individual
        base = rng.choice(pop)
        ind = [list(r) for r in base]
        # Perturb a few gaps
        for _ in range(rng.randint(1, 4)):
            ri = rng.randint(0, n_rows - 1)
            gi = rng.randint(0, n_gaps - 1)
            row = ind[ri]
            old_gap = row[gi + 1] - row[gi]
            new_gap = max(1, old_gap + rng.randint(-3, 3))
            # Shift all subsequent elements
            shift = new_gap - old_gap
            for j in range(gi + 1, k + 1):
                row[j] += shift
            # Ensure strictly increasing
            if any(row[j + 1] <= row[j] for j in range(k)):
                # Fix: rebuild from gi+1
                row = list(base[ri][:gi + 1])
                x = row[-1]
                for j in range(gi + 1, k + 1):
                    x += max(1, rng.randint(1, 5))
                    row.append(x)
                ind[ri] = row
        pop.append(ind)

    # Evaluate initial population
    def evaluate(ind):
        scope, coll, cost = _pop_cost(ind, W)
        return -cost, scope, coll

    pop_fit = [evaluate(ind) for ind in pop]

    best_fit = max(pop_fit, key=lambda x: x[0])
    best_ind = pop[pop_fit.index(best_fit)]
    best_scope = best_fit[1]

    gen = 0
    stagnant = 0
    MAX_STAGNANT = 15

    while time.time() < deadline:
        gen += 1
        # Sort by fitness
        order = sorted(range(len(pop)), key=lambda i: pop_fit[i][0], reverse=True)
        pop = [pop[i] for i in order]
        pop_fit = [pop_fit[i] for i in order]

        # Update best
        if pop_fit[0][0] > best_fit[0]:
            best_fit = pop_fit[0]
            best_ind = [list(r) for r in pop[0]]
            best_scope = best_fit[1]
            stagnant = 0
        else:
            stagnant += 1

        # Selection: tournament
        def tournament():
            t = rng.sample(range(len(pop)), min(3, len(pop)))
            best_i = max(t, key=lambda i: pop_fit[i][0])
            return pop[best_i]

        # Crossover on gap representation
        def crossover(parent1, parent2):
            # Uniform crossover on gaps per row
            child = []
            for ri in range(n_rows):
                gaps1 = _row_to_gaps(parent1[ri])
                gaps2 = _row_to_gaps(parent2[ri])
                child_gaps = []
                for gi in range(n_gaps):
                    if rng.random() < 0.5:
                        child_gaps.append(gaps1[gi])
                    else:
                        child_gaps.append(gaps2[gi])
                # Ensure positive gaps
                child_gaps = [max(1, g) for g in child_gaps]
                child.append(_gaps_to_row(child_gaps))
            return child

        # Mutation
        def mutate(ind, rate=0.15):
            child = [list(r) for r in ind]
            for ri in range(n_rows):
                if rng.random() < rate:
                    # Mutate one gap in this row
                    gi = rng.randint(0, n_gaps - 1)
                    row = child[ri]
                    old_gap = row[gi + 1] - row[gi]
                    # Random new gap
                    new_gap = max(1, old_gap + rng.randint(-4, 4))
                    shift = new_gap - old_gap
                    for j in range(gi + 1, k + 1):
                        row[j] += shift
                    # Fix strict increase if violated
                    for j in range(k):
                        if row[j + 1] <= row[j]:
                            row[j + 1] = row[j] + 1
                    child[ri] = row
            return child

        # Elitism + new generation
        elite_size = max(2, pop_size // 10)
        new_pop = [list(r) for r in pop[:elite_size]]
        new_fit = pop_fit[:elite_size]

        while len(new_pop) < pop_size:
            p1 = tournament()
            p2 = tournament()
            child = crossover(p1, p2)
            child = mutate(child)
            # Validate: ensure each row starts at 0 and is strictly increasing
            valid = True
            for r in child:
                if r[0] != 0 or any(r[j + 1] <= r[j] for j in range(k)):
                    valid = False
                    break
            if not valid:
                # Rebuild invalid rows
                for ri in range(n_rows):
                    r = child[ri]
                    if r[0] != 0 or any(r[j + 1] <= r[j] for j in range(k)):
                        # Greedy rebuild
                        used_local = 0
                        new_r = [0]
                        x = 0
                        while len(new_r) < k + 1:
                            x += 1
                            nb = 0
                            for a in new_r:
                                nb |= (1 << (x - a))
                            if (used_local & nb) == 0:
                                new_r.append(x)
                                used_local |= nb
                        child[ri] = new_r
            fit = evaluate(child)
            new_pop.append(child)
            new_fit.append(fit)

        pop = new_pop
        pop_fit = new_fit

        # Stagnation: inject random individuals
        if stagnant > MAX_STAGNANT:
            for _ in range(pop_size // 5):
                # Random greedy
                used = 0
                rows = []
                for _ in range(n_rows):
                    row = [0]
                    x = 0
                    while len(row) < k + 1:
                        x += 1
                        nb = 0
                        for a in row:
                            nb |= (1 << (x - a))
                        if (used & nb) == 0:
                            row.append(x)
                            used |= nb
                    rows.append(row)
                # Replace worst individual
                worst_i = len(pop) - 1
                pop[worst_i] = rows
                pop_fit[worst_i] = evaluate(rows)
            stagnant = 0

    return best_scope, best_ind


def construct(n_rows, k):
    global _BEST, _DEADLINE
    _BEST = None
    _DEADLINE = time.time() + 140.0  # safety margin before 150s kill

    def consider(rows):
        global _BEST
        if not _valid(rows, k):
            return
        s = _scope(rows)
        if _BEST is None or s < _BEST[0]:
            _BEST = (s, [list(r) for r in rows])

    rng = random.Random(42)

    # Generate diverse initial candidates
    candidates = []
    candidates.append(_fast_greedy(k, n_rows, rng, spread=False))
    candidates.append(_fast_greedy(k, n_rows, rng, spread=True))
    for _ in range(4):
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                found = False
                max_step = rng.randint(1, 15)
                for t in range(x + 1, x + 1 + max_step * 3):
                    new = 0
                    for a in row:
                        new |= (1 << (t - a))
                    if (used & new) == 0:
                        row.append(t)
                        used |= new
                        x = t
                        found = True
                        break
                if not found:
                    t = x + 1
                    new = 0
                    for a in row:
                        new |= (1 << (t - a))
                    while used & new:
                        t += 1
                        new = 0
                        for a in row:
                            new |= (1 << (t - a))
                    row.append(t)
                    used |= new
                    x = t
            rows.append(row)
        candidates.append(rows)

    candidates.sort(key=_scope)

    # Run GA on multiple initial populations
    time_per_run = 30.0
    for idx, cand in enumerate(candidates[:4]):
        if time.time() > _DEADLINE:
            break
        remaining = _DEADLINE - time.time()
        if remaining < 5:
            break
        ga_deadline = time.time() + min(time_per_run, remaining - 2)
        s, rows = _ga_optimize(cand, k, n_rows, ga_deadline, rng, pop_size=50)
        consider(rows)

    # If we have a valid best, try one more GA run with larger population on best
    if _BEST is not None and time.time() < _DEADLINE - 10:
        remaining = _DEADLINE - time.time()
        if remaining > 10:
            s, rows = _ga_optimize(_BEST[1], k, n_rows, _DEADLINE - 2, rng, pop_size=80)
            consider(rows)

    if _BEST is not None:
        return _BEST[1]
    return candidates[0]
