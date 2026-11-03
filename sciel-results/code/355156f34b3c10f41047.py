import time
import random
import math
from collections import Counter

def _scope(rows):
    if not rows:
        return 0
    return max(a for r in rows for a in r[1:])

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

def _diffs_for_row(row):
    diffs = []
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.append(row[j] - row[i])
    return diffs

def _build_diff_counter(rows):
    c = Counter()
    for r in rows:
        for d in _diffs_for_row(r):
            c[d] += 1
    return c

def _collision_count_from_counter(counter):
    return sum(max(0, v - 1) for v in counter.values())

def _fast_greedy(n_rows, k, rng, scope_limit=None):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if scope_limit is not None and x > scope_limit:
                return None
            conflict = False
            for a in row:
                if (x - a) in used:
                    conflict = True
                    break
            if not conflict:
                row.append(x)
                for a in row[:-1]:
                    used.add(x - a)
        rows.append(row)
    return rows

def _decode(diffs, k):
    """
    Reconstruct a strictly increasing row of length k+1 starting at 0 from a
    sorted multiset of pairwise differences.  A valid multiset must contain
    exactly one copy of the largest gap (the row span).  We greedily assign the
    next element to the smallest unused gap whose complement (span - gap) is
    also present in the multiset; this is the only locally consistent choice.
    Returns the row or None if the multiset is not realizable.
    """
    n = k + 1
    if len(diffs) != k * (k + 1) // 2:
        return None
    d = sorted(diffs)
    if not d or d[0] <= 0:
        return None
    span = d[-1]
    if d.count(span) != 1:
        return None
    avail = Counter(d)
    row = [0]
    remaining = n - 1
    while remaining:
        placed = None
        for g in d:
            if avail[g] <= 0:
                continue
            if span - g in avail and (span - g) != g or (span - g == g and avail[g] >= 1):
                # need both g and span-g to exist
                if span - g in avail:
                    placed = g
                    break
        if placed is None:
            return None
        avail[placed] -= 1
        avail[span - placed] -= 1
        row.append(placed)
        remaining -= 1
    if any(v < 0 for v in avail.values()):
        return None
    # verify strictly increasing (guaranteed by construction but be safe)
    for i in range(1, n):
        if row[i] <= row[i - 1]:
            return None
    # verify the multiset of diffs matches
    got = Counter(_diffs_for_row(row))
    if got != Counter(d):
        return None
    return row

def _sa_optimize(rows, k, deadline, rng, max_iters=None):
    n_rows = len(rows)
    if max_iters is None:
        max_iters = 2000000
    current = [list(r) for r in rows]
    current_scope = _scope(current)
    diff_counter = _build_diff_counter(current)
    collisions = _collision_count_from_counter(diff_counter)
    best = [list(r) for r in current]
    best_scope = current_scope
    best_collisions = collisions
    collision_weight = 100000
    def cost(scope, coll):
        return coll * collision_weight + scope
    T = 50.0
    cooling = 0.9999
    min_T = 0.01
    it = 0
    while it < max_iters:
        if time.time() > deadline:
            break
        it += 1
        i = rng.randint(0, n_rows - 1)
        j = rng.randint(1, k)
        old_val = current[i][j]
        prev_val = current[i][j-1]
        next_val = current[i][j+1] if j < k else old_val + 10000
        move_type = rng.random()
        if move_type < 0.5:
            upper = old_val - 1 if old_val > prev_val + 1 else prev_val
            if upper <= prev_val:
                continue
            new_val = rng.randint(prev_val + 1, upper)
        elif move_type < 0.8:
            delta = rng.randint(-8, 8)
            new_val = old_val + delta
        else:
            new_val = rng.randint(prev_val + 1, max(prev_val + 1, old_val + 30))
        if new_val <= prev_val or (j < k and new_val >= next_val):
            continue
        if new_val == old_val:
            continue
        old_row = current[i]
        old_diffs = _diffs_for_row(old_row)
        current[i][j] = new_val
        new_row = current[i]
        new_diffs = _diffs_for_row(new_row)
        if len(new_diffs) != len(set(new_diffs)):
            current[i][j] = old_val
            continue
        for d in old_diffs:
            diff_counter[d] -= 1
            if diff_counter[d] == 0:
                del diff_counter[d]
        for d in new_diffs:
            diff_counter[d] = diff_counter.get(d, 0) + 1
        new_collisions = _collision_count_from_counter(diff_counter)
        new_scope = current_scope
        if new_val > current_scope:
            new_scope = new_val
        elif old_val == current_scope and new_val < old_val:
            new_scope = _scope(current)
        old_c = cost(current_scope, collisions)
        new_c = cost(new_scope, new_collisions)
        if new_c <= old_c:
            accept = True
        else:
            delta_c = new_c - old_c
            prob = math.exp(-delta_c / max(T, min_T))
            accept = rng.random() < prob
        if accept:
            current_scope = new_scope
            collisions = new_collisions
            if new_scope < best_scope or (new_scope == best_scope and new_collisions < best_collisions):
                best_scope = new_scope
                best_collisions = new_collisions
                best = [list(r) for r in current]
        else:
            current[i][j] = old_val
            for d in new_diffs:
                diff_counter[d] -= 1
                if diff_counter[d] == 0:
                    del diff_counter[d]
            for d in old_diffs:
                diff_counter[d] = diff_counter.get(d, 0) + 1
        T = max(T * cooling, min_T)
    return best_scope, best

def _ga_diff_space(rows, k, deadline, rng, pop_size=40, max_gens=400):
    """
    Genetic algorithm in difference space.  Each individual is a list of
    per-row sorted difference multisets.  Crossover/mutation act on the
    difference values; a prefix-sum decoder reconstructs rows and rejects
    inconsistent genomes (heavy penalty).  Fitness = collision penalty + scope.
    """
    n_rows = len(rows)
    n_d = k * (k + 1) // 2
    collision_weight = 100000

    def encode(ind):
        return [sorted(_diffs_for_row(r)) for r in ind]

    def decode_all(enc):
        out = []
        for d in enc:
            row = _decode(d, k)
            if row is None:
                return None
            out.append(row)
        return out

    def fitness(ind):
        # ind is a list of rows
        for r in ind:
            if r[0] != 0 or len(r) != k + 1:
                return float('inf')
            ds = _diffs_for_row(r)
            if len(ds) != len(set(ds)):
                return float('inf')
        c = _build_diff_counter(ind)
        coll = _collision_count_from_counter(c)
        return coll * collision_weight + _scope(ind)

    # initialize population from best with diff-space perturbations
    base_enc = encode(rows)
    population = []
    for _ in range(pop_size):
        enc = [list(d) for d in base_enc]
        for i in range(n_rows):
            for _ in range(rng.randint(0, 4)):
                p = rng.randrange(n_d)
                old = enc[i][p]
                new = max(1, old + rng.randint(-6, 6))
                enc[i][p] = new
            enc[i].sort()
        population.append(enc)
    population[0] = [list(d) for d in base_enc]

    def eval_enc(enc):
        ind = decode_all(enc)
        if ind is None:
            return float('inf')
        return fitness(ind)

    fitnesses = [eval_enc(e) for e in population]
    best_enc = population[fitnesses.index(min(fitnesses))]
    best_fit = min(fitnesses)

    for gen in range(max_gens):
        if time.time() > deadline:
            break
        fitnesses = [eval_enc(e) for e in population]
        bi = fitnesses.index(min(fitnesses))
        if fitnesses[bi] < best_fit:
            best_fit = fitnesses[bi]
            best_enc = [list(d) for d in population[bi]]

        def select():
            a, b = rng.randrange(pop_size), rng.randrange(pop_size)
            return population[a] if fitnesses[a] <= fitnesses[b] else population[b]

        new_population = [population[bi]]  # elitism
        while len(new_population) < pop_size:
            p1 = select()
            p2 = select()
            child = []
            for i in range(n_rows):
                # uniform crossover on difference multisets
                cd = [p1[i][t] if rng.random() < 0.5 else p2[i][t] for t in range(n_d)]
                # mutation: small perturbations to individual differences
                for _ in range(rng.randint(0, 3)):
                    t = rng.randrange(n_d)
                    cd[t] = max(1, cd[t] + rng.randint(-4, 4))
                cd.sort()
                child.append(cd)
            new_population.append(child)
        population = new_population

    best_ind = decode_all(best_enc)
    if best_ind is None:
        return best_fit, None
    return fitness(best_ind), best_ind

def _scope_squeeze(rows, k, deadline, rng):
    rows = [list(r) for r in rows]
    current_scope = _scope(rows)
    for _ in range(100):
        if time.time() > deadline:
            break
        max_rows = [i for i, r in enumerate(rows) if r[-1] == current_scope]
        if not max_rows:
            break
        i = rng.choice(max_rows)
        row = rows[i]
        old_max = row[-1]
        prev = row[k-1]
        found = False
        for new_val in range(old_max - 1, prev, -1):
            row[k] = new_val
            new_diffs = _diffs_for_row(row)
            if len(new_diffs) != len(set(new_diffs)):
                continue
            conflict = False
            for other_idx, other_row in enumerate(rows):
                if other_idx == i:
                    continue
                other_diffs_set = set(_diffs_for_row(other_row))
                for d in new_diffs:
                    if d in other_diffs_set:
                        conflict = True
                        break
                if conflict:
                    break
            if not conflict:
                current_scope = new_val
                found = True
                break
        if not found:
            row[k] = old_max
    return rows

def construct(n_rows, k):
    deadline = time.time() + 145.0
    rng = random.Random(12345)
    best_result = None
    best_score = float('inf')

    # Phase 1: greedy seed
    rows = _fast_greedy(n_rows, k, rng)
    if rows is None:
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                conflict = False
                for a in row:
                    if (x - a) in used:
                        conflict = True
                        break
                if not conflict:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
            rows.append(row)
    best_result = rows
    best_score = _scope(rows)

    # Phase 2: Simulated Annealing (value space)
    if time.time() < deadline - 40:
        sa_deadline = time.time() + (deadline - time.time()) * 0.35
        s, rows = _sa_optimize(best_result, k, sa_deadline, rng)
        if _valid(rows, k) and s < best_score:
            best_score = s
            best_result = rows

    # Phase 3: Difference-space Genetic Algorithm (new structural search)
    if time.time() < deadline - 20:
        ga_deadline = time.time() + (deadline - time.time()) * 0.55
        f, rows = _ga_diff_space(best_result, k, ga_deadline, rng, pop_size=40, max_gens=400)
        if _valid(rows, k):
            s = _scope(rows)
            if s < best_score:
                best_score = s
                best_result = rows

    # Phase 4: Scope squeeze
    if time.time() < deadline - 5:
        squeeze_deadline = time.time() + (deadline - time.time()) * 0.8
        squeezed = _scope_squeeze(best_result, k, squeeze_deadline, rng)
        if _valid(squeezed, k):
            s = _scope(squeezed)
            if s < best_score:
                best_score = s
                best_result = squeezed

    # Final validation / fallback
    if best_result is None or not _valid(best_result, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                conflict = False
                for a in row:
                    if (x - a) in used:
                        conflict = True
                        break
                if not conflict:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
            rows.append(row)
        best_result = rows
    return best_result
