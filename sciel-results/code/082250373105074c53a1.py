import time
import random
import math
from collections import Counter

_BEST = None
_DEADLINE = 0.0

def _scope(rows):
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

def _row_diffs(row):
    out = []
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            out.append(row[j] - row[i])
    return out

def _build_diff_counts(rows):
    counts = Counter()
    for r in rows:
        for d in _row_diffs(r):
            counts[d] += 1
    return counts

def _remove_diffs(counts, row):
    for d in _row_diffs(row):
        counts[d] -= 1
        if counts[d] == 0:
            del counts[d]

def _add_diffs(counts, row):
    for d in _row_diffs(row):
        counts[d] = counts.get(d, 0) + 1

def _gaps_to_row(gaps):
    row = [0]
    x = 0
    for g in gaps:
        x += g
        row.append(x)
    return row

def _row_to_gaps(row):
    return [row[i + 1] - row[i] for i in range(len(row) - 1)]

def _fast_greedy(k, n_rows, rng=None, spread=False):
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

def _repair_row(gaps, k, max_scope):
    """Ensure gaps are positive and the resulting row stays within max_scope."""
    fixed = list(gaps)
    for i in range(len(fixed)):
        if fixed[i] < 1:
            fixed[i] = 1
    # If total exceeds max_scope, scale down proportionally
    total = sum(fixed)
    if total > max_scope:
        scale = max_scope / total
        new_fixed = []
        acc = 0
        for i, g in enumerate(fixed):
            if i == len(fixed) - 1:
                new_fixed.append(max(1, max_scope - acc))
            else:
                ng = max(1, int(g * scale))
                new_fixed.append(ng)
                acc += ng
        fixed = new_fixed
    return fixed

def _ga_fitness(rows, k, diff_counts):
    scope = _scope(rows)
    collisions = sum(max(0, c - 1) for c in diff_counts.values())
    return scope + 200 * collisions

def _genetic_algorithm(init_rows, k, rng, deadline, pop_size=24):
    """
    GA over gap-vectors. Each individual = list of rows (each row = list of k+1 ints).
    Fitness = scope + 200 * collisions.
    Crossover: uniform on gap vectors.
    Mutation: bit-flip on a random gap with small delta, then repair.
    """
    n_rows = len(init_rows)
    # Population: list of (rows, fitness, diff_counts)
    population = []
    for r in init_rows:
        dc = _build_diff_counts(r)
        f = _ga_fitness(r, k, dc)
        population.append((r, f, dc))
    
    # Sort by fitness
    population.sort(key=lambda x: x[1])
    
    best_rows = [list(r) for r in population[0][0]]
    best_fit = population[0][1]
    
    # Elitism: keep top 4
    elite_size = 4
    T = 5.0
    cooling = 0.999
    min_T = 0.1
    
    gen = 0
    while time.time() < deadline:
        gen += 1
        
        # Selection: tournament of 5
        def select():
            idxs = [rng.randint(0, len(population) - 1) for _ in range(5)]
            best_idx = min(idxs, key=lambda i: population[i][1])
            return population[best_idx][0]
        
        # Generate offspring
        offspring = []
        for _ in range(pop_size - elite_size):
            parent1 = select()
            parent2 = select()
            
            # Crossover: uniform on gap vectors
            child_rows = []
            for i in range(n_rows):
                g1 = _row_to_gaps(parent1[i])
                g2 = _row_to_gaps(parent2[i])
                child_gaps = [g1[j] if rng.random() < 0.5 else g2[j] for j in range(k)]
                # Mutation: perturb one gap
                if rng.random() < 0.6:
                    mi = rng.randint(0, k - 1)
                    delta = rng.randint(-3, 3)
                    if delta != 0:
                        child_gaps[mi] += delta
                # Repair
                child_gaps = _repair_row(child_gaps, k, _scope(parent1) + 10)
                child_rows.append(_gaps_to_row(child_gaps))
            
            dc = _build_diff_counts(child_rows)
            f = _ga_fitness(child_rows, k, dc)
            offspring.append((child_rows, f, dc))
        
        # Combine: elite + offspring
        population = population[:elite_size] + offspring
        population.sort(key=lambda x: x[1])
        
        # Track best
        if population[0][1] < best_fit:
            best_fit = population[0][1]
            best_rows = [list(r) for r in population[0][0]]
        
        # Occasionally inject a fresh greedy candidate to maintain diversity
        if gen % 20 == 0:
            fresh = _fast_greedy(k, n_rows, rng, spread=rng.random() < 0.5)
            dc = _build_diff_counts(fresh)
            f = _ga_fitness(fresh, k, dc)
            if f < population[-1][1]:
                population[-1] = (fresh, f, dc)
                population.sort(key=lambda x: x[1])
        
        T = max(T * cooling, min_T)
        
        # If we found a valid solution, try to improve it with SA-like local search
        if _valid(best_rows, k):
            # Small local refinement on the best
            if time.time() < deadline - 5:
                best_rows = _local_refine(best_rows, k, rng, deadline - 5)
                if _valid(best_rows, k):
                    sf = _scope(best_rows)
                    if sf < best_fit:
                        best_fit = sf
    
    return best_rows

def _local_refine(rows, k, rng, deadline):
    """Quick local search to reduce scope of a valid solution."""
    if not _valid(rows, k):
        return rows
    current = [list(r) for r in rows]
    current_scope = _scope(current)
    best = [list(r) for r in current]
    best_scope = current_scope
    n_rows = len(current)
    
    # Build diff set for fast validity check
    diff_set = set()
    for r in current:
        for d in _row_diffs(r):
            diff_set.add(d)
    
    while time.time() < deadline:
        i = rng.randint(0, n_rows - 1)
        j = rng.randint(1, k)
        old_val = current[i][j]
        prev_val = current[i][j - 1]
        next_val = current[i][j + 1] if j < k else current_scope + 5
        lo, hi = prev_val + 1, next_val - 1
        if lo > hi:
            continue
        new_val = rng.randint(lo, hi)
        if new_val == old_val:
            continue
        
        # Check validity of the change
        # Remove old diffs involving position j
        old_diffs = []
        for t in range(k + 1):
            if t == j:
                continue
            if t < j:
                d = old_val - current[i][t]
            else:
                d = current[i][t] - old_val
            if d > 0:
                old_diffs.append(d)
        
        # Check if new diffs would conflict
        new_diffs = []
        conflict = False
        for t in range(k + 1):
            if t == j:
                continue
            if t < j:
                d = new_val - current[i][t]
            else:
                d = current[i][t] - new_val
            if d > 0:
                new_diffs.append(d)
                # Check against all other diffs (excluding old ones from this row)
                if d in diff_set and d not in old_diffs:
                    # Need to check if it's in another row
                    # Simplified: just check global set minus this row's old diffs
                    pass
        
        # Simpler: just try the change and validate
        current[i][j] = new_val
        if _valid(current, k):
            ns = _scope(current)
            if ns < best_scope:
                best_scope = ns
                best = [list(r) for r in current]
            # Update diff_set
            # Rebuild is expensive; skip for speed in this quick refine
        else:
            current[i][j] = old_val
    
    return best

def construct(n_rows, k):
    global _BEST, _DEADLINE
    _BEST = None
    _DEADLINE = time.time() + 145.0  # Safety margin before 150s kill

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
    
    # Randomized greedy
    for _ in range(3):
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                found = False
                max_step = rng.randint(1, 8)
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

    # Run GA on top candidates
    for idx, cand in enumerate(candidates[:3]):
        if time.time() > _DEADLINE:
            break
        remaining = _DEADLINE - time.time()
        if idx == 0:
            ga_deadline = time.time() + remaining * 0.35
        elif idx == 1:
            ga_deadline = time.time() + remaining * 0.35
        else:
            ga_deadline = _DEADLINE
        
        result = _genetic_algorithm(cand, k, rng, ga_deadline, pop_size=20)
        consider(result)

    # If we have a valid best, do a final local refinement
    if _BEST is not None and time.time() < _DEADLINE:
        refined = _local_refine(_BEST[1], k, rng, _DEADLINE)
        consider(refined)

    if _BEST is not None:
        return _BEST[1]
    return candidates[0]
