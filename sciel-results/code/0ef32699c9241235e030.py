import time
import random
import sys
import math
from functools import lru_cache

def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]

    deadline = time.time() + 140.0
    m = k + 1  # entries per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    def time_left():
        return deadline - time.time()

    def compute_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        if len(rows) != n_rows:
            return False
        seen = set()
        for row in rows:
            if len(row) != m or row[0] != 0:
                return False
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    def greedy_construct(used, rng, max_val=None, n=n_rows, kk=k):
        """Build n rows greedily avoiding used diffs."""
        rows = []
        for _ in range(n):
            row = [0]
            for pos in range(kk):
                found = False
                if max_val is not None:
                    for _ in range(80):
                        x = rng.randint(row[-1] + 1, max_val)
                        ok = True
                        for a in row:
                            if (x - a) in used:
                                ok = False
                                break
                        if ok:
                            for a in row:
                                used.add(x - a)
                            row.append(x)
                            found = True
                            break
                else:
                    x = row[-1]
                    while True:
                        x += 1
                        ok = True
                        for a in row:
                            if (x - a) in used:
                                ok = False
                                break
                        if ok:
                            for a in row:
                                used.add(x - a)
                            row.append(x)
                            found = True
                            break
                if not found:
                    return None
            rows.append(row)
        return rows

    # Generate many candidate difference sets (Sidon sets of size m)
    # Each is a sorted list starting at 0 with distinct pairwise differences
    def gen_sidon(rng, max_val):
        """Generate a random Sidon set of size m starting at 0 within [0, max_val]."""
        row = [0]
        for _ in range(k):
            tries = 0
            while tries < 200:
                x = rng.randint(row[-1] + 1, max_val)
                ok = True
                for a in row:
                    if (x - a) in set(row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))):
                        ok = False
                        break
                if ok:
                    row.append(x)
                    break
                tries += 1
            else:
                return None
        return row

    # Strategy: build a pool of candidate rows, then select disjoint-diff subsets
    def build_pool(rng, pool_size, max_val):
        pool = []
        seen_diffs_pool = set()
        attempts = 0
        while len(pool) < pool_size and attempts < pool_size * 30 and time_left() > 20:
            row = gen_sidon(rng, max_val)
            if row is None:
                attempts += 1
                continue
            d = compute_diffs(row)
            # check internal distinctness (gen_sidon ensures this)
            pool.append((row, d))
            attempts += 1
        return pool

    def select_from_pool(pool, rng, n_sel):
        """Select n_sel rows from pool with disjoint difference sets, minimizing max entry."""
        # Sort pool by max entry ascending
        pool_sorted = sorted(pool, key=lambda x: max(x[0]))
        best = None
        best_scope = float('inf')
        # Try greedy selection with randomization
        for trial in range(30):
            if time_left() < 5:
                break
            used = set()
            selected = []
            # Shuffle slightly
            shuffled = pool_sorted[:]
            if trial > 0:
                # bias towards smaller
                for i in range(len(shuffled) - 1, 0, -1):
                    j = rng.randint(0, i)
                    shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
            for row, d in shuffled:
                if len(selected) == n_sel:
                    break
                if not (d & used):
                    used |= d
                    selected.append(row)
            if len(selected) == n_sel:
                scope = max(max(r) for r in selected)
                if scope < best_scope:
                    best_scope = scope
                    best = [list(r) for r in selected]
        return best, best_scope

    # Local improvement: try to reduce max entry by replacing a row
    def improve(rows, rng, max_iter=2000):
        rows = [list(r) for r in rows]
        used = set()
        for row in rows:
            used |= compute_diffs(row)
        best_rows = [list(r) for r in rows]
        best_scope = max(max(r) for r in rows)
        temperature = 5.0
        cooling = 0.998
        current_scope = best_scope
        for _ in range(max_iter):
            if time_left() < 10:
                break
            idx = rng.randint(0, len(rows) - 1)
            # Try to regenerate row idx with a tighter bound
            other_used = set()
            for i, row in enumerate(rows):
                if i != idx:
                    other_used |= compute_diffs(row)
            # Try smaller max
            cur_max = max(rows[idx])
            new_max = max(1, cur_max - rng.randint(0, 5))
            row = None
            for _ in range(30):
                candidate = gen_sidon(rng, new_max)
                if candidate is not None:
                    d = compute_diffs(candidate)
                    if not (d & other_used):
                        row = candidate
                        break
            if row is not None:
                new_scope = max(max(r) for i, r in enumerate(rows) if i != idx)
                new_scope = max(new_scope, max(row))
                if new_scope < current_scope:
                    rows[idx] = row
                    current_scope = new_scope
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [list(r) for r in rows]
                else:
                    if rng.random() < math.exp(-1.0 / max(temperature, 0.1)):
                        rows[idx] = row
                        current_scope = new_scope
            temperature *= cooling
        return best_rows

    rng = random.Random(777)

    # Initial greedy solution
    used = set()
    best_rows = greedy_construct(used, rng)
    if best_rows is None:
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < m:
                x += 1
                ok = True
                for a in row:
                    if (x - a) in used:
                        ok = False
                        break
                if ok:
                    for a in row:
                        used.add(x - a)
                    row.append(x)
            best_rows.append(row)

    best_scope = max(max(r) for r in best_rows)

    # Build pool and select
    if time_left() > 30:
        pool = build_pool(rng, pool_size=400, max_val=best_scope)
        if len(pool) >= n_rows:
            sel, sel_scope = select_from_pool(pool, rng, n_rows)
            if sel is not None and sel_scope < best_scope:
                best_rows = sel
                best_scope = sel_scope

    # Local improvement
    if time_left() > 20:
        improved = improve(best_rows, rng, max_iter=3000)
        if is_valid(improved):
            imp_scope = max(max(r) for r in improved)
            if imp_scope < best_scope:
                best_rows = improved
                best_scope = imp_scope

    # More restarts
    for trial in range(8):
        if time_left() < 15:
            break
        used = set()
        rows = greedy_construct(used, rng, max_val=best_scope)
        if rows is not None and is_valid(rows):
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                best_rows = rows
                best_scope = scope
                if time_left() > 15:
                    improved = improve(best_rows, rng, max_iter=1500)
                    if is_valid(improved):
                        imp_scope = max(max(r) for r in improved)
                        if imp_scope < best_scope:
                            best_rows = improved
                            best_scope = imp_scope

    return best_rows

if __name__ == "__main__":
    sol = construct(4, 7)
    used = set()
    valid = True
    for row in sol:
        if len(row) != 8 or row[0] != 0:
            valid = False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    valid = False
                used.add(d)
    print("n_rows=4 k=7 valid:", valid, "scope:", max(max(r) for r in sol))
