import time
import random
import math
from collections import Counter

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = t0 + 118.0
    row_len = k + 1

    # --- Bitmask helpers ---
    def row_mask(row):
        m = 0
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                m |= (1 << (row[j] - ri))
        return m

    def pairwise_diffs(row):
        L = len(row)
        diffs = []
        for i in range(L):
            ri = row[i]
            for j in range(i + 1, L):
                diffs.append(row[j] - ri)
        return diffs

    def scope_of(rows):
        return max(r[-1] for r in rows) if rows else 0

    # --- Robust Greedy Row Construction ---
    def build_greedy_row(used_mask):
        row = [0]
        x = 0
        # We need to ensure we can always find a next element.
        # If we hit a dead end, we just pick the next available integer globally, 
        # expanding the scope if necessary.
        while len(row) < row_len:
            found = False
            # Search up to a reasonably large bound
            # If we don't find one, we extend the bound
            upper = 200 
            while not found:
                for t in range(x + 1, upper + 1):
                    new_mask = 0
                    for a in row:
                        new_mask |= (1 << (t - a))
                    if (new_mask & used_mask) == 0:
                        row.append(t)
                        used_mask |= new_mask
                        x = t
                        found = True
                        break
                if not found:
                    upper += 50
        return row, used_mask

    # --- Simulated Annealing ---
    def sa_optimize(rows, iters, rng, deadline):
        n = len(rows)
        cur_rows = [list(r) for r in rows]
        cur_scope = scope_of(cur_rows)
        best_rows = [list(r) for r in cur_rows]
        best_scope = cur_scope

        diff_counts = Counter()
        for r in cur_rows:
            for d in pairwise_diffs(r):
                diff_counts[d] += 1
        
        row_masks = [row_mask(r) for r in cur_rows]

        def compute_used_without(i):
            u = 0
            for ri in range(n):
                if ri != i:
                    u |= row_masks[ri]
            return u

        def get_cost(rows, dc):
            collisions = sum(max(0, c - 1) for c in dc.values())
            scope = scope_of(rows)
            return scope + 1000 * collisions

        cur_cost = get_cost(cur_rows, diff_counts)
        best_cost = cur_cost

        T = 10.0
        cooling = 0.9995
        min_T = 0.05

        for it in range(iters):
            if time.time() > deadline:
                break

            i = rng.randint(0, n - 1)
            old_row = list(cur_rows[i])
            old_mask = row_masks[i]
            used_without = compute_used_without(i)

            # Randomly perturb the row
            # We regenerate the row using a randomized greedy approach
            # or just pick a random valid row by searching for a new one
            # To keep it simple and fast, we try to modify one element or rebuild
            
            # Strategy: Rebuild row i with a randomized greedy
            new_row = [0]
            x = 0
            local_mask = 0
            success = True
            for j in range(1, row_len):
                found = False
                # Randomize start point slightly
                start = x + 1
                if rng.random() < 0.3:
                    start = x + rng.randint(1, 3)
                
                # Search in a window
                upper = cur_scope + 10
                for t in range(start, upper + 1):
                    new_mask = 0
                    for a in new_row:
                        new_mask |= (1 << (t - a))
                    if (new_mask & used_without) or (new_mask & local_mask):
                        continue
                    new_row.append(t)
                    local_mask |= new_mask
                    x = t
                    found = True
                    break
                if not found:
                    success = False
                    break
            
            if not success:
                continue

            new_mask = row_mask(new_row)
            
            # Update counts
            for d in pairwise_diffs(old_row):
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
            
            for d in pairwise_diffs(new_row):
                diff_counts[d] = diff_counts.get(d, 0) + 1

            cur_rows[i] = new_row
            row_masks[i] = new_mask
            new_cost = get_cost(cur_rows, diff_counts)

            if new_cost < cur_cost or rng.random() < math.exp(-(new_cost - cur_cost) / max(T, min_T)):
                cur_cost = new_cost
                cur_scope = scope_of(cur_rows)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_scope = cur_scope
                    best_rows = [list(r) for r in cur_rows]
            else:
                # Revert
                for d in pairwise_diffs(new_row):
                    diff_counts[d] -= 1
                    if diff_counts[d] == 0:
                        del diff_counts[d]
                
                for d in pairwise_diffs(old_row):
                    diff_counts[d] = diff_counts.get(d, 0) + 1

                cur_rows[i] = old_row
                row_masks[i] = old_mask

            T = max(min_T, T * cooling)

        return best_scope, best_rows

    # --- Main Logic ---
    rng = random.Random(42)
    
    # 1. Build initial valid set
    used_mask = 0
    rows = []
    for _ in range(n_rows):
        if time.time() > DEADLINE - 10:
            break
        row, used_mask = build_greedy_row(used_mask)
        rows.append(row)
    
    # If we ran out of time building, fill rest with greedy (fast)
    while len(rows) < n_rows:
        row, used_mask = build_greedy_row(used_mask)
        rows.append(row)
        
    best_rows = [r[:] for r in rows]
    best_scope = scope_of(rows)

    # 2. Optimize
    if time.time() < DEADLINE - 5:
        sa_scope, sa_rows = sa_optimize(rows, 20000, rng, DEADLINE - 2)
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_rows = [r[:] for r in sa_rows]

    return best_rows
