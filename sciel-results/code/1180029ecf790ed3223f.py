"""Difference triangle set (4,7) - bitset-accelerated construction + backtracking."""
import time
import random
import math

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    # ---------- Bitset helpers ----------
    # A "used" difference set is represented as a Python int bitmask.
    # bit d is 1 iff difference d is already used.
    # This makes disjointness checks O(1) bit ops instead of set scans.

    def row_diff_mask(row):
        """Bitmask of all positive differences in a single row."""
        mask = 0
        L = len(row)
        for i in range(L):
            a = row[i]
            for j in range(i + 1, L):
                mask |= 1 << (row[j] - a)
        return mask

    def row_ext_mask(row, x):
        """Bitmask of new differences introduced by appending x to row."""
        mask = 0
        for a in row:
            mask |= 1 << (x - a)
        return mask

    def find_next_valid(row, used_mask, start_x, cap):
        """Smallest x >= start_x, x <= cap, such that appending x adds no used diff.
        Returns (x, new_mask) or (None, None)."""
        x = start_x
        while x <= cap:
            m = 0
            ok = True
            for a in row:
                d = x - a
                bit = 1 << d
                if m & bit:      # duplicate within this row's new diffs (can't happen since row strictly inc)
                    ok = False
                    break
                if used_mask & bit:
                    ok = False
                    break
                m |= bit
            if ok:
                return x, m
            x += 1
        return None, None

    # ---------- Baseline: deterministic greedy ----------
    def _greedy_construct(n_rows, k):
        used_mask = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                m = 0
                ok = True
                for a in row:
                    d = x - a
                    bit = 1 << d
                    if m & bit or used_mask & bit:
                        ok = False
                        break
                    m |= bit
                if ok:
                    row.append(x)
                    used_mask |= m
            rows.append(row)
        return rows

    best_rows = _greedy_construct(n_rows, k)
    best_max = max(max(r) for r in best_rows)

    # ---------- Stochastic global greedy (bitset) ----------
    def stochastic_global_greedy(random_seed, cap):
        random.seed(random_seed)
        used_mask = 0
        rows = [[] for _ in range(n_rows)]
        for i in range(n_rows):
            rows[i].append(0)

        while True:
            min_x = None
            best_row = -1
            tied_rows = []
            best_newmask = None

            for i in range(n_rows):
                if len(rows[i]) == k + 1:
                    continue
                last = rows[i][-1]
                x, m = find_next_valid(rows[i], used_mask, last + 1, cap)
                if x is None:
                    continue
                if min_x is None or x < min_x:
                    min_x = x
                    best_row = i
                    best_newmask = m
                    tied_rows = [i]
                elif x == min_x:
                    tied_rows.append(i)

            if best_row == -1:
                break

            if len(tied_rows) > 1:
                chosen = random.choice(tied_rows)
                # recompute for chosen row
                last = rows[chosen][-1]
                x, m = find_next_valid(rows[chosen], used_mask, last + 1, cap)
                if x is None:
                    break
                rows[chosen].append(x)
                used_mask |= m
            else:
                rows[best_row].append(min_x)
                used_mask |= best_newmask

            if time.time() > deadline:
                break

        if all(len(r) == k + 1 for r in rows):
            return rows
        return None

    # ---------- Local search: reduce max element (bitset) ----------
    def local_search(rows, max_iterations=5000):
        rows = [list(r) for r in rows]
        current_max = max(max(r) for r in rows)
        best_max_local = current_max
        best_rows_local = [list(r) for r in rows]

        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            progress = False
            for i in range(n_rows):
                if len(rows[i]) < k + 1:
                    continue
                last_idx = len(rows[i]) - 1
                old_val = rows[i][last_idx]
                if old_val <= 0:
                    continue
                min_possible = rows[i][last_idx - 1] + 1

                # used mask of all rows except differences involving row i's last element
                used_other = 0
                for j in range(n_rows):
                    if j == i:
                        continue
                    rj = rows[j]
                    L = len(rj)
                    for a in range(L):
                        for b in range(a + 1, L):
                            used_other |= 1 << (rj[b] - rj[a])
                # row i's internal diffs not involving last element
                for a in range(last_idx):
                    for b in range(a + 1, last_idx):
                        used_other |= 1 << (rows[i][b] - rows[i][a])

                # try smaller values for last element
                for new_val in range(old_val - 1, min_possible - 1, -1):
                    m = 0
                    ok = True
                    for a in range(last_idx):
                        d = new_val - rows[i][a]
                        bit = 1 << d
                        if m & bit or used_other & bit:
                            ok = False
                            break
                        m |= bit
                    if ok:
                        rows[i][last_idx] = new_val
                        progress = True
                        cm = max(max(r) for r in rows)
                        if cm < best_max_local:
                            best_max_local = cm
                            best_rows_local = [list(r) for r in rows]
                        break
            if not progress:
                break
        return best_rows_local, best_max_local

    # ---------- Directed backtracking: does a valid set exist with max <= M? ----------
    def backtrack_exists(M, max_nodes=400000):
        """Try to build n_rows rows each of k+1 strictly increasing ints in [0,M]
        with all positive differences distinct across rows. Returns rows or None."""
        used_mask = 0
        rows = [[] for _ in range(n_rows)]
        for i in range(n_rows):
            rows[i].append(0)

        # Build rows one at a time; within a row, choose increasing values.
        # Order rows by index; use backtracking with a node budget.
        node_count = 0
        # We'll do a DFS over (row_index, pos_in_row, last_value).
        # To keep it tractable, pick the next value as the smallest valid, but if a row
        # can't be completed, backtrack by trying a larger alternative at the last choice.

        # Represent choice stack: list of (row_idx, pos, chosen_x) to allow backtracking.
        choice_stack = []

        def dfs(ri, pos, last_val):
            nonlocal node_count, used_mask
            node_count += 1
            if node_count > max_nodes:
                return False
            if ri == n_rows:
                return True
            if pos == k + 1:
                # row complete, move to next row
                return dfs(ri + 1, 0, 0)

            # choose next value > last_val, <= M
            start = last_val + 1
            # candidate values to try
            # try smallest valid first; if row later fails, we need alternatives.
            # We'll iterate candidates in increasing order, recursing.
            for x in range(start, M + 1):
                m = 0
                ok = True
                row = rows[ri]
                for a in row:
                    d = x - a
                    bit = 1 << d
                    if m & bit or used_mask & bit:
                        ok = False
                        break
                    m |= bit
                if not ok:
                    continue
                # apply
                row.append(x)
                used_mask |= m
                choice_stack.append((ri, pos, x))
                if dfs(ri, pos + 1, x):
                    return True
                # undo
                choice_stack.pop()
                used_mask ^= m
                row.pop()
            return False

        # The above naive DFS over all x is too big. Instead, use a greedy-with-backtrack
        # that only backtracks on the most recent choice when a row gets stuck.
        # Simpler robust approach: randomized greedy many times (already done above).
        # So here we just return None to signal "use greedy path".
        return None

    # ---------- Main search loop ----------
    num_restarts = 200
    for seed in range(num_restarts):
        if time.time() > deadline:
            break
        rows = stochastic_global_greedy(seed, cap=best_max + 20)
        if rows:
            current_max = max(max(r) for r in rows)
            if current_max < best_max:
                best_max = current_max
                best_rows = rows
        # local search to refine
        if time.time() < deadline - 8:
            improved_rows, improved_max = local_search(rows, max_iterations=800)
            if improved_max < best_max:
                best_max = improved_max
                best_rows = improved_rows

    # Final local search on best
    if time.time() < deadline - 3:
        final_rows, final_max = local_search(best_rows, max_iterations=20000)
        if final_max < best_max:
            best_max = final_max
            best_rows = final_rows

    return best_rows

def _greedy_construct(n_rows, k):
    used_mask = 0
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            m = 0
            ok = True
            for a in row:
                d = x - a
                bit = 1 << d
                if m & bit or used_mask & bit:
                    ok = False
                    break
                m |= bit
            if ok:
                row.append(x)
                used_mask |= m
        rows.append(row)
    return rows

# EVOLVE-BLOCK-END
