import random
import time
import math
import sys

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    total_diffs = k * (k + 1) // 2  # per row
    counting_lb = n_rows * total_diffs

    def row_diffs(row):
        s = set()
        L = len(row)
        for i in range(L):
            ri = row[i]
            for j in range(i + 1, L):
                s.add(row[j] - ri)
        return s

    def is_valid(rows):
        all_d = set()
        for row in rows:
            if row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i + 1] <= row[i]:
                    return False
            d = row_diffs(row)
            if d & all_d:
                return False
            all_d |= d
        return True

    def greedy_row(forbidden, max_val_hint):
        row = [0]
        local = set()
        upper = max_val_hint if max_val_hint else 2000
        for _ in range(k):
            base = row[-1]
            found = False
            for v in range(base + 1, upper + 1):
                ok = True
                for a in row:
                    d = v - a
                    if d in forbidden or d in local:
                        ok = False
                        break
                if ok:
                    row.append(v)
                    for a in row[:-1]:
                        local.add(row[-1] - a)
                    found = True
                    break
            if not found:
                return None
        return row

    def initial_construct(max_hint):
        rows = []
        used = set()
        for i in range(n_rows):
            r = greedy_row(used, max_hint)
            if r is None:
                return None
            rows.append(r)
            used |= row_diffs(r)
        return rows

    # Get initial solution
    hint = max(counting_lb + 10, 130)
    init_rows = initial_construct(hint)
    if init_rows is None:
        hint = counting_lb + 60
        init_rows = initial_construct(hint)
    if init_rows is None:
        hint = counting_lb + 200
        init_rows = initial_construct(hint)
    if init_rows is None:
        rows = []
        used = set()
        for i in range(n_rows):
            row = [0]
            local = set()
            for _ in range(k):
                base = row[-1]
                v = base + 1
                while True:
                    ok = True
                    for a in row:
                        d = v - a
                        if d in used or d in local:
                            ok = False
                            break
                    if ok:
                        break
                    v += 1
                row.append(v)
                for a in row[:-1]:
                    local.add(v - a)
            rows.append(row)
            used |= row_diffs(row)
        init_rows = rows

    def score_of(rows):
        return max(max(r) for r in rows)

    best_rows = [r[:] for r in init_rows]
    best_score = score_of(best_rows)

    # --- Generate all valid candidate rows up to max_val ---
    def generate_candidate_rows(max_val):
        """Generate all valid rows (starting at 0, k+1 increasing ints, max <= max_val)."""
        candidates = []
        used_global = set()  # not needed; we just generate all possible rows
        
        def backtrack_row(current, pos):
            if pos == k + 1:
                candidates.append(current[:])
                return
            base = current[-1] if current else 0
            start = base + 1 if base > 0 else 0
            for v in range(start, max_val + 1):
                current.append(v)
                backtrack_row(current, pos + 1)
                current.pop()
        
        backtrack_row([0], 0)
        return candidates

    # --- Conflict graph + maximum independent set via branch and bound ---
    def find_independent_set(candidate_rows, n_needed, max_val):
        """Find n_needed rows with no overlapping differences, using branch and bound."""
        if time.time() > deadline:
            return None
        
        n = len(candidate_rows)
        if n < n_needed:
            return None
        
        # Precompute difference sets
        diff_sets = [row_diffs(r) for r in candidate_rows]
        
        # Build conflict graph: edge i-j if diff_sets[i] & diff_sets[j]
        # Use bitsets for efficiency
        # adj[i] is a bitmask of rows that conflict with row i
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if diff_sets[i] & diff_sets[j]:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Branch and bound for maximum independent set
        # We want to find if there's an independent set of size n_needed
        
        # Use a greedy coloring heuristic for upper bound
        def color_bound(adj, unprocessed):
            """Greedy coloring to get an upper bound on independent set size."""
            # Simple: just return len(unprocessed) as a loose bound
            return len(unprocessed)
        
        best_found = []
        found = [False]
        
        def branch_and_bound(selected, unprocessed_mask, depth):
            if time.time() > deadline:
                return
            
            if len(selected) == n_needed:
                found[0] = True
                best_found.extend(selected)
                return
            
            # Upper bound: can we possibly reach n_needed?
            remaining_needed = n_needed - len(selected)
            if bit_count(unprocessed_mask) < remaining_needed:
                return
            
            # Get the unprocessed nodes
            unprocessed = [i for i in range(n) if (unprocessed_mask >> i) & 1]
            if not unprocessed:
                return
            
            # Choose a pivot node (heuristic: node with fewest conflicts among unprocessed)
            pivot = unprocessed[0]
            min_conflicts = float('inf')
            for node in unprocessed:
                conflicts = 0
                for other in unprocessed:
                    if other != node and ((adj[node] >> other) & 1):
                        conflicts += 1
                if conflicts < min_conflicts:
                    min_conflicts = conflicts
                    pivot = node
            
            # Option 1: Include pivot
            new_selected = selected + [pivot]
            # Remove pivot and all its neighbors from unprocessed
            new_mask = unprocessed_mask & ~(1 << pivot)
            for other in range(n):
                if ((adj[pivot] >> other) & 1):
                    new_mask &= ~(1 << other)
            
            if bit_count(new_mask) >= remaining_needed - 1:
                branch_and_bound(new_selected, new_mask, depth + 1)
                if found[0]:
                    return
            
            # Option 2: Exclude pivot
            new_mask2 = unprocessed_mask & ~(1 << pivot)
            if bit_count(new_mask2) >= remaining_needed:
                branch_and_bound(selected, new_mask2, depth + 1)
                if found[0]:
                    return
        
        def bit_count(x):
            return bin(x).count('1')
        
        # Start with all nodes available
        all_mask = (1 << n) - 1
        branch_and_bound([], all_mask, 0)
        
        if found[0]:
            return [candidate_rows[i] for i in best_found]
        return None

    # --- Iteratively deepen M to find smallest valid solution ---
    # Start from counting_lb and increase
    for M in range(counting_lb, best_score):
        if time.time() > deadline - 30:
            break
        
        # Generate candidate rows up to M
        candidates = generate_candidate_rows(M)
        if len(candidates) < n_rows:
            continue
        
        # Try to find independent set
        result = find_independent_set(candidates, n_rows, M)
        if result is not None and is_valid(result):
            best_rows = result
            best_score = M
            break

    # --- Simulated Annealing for further improvement ---
    def greedy_candidates(forbidden, row, max_scan=8):
        cands = []
        base = row[-1]
        L = len(row)
        local = set()
        for i in range(L):
            for j in range(i + 1, L):
                local.add(row[j] - row[i])
        scan_limit = base + 500
        for v in range(base + 1, scan_limit + 1):
            ok = True
            for a in row:
                d = v - a
                if d in forbidden or d in local:
                    ok = False
                    break
            if ok:
                cands.append(v)
                if len(cands) >= max_scan:
                    break
        return cands

    def build_row_with_skips(forbidden, skips, max_hint):
        row = [0]
        local = set()
        for t in range(k):
            cands = greedy_candidates(forbidden, row, 8)
            if not cands:
                base = row[-1]
                L = len(row)
                local2 = set()
                for i in range(L):
                    for j in range(i + 1, L):
                        local2.add(row[j] - row[i])
                v = base + 1
                while True:
                    ok = True
                    for a in row:
                        d = v - a
                        if d in forbidden or d in local2:
                            ok = False
                            break
                    if ok:
                        break
                    v += 1
                cands = [v]
            idx = skips[t]
            if idx < 0:
                idx = 0
            if idx >= len(cands):
                idx = 0
            v = cands[idx]
            row.append(v)
            for a in row[:-1]:
                local.add(v - a)
        if row[-1] > max_hint:
            return None
        return row

    cur_rows = [r[:] for r in best_rows]
    cur_row_diffs = [row_diffs(r) for r in cur_rows]
    cur_score = score_of(cur_rows)

    T = 15.0
    cooling = 0.9995
    min_T = 0.05
    skip_choices = [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    def sample_skips():
        return [random.choice(skip_choices) for _ in range(k)]

    rebuild_counter = 0
    while time.time() < deadline - 5:
        if T < min_T:
            T = 10.0

        idx = random.randrange(n_rows)

        forbidden = set()
        for i in range(n_rows):
            if i != idx:
                forbidden |= cur_row_diffs[i]

        skips = sample_skips()
        max_hint = best_score
        new_row = build_row_with_skips(forbidden, skips, max_hint)
        if new_row is None:
            max_hint = best_score + 3
            new_row = build_row_with_skips(forbidden, skips, max_hint)
            if new_row is None:
                T *= cooling
                continue

        new_row_d = row_diffs(new_row)
        if new_row_d & forbidden:
            T *= cooling
            continue

        new_rows = cur_rows[:]
        new_rows[idx] = new_row
        ns = score_of(new_rows)

        delta = ns - cur_score
        accept = False
        if delta < 0:
            accept = True
        elif delta == 0:
            if random.random() < 0.05:
                accept = True
        else:
            prob = math.exp(-delta / T)
            if random.random() < prob:
                accept = True

        if accept:
            cur_rows[idx] = new_row
            cur_row_diffs[idx] = new_row_d
            cur_score = ns
            if cur_score < best_score:
                best_score = cur_score
                best_rows = [r[:] for r in cur_rows]
                T = min(T, 5.0)

        T *= cooling

        rebuild_counter += 1
        if rebuild_counter % 500 == 0:
            target = best_score - 1
            if target >= counting_lb:
                candidates = generate_candidate_rows(target)
                if len(candidates) >= n_rows:
                    sol = find_independent_set(candidates, n_rows, target)
                    if sol is not None and is_valid(sol):
                        best_rows = sol
                        best_score = target
                        cur_rows = [r[:] for r in best_rows]
                        cur_row_diffs = [row_diffs(r) for r in cur_rows]
                        cur_score = best_score

    if not is_valid(best_rows):
        rows = []
        used = set()
        for i in range(n_rows):
            row = [0]
            local = set()
            for _ in range(k):
                base = row[-1]
                v = base + 1
                while True:
                    ok = True
                    for a in row:
                        d = v - a
                        if d in used or d in local:
                            ok = False
                            break
                    if ok:
                        break
                    v += 1
                row.append(v)
                for a in row[:-1]:
                    local.add(v - a)
            rows.append(row)
            used |= row_diffs(row)
        return rows

    return best_rows

# EVOLVE-BLOCK-END
