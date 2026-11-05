"""Rewritten program: row-level simulated annealing with conflict-aware repair."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    deadline = time.time() + 138.0
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * num_diffs_per_row

    def row_diffs_set(row):
        s = set()
        for i in range(len(row)):
            ri = row[i]
            for j in range(i + 1, len(row)):
                s.add(row[j] - ri)
        return s

    def all_diffs_list(rows):
        out = []
        for row in rows:
            for i in range(len(row)):
                ri = row[i]
                for j in range(i + 1, len(row)):
                    out.append(row[j] - ri)
        return out

    def count_violations(diffs):
        if not diffs:
            return 0
        seen = {}
        v = 0
        for d in diffs:
            if d in seen:
                v += 1
            else:
                seen[d] = 1
        return v

    def max_entry(rows):
        return max(max(row) for row in rows)

    def is_valid(rows):
        d = all_diffs_list(rows)
        return len(d) == len(set(d))

    def compute_cost(rows):
        d = all_diffs_list(rows)
        v = count_violations(d)
        m = max_entry(rows)
        return v * 100000 + m

    # Greedy init: build rows one by one, picking smallest x that avoids used diffs
    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    best_rows = greedy_init()
    best_cost = compute_cost(best_rows)
    best_max = max_entry(best_rows)

    # Build a global used-diff multiset for fast conflict checks
    def build_used(rows):
        used = {}
        for row in rows:
            for i in range(len(row)):
                ri = row[i]
                for j in range(i + 1, len(row)):
                    d = row[j] - ri
                    used[d] = used.get(d, 0) + 1
        return used

    # Try to construct a row avoiding a given used set (greedy, with randomness)
    def build_row_avoiding(used, start_randomness=0):
        row = [0]
        x = 0
        local_new = set()
        attempts = 0
        max_attempts = 5000
        while len(row) < k + 1 and attempts < max_attempts:
            x += 1
            if start_randomness and random.random() < start_randomness:
                x += random.randint(0, 3)
            new = {x - a for a in row}
            conflict = False
            for d in new:
                if used.get(d, 0) > 0:
                    conflict = True
                    break
            if not conflict:
                row.append(x)
                local_new |= new
            attempts += 1
        if len(row) < k + 1:
            return None
        return row, local_new

    # Conflict-driven repair: for each row, try to rebuild it avoiding others' diffs
    def repair_row(rows, r, used, rng):
        # Remove row r's diffs from used
        row = rows[r]
        for i in range(len(row)):
            ri = row[i]
            for j in range(i + 1, len(row)):
                d = row[j] - ri
                used[d] -= 1
                if used[d] == 0:
                    del used[d]
        # Try to build a new row avoiding the remaining used
        new_row, new_local = build_row_avoiding(used, start_randomness=0.15)
        if new_row is None:
            # Restore original
            for i in range(len(row)):
                ri = row[i]
                for j in range(i + 1, len(row)):
                    d = row[j] - ri
                    used[d] = used.get(d, 0) + 1
            return False, 0
        # Check that new_row's diffs don't conflict with used
        conflict = False
        for d in new_local:
            if used.get(d, 0) > 0:
                conflict = True
                break
        if conflict:
            for i in range(len(row)):
                ri = row[i]
                for j in range(i + 1, len(row)):
                    d = row[j] - ri
                    used[d] = used.get(d, 0) + 1
            return False, 0
        # Commit
        rows[r] = new_row
        for d in new_local:
            used[d] = used.get(d, 0) + 1
        return True, len(new_local)

    # Simulated annealing with row-level replacement
    current_rows = [r[:] for r in best_rows]
    current_used = build_used(current_rows)
    current_cost = best_cost

    T = 50.0
    T_min = 0.01
    cooling = 0.99997
    step = 0

    while time.time() < deadline - 5:
        step += 1
        if T < T_min:
            T = T_min

        # Choose a random row to replace
        r = random.randrange(n_rows)

        # Remove row r from used
        row = current_rows[r]
        for i in range(len(row)):
            ri = row[i]
            for j in range(i + 1, len(row)):
                d = row[j] - ri
                current_used[d] -= 1
                if current_used[d] == 0:
                    del current_used[d]

        # Build a new candidate row avoiding current_used
        cand, cand_local = build_row_avoiding(current_used, start_randomness=0.2)
        if cand is None:
            # Restore
            for i in range(len(row)):
                ri = row[i]
                for j in range(i + 1, len(row)):
                    d = row[j] - ri
                    current_used[d] = current_used.get(d, 0) + 1
            T *= cooling
            continue

        # Check conflicts
        conflict = False
        for d in cand_local:
            if current_used.get(d, 0) > 0:
                conflict = True
                break
        if conflict:
            for i in range(len(row)):
                ri = row[i]
                for j in range(i + 1, len(row)):
                    d = row[j] - ri
                    current_used[d] = current_used.get(d, 0) + 1
            T *= cooling
            continue

        # Compute cost change
        old_m = max_entry(current_rows)
        current_rows[r] = cand
        new_m = max_entry(current_rows)
        # violations change: we removed old row's diffs (all were unique among themselves? not necessarily)
        # Simpler: recompute cost
        new_cost = compute_cost(current_rows)
        delta = new_cost - current_cost

        accept = False
        if delta <= 0:
            accept = True
        else:
            if T > T_min:
                if random.random() < math.exp(-delta / T):
                    accept = True

        if accept:
            for d in cand_local:
                current_used[d] = current_used.get(d, 0) + 1
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_rows = [rr[:] for rr in current_rows]
                best_max = new_m
        else:
            # Restore old row
            current_rows[r] = row
            for i in range(len(row)):
                ri = row[i]
                for j in range(i + 1, len(row)):
                    d = row[j] - ri
                    current_used[d] = current_used.get(d, 0) + 1

        T *= cooling

        # Periodic: try to shrink the largest row
        if step % 200 == 0 and time.time() < deadline - 10:
            # Find row with max entry
            max_r = 0
            max_val = -1
            for i, rr in enumerate(current_rows):
                if rr[-1] > max_val:
                    max_val = rr[-1]
                    max_r = i
            # Try to rebuild that row more tightly
            row = current_rows[max_r]
            for i in range(len(row)):
                ri = row[i]
                for j in range(i + 1, len(row)):
                    d = row[j] - ri
                    current_used[d] -= 1
                    if current_used[d] == 0:
                        del current_used[d]
            # Greedy tight build
            cand2, cand2_local = build_row_avoiding(current_used, start_randomness=0.0)
            if cand2 is not None and cand2[-1] < max_val:
                conflict2 = False
                for d in cand2_local:
                    if current_used.get(d, 0) > 0:
                        conflict2 = True
                        break
                if not conflict2:
                    current_rows[max_r] = cand2
                    for d in cand2_local:
                        current_used[d] = current_used.get(d, 0) + 1
                    new_cost2 = compute_cost(current_rows)
                    if new_cost2 < best_cost:
                        best_cost = new_cost2
                        best_rows = [rr[:] for rr in current_rows]
                        best_max = max_entry(best_rows)
                    current_cost = new_cost2
                else:
                    # restore
                    for i in range(len(row)):
                        ri = row[i]
                        for j in range(i + 1, len(row)):
                            d = row[j] - ri
                            current_used[d] = current_used.get(d, 0) + 1
            else:
                for i in range(len(row)):
                    ri = row[i]
                    for j in range(i + 1, len(row)):
                        d = row[j] - ri
                        current_used[d] = current_used.get(d, 0) + 1

    # Final: try a few targeted repairs to reduce max entry
    if time.time() < deadline - 3:
        for attempt in range(30):
            if time.time() > deadline - 3:
                break
            rows = [r[:] for r in best_rows]
            used = build_used(rows)
            # Find row with max entry
            max_r = 0
            max_val = -1
            for i, rr in enumerate(rows):
                if rr[-1] > max_val:
                    max_val = rr[-1]
                    max_r = i
            # Remove it
            row = rows[max_r]
            for i in range(len(row)):
                ri = row[i]
                for j in range(i + 1, len(row)):
                    d = row[j] - ri
                    used[d] -= 1
                    if used[d] == 0:
                        del used[d]
            # Try several random rebuilds
            for trial in range(20):
                cand, cand_local = build_row_avoiding(used, start_randomness=0.3)
                if cand is None:
                    continue
                conflict = False
                for d in cand_local:
                    if used.get(d, 0) > 0:
                        conflict = True
                        break
                if conflict:
                    continue
                if cand[-1] < max_val:
                    rows[max_r] = cand
                    for d in cand_local:
                        used[d] = used.get(d, 0) + 1
                    if is_valid(rows):
                        if max_entry(rows) < best_max:
                            best_rows = [rr[:] for rr in rows]
                            best_max = max_entry(best_rows)
                            best_cost = compute_cost(best_rows)
                            break
                    # restore for next trial
                    for d in cand_local:
                        used[d] -= 1
                        if used[d] == 0:
                            del used[d]
                    rows[max_r] = row
                    for i in range(len(row)):
                        ri = row[i]
                        for j in range(i + 1, len(row)):
                            d = row[j] - ri
                            used[d] = used.get(d, 0) + 1

    return best_rows
# EVOLVE-BLOCK-END
