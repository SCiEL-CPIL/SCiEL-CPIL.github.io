import time
import random
import math
from collections import defaultdict

def construct(n_rows, k):
    row_len = k + 1
    deadline = time.time() + 115.0
    rng = random.Random(42)

    # Counting bound: m >= k*(k+1)/2
    counting_bound = k * (k + 1) // 2

    def get_diffs(row):
        """Get all positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        all_diffs = []
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    all_diffs.append(r[j] - r[i])
        return len(set(all_diffs)) == len(all_diffs)

    def scope_of(rows):
        return max(r[-1] for r in rows)

    def greedy_construct():
        """Greedy: pick smallest possible next value that doesn't conflict."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_len:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    def greedy_construct_randomized(max_val):
        """Randomized greedy with constraint that max entry <= max_val."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            # Try to build row with backtracking
            def dfs(pos, last_val):
                if pos == row_len:
                    return True
                if last_val + (row_len - pos) > max_val:
                    return False
                start = last_val + 1
                end = max_val + 1
                # Shuffle for randomization
                candidates = list(range(start, end))
                rng.shuffle(candidates)
                for v in candidates:
                    new_diffs = {v - a for a in row}
                    if new_diffs & used:
                        continue
                    row.append(v)
                    used.update(new_diffs)
                    if dfs(pos + 1, v):
                        return True
                    used.difference_update(new_diffs)
                    row.pop()
                return False

            if dfs(1, 0):
                rows.append(row)
            else:
                return None
        return rows

    def polish(rows, max_time=3.0):
        """Try to reduce scope by decrementing entries where possible."""
        start_t = time.time()
        rows = [r[:] for r in rows]

        def build_counts(rws):
            counts = defaultdict(int)
            for r in rws:
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        counts[r[j] - r[i]] += 1
            return counts

        counts = build_counts(rows)

        def get_collisions(ct):
            return sum(c - 1 for c in ct.values() if c > 1)

        if get_collisions(counts) > 0:
            return rows

        improved = True
        while improved and (time.time() - start_t) < max_time:
            improved = False
            for r_idx in range(n_rows):
                for i_idx in range(row_len - 1, 0, -1):
                    old_val = rows[r_idx][i_idx]
                    # Try to reduce by 1, 2, ..., up to 3
                    for delta in range(1, 4):
                        new_val = old_val - delta
                        if new_val <= rows[r_idx][i_idx - 1]:
                            break
                        if i_idx < row_len - 1 and new_val >= rows[r_idx][i_idx + 1]:
                            break

                        # Check validity
                        rows[r_idx][i_idx] = new_val
                        new_counts = build_counts(rows)
                        if get_collisions(new_counts) == 0:
                            counts = new_counts
                            improved = True
                            break
                        else:
                            rows[r_idx][i_idx] = old_val
        return rows

    def simulate_annealing(init_rows, time_limit):
        rows = [r[:] for r in init_rows]
        counts = defaultdict(int)
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    counts[r[j] - r[i]] += 1

        def get_collisions():
            return sum(c - 1 for c in counts.values() if c > 1)

        current_collisions = get_collisions()
        current_max = scope_of(rows)
        current_cost = current_collisions * 100000 + current_max

        best_local_cost = current_cost
        best_local_rows = [r[:] for r in rows]
        best_local_max = current_max

        T = 50.0
        cooling = 0.99997
        start_time = time.time()
        steps = 0

        while time.time() - start_time < time_limit:
            r_idx = rng.randint(0, n_rows - 1)
            i_idx = rng.randint(1, row_len - 1)

            lower = rows[r_idx][i_idx - 1] + 1
            upper_limit = best_local_max + 15
            if i_idx < row_len - 1:
                upper = min(rows[r_idx][i_idx + 1] - 1, upper_limit)
            else:
                upper = upper_limit

            if lower > upper:
                continue

            new_val = rng.randint(lower, upper)
            if new_val == rows[r_idx][i_idx]:
                continue

            old_val = rows[r_idx][i_idx]

            # Compute delta
            delta_coll = 0
            for j in range(row_len):
                if j == i_idx:
                    continue
                old_d = abs(old_val - rows[r_idx][j])
                new_d = abs(new_val - rows[r_idx][j])
                if old_d != new_d:
                    oc = counts.get(old_d, 0)
                    if oc > 0:
                        if oc == 1:
                            del counts[old_d]
                        else:
                            counts[old_d] = oc - 1
                            delta_coll -= 1
                    nc = counts.get(new_d, 0)
                    if nc == 0:
                        counts[new_d] = 1
                    else:
                        counts[new_d] = nc + 1
                        delta_coll += 1

            rows[r_idx][i_idx] = new_val
            new_max = scope_of(rows)
            new_collisions = current_collisions + delta_coll
            new_cost = new_collisions * 100000 + new_max

            if new_cost < current_cost:
                current_cost = new_cost
                current_collisions = new_collisions
                current_max = new_max
                if new_cost < best_local_cost:
                    best_local_cost = new_cost
                    best_local_max = new_max
                    best_local_rows = [r[:] for r in rows]
            else:
                delta = new_cost - current_cost
                if delta == 0:
                    if rng.random() < 0.3:
                        current_cost = new_cost
                        current_collisions = new_collisions
                        current_max = new_max
                    else:
                        rows[r_idx][i_idx] = old_val
                        for j in range(row_len):
                            if j == i_idx:
                                continue
                            old_d = abs(old_val - rows[r_idx][j])
                            new_d = abs(new_val - rows[r_idx][j])
                            if old_d != new_d:
                                nc = counts.get(new_d, 0)
                                if nc == 1:
                                    del counts[new_d]
                                else:
                                    counts[new_d] = nc - 1
                                oc = counts.get(old_d, 0)
                                if oc == 0:
                                    counts[old_d] = 1
                                else:
                                    counts[old_d] = oc + 1
                else:
                    if delta < 10000:
                        p = math.exp(-delta / T)
                    else:
                        p = 0.0

                    if rng.random() < p:
                        current_cost = new_cost
                        current_collisions = new_collisions
                        current_max = new_max
                    else:
                        rows[r_idx][i_idx] = old_val
                        for j in range(row_len):
                            if j == i_idx:
                                continue
                            old_d = abs(old_val - rows[r_idx][j])
                            new_d = abs(new_val - rows[r_idx][j])
                            if old_d != new_d:
                                nc = counts.get(new_d, 0)
                                if nc == 1:
                                    del counts[new_d]
                                else:
                                    counts[new_d] = nc - 1
                                oc = counts.get(old_d, 0)
                                if oc == 0:
                                    counts[old_d] = 1
                                else:
                                    counts[old_d] = oc + 1

            T *= cooling
            if T < 0.05:
                T = 0.05
            steps += 1

            if steps % 3000 == 0:
                actual_coll = get_collisions()
                if actual_coll != current_collisions:
                    # Rebuild counts
                    counts = defaultdict(int)
                    for r in rows:
                        for i in range(len(r)):
                            for j in range(i + 1, len(r)):
                                counts[r[j] - r[i]] += 1
                    current_collisions = get_collisions()
                    current_max = scope_of(rows)
                    current_cost = current_collisions * 100000 + current_max

        return best_local_rows, best_local_max, best_local_cost

    # Start with greedy
    best_rows = greedy_construct()
    best_max = scope_of(best_rows)
    best_cost = best_max

    # Try randomized greedy with tighter bounds
    for attempt in range(5):
        trial_max = best_max - attempt
        if trial_max < counting_bound:
            break
        csp_rows = greedy_construct_randomized(trial_max)
        if csp_rows is not None:
            csp_scope = scope_of(csp_rows)
            if csp_scope < best_max:
                best_rows = csp_rows
                best_max = csp_scope
                best_cost = csp_scope

    # Main SA loop with restarts
    while time.time() < deadline - 5.0:
        remaining = deadline - time.time() - 5.0
        if remaining <= 0:
            break

        if is_valid(best_rows):
            init_rows = best_rows
        else:
            # Random restart
            init_rows = []
            for _ in range(n_rows):
                pts = random.sample(range(1, best_max + 1), k)
                pts.sort()
                init_rows.append([0] + pts)

        sa_time = min(remaining, 5.0)
        if sa_time <= 0:
            break

        local_rows, local_max, local_cost = simulate_annealing(init_rows, sa_time)

        if local_cost < 100000:
            if local_max < best_max:
                best_max = local_max
                best_rows = local_rows
                best_cost = local_max

    # Final polish pass
    if time.time() < deadline - 2.0:
        polished = polish(best_rows, max_time=min(3.0, deadline - time.time() - 2.0))
        if is_valid(polished):
            pol_scope = scope_of(polished)
            if pol_scope < best_max:
                best_rows = polished
                best_max = pol_scope

    return best_rows
