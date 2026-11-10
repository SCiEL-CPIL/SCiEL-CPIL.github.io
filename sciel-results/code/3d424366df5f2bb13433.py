import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin vs 150s kill

    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = int(math.ceil((math.sqrt(1 + 8 * total_diffs) - 1) / 2))

    def validate(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def compute_diffs(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                ri = row[i]
                for j in range(i + 1, len(row)):
                    used.add(row[j] - ri)
        return used

    def scope_of(rows):
        return max(max(r) for r in rows)

    # ---- MRV-style interleaved construction (greedy with randomness) ----
    def build_interleaved(seed):
        random.seed(seed)
        rows = [[0] for _ in range(n_rows)]
        used = set()
        total_steps = n_rows * k
        steps = 0
        current_max = 0

        while steps < total_steps:
            if time.time() > deadline - 5:
                return None
            candidates = []
            for r in range(n_rows):
                if len(rows[r]) == k + 1:
                    continue
                cur = rows[r]
                last = cur[-1]
                valid_xs = []
                x = last + 1
                checks = 0
                while len(valid_xs) < 6 and checks < 200:
                    if x in used:
                        x += 1
                        checks += 1
                        continue
                    conflict = False
                    for a in cur[1:]:
                        if (x - a) in used:
                            conflict = True
                            break
                    if not conflict:
                        valid_xs.append(x)
                    x += 1
                    checks += 1
                if not valid_xs:
                    return None
                candidates.append((valid_xs[0], r, valid_xs))

            if not candidates:
                break
            candidates.sort(key=lambda item: item[0])

            if random.random() < 0.85:
                chosen_idx = 0
            else:
                chosen_idx = random.randint(0, min(3, len(candidates) - 1))
            _, r, valid_xs = candidates[chosen_idx]

            if random.random() < 0.8:
                chosen_x = valid_xs[0]
            else:
                chosen_x = valid_xs[random.randint(0, min(3, len(valid_xs) - 1))]

            rows[r].append(chosen_x)
            for a in rows[r][:-1]:
                used.add(chosen_x - a)
            if chosen_x > current_max:
                current_max = chosen_x
            steps += 1

        if not validate(rows):
            return None
        return rows

    # ---- Local search refinement ----
    def local_search(rows, time_budget):
        rows = [r[:] for r in rows]
        diffs = compute_diffs(rows)
        scope = scope_of(rows)
        best_rows = [r[:] for r in rows]
        best_scope = scope
        end = time.time() + time_budget
        temp = max(1.0, (scope - lower_bound) / 8.0)

        while time.time() < end:
            r = random.randint(0, n_rows - 1)
            row = rows[r]
            if len(row) < 2:
                continue
            idx = random.randint(1, len(row) - 1)
            deltas = list(range(-20, 21))
            random.shuffle(deltas)
            improved = False
            for delta in deltas:
                if delta == 0:
                    continue
                new_val = row[idx] + delta
                if new_val <= row[idx - 1]:
                    continue
                if idx < len(row) - 1 and new_val >= row[idx + 1]:
                    continue
                old_diffs = set()
                new_diffs = set()
                for j in range(idx):
                    old_diffs.add(row[idx] - row[j])
                    new_diffs.add(new_val - row[j])
                for j in range(idx + 1, len(row)):
                    old_diffs.add(row[j] - row[idx])
                    new_diffs.add(row[j] - new_val)
                if new_diffs & (diffs - old_diffs):
                    continue
                new_scope = max(scope, new_val)
                new_diffs_set = (diffs - old_diffs) | new_diffs
                # accept if scope improves or (equal and random with temp)
                if new_scope < scope or (new_scope == scope and random.random() < 0.2):
                    rows[r][idx] = new_val
                    diffs = new_diffs_set
                    scope = new_scope
                    if scope < best_scope:
                        best_scope = scope
                        best_rows = [rr[:] for rr in rows]
                    improved = True
                    break
            if not improved:
                temp *= 0.999

        return best_rows, best_scope

    # ---- Fallback simple greedy (row-by-row) ----
    def greedy_fallback():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                nd = {x - a for a in row}
                if not (nd & used):
                    row.append(x)
                    used |= nd
            rows.append(row)
        return rows

    best_rows = None
    best_scope = float('inf')

    # Phase 1: many interleaved constructions
    seed = 1
    while time.time() < deadline - 20:
        rows = build_interleaved(seed)
        if rows is not None:
            s = scope_of(rows)
            if s < best_scope:
                best_scope = s
                best_rows = [r[:] for r in rows]
                if best_scope <= lower_bound:
                    return best_rows
        seed += 1

    if best_rows is None:
        best_rows = greedy_fallback()
        best_scope = scope_of(best_rows)

    # Phase 2: local search on the best found
    if time.time() < deadline - 2:
        ls_rows, ls_scope = local_search(best_rows, max(0.0, deadline - 2 - time.time()))
        if ls_scope < best_scope:
            best_rows = ls_rows
            best_scope = ls_scope

    return best_rows
