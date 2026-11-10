import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin vs 120s kill

    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = int(math.ceil((math.sqrt(1 + 8 * total_diffs) - 1) / 2))

    # Max possible scope for bytearray sizing
    max_scope_est = lower_bound + 200

    def validate(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                ri = row[i]
                for j in range(i + 1, len(row)):
                    d = row[j] - ri
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

    # ---- Fast interleaved construction with bytearray for used diffs ----
    def build_interleaved(seed):
        random.seed(seed)
        rows = [[0] for _ in range(n_rows)]
        # Use a bytearray for fast membership testing
        # Size it generously
        ba_size = max_scope_est + 50
        used_ba = bytearray(ba_size)
        used_set = set()  # fallback for large values
        total_steps = n_rows * k
        steps = 0

        def is_used(d):
            if d < ba_size:
                return used_ba[d] == 1
            return d in used_set

        def mark_used(d):
            if d < ba_size:
                used_ba[d] = 1
            else:
                used_set.add(d)

        while steps < total_steps:
            if time.time() > deadline - 30:
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
                # Wider candidate window
                while len(valid_xs) < 15 and checks < 300:
                    if is_used(x):
                        x += 1
                        checks += 1
                        continue
                    conflict = False
                    for a in cur[1:]:
                        if is_used(x - a):
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

            # Pick the most constrained with slight randomness
            if random.random() < 0.75:
                chosen_idx = 0
            else:
                chosen_idx = random.randint(0, min(5, len(candidates) - 1))
            _, r, valid_xs = candidates[chosen_idx]

            # Choose a value; mostly the tightest, occasionally explore
            if random.random() < 0.70:
                chosen_x = valid_xs[0]
            else:
                chosen_x = valid_xs[random.randint(0, min(7, len(valid_xs) - 1))]

            rows[r].append(chosen_x)
            for a in rows[r][:-1]:
                mark_used(chosen_x - a)
            steps += 1

        if not validate(rows):
            return None
        return rows

    # ---- Local search with multiple move types ----
    def local_search(rows, time_budget):
        rows = [r[:] for r in rows]
        diffs = compute_diffs(rows)
        t = scope_of(rows)
        best_rows = [r[:] for r in rows]
        best_t = t
        end = time.time() + time_budget
        no_improve_count = 0
        max_no_improve = 400

        while time.time() < end:
            if no_improve_count > max_no_improve:
                break
            r = random.randint(0, n_rows - 1)
            row = rows[r]
            if len(row) < 2:
                no_improve_count += 1
                continue

            # Choose move type
            move_type = random.random()
            improved = False

            if move_type < 0.5:
                # Single element move
                idx = random.randint(1, len(row) - 1)
                deltas = list(range(-30, 31))
                random.shuffle(deltas)
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
                    new_t = max(t, new_val)
                    if new_t < t or (new_t == t and random.random() < 0.1):
                        rows[r][idx] = new_val
                        diffs = (diffs - old_diffs) | new_diffs
                        t = new_t
                        if t < best_t:
                            best_t = t
                            best_rows = [rr[:] for rr in rows]
                        improved = True
                        no_improve_count = 0
                        break

            elif move_type < 0.8:
                # Swap two adjacent non-zero elements
                if len(row) < 3:
                    no_improve_count += 1
                    continue
                i = random.randint(1, len(row) - 2)
                j = i + 1
                # Try swapping row[i] and row[j]
                old_i, old_j = row[i], row[j]
                # After swap, we need row[i] < row[j] still
                # So we need old_j < old_i which means we can't just swap
                # Instead, try to move element at i to position j's value and vice versa
                # This only works if they can be rearranged
                # Actually, let's try: set row[i]=new_i, row[j]=new_j where new_i < new_j
                # and new_i != old_i or new_j != old_j
                # Simpler: try moving row[i] by delta and row[j] by delta2
                d1 = random.randint(-15, 15)
                d2 = random.randint(-15, 15)
                if d1 == 0 and d2 == 0:
                    no_improve_count += 1
                    continue
                new_i = old_i + d1
                new_j = old_j + d2
                if new_i <= row[i-1] or new_j <= new_i or (j < len(row)-1 and new_j >= row[j+1]):
                    no_improve_count += 1
                    continue
                # Compute old and new diffs
                old_diffs = set()
                new_diffs = set()
                # Diffs involving position i
                for jj in range(i):
                    old_diffs.add(old_i - row[jj])
                    new_diffs.add(new_i - row[jj])
                for jj in range(i+1, len(row)):
                    if jj == j:
                        old_diffs.add(old_j - old_i)
                        new_diffs.add(new_j - new_i)
                    else:
                        old_diffs.add(row[jj] - old_i)
                        new_diffs.add(row[jj] - new_i)
                # Diffs involving position j (excluding i which we handled)
                for jj in range(j+1, len(row)):
                    old_diffs.add(row[jj] - old_j)
                    new_diffs.add(row[jj] - new_j)
                # Diffs between i and j handled above
                # Check conflicts
                if new_diffs & (diffs - old_diffs):
                    no_improve_count += 1
                    continue
                new_t = max(t, new_i, new_j)
                if new_t < t or (new_t == t and random.random() < 0.05):
                    rows[r][i] = new_i
                    rows[r][j] = new_j
                    diffs = (diffs - old_diffs) | new_diffs
                    t = new_t
                    if t < best_t:
                        best_t = t
                        best_rows = [rr[:] for rr in rows]
                    improved = True
                    no_improve_count = 0

            else:
                # Try to shift a suffix of the row left
                if len(row) < 3:
                    no_improve_count += 1
                    continue
                start_idx = random.randint(1, len(row) - 2)
                shift = random.randint(-10, 0)  # shift left only
                if shift == 0:
                    no_improve_count += 1
                    continue
                # Check if all elements from start_idx to end can be shifted by 'shift'
                old_suffix = row[start_idx:]
                new_suffix = [v + shift for v in old_suffix]
                # Check ordering
                if new_suffix[0] <= row[start_idx - 1]:
                    no_improve_count += 1
                    continue
                for idx2 in range(1, len(new_suffix)):
                    if new_suffix[idx2] <= new_suffix[idx2-1]:
                        no_improve_count += 1
                        break
                else:
                    old_diffs = set()
                    new_diffs = set()
                    for ii in range(start_idx):
                        for jj in range(start_idx, len(row)):
                            old_diffs.add(row[jj] - row[ii])
                            new_diffs.add(new_suffix[jj - start_idx] - row[ii])
                    for ii in range(start_idx, len(row)):
                        for jj in range(ii+1, len(row)):
                            old_diffs.add(row[jj] - row[ii])
                            new_diffs.add(new_suffix[jj - start_idx] - new_suffix[ii - start_idx])
                    if new_diffs & (diffs - old_diffs):
                        no_improve_count += 1
                        continue
                    new_t = max(t, new_suffix[-1])
                    if new_t < t:
                        for idx2 in range(start_idx, len(row)):
                            rows[r][idx2] = new_suffix[idx2 - start_idx]
                        diffs = (diffs - old_diffs) | new_diffs
                        t = new_t
                        if t < best_t:
                            best_t = t
                            best_rows = [rr[:] for rr in rows]
                        improved = True
                        no_improve_count = 0

            if not improved:
                no_improve_count += 1

        return best_rows, best_t

    # ---- High-precision polish pass ----
    def polish(rows, time_budget):
        rows = [r[:] for r in rows]
        diffs = compute_diffs(rows)
        t = scope_of(rows)
        best_rows = [r[:] for r in rows]
        best_t = t
        end = time.time() + time_budget
        no_improve_count = 0
        max_no_improve = 250

        while time.time() < end:
            if no_improve_count > max_no_improve:
                break
            r = random.randint(0, n_rows - 1)
            row = rows[r]
            if len(row) < 2:
                no_improve_count += 1
                continue
            idx = random.randint(1, len(row) - 1)
            deltas = list(range(-6, 7))
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
                new_t = max(t, new_val)
                if new_t < t:
                    new_diffs_set = (diffs - old_diffs) | new_diffs
                    rows[r][idx] = new_val
                    diffs = new_diffs_set
                    t = new_t
                    if t < best_t:
                        best_t = t
                        best_rows = [rr[:] for rr in rows]
                    improved = True
                    no_improve_count = 0
                    break
            if not improved:
                no_improve_count += 1
        return best_rows, best_t

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

    # Phase 1: many interleaved MRV constructions
    seed = 1
    while time.time() < deadline - 40:
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
    if time.time() < deadline - 20:
        ls_rows, ls_scope = local_search(best_rows, max(0.0, deadline - 20 - time.time()))
        if ls_scope < best_scope:
            best_rows = ls_rows
            best_scope = ls_scope

    # Phase 3: high-precision polish pass
    if time.time() < deadline - 2:
        pl_rows, pl_scope = polish(best_rows, max(0.0, deadline - 2 - time.time()))
        if pl_scope < best_scope:
            best_rows = pl_rows
            best_scope = pl_scope

    return best_rows
