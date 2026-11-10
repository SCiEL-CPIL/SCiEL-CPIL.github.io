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

    # ---- Interleaved construction with adaptive candidate selection ----
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
                while len(valid_xs) < 8 and checks < 300:
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

            # Adaptive randomness: more randomization early, less later
            progress = steps / total_steps
            if random.random() < 0.85 - 0.3 * progress:
                chosen_idx = 0
            else:
                chosen_idx = random.randint(0, min(3, len(candidates) - 1))
            _, r, valid_xs = candidates[chosen_idx]

            if random.random() < 0.8 - 0.3 * progress:
                chosen_x = valid_xs[0]
            else:
                chosen_x = valid_xs[random.randint(0, min(5, len(valid_xs) - 1))]

            rows[r].append(chosen_x)
            for a in rows[r][:-1]:
                used.add(chosen_x - a)
            if chosen_x > current_max:
                current_max = chosen_x
            steps += 1

        if not validate(rows):
            return None
        return rows

    # ---- Enhanced local search with multi-point moves and simulated annealing ----
    def local_search(rows, time_budget):
        rows = [r[:] for r in rows]
        diffs = compute_diffs(rows)
        scope = scope_of(rows)
        best_rows = [r[:] for r in rows]
        best_scope = scope
        end = time.time() + time_budget
        temp = max(1.0, (scope - lower_bound) / 4.0)
        min_temp = 0.1

        while time.time() < end:
            r = random.randint(0, n_rows - 1)
            row = rows[r]
            if len(row) < 2:
                continue
            
            # Try single-point move first
            idx = random.randint(1, len(row) - 1)
            improved = False
            
            # Generate candidate deltas with bias toward smaller values
            deltas = []
            base_range = 30
            for d in range(-base_range, base_range + 1):
                if d != 0:
                    deltas.append(d)
            random.shuffle(deltas)
            
            for delta in deltas:
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
                
                # Acceptance criterion: scope improvement, or equal scope with probability
                if new_scope < scope:
                    rows[r][idx] = new_val
                    diffs = new_diffs_set
                    scope = new_scope
                    if scope < best_scope:
                        best_scope = scope
                        best_rows = [rr[:] for rr in rows]
                    improved = True
                    break
                elif new_scope == scope and random.random() < min(0.3, temp / max(1.0, scope - lower_bound + 1)):
                    rows[r][idx] = new_val
                    diffs = new_diffs_set
                    improved = True
                    break
            
            if not improved:
                temp = max(min_temp, temp * 0.998)
            
            # Occasionally try a two-point move for better exploration
            if random.random() < 0.15 and len(row) >= 3:
                idx1 = random.randint(1, len(row) - 2)
                idx2 = random.randint(idx1 + 1, len(row) - 1)
                for d1 in range(-15, 16):
                    if d1 == 0:
                        continue
                    v1 = row[idx1] + d1
                    if v1 <= row[idx1 - 1]:
                        continue
                    if v1 >= row[idx1 + 1]:
                        continue
                    for d2 in range(-15, 16):
                        if d2 == 0:
                            continue
                        v2 = row[idx2] + d2
                        if v2 <= row[idx2 - 1]:
                            continue
                        if v2 >= row[idx2 + 1]:
                            continue
                        if v1 >= v2:
                            continue
                        
                        old_diffs = set()
                        new_diffs = set()
                        # Compute all diffs involving idx1 or idx2
                        for j in range(len(row)):
                            if j == idx1 or j == idx2:
                                continue
                            if j < idx1:
                                old_diffs.add(row[idx1] - row[j])
                                new_diffs.add(v1 - row[j])
                            elif j > idx1:
                                old_diffs.add(row[j] - row[idx1])
                                new_diffs.add(row[j] - v1)
                            if j < idx2:
                                old_diffs.add(row[idx2] - row[j])
                                new_diffs.add(v2 - row[j])
                            elif j > idx2:
                                old_diffs.add(row[j] - row[idx2])
                                new_diffs.add(row[j] - v2)
                        # Also handle the diff between idx1 and idx2
                        old_diffs.add(row[idx2] - row[idx1])
                        new_diffs.add(v2 - v1)
                        
                        if new_diffs & (diffs - old_diffs):
                            continue
                        
                        new_scope = max(scope, v1, v2)
                        new_diffs_set = (diffs - old_diffs) | new_diffs
                        
                        if new_scope < scope or (new_scope == scope and random.random() < 0.2):
                            row[idx1] = v1
                            row[idx2] = v2
                            diffs = new_diffs_set
                            scope = new_scope
                            if scope < best_scope:
                                best_scope = scope
                                best_rows = [rr[:] for rr in rows]
                            break
                    else:
                        continue
                    break

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

    # Phase 1: many interleaved constructions with varied seeds
    seed = 1
    max_seeds = 500
    while time.time() < deadline - 25 and seed <= max_seeds:
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
