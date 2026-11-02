import time
import random
import sys
import math

sys.setrecursionlimit(10000)


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 117  # Safety margin for 150s kill

    # Generate initial solution using fast greedy construction
    best_solution = _greedy_construct_fast(n_rows, k)
    if not best_solution:
        best_solution = _greedy_construct(n_rows, k)
    if not best_solution:
        return []

    best_score = max(max(r) for r in best_solution)

    # Phase 1: Simulated Annealing with both element-level moves and
    # row-regeneration moves (the key structural addition from the inspiration).
    improved = _simulated_annealing(best_solution, n_rows, k, deadline - 35)
    if improved:
        curr_score = max(max(r) for r in improved)
        if curr_score < best_score:
            best_score = curr_score
            best_solution = improved

    # Phase 2: Iterated Local Search with perturbations
    no_improve_count = 0
    while time.time() < deadline - 22:
        candidate = [r[:] for r in best_solution]

        if no_improve_count >= 4:
            # Strong perturbation: rebuild 1-2 rows greedily
            n_rebuild = min(2, n_rows)
            for _ in range(n_rebuild):
                r_idx = random.randint(0, n_rows - 1)
                other_rows = [r for i, r in enumerate(candidate) if i != r_idx]
                used_diffs_other = _compute_all_diffs(other_rows)
                new_row = _greedy_row_given_diffs(used_diffs_other, k)
                if new_row is not None:
                    candidate[r_idx] = new_row
            no_improve_count = 0
        else:
            # Mild perturbation: several random valid moves
            n_moves = random.randint(2, 5)
            for _ in range(n_moves):
                r_idx = random.randint(0, n_rows - 1)
                pos = random.randint(1, k)
                new_val = _find_random_valid_value(candidate, r_idx, pos, n_rows, k)
                if new_val is not None and new_val != candidate[r_idx][pos]:
                    candidate[r_idx][pos] = new_val

        improved = _local_search_minimize(candidate, n_rows, k, deadline - 18)

        if improved:
            curr_score = max(max(r) for r in improved)
            if curr_score < best_score:
                best_score = curr_score
                best_solution = improved
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1

    # Phase 3: Final high-precision polish with remaining time
    if time.time() < deadline - 1:
        polished = _local_search_minimize(best_solution, n_rows, k, deadline - 0.5)
        if polished:
            curr_score = max(max(r) for r in polished)
            if curr_score < best_score:
                best_score = curr_score
                best_solution = polished

    return best_solution


def _greedy_row_given_diffs(used_diffs, k):
    row = [0]
    current_max = 0
    while len(row) < k + 1:
        found = False
        for x in range(current_max + 1, current_max + 3000):
            valid = True
            for a in row:
                d = x - a
                if d in used_diffs:
                    valid = False
                    break
            if valid:
                row.append(x)
                for a in row[:-1]:
                    used_diffs.add(x - a)
                current_max = x
                found = True
                break
        if not found:
            return None
    return row


def _row_diffs(row):
    s = set()
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            s.add(row[j] - ri)
    return s


def _compute_all_diffs(rows):
    all_diffs = set()
    for r in rows:
        n = len(r)
        for i in range(n):
            ri = r[i]
            for j in range(i + 1, n):
                all_diffs.add(r[j] - ri)
    return all_diffs


def _find_random_valid_value(rows, r_idx, pos, n_rows, k):
    row = rows[r_idx]
    old_val = row[pos]
    n = len(row)
    all_diffs = _compute_all_diffs(rows)
    diffs_involving_old = set()
    for i in range(n):
        if i == pos:
            continue
        if i < pos:
            diffs_involving_old.add(old_val - row[i])
        else:
            diffs_involving_old.add(row[i] - old_val)
    forbidden_diffs = all_diffs - diffs_involving_old
    lower_bound = row[pos - 1] + 1
    current_max = max(max(r) for r in rows)
    upper_bound = min(current_max + 30, 250)
    if pos + 1 < n:
        upper_bound = min(upper_bound, row[pos + 1] - 1)
    if upper_bound < lower_bound:
        return None
    valid_candidates = []
    for new_val in range(lower_bound, upper_bound + 1):
        if new_val == old_val:
            continue
        valid = True
        new_diffs = set()
        for i in range(n):
            if i == pos:
                continue
            if i < pos:
                d = new_val - row[i]
            else:
                d = row[i] - new_val
            if d in forbidden_diffs or d in new_diffs:
                valid = False
                break
            new_diffs.add(d)
        if valid:
            valid_candidates.append(new_val)
    if not valid_candidates:
        return None
    if len(valid_candidates) == 1:
        return valid_candidates[0]
    max_candidate = max(valid_candidates)
    weights = []
    total_weight = 0
    for v in valid_candidates:
        w = (max_candidate - v + 1) ** 2
        weights.append(w)
        total_weight += w
    r = random.uniform(0, total_weight)
    cumulative = 0
    for i, w in enumerate(weights):
        cumulative += w
        if r <= cumulative:
            return valid_candidates[i]
    return valid_candidates[-1]


def _build_static_diffs(row, k):
    n = k + 1
    static = [None] * (k + 1)
    for pi in range(1, k + 1):
        s = set()
        for i in range(n):
            if i == pi:
                continue
            for j in range(i + 1, n):
                if j == pi:
                    continue
                s.add(row[j] - row[i])
        static[pi] = s
    return static


def _compute_new_diffs_fast(row, pi, new_val, static_diffs_pi):
    n = len(row)
    new_diffs = set(static_diffs_pi)
    valid = True
    for i in range(n):
        if i == pi:
            continue
        if i < pi:
            d = new_val - row[i]
        else:
            d = row[i] - new_val
        if d in new_diffs:
            valid = False
            break
        new_diffs.add(d)
    return new_diffs, valid


def _random_valid_row(forbidden, max_limit, k, rng):
    """Regenerate a whole row of length k+1 starting at 0 with max <= max_limit,
    using only differences disjoint from `forbidden` and internally distinct.
    Returns the row or None."""
    row = [0]
    used_in_row = set()
    for _ in range(k):
        start = row[-1] + 1
        end = max_limit + 1
        if end <= start:
            return None
        candidates = []
        for x in range(start, end):
            nd = set()
            ok = True
            for a in row:
                d = x - a
                if d in forbidden or d in used_in_row or d in nd:
                    ok = False
                    break
                nd.add(d)
            if ok:
                candidates.append(x)
                if len(candidates) >= 6:
                    break
        if not candidates:
            return None
        # Bias toward smaller values to keep the row compact
        chosen = candidates[0]
        if rng.random() < 0.5:
            chosen = rng.choice(candidates)
        row.append(chosen)
        for a in row[:-1]:
            used_in_row.add(chosen - a)
    return row


def _simulated_annealing(rows, n_rows, k, deadline):
    current_rows = [r[:] for r in rows]
    all_used = _compute_all_diffs(current_rows)
    row_diffs = [_row_diffs(r) for r in current_rows]
    static_diffs = [_build_static_diffs(r, k) for r in current_rows]
    row_maxes = [max(r) for r in current_rows]

    current_val = max(row_maxes)
    best_rows = [r[:] for r in current_rows]
    best_val = current_val

    T = 120.0
    T_min = 0.3
    cool = 0.99998

    no_improve = 0
    max_no_improve = 80000

    # Row-regeneration move probability
    p_row_move = 0.30

    while time.time() < deadline - 2 and no_improve < max_no_improve:
        if random.random() < p_row_move:
            # ---- ROW REGENERATION MOVE (structural addition) ----
            ri = random.randrange(n_rows)
            forbidden = all_used - row_diffs[ri]
            # Target a tighter bound than current to push the max down
            target = current_val
            new_row = _random_valid_row(forbidden, target, k, random)
            if new_row is None:
                # allow a slightly looser bound for exploration
                new_row = _random_valid_row(forbidden, target + 8, k, random)
            if new_row is None:
                no_improve += 1
                T = max(T_min, T * cool)
                continue

            new_row_max = max(new_row)
            other_max = 0
            for i in range(n_rows):
                if i != ri:
                    m = row_maxes[i]
                    if m > other_max:
                        other_max = m
            new_global_max = max(other_max, new_row_max)
            delta = new_global_max - current_val

            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if random.random() < 0.2:
                    accept = True
            else:
                if T > T_min:
                    prob = math.exp(-delta / max(T, 0.1))
                    if random.random() < prob:
                        accept = True

            if accept:
                all_used -= row_diffs[ri]
                new_rd = _row_diffs(new_row)
                all_used |= new_rd
                current_rows[ri] = new_row
                row_diffs[ri] = new_rd
                static_diffs[ri] = _build_static_diffs(new_row, k)
                row_maxes[ri] = new_row_max
                current_val = new_global_max
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                else:
                    no_improve += 1
        else:
            # ---- ELEMENT-LEVEL MOVE (fast static-diff conflict check) ----
            ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)
            old_val = current_rows[ri][pi]
            candidates = []
            min_allowed = current_rows[ri][pi - 1] + 1
            max_decrease = min(40, old_val - min_allowed)
            for d in range(1, max_decrease + 1):
                candidates.append(old_val - d)
            max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 25
            max_increase = min(12, max_allowed - old_val)
            for d in range(1, max_increase + 1):
                candidates.append(old_val + d)
            if random.random() < 0.12:
                low = current_rows[ri][pi - 1] + 1
                high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 20
                if low <= high:
                    candidates.append(random.randint(low, high))
            if not candidates:
                no_improve += 1
                T = max(T_min, T * cool)
                continue
            random.shuffle(candidates)
            static_pi = static_diffs[ri][pi]
            moved = False
            for new_val in candidates:
                if new_val <= current_rows[ri][pi - 1]:
                    continue
                if pi < k and new_val >= current_rows[ri][pi + 1]:
                    continue
                new_diffs, valid_internal = _compute_new_diffs_fast(
                    current_rows[ri], pi, new_val, static_pi)
                if not valid_internal:
                    continue
                old_diffs = row_diffs[ri]
                conflict = False
                for d in new_diffs:
                    if d in all_used and d not in old_diffs:
                        conflict = True
                        break
                if conflict:
                    continue
                if pi == k:
                    cand_row_max = new_val
                else:
                    cand_row_max = current_rows[ri][k]
                old_row_max = row_maxes[ri]
                row_maxes[ri] = cand_row_max
                new_global_max = max(row_maxes)
                row_maxes[ri] = old_row_max
                delta = new_global_max - current_val
                accept = False
                if delta < 0:
                    accept = True
                elif delta == 0:
                    if random.random() < 0.15:
                        accept = True
                else:
                    if T > T_min:
                        prob = math.exp(-delta / max(T, 0.1))
                        if random.random() < prob:
                            accept = True
                if accept:
                    all_used -= old_diffs
                    all_used |= new_diffs
                    current_rows[ri][pi] = new_val
                    row_diffs[ri] = new_diffs
                    row_maxes[ri] = cand_row_max
                    current_val = new_global_max
                    static_diffs[ri] = _build_static_diffs(current_rows[ri], k)
                    if current_val < best_val:
                        best_val = current_val
                        best_rows = [r[:] for r in current_rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                    moved = True
                    break
            if not moved:
                no_improve += 1

        T = max(T_min, T * cool)

        # Periodic strong restart if stuck
        if no_improve > 30000 and random.random() < 0.05:
            for _ in range(3):
                r_idx = random.randint(0, n_rows - 1)
                pos = random.randint(1, k)
                new_val = _find_random_valid_value(current_rows, r_idx, pos, n_rows, k)
                if new_val is not None and new_val != current_rows[r_idx][pos]:
                    current_rows[r_idx][pos] = new_val
            all_used = _compute_all_diffs(current_rows)
            row_diffs = [_row_diffs(r) for r in current_rows]
            static_diffs = [_build_static_diffs(r, k) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            current_val = max(row_maxes)
            no_improve = 0
            T = max(T, 40.0)

    return best_rows


def _local_search_minimize(rows, n_rows, k, deadline):
    best_rows = [r[:] for r in rows]
    used_diffs = _compute_all_diffs(best_rows)

    while time.time() < deadline - 1:
        progress = False
        row_order = list(range(n_rows))
        random.shuffle(row_order)

        # Pass 1: right-to-left lowering
        for r_idx in row_order:
            if time.time() > deadline - 1:
                break
            row = best_rows[r_idx]
            n = len(row)
            for pos in range(k, 0, -1):
                old_val = row[pos]
                diffs_involving_old = set()
                for i in range(n):
                    if i == pos:
                        continue
                    if i < pos:
                        diffs_involving_old.add(old_val - row[i])
                    else:
                        diffs_involving_old.add(row[i] - old_val)
                forbidden_diffs = used_diffs - diffs_involving_old
                lower_bound = row[pos - 1] + 1
                upper_bound = old_val - 1
                if pos + 1 < n:
                    upper_bound = min(upper_bound, row[pos + 1] - 1)
                if upper_bound < lower_bound:
                    continue
                found = None
                new_diffs = None
                for new_val in range(lower_bound, upper_bound + 1):
                    valid = True
                    nd = set()
                    for i in range(n):
                        if i == pos:
                            continue
                        if i < pos:
                            d = new_val - row[i]
                        else:
                            d = row[i] - new_val
                        if d in forbidden_diffs or d in nd:
                            valid = False
                            break
                        nd.add(d)
                    if valid:
                        found = new_val
                        new_diffs = nd
                        break
                if found is not None and found < old_val:
                    for d in diffs_involving_old:
                        used_diffs.discard(d)
                    for d in new_diffs:
                        used_diffs.add(d)
                    row[pos] = found
                    progress = True

        # Pass 2: left-to-right
        if time.time() < deadline - 1:
            for r_idx in range(n_rows):
                if time.time() > deadline - 1:
                    break
                row = best_rows[r_idx]
                n = len(row)
                for pos in range(1, k + 1):
                    old_val = row[pos]
                    diffs_involving_old = set()
                    for i in range(n):
                        if i == pos:
                            continue
                        if i < pos:
                            diffs_involving_old.add(old_val - row[i])
                        else:
                            diffs_involving_old.add(row[i] - old_val)
                    forbidden_diffs = used_diffs - diffs_involving_old
                    lower_bound = row[pos - 1] + 1
                    upper_bound = old_val - 1
                    if pos + 1 < n:
                        upper_bound = min(upper_bound, row[pos + 1] - 1)
                    if upper_bound < lower_bound:
                        continue
                    found = None
                    new_diffs = None
                    for new_val in range(lower_bound, upper_bound + 1):
                        valid = True
                        nd = set()
                        for i in range(n):
                            if i == pos:
                                continue
                            if i < pos:
                                d = new_val - row[i]
                            else:
                                d = row[i] - new_val
                            if d in forbidden_diffs or d in nd:
                                valid = False
                                break
                            nd.add(d)
                        if valid:
                            found = new_val
                            new_diffs = nd
                            break
                    if found is not None and found < old_val:
                        for d in diffs_involving_old:
                            used_diffs.discard(d)
                        for d in new_diffs:
                            used_diffs.add(d)
                        row[pos] = found
                        progress = True

        if not progress:
            break

    return best_rows


def _greedy_construct(n_rows, k):
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            found = False
            for x in range(current_max + 1, current_max + 3000):
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    for a in row[:-1]:
                        used_diffs.add(x - a)
                    current_max = x
                    found = True
                    break
            if not found:
                return None
        rows.append(row)
    return rows


def _greedy_construct_fast(n_rows, k):
    best_rows = None
    best_max = float('inf')
    for attempt in range(25):
        used_diffs = set()
        rows = []
        success = True
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                found = False
                search_limit = current_max + 600
                for x in range(current_max + 1, search_limit):
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        for a in row[:-1]:
                            used_diffs.add(x - a)
                        current_max = x
                        found = True
                        break
                if not found:
                    for x in range(search_limit, current_max + 4000):
                        valid = True
                        for a in row:
                            d = x - a
                            if d in used_diffs:
                                valid = False
                                break
                        if valid:
                            row.append(x)
                            for a in row[:-1]:
                                used_diffs.add(x - a)
                            current_max = x
                            found = True
                            break
                if not found:
                    success = False
                    break
            if not success:
                break
            rows.append(row)
        if success and rows:
            curr_max = max(max(r) for r in rows)
            if curr_max < best_max:
                best_max = curr_max
                best_rows = rows
    if best_rows:
        return best_rows
    return _greedy_construct(n_rows, k)
