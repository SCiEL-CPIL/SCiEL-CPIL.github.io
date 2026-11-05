import time
import random
import math
import bisect

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    num_diffs_per_row = (k + 1) * k // 2
    lower_bound = n_rows * num_diffs_per_row

    best_scope = float('inf')
    best_rows = None

    # Phase 1: Generate initial solutions with best-fit greedy
    seed = 0
    while time.time() < deadline - 75:  # Reserve 75s for SA + polish
        rows = _greedy_construct(n_rows, k, seed)
        if rows is not None:
            scope = _scope(rows)

            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                if best_scope == lower_bound:
                    break

        seed += 1

    if best_rows is None:
        best_rows = _greedy_construct(n_rows, k, 0)
        if best_rows is None:
            # Fallback: construct a trivial (possibly invalid) solution
            best_rows = [[list(range(k + 1))] for _ in range(n_rows)]
        best_scope = _scope(best_rows)

    # Phase 2: Local search with maintained diff-count (fast validity checks)
    if time.time() < deadline - 12:
        improved = _simulated_annealing(best_rows, n_rows, k, start_time, deadline)
        if improved is not None:
            imp_scope = _scope(improved)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved

    # Phase 3: Final polish pass targeting the maximum element
    if time.time() < deadline - 2 and best_rows is not None:
        polished = _polish_max(best_rows, n_rows, k, start_time, deadline)
        if polished is not None:
            pol_scope = _scope(polished)
            if pol_scope < best_scope:
                best_scope = pol_scope
                best_rows = polished

    return best_rows

def _scope(rows):
    if not rows:
        return 0
    return max(row[-1] for row in rows if row)

def _greedy_construct(n_rows, k, seed):
    """Best-fit greedy: prefer values whose differences pack tightly into used space."""
    rng = random.Random(seed)
    used = set()
    used_sorted = []
    rows = []

    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            valid_xs = []
            scored_xs = []
            last_x = row[-1]
            limit = 500

            x = last_x + 1
            while len(valid_xs) < 40 and x <= limit:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    valid_xs.append(x)
                    score = _packing_score(new_diffs, used_sorted, x)
                    scored_xs.append((score, x))
                x += 1

            if not valid_xs:
                x = last_x + 1
                while len(valid_xs) < 40:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        valid_xs.append(x)
                        score = _packing_score(new_diffs, used_sorted, x)
                        scored_xs.append((score, x))
                    x += 1

            if not valid_xs:
                return None

            scored_xs.sort(key=lambda t: t[0])
            top_n = min(5, len(scored_xs))
            chosen_x = scored_xs[rng.randint(0, top_n - 1)][1]
            row.append(chosen_x)
            new_diffs = {chosen_x - a for a in row[:-1]}
            used.update(new_diffs)
            for d in new_diffs:
                bisect.insort(used_sorted, d)

        rows.append(row)

    return rows

def _packing_score(new_diffs, used, x):
    """
    Score a candidate x by how well its differences 'pack' into the used difference space.
    Lower score = better fit (less fragmentation).
    """
    if not used:
        return x * 1.0

    if not isinstance(used, list):
        used_sorted = sorted(used)
    else:
        used_sorted = used

    total_dist = 0.0
    for d in new_diffs:
        idx = bisect.bisect_left(used_sorted, d)
        min_dist = float('inf')
        if idx < len(used_sorted):
            min_dist = min(min_dist, abs(d - used_sorted[idx]))
        if idx > 0:
            min_dist = min(min_dist, abs(d - used_sorted[idx - 1]))
        total_dist += min_dist

    return total_dist / max(len(new_diffs), 1) + 0.1 * x

def _simulated_annealing(initial_rows, n_rows, k, start_time, deadline):
    """
    Local search with maintained diff-count for O(1) validity checks,
    plus a dedicated shrink-the-max-row phase.
    """
    rng = random.Random(42)
    rows = [list(r) for r in initial_rows]
    current_scope = _scope(rows)
    best_scope = current_scope
    best_rows = [list(r) for r in rows]

    # Precompute difference sets for each row
    row_diffs = []
    for row in rows:
        n = len(row)
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        row_diffs.append(diffs)

    # Maintain a global count of how many rows use each difference value.
    diff_count = {}
    for diffs in row_diffs:
        for d in diffs:
            diff_count[d] = diff_count.get(d, 0) + 1

    def _row_is_valid(new_diffs, exclude_idx):
        for d in new_diffs:
            cnt = diff_count.get(d, 0)
            if d in row_diffs[exclude_idx]:
                cnt -= 1
            if cnt > 0:
                return False
        return True

    def _commit_row(row_idx, new_row, new_diffs):
        nonlocal rows, row_diffs, diff_count
        for d in row_diffs[row_idx]:
            diff_count[d] -= 1
            if diff_count[d] == 0:
                del diff_count[d]
        for d in new_diffs:
            diff_count[d] = diff_count.get(d, 0) + 1
        rows[row_idx] = new_row
        row_diffs[row_idx] = new_diffs

    # Phase A: General local search
    T_start = 20.0
    T_end = 0.3
    total_steps = 40000
    cool_rate = (T_end / T_start) ** (1.0 / total_steps)
    T = T_start
    steps_since_improve = 0
    max_no_improve = 4000

    for step in range(total_steps):
        if time.time() > deadline - 25:
            break

        if rng.random() < 0.4:
            row_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        else:
            row_idx = rng.randint(0, n_rows - 1)

        new_row = _generate_candidate_row(rows, row_idx, k, rng)
        if new_row is None:
            continue

        n = len(new_row)
        new_diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                new_diffs.add(new_row[j] - new_row[i])

        if not _row_is_valid(new_diffs, row_idx):
            continue

        new_scope = max(rows[i][-1] for i in range(n_rows) if i != row_idx)
        new_scope = max(new_scope, new_row[-1])

        delta = new_scope - current_scope
        if delta <= 0:
            _commit_row(row_idx, new_row, new_diffs)
            current_scope = new_scope
            steps_since_improve = 0
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in rows]
        else:
            prob = math.exp(-delta / T)
            if rng.random() < prob:
                _commit_row(row_idx, new_row, new_diffs)
                current_scope = new_scope
            steps_since_improve += 1

        T *= cool_rate

        if steps_since_improve > max_no_improve:
            pert_idx = rng.randint(0, n_rows - 1)
            pert_row = _generate_candidate_row(rows, pert_idx, k, rng)
            if pert_row is not None:
                n = len(pert_row)
                pert_diffs = set()
                for i in range(n):
                    for j in range(i + 1, n):
                        pert_diffs.add(pert_row[j] - pert_row[i])
                if _row_is_valid(pert_diffs, pert_idx):
                    _commit_row(pert_idx, pert_row, pert_diffs)
                    current_scope = _scope(rows)
            steps_since_improve = 0

    # Phase B: Dedicated shrink-the-max-row loop
    for attempt in range(50):
        if time.time() > deadline - 5:
            break

        max_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        current_max = rows[max_idx][-1]

        target_max = current_max - 1
        found = False
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            new_row = _build_constrained_row(rows, max_idx, k, rng, target_max)
            if new_row is None:
                target_max += 1
                continue
            n = len(new_row)
            new_diffs = set()
            for i in range(n):
                for j in range(i + 1, n):
                    new_diffs.add(new_row[j] - new_row[i])
            if _row_is_valid(new_diffs, max_idx):
                _commit_row(max_idx, new_row, new_diffs)
                current_scope = _scope(rows)
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [list(r) for r in rows]
                found = True
                break
            target_max += 1
        if not found:
            for _ in range(10):
                if time.time() > deadline - 5:
                    break
                new_row = _generate_candidate_row(rows, max_idx, k, rng)
                if new_row is None:
                    continue
                n = len(new_row)
                new_diffs = set()
                for i in range(n):
                    for j in range(i + 1, n):
                        new_diffs.add(new_row[j] - new_row[i])
                if not _row_is_valid(new_diffs, max_idx):
                    continue
                if new_row[-1] < current_max:
                    _commit_row(max_idx, new_row, new_diffs)
                    current_scope = _scope(rows)
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [list(r) for r in rows]
                    break

    return best_rows

def _build_constrained_row(rows, exclude_idx, k, rng, target_max):
    """Build a row with all elements <= target_max using best-fit packing."""
    used_diffs = set()
    for i, row in enumerate(rows):
        if i == exclude_idx:
            continue
        n = len(row)
        for a in range(n):
            for b in range(a + 1, n):
                used_diffs.add(row[b] - row[a])

    used_sorted = sorted(used_diffs)

    row = [0]
    for _ in range(k):
        valid_xs = []
        scored_xs = []
        last_x = row[-1]

        x = last_x + 1
        while x <= target_max:
            new_diffs = {x - a for a in row}
            if not (new_diffs & used_diffs):
                valid_xs.append(x)
                score = _packing_score(new_diffs, used_sorted, x)
                scored_xs.append((score, x))
            x += 1

        if not valid_xs:
            return None

        scored_xs.sort(key=lambda t: t[0])
        top_n = min(3, len(scored_xs))
        chosen_x = scored_xs[rng.randint(0, top_n - 1)][1]
        row.append(chosen_x)
        new_diffs = {chosen_x - a for a in row[:-1]}
        used_diffs.update(new_diffs)
        for d in new_diffs:
            bisect.insort(used_sorted, d)

    return row

def _generate_candidate_row(rows, exclude_idx, k, rng):
    """Generate a new candidate row using best-fit packing heuristic."""
    used_diffs = set()
    for i, row in enumerate(rows):
        if i == exclude_idx:
            continue
        n = len(row)
        for a in range(n):
            for b in range(a + 1, n):
                used_diffs.add(row[b] - row[a])

    used_sorted = sorted(used_diffs)

    row = [0]
    for _ in range(k):
        valid_xs = []
        scored_xs = []
        last_x = row[-1]
        limit = 500

        x = last_x + 1
        while len(valid_xs) < 30 and x <= limit:
            new_diffs = {x - a for a in row}
            if not (new_diffs & used_diffs):
                valid_xs.append(x)
                score = _packing_score(new_diffs, used_sorted, x)
                scored_xs.append((score, x))
            x += 1

        if not valid_xs:
            x = last_x + 1
            while len(valid_xs) < 30:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    valid_xs.append(x)
                    score = _packing_score(new_diffs, used_sorted, x)
                    scored_xs.append((score, x))
                x += 1

        if not valid_xs:
            return None

        scored_xs.sort(key=lambda t: t[0])
        top_n = min(4, len(scored_xs))
        chosen_x = scored_xs[rng.randint(0, top_n - 1)][1]

        row.append(chosen_x)
        new_diffs = {chosen_x - a for a in row[:-1]}
        used_diffs.update(new_diffs)
        for d in new_diffs:
            bisect.insort(used_sorted, d)

    return row

def _polish_max(rows, n_rows, k, start_time, deadline):
    """
    Final polish: repeatedly try to reduce the maximum element using
    maintained diff-count for fast validity checks.
    """
    rng = random.Random(123)
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)

    row_diffs = []
    diff_count = {}
    for row in rows:
        n = len(row)
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        row_diffs.append(diffs)
        for d in diffs:
            diff_count[d] = diff_count.get(d, 0) + 1

    def _valid(new_diffs, excl):
        for d in new_diffs:
            cnt = diff_count.get(d, 0)
            if d in row_diffs[excl]:
                cnt -= 1
            if cnt > 0:
                return False
        return True

    def _commit(idx, new_row, new_diffs):
        nonlocal rows, row_diffs, diff_count
        for d in row_diffs[idx]:
            diff_count[d] -= 1
            if diff_count[d] == 0:
                del diff_count[d]
        for d in new_diffs:
            diff_count[d] = diff_count.get(d, 0) + 1
        rows[idx] = new_row
        row_diffs[idx] = new_diffs

    for iteration in range(100):
        if time.time() > deadline - 2:
            break

        max_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        current_max = rows[max_idx][-1]

        target = current_max - 1
        improved = False
        for _ in range(10):
            if time.time() > deadline - 2:
                break
            new_row = _build_constrained_row(rows, max_idx, k, rng, target)
            if new_row is None:
                target += 1
                continue
            n = len(new_row)
            new_diffs = set()
            for i in range(n):
                for j in range(i + 1, n):
                    new_diffs.add(new_row[j] - new_row[i])
            if _valid(new_diffs, max_idx):
                _commit(max_idx, new_row, new_diffs)
                new_scope = _scope(rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in rows]
                improved = True
                break
            target += 1

        if not improved:
            new_row = _generate_candidate_row(rows, max_idx, k, rng)
            if new_row is not None and new_row[-1] < current_max:
                n = len(new_row)
                new_diffs = set()
                for i in range(n):
                    for j in range(i + 1, n):
                        new_diffs.add(new_row[j] - new_row[i])
                if _valid(new_diffs, max_idx):
                    _commit(max_idx, new_row, new_diffs)
                    new_scope = _scope(rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in rows]

    return best_rows

def _reduce_max_row(rows, row_idx, k, rng, current_max):
    """
    Try to find a replacement for rows[row_idx] with a smaller maximum element.
    """
    used_diffs = set()
    for i, row in enumerate(rows):
        if i == row_idx:
            continue
        n = len(row)
        for a in range(n):
            for b in range(a + 1, n):
                used_diffs.add(row[b] - row[a])

    used_sorted = sorted(used_diffs)

    target_max = current_max - 1
    for attempt in range(15):
        row = [0]
        success = True
        local_used = set(used_diffs)
        local_sorted = list(used_sorted)

        for _ in range(k):
            valid_xs = []
            scored_xs = []
            last_x = row[-1]

            x = last_x + 1
            while x <= target_max:
                new_diffs = {x - a for a in row}
                if not (new_diffs & local_used):
                    valid_xs.append(x)
                    score = _packing_score(new_diffs, local_sorted, x)
                    scored_xs.append((score, x))
                x += 1

            if not valid_xs:
                success = False
                break

            scored_xs.sort(key=lambda t: t[0])
            top_n = min(3, len(scored_xs))
            chosen_x = scored_xs[rng.randint(0, top_n - 1)][1]
            row.append(chosen_x)
            new_diffs = {chosen_x - a for a in row[:-1]}
            local_used.update(new_diffs)
            for d in new_diffs:
                bisect.insort(local_sorted, d)

        if success and len(row) == k + 1:
            return row

        target_max = min(target_max + 2, current_max - 1)

    return None
