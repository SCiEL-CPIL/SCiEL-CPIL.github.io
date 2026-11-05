import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = time.time() + 140  # 10s margin from 150s limit

    def _row_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def _all_diffs(rows):
        used = set()
        for row in rows:
            used |= _row_diffs(row)
        return used

    def _valid(rows):
        all_d = set()
        for row in rows:
            if not row or row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i + 1] <= row[i]:
                    return False
            rd = _row_diffs(row)
            if rd & all_d:
                return False
            all_d |= rd
        return True

    def _scope(rows):
        return max(max(r) for r in rows)

    def _greedy_construct(seed_offset=0, randomize=False):
        rng = random.Random(42 + seed_offset)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                if randomize:
                    max_step = max(1, int((k + 1) * 2.5))
                    found = False
                    for _attempt in range(30):
                        step = rng.randint(1, max_step)
                        x = row[-1] + step
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            found = True
                            break
                    if not found:
                        x = row[-1] + 1
                        while True:
                            new_diffs = {x - a for a in row}
                            if not new_diffs & used:
                                break
                            x += 1
                else:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if new_diffs & used:
                        continue
                row.append(x)
                used |= new_diffs
            rows.append(row)
        return rows

    def _local_search_shrink(rows, deadline, rng):
        rows = [list(r) for r in rows]
        used = _all_diffs(rows)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_scope = _scope(rows)
            for ri in range(len(rows)):
                if time.time() > deadline:
                    break
                row = rows[ri]
                for pos in range(len(row) - 1, 0, -1):
                    if time.time() > deadline:
                        break
                    old_val = row[pos]
                    if old_val < current_scope:
                        continue
                    low = row[pos - 1] + 1
                    high = old_val - 1
                    if pos + 1 < len(row):
                        high = min(high, row[pos + 1] - 1)
                    if low > high:
                        continue
                    candidates = list(range(high, low - 1, -1))
                    rng.shuffle(candidates)
                    for new_val in candidates:
                        if time.time() > deadline:
                            break
                        if new_val >= old_val:
                            continue
                        old_diffs = set()
                        for i in range(pos):
                            old_diffs.add(old_val - row[i])
                        for i in range(pos + 1, len(row)):
                            old_diffs.add(row[i] - old_val)
                        new_diffs = set()
                        for i in range(pos):
                            new_diffs.add(new_val - row[i])
                        for i in range(pos + 1, len(row)):
                            new_diffs.add(row[i] - new_val)
                        conflict = new_diffs & (used - old_diffs)
                        if conflict:
                            continue
                        row[pos] = new_val
                        used -= old_diffs
                        used |= new_diffs
                        improved = True
                        break
        return rows

    def _simulated_annealing(rows, k, deadline, rng):
        """Simulated annealing: jointly optimize all rows via coordinated moves."""
        rows = [list(r) for r in rows]
        current_scope = _scope(rows)
        best_rows = [r[:] for r in rows]
        best_scope = current_scope
        used = _all_diffs(rows)

        # Build diff_map for quick conflict checking
        def _build_diff_map(rows):
            dm = {}
            for ri, row in enumerate(rows):
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        dm[row[j] - row[i]] = (ri, i, j)
            return dm

        diff_map = _build_diff_map(rows)

        # Temperature schedule
        T = float(current_scope)
        T_min = 1.0
        cooling = 0.995
        steps_per_iter = 50

        step = 0
        while time.time() < deadline:
            T *= cooling
            if T < T_min:
                T = T_min

            for _ in range(steps_per_iter):
                if time.time() > deadline:
                    break
                step += 1

                # Random move: pick a row, position, and new value
                ri = rng.randint(0, len(rows) - 1)
                row = rows[ri]
                pos = rng.randint(1, k)

                old_val = row[pos]
                low = row[pos - 1] + 1
                high = min(current_scope + 5, row[pos - 1] + current_scope)
                if pos + 1 < len(row):
                    high = min(high, row[pos + 1] - 1)

                if low > high:
                    continue

                new_val = rng.randint(low, high)
                if new_val == old_val:
                    continue

                # Compute diff changes
                old_diffs = set()
                for i in range(pos):
                    old_diffs.add(old_val - row[i])
                for i in range(pos + 1, len(row)):
                    old_diffs.add(row[i] - old_val)

                new_diffs = set()
                for i in range(pos):
                    new_diffs.add(new_val - row[i])
                for i in range(pos + 1, len(row)):
                    new_diffs.add(row[i] - new_val)

                # Check conflicts
                removed = old_diffs - new_diffs
                added = new_diffs - old_diffs
                conflict = added & (used - old_diffs)
                if conflict:
                    continue

                # Apply move
                row[pos] = new_val
                used -= old_diffs
                used |= new_diffs

                new_scope = _scope(rows)
                delta = new_scope - current_scope

                if delta <= 0 or rng.random() < pow(2.718281828, -delta / max(T, 0.01)):
                    current_scope = new_scope
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [r[:] for r in rows]
                else:
                    # Revert
                    row[pos] = old_val
                    used -= new_diffs
                    used |= old_diffs

        return best_rows, best_scope

    def _backtrack_row(used, max_val, row, depth, rng, deadline):
        if time.time() > deadline:
            return None
        if depth == k + 1:
            return row[:]

        last = row[-1] if row else 0
        start = last + 1
        candidates = list(range(start, max_val + 1))
        rng.shuffle(candidates)

        for x in candidates:
            new_diffs = {x - a for a in row}
            if new_diffs & used:
                continue
            row.append(x)
            used_copy = used | new_diffs
            result = _backtrack_row(used_copy, max_val, row, depth + 1, rng, deadline)
            if result is not None:
                return result
            row.pop()
        return None

    def _backtrack_construct(max_val, num_rows, rng, deadline):
        used = set()
        rows = []
        for i in range(num_rows):
            if time.time() > deadline:
                return None
            row = [0]
            result = _backtrack_row(used, max_val, row, 1, rng, deadline)
            if result is None:
                return None
            rows.append(result)
            used |= _row_diffs(result)
        return rows

    # Phase 1: Generate diverse initial solutions via greedy
    best_rows = _greedy_construct(seed_offset=0, randomize=False)
    best_scope = _scope(best_rows)

    for seed in range(10):
        if time.time() > deadline - 30:
            break
        rows = _greedy_construct(seed_offset=seed, randomize=(seed % 2 == 0))
        scope = _scope(rows)
        if scope < best_scope:
            best_rows = rows
            best_scope = scope

    # Phase 2: Local search shrink on best solution
    rng = random.Random(123)
    if time.time() < deadline - 20:
        polished = _local_search_shrink(best_rows, deadline - 20, rng)
        polished_scope = _scope(polished)
        if polished_scope < best_scope:
            best_rows = polished
            best_scope = polished_scope

    # Phase 3: Simulated annealing for coordinated multi-row optimization
    if time.time() < deadline - 10:
        sa_rows, sa_scope = _simulated_annealing(best_rows, k, deadline - 10, rng)
        if sa_scope < best_scope:
            best_rows = sa_rows
            best_scope = sa_scope

    # Phase 4: Local search shrink after SA
    if time.time() < deadline - 5:
        rng2 = random.Random(999)
        polished = _local_search_shrink(best_rows, deadline - 5, rng2)
        polished_scope = _scope(polished)
        if polished_scope < best_scope:
            best_rows = polished
            best_scope = polished_scope

    # Phase 5: Try backtracking at progressively lower scopes
    current_max = best_scope
    lower_bound = max(n_rows * k * (k + 1) // 2, 50)

    while time.time() < deadline - 3:
        target_scope = current_max - 1
        if target_scope < lower_bound:
            break
        rng3 = random.Random(int(time.time() * 1000) % 10000)
        result = _backtrack_construct(target_scope, n_rows, rng3, deadline - 3)
        if result is not None:
            best_rows = result
            best_scope = target_scope
            current_max = target_scope
            if time.time() < deadline - 3:
                rng4 = random.Random(777)
                polished = _local_search_shrink(best_rows, deadline - 3, rng4)
                polished_scope = _scope(polished)
                if polished_scope < best_scope:
                    best_rows = polished
                    best_scope = polished_scope
                    current_max = polished_scope
        else:
            current_max = target_scope

    # Final validation
    if not _valid(best_rows):
        # Fallback to a known-valid construction
        best_rows = _greedy_construct(seed_offset=0, randomize=False)

    return best_rows
