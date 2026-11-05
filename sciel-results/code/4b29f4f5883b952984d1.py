import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = time.time() + 140

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

    def greedy_construct(seed_offset=0):
        rng = random.Random(42 + seed_offset)
        used = set()
        rows = []

        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)

        return rows

    def greedy_construct_randomized(seed_offset=0):
        rng = random.Random(42 + seed_offset)
        used = set()
        rows = []

        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
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
                row.append(x)
                used |= new_diffs
            rows.append(row)

        return rows

    def local_search_shrink(rows, deadline, rng):
        """Local search: try to reduce the maximum entry by modifying individual elements."""
        rows = [list(r) for r in rows]
        best_scope = _scope(rows)
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

    def simulated_annealing(rows, deadline, seed_offset=0):
        """SA that jointly optimizes all rows with coordinated moves."""
        rng = random.Random(123 + seed_offset)
        current = [list(r) for r in rows]
        best = [list(r) for r in current]
        best_scope = _scope(best)
        current_scope = best_scope

        used = _all_diffs(current)
        T = max(1.0, current_scope * 0.5)
        T_min = 0.01
        cooling = 0.995
        max_iters = 50000

        for it in range(max_iters):
            if time.time() > deadline:
                break
            if T < T_min:
                break

            ri = rng.randint(0, len(current) - 1)
            pi = rng.randint(1, k)

            old_val = current[ri][pi]
            low = current[ri][pi - 1] + 1
            high = current_scope + 5
            if pi < k:
                high = min(high, current[ri][pi + 1] - 1)

            if low > high:
                continue

            new_val = rng.randint(low, high)
            if new_val == old_val:
                continue

            # Compute old and new diffs for this element
            old_diffs = set()
            for i in range(pi):
                old_diffs.add(old_val - current[ri][i])
            for i in range(pi + 1, len(current[ri])):
                old_diffs.add(current[ri][i] - old_val)

            new_diffs = set()
            for i in range(pi):
                new_diffs.add(new_val - current[ri][i])
            for i in range(pi + 1, len(current[ri])):
                new_diffs.add(current[ri][i] - new_val)

            # Check conflicts
            remaining_used = used - old_diffs
            conflict = new_diffs & remaining_used
            if conflict:
                # Penalty for conflicts
                delta = len(conflict) * 10 + (new_val - old_val) * 0.1
            else:
                new_scope = max(max(r) for r in current if r is not current[ri])
                new_scope = max(new_scope, max(current[ri]))
                if pi + 1 < len(current[ri]):
                    new_scope = max(new_scope, new_val)
                delta = new_scope - current_scope

            if delta <= 0 or rng.random() < math.exp(-delta / T):
                # Accept
                current[ri][pi] = new_val
                used -= old_diffs
                used |= new_diffs
                current_scope += delta

                if current_scope < best_scope:
                    best_scope = current_scope
                    best = [list(r) for r in current]

            T *= cooling

        return best, best_scope

    def diff_packing_construct(M, seed_offset=0):
        """Difference-space list scheduling: assign differences to rows."""
        rng = random.Random(42 + seed_offset)
        # Total differences needed: n_rows * k*(k+1)/2
        total_diffs = n_rows * k * (k + 1) // 2
        if M < total_diffs:
            return None

        # We need to partition [1, M] into groups of k*(k+1)/2 for each row
        # such that each group can form a valid row starting at 0
        # This is complex; use a simpler approach: try to build rows with max entry <= M

        used = set()
        rows = []

        for _ in range(n_rows):
            row = [0]
            x = 0
            found = False
            while len(row) < k + 1:
                x += 1
                if x > M:
                    return None
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
                    found = True
            rows.append(row)

        return rows

    # Start with greedy constructions
    best_rows = greedy_construct(0)
    best_scope = _scope(best_rows)

    # Try multiple greedy variants
    for seed in range(20):
        if time.time() > deadline - 30:
            break
        rows = greedy_construct(seed)
        scope = _scope(rows)
        if scope < best_scope:
            best_rows = rows
            best_scope = scope

    for seed in range(10):
        if time.time() > deadline - 30:
            break
        rows = greedy_construct_randomized(seed)
        scope = _scope(rows)
        if scope < best_scope:
            best_rows = rows
            best_scope = scope

    # Local search shrink
    rng = random.Random(123)
    if time.time() < deadline - 20:
        polished = local_search_shrink(best_rows, deadline - 10, rng)
        polished_scope = _scope(polished)
        if polished_scope < best_scope:
            best_rows = polished
            best_scope = polished_scope

    # Simulated annealing
    if time.time() < deadline - 15:
        sa_rows, sa_scope = simulated_annealing(best_rows, deadline - 5, seed_offset=0)
        if sa_scope < best_scope:
            best_rows = sa_rows
            best_scope = sa_scope

    # Try diff packing with various M values
    for M in range(max(50, best_scope - 10), best_scope + 1):
        if time.time() > deadline - 5:
            break
        for seed in range(5):
            rows = diff_packing_construct(M, seed)
            if rows is not None:
                scope = _scope(rows)
                if scope < best_scope:
                    best_rows = rows
                    best_scope = scope

    # Final polish
    if time.time() < deadline - 2:
        rng = random.Random(999)
        polished = local_search_shrink(best_rows, deadline - 2, rng)
        polished_scope = _scope(polished)
        if polished_scope < best_scope:
            best_rows = polished
            best_scope = polished_scope

    return best_rows
