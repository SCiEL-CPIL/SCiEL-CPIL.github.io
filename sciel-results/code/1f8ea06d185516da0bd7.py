import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 150s kill

    num_diffs_per_row = (k + 1) * k // 2
    lower_bound = n_rows * num_diffs_per_row

    best_scope = float('inf')
    best_rows = None

    # Try multiple seeds with backtracking + local search
    seed = 0
    while time.time() < deadline - 25:
        rows = _backtrack_construct(n_rows, k, seed, deadline - 20)
        if rows is not None:
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                if best_scope == lower_bound:
                    break
        seed += 1

    # Polish the best solution found
    if best_rows is not None and time.time() < deadline - 5:
        polished = _local_search(best_rows, n_rows, k, deadline - 3)
        if polished is not None:
            scope = max(max(r) for r in polished)
            if scope <= best_scope:
                best_rows = polished
                best_scope = scope

    # Fallback: simple greedy if nothing found
    if best_rows is None:
        best_rows = _simple_greedy(n_rows, k)

    return best_rows


def _backtrack_construct(n_rows, k, seed, deadline):
    """
    Backtracking search that builds rows one at a time.
    For each row, we try to build it using a constraint-aware heuristic:
    - At each position, prefer candidates that leave more room for future rows.
    - Use randomized restarts with different seeds.
    """
    rng = random.Random(seed)
    used = set()
    rows = []

    def _try_build_row(row, pos, max_val):
        """Try to extend row at position pos. Returns True if success."""
        if pos == k + 1:
            return True

        last = row[-1] if row else 0
        # Try candidates from last+1 up to max_val
        candidates = []
        for x in range(last + 1, max_val + 1):
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                candidates.append(x)

        if not candidates:
            return False

        # Shuffle with bias toward smaller values (to keep scope small)
        # but also consider spread
        rng.shuffle(candidates)

        # Try candidates in order of a heuristic score
        def score(x):
            # Prefer smaller x (keeps scope down)
            # but also prefer x that doesn't create "tight" differences
            s = x
            for a in row:
                d = x - a
                # Penalize if d is close to existing used diffs
                if used:
                    min_dist = min(abs(d - u) for u in used) if used else 0
                    s += min_dist * 0.1
            return s

        candidates.sort(key=score)

        for x in candidates:
            new_diffs = {x - a for a in row}
            if new_diffs & used:
                continue

            # Temporarily add
            old_used = set(used)
            used.update(new_diffs)
            row.append(x)

            if _try_build_row(row, pos + 1, max_val):
                return True

            # Backtrack
            row.pop()
            used.clear()
            used.update(old_used)

        return False

    # Build each row
    for r_idx in range(n_rows):
        if time.time() > deadline:
            return None

        # Estimate max_val: we need enough room for this row
        # The remaining rows also need space, so be conservative
        remaining_rows = n_rows - r_idx
        # Each row needs at least k+1 elements, and differences must be distinct
        # A rough upper bound: lower_bound / remaining_rows + some slack
        max_val = int(lower_bound / remaining_rows) + 30 + r_idx * 5

        row = [0]
        if not _try_build_row(row, 1, max_val):
            return None

        # Commit the row
        for a in row:
            for b in row:
                if a < b:
                    d = b - a
                    if d in used:
                        return None  # Should not happen
                    used.add(d)

        rows.append(row)

    return rows


def _local_search(rows, n_rows, k, deadline):
    """
    Local search to reduce the scope.
    Try to move elements to smaller positions while maintaining validity.
    """
    rng = random.Random(12345)
    current_rows = [list(r) for r in rows]
    current_scope = max(max(r) for r in current_rows)

    def _check_validity(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    iterations = 0
    max_iterations = 5000

    while iterations < max_iterations and time.time() < deadline:
        iterations += 1

        # Pick a random row and position
        row_idx = rng.randint(0, n_rows - 1)
        pos_idx = rng.randint(1, k)

        old_val = current_rows[row_idx][pos_idx]

        # Try to move to a smaller value
        lower = current_rows[row_idx][pos_idx - 1] + 1
        upper = current_rows[row_idx][pos_idx + 1] - 1 if pos_idx < k else old_val - 1

        if upper < lower:
            continue

        # Try several candidate values
        try_count = 0
        for new_val in range(upper, lower - 1, -1):
            if new_val >= old_val:
                continue
            try_count += 1
            if try_count > 20:
                break

            current_rows[row_idx][pos_idx] = new_val

            if _check_validity(current_rows):
                new_scope = max(max(r) for r in current_rows)
                if new_scope < current_scope:
                    current_scope = new_scope
                    break
                else:
                    current_rows[row_idx][pos_idx] = old_val
            else:
                current_rows[row_idx][pos_idx] = old_val

    return current_rows


def _simple_greedy(n_rows, k):
    """Simple greedy fallback."""
    used = set()
    rows = []

    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            x = row[-1] + 1
            while True:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    break
                x += 1
            row.append(x)
            new_diffs = {x - a for a in row[:-1]}
            used.update(new_diffs)
        rows.append(row)

    return rows
