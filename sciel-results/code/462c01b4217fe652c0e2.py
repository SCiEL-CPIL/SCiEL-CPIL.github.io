"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def _build_greedy(n_rows, k):
    """Fast greedy: each row takes smallest next mark with no difference collision."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _valid(rows, n_rows, k):
    """Check all rows are valid and all differences are globally distinct."""
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            if i > 0 and row[i] <= row[i - 1]:
                return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _max_entry(rows):
    return max(max(r) for r in rows)

def _row_diffs_set(row):
    """Return set of all positive differences in a row."""
    s = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s

def _row_diffs_list(row):
    """Return list of all positive differences in a row."""
    out = []
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            out.append(row[j] - row[i])
    return out

def construct(n_rows, k):
    deadline = time.time() + 105  # safety margin before 120s limit
    random.seed(42)

    # Start from greedy
    rows = _build_greedy(n_rows, k)
    best = [r[:] for r in rows]
    best_val = _max_entry(best)

    # Build initial used set
    cur = [r[:] for r in best]
    cur_val = best_val
    cur_used = set()
    for r in cur:
        cur_used |= _row_diffs_set(r)

    # Simulated annealing with incremental diff tracking
    T = 20.0
    T_min = 0.05
    cool = 0.999

    def try_move(ri, pi, new_val):
        """Try to move position pi in row ri to new_val. Returns (cand_row, valid) if valid."""
        # Check strict increase within row
        if pi > 0 and new_val <= cur[ri][pi - 1]:
            return None, False
        if pi < k and new_val >= cur[ri][pi + 1]:
            return None, False
        if new_val <= 0:
            return None, False

        # Build candidate row
        cand_row = cur[ri][:]
        cand_row[pi] = new_val

        # Compute new diffs for this row
        new_diffs = _row_diffs_set(cand_row)

        # Check for collisions with other rows' diffs
        # Get old diffs for this row
        old_diffs = _row_diffs_set(cur[ri])

        # Diffs that are removed from cur_used
        removed = old_diffs - new_diffs
        # Diffs that are added to cur_used
        added = new_diffs - old_diffs

        # Check if any added diff is already in cur_used (from other rows or same row)
        for d in added:
            if d in cur_used and d not in removed:
                # This diff is already used elsewhere
                return None, False

        return cand_row, True

    step = 0
    no_improve = 0
    while time.time() < deadline:
        step += 1

        # Strategy: target the row with the largest entry, or random
        if random.random() < 0.6:
            # Target the row with max entry
            max_row_idx = max(range(n_rows), key=lambda i: cur[i][-1])
            ri = max_row_idx
        else:
            ri = random.randrange(n_rows)

        # Choose position: bias toward later positions (larger values)
        if random.random() < 0.5:
            pi = random.randrange(k // 2, k + 1)  # upper half
        else:
            pi = random.randrange(1, k + 1)

        old_val = cur[ri][pi]

        # Try decreasing first (to reduce max), then increasing
        deltas = []
        # Prefer negative deltas to reduce values
        for d in range(-1, -20, -1):
            deltas.append(d)
        for d in range(1, 10):
            deltas.append(d)
        random.shuffle(deltas)

        moved = False
        for d in deltas:
            new_val = old_val + d
            if new_val <= 0:
                continue

            cand_row, valid = try_move(ri, pi, new_val)
            if not valid:
                continue

            # Compute new max
            if ri == max(range(n_rows), key=lambda i: cur[i][-1]) or new_val > cur_val:
                # Recompute max if this might affect it
                new_max = max(max(r) for r in cur if r is not cand_row)
                new_max = max(new_max, max(cand_row))
            else:
                new_max = cur_val
                if new_val > new_max:
                    new_max = new_val

            # Acceptance criterion
            if new_max < cur_val:
                # Always accept improvement
                old_diffs = _row_diffs_set(cur[ri])
                new_diffs = _row_diffs_set(cand_row)
                cur_used -= old_diffs
                cur_used |= new_diffs
                cur[ri] = cand_row
                cur_val = new_max
                if cur_val < best_val:
                    best_val = cur_val
                    best = [r[:] for r in cur]
                    no_improve = 0
                moved = True
                break
            elif new_max == cur_val:
                # Accept lateral move with some probability
                if random.random() < 0.4:
                    old_diffs = _row_diffs_set(cur[ri])
                    new_diffs = _row_diffs_set(cand_row)
                    cur_used -= old_diffs
                    cur_used |= new_diffs
                    cur[ri] = cand_row
                    moved = True
                    break
            else:
                # Worse move: SA acceptance
                delta_e = new_max - cur_val
                if random.random() < (1.0 / (1.0 + T * delta_e)) * 0.05:
                    old_diffs = _row_diffs_set(cur[ri])
                    new_diffs = _row_diffs_set(cand_row)
                    cur_used -= old_diffs
                    cur_used |= new_diffs
                    cur[ri] = cand_row
                    cur_val = new_max
                    moved = True
                    break

        if not moved:
            no_improve += 1
            # Restart if stuck too long
            if no_improve > 500:
                # Perturb: pick a random row and shuffle it
                ri = random.randrange(n_rows)
                # Try to rebuild this row with a different greedy start
                # Remove this row's diffs
                old_diffs = _row_diffs_set(cur[ri])
                cur_used -= old_diffs

                # Try to build a new row
                row = [0]
                x = 0
                attempts = 0
                while len(row) < k + 1 and attempts < 1000:
                    x += 1
                    new = {x - a for a in row}
                    if not new & cur_used:
                        row.append(x)
                        cur_used |= new
                    attempts += 1

                if len(row) == k + 1:
                    cur[ri] = row
                    cur_val = max(max(r) for r in cur)
                    if cur_val < best_val:
                        best_val = cur_val
                        best = [r[:] for r in cur]
                else:
                    # Restore
                    for d in _row_diffs_set(cur[ri]):
                        cur_used.discard(d)
                    # Rebuild original
                    pass

                no_improve = 0

        # Cool down
        T = max(T_min, T * cool)

    # Ensure best is valid
    if not _valid(best, n_rows, k):
        best = _build_greedy(n_rows, k)
    return best
# EVOLVE-BLOCK-END
