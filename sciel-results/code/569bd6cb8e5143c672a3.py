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

def _row_diffs(row):
    """Return set of all positive differences in a row."""
    s = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s

def _max_entry(rows):
    return max(max(r) for r in rows)

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

def construct(n_rows, k):
    deadline = time.time() + 110  # safety margin before 120s limit
    random.seed(42)

    # Start from greedy
    rows = _build_greedy(n_rows, k)
    best = [r[:] for r in rows]
    best_val = _max_entry(best)

    # Maintain global used set incrementally
    cur = [r[:] for r in best]
    cur_val = best_val
    row_diffs = [_row_diffs(r) for r in cur]
    cur_used = set()
    for rd in row_diffs:
        cur_used |= rd

    T = 50.0
    T_min = 0.01
    cool = 0.9998

    steps = 0
    no_improve = 0
    max_no_improve = 20000

    while time.time() < deadline:
        steps += 1
        
        # Occasionally restart from greedy if stuck
        if no_improve > max_no_improve:
            random.seed(random.randint(0, 100000))
            rows = _build_greedy(n_rows, k)
            cur = [r[:] for r in rows]
            cur_val = _max_entry(cur)
            row_diffs = [_row_diffs(r) for r in cur]
            cur_used = set()
            for rd in row_diffs:
                cur_used |= rd
            no_improve = 0
            T = 50.0
            continue

        # Pick a random row and position to perturb
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)
        old_val = cur[ri][pi]

        # Generate candidate values to try
        candidates = []
        # Small deltas
        for d in range(-15, 16):
            if d == 0:
                continue
            nv = old_val + d
            if nv > 0:
                candidates.append(nv)
        # Larger jumps
        for _ in range(5):
            nv = random.randint(1, max(old_val + 30, 50))
            candidates.append(nv)
        # Try to reduce max: if this is the max position, try smaller values
        if old_val == cur_val:
            for d in range(1, min(old_val - 1, 30)):
                candidates.append(old_val - d)

        random.shuffle(candidates)
        moved = False

        for new_val in candidates:
            if new_val < 1:
                continue
            # Check strict increase within row
            if pi > 0 and new_val <= cur[ri][pi - 1]:
                continue
            if pi < k and new_val >= cur[ri][pi + 1]:
                continue

            # Build candidate row
            cand_row = cur[ri][:]
            cand_row[pi] = new_val

            # Check if row is strictly increasing
            if any(cand_row[j] <= cand_row[j-1] for j in range(1, len(cand_row))):
                continue

            # Compute new diffs for this row
            new_diffs = _row_diffs(cand_row)
            old_diffs = row_diffs[ri]

            # Check for collisions: new_diffs must not intersect (cur_used - old_diffs)
            other_used = cur_used - old_diffs
            if new_diffs & other_used:
                continue

            # Valid! Compute new max
            cand_max = max(max(cur[r]) for r in range(n_rows) if r != ri)
            cand_max = max(cand_max, max(cand_row))

            # Acceptance
            if cand_max < cur_val:
                accept = True
            elif cand_max == cur_val:
                accept = random.random() < 0.5
            else:
                delta_e = cand_max - cur_val
                accept = random.random() < (1.0 / (1.0 + T * 0.1)) * 0.05

            if accept:
                # Update state
                cur_used -= old_diffs
                cur_used |= new_diffs
                row_diffs[ri] = new_diffs
                cur[ri] = cand_row
                cur_val = cand_max

                if cur_val < best_val:
                    best_val = cur_val
                    best = [r[:] for r in cur]
                    no_improve = 0
                else:
                    no_improve += 1

                moved = True
                break

        # Cool down
        T = max(T_min, T * cool)

    # Ensure best is valid
    if not _valid(best, n_rows, k):
        best = _build_greedy(n_rows, k)
    return best
# EVOLVE-BLOCK-END
