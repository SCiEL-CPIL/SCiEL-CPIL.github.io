import time
import random
import math

# EVOLVE-BLOCK-START

def _greedy_init(n_rows, k, gap_factor=10):
    """
    Standard greedy initialization: each row takes the smallest next mark with no difference collision.
    """
    rows = []
    used = set()
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    
    return rows

def _row_diffs_set(row):
    """Compute the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _compute_all_used_set(rows):
    """Compute the union of all differences across all rows (as a set)."""
    all_used = set()
    for row in rows:
        all_used |= _row_diffs_set(row)
    return all_used

def _compute_all_used_ba(rows, max_diff):
    """Compute the union of all differences across all rows as a bytearray."""
    ba = bytearray(max_diff + 1)
    for row in rows:
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                d = row[j] - ri
                if d <= max_diff:
                    ba[d] = 1
    return ba

def _global_max(rows):
    """Compute the maximum entry across all rows."""
    return max(max(row) for row in rows)

def _build_static_diffs(row, k, pair_idx):
    """
    For each position pi (1..k), compute the set of differences in the row
    that do NOT involve position pi.
    """
    n = k + 1
    static = [None] * (k + 1)
    for pi in range(1, k + 1):
        s = set()
        for (i, j) in pair_idx:
            if i == pi or j == pi:
                continue
            s.add(row[j] - row[i])
        static[pi] = s
    return static

def _pair_indices(k):
    """Precompute (i, j) pairs for a row of length k+1."""
    n = k + 1
    return [(i, j) for i in range(n) for j in range(i + 1, n)]

def _try_move_fast(current_rows, ri, pi, new_val, static_diffs, row_diffs, all_used_ba,
                   row_maxes, k, max_diff):
    """
    Try to apply a move. Returns (success, new_diffs, cand_row_max, new_global_max).
    """
    row = current_rows[ri]
    if new_val <= row[pi - 1]:
        return False, None, None, None
    if pi < k and new_val >= row[pi + 1]:
        return False, None, None, None

    static_pi = static_diffs[ri][pi]
    old_diffs = row_diffs[ri]

    # Compute the k diffs involving pi and check for internal conflicts
    new_diffs = set(static_pi)
    valid = True
    for i in range(len(row)):
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

    if not valid:
        return False, None, None, None

    # Check for conflicts with other rows via the bytearray
    for d in new_diffs:
        if d in old_diffs:
            continue
        if d <= max_diff and all_used_ba[d]:
            return False, None, None, None

    if pi == k:
        cand_row_max = new_val
    else:
        cand_row_max = row[k]

    old_row_max = row_maxes[ri]
    row_maxes[ri] = cand_row_max
    new_global_max = max(row_maxes)
    row_maxes[ri] = old_row_max

    return True, new_diffs, cand_row_max, new_global_max

def _apply_move_fast(current_rows, ri, pi, new_val, new_diffs, cand_row_max,
                     row_diffs, all_used_ba, row_maxes, static_diffs, k, old_diffs,
                     pair_idx, max_diff):
    """Apply a valid move to the state."""
    for d in old_diffs:
        if d <= max_diff:
            all_used_ba[d] = 0
    for d in new_diffs:
        if d <= max_diff:
            all_used_ba[d] = 1
    current_rows[ri][pi] = new_val
    row_diffs[ri] = new_diffs
    row_maxes[ri] = cand_row_max
    static_diffs[ri] = _build_static_diffs(current_rows[ri], k, pair_idx)
    return new_global_max if False else max(row_maxes)

def _sa_phase(current_rows, k, n_rows, all_used_ba, row_diffs, static_diffs, row_maxes,
              current_val, best_rows, best_val, T, T_min, cool, max_no_improve,
              deadline, pair_idx, max_diff):
    """Run simulated annealing phase with single and multi-position moves."""
    no_improve = 0
    multi_pos_prob = 0.15  # probability of attempting a 2-position move

    while time.time() < deadline and no_improve < max_no_improve:
        # Pick a row and position
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)

        old_val = current_rows[ri][pi]
        old_diffs = row_diffs[ri]

        # Generate candidates
        candidates = []

        # Decreasing candidates
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(60, old_val - min_allowed)
        for d in range(1, max_decrease + 1):
            candidates.append(old_val - d)

        # Increasing candidates
        max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 40
        max_increase = min(20, max_allowed - old_val)
        for d in range(1, max_increase + 1):
            candidates.append(old_val + d)

        # Random jump
        if random.random() < 0.2:
            low = current_rows[ri][pi - 1] + 1
            high = (current_rows[ri][pi + 1] - 1) if pi < k else current_val + 30
            if low <= high:
                candidates.append(random.randint(low, high))

        if not candidates:
            no_improve += 1
            T = max(T_min, T * cool)
            continue

        random.shuffle(candidates)

        moved = False
        for new_val in candidates:
            success, new_diffs, cand_row_max, new_global_max = _try_move_fast(
                current_rows, ri, pi, new_val, static_diffs, row_diffs, all_used_ba,
                row_maxes, k, max_diff)

            if not success:
                continue

            delta = new_global_max - current_val

            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if random.random() < 0.25:
                    accept = True
            else:
                if T > T_min:
                    prob = math.exp(-delta / max(T, 0.1))
                    if random.random() < prob:
                        accept = True

            if accept:
                current_val = _apply_move_fast(
                    current_rows, ri, pi, new_val, new_diffs, cand_row_max,
                    row_diffs, all_used_ba, row_maxes, static_diffs, k, old_diffs,
                    pair_idx, max_diff)

                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                else:
                    no_improve += 1

                moved = True
                break

        if not moved:
            # Try a multi-position move (change 2 positions in the same row)
            if random.random() < multi_pos_prob:
                moved = _try_multi_pos_move(
                    current_rows, ri, pi, k, n_rows, all_used_ba, row_diffs,
                    static_diffs, row_maxes, current_val, best_val, T, T_min,
                    pair_idx, max_diff)
                if moved:
                    no_improve = 0
                    # Update best if improved
                    cv = max(row_maxes)
                    if cv < best_val:
                        best_val = cv
                        best_rows = [r[:] for r in current_rows]
                    current_val = cv
                else:
                    no_improve += 1
            else:
                no_improve += 1

        T = max(T_min, T * cool)

        # Periodic restart if stuck
        if no_improve > 15000:
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            all_used_ba = _compute_all_used_ba(current_rows, max_diff)
            row_diffs = [_row_diffs_set(r) for r in current_rows]
            static_diffs = [_build_static_diffs(r, k, pair_idx) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            T = 200.0
            no_improve = 0

    return current_rows, current_val, all_used_ba, row_diffs, static_diffs, row_maxes, best_rows, best_val


def _try_multi_pos_move(current_rows, ri, pi, k, n_rows, all_used_ba, row_diffs,
                        static_diffs, row_maxes, current_val, best_val, T, T_min,
                        pair_idx, max_diff):
    """
    Attempt to move two positions in the same row simultaneously.
    Returns True if a move was accepted.
    """
    row = current_rows[ri]
    n = k + 1

    # Pick a second position different from pi
    pi2 = random.randrange(1, k + 1)
    if pi2 == pi:
        pi2 = (pi + 1) % k  # shift to next valid position

    old_val1 = row[pi]
    old_val2 = row[pi2]
    old_diffs = row_diffs[ri]

    # Generate small perturbations for each position
    def gen_candidates(pos, old_v):
        cands = []
        lo = row[pos - 1] + 1
        hi = (row[pos + 1] - 1) if pos < k else max_diff
        span = hi - lo + 1
        if span <= 0:
            return cands
        # Decreasing
        for d in range(1, min(15, old_v - lo + 1)):
            cands.append(old_v - d)
        # Increasing
        for d in range(1, min(10, hi - old_v + 1)):
            cands.append(old_v + d)
        return cands

    cands1 = gen_candidates(pi, old_val1)
    cands2 = gen_candidates(pi2, old_val2)

    if not cands1 or not cands2:
        return False

    # Try a few random combinations
    for _ in range(3):
        nv1 = random.choice(cands1)
        nv2 = random.choice(cands2)

        # Check ordering constraints
        if nv1 <= row[pi - 1] or (pi < k and nv1 >= row[pi + 1]):
            continue
        if nv2 <= row[pi2 - 1] or (pi2 < k and nv2 >= row[pi2 + 1]):
            continue
        # Ensure the two new values don't violate relative ordering with each other
        if pi < pi2 and nv1 >= nv2:
            continue
        if pi2 < pi and nv2 >= nv1:
            continue

        # Temporarily apply and check validity
        saved_row = row[:]
        saved_diffs = old_diffs

        # Build new row
        row[pi] = nv1
        row[pi2] = nv2

        # Check all diffs in this row are valid (no internal conflicts)
        new_diffs = set()
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            if not valid:
                break

        if not valid:
            row[:] = saved_row
            continue

        # Check against other rows
        conflict = False
        for d in new_diffs:
            if d in old_diffs:
                continue
            if d <= max_diff and all_used_ba[d]:
                conflict = True
                break

        if conflict:
            row[:] = saved_row
            continue

        # Compute new global max
        new_row_max = max(row)
        old_row_max = row_maxes[ri]
        row_maxes[ri] = new_row_max
        new_global_max = max(row_maxes)
        row_maxes[ri] = old_row_max

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
            # Actually apply the move
            for d in old_diffs:
                if d <= max_diff:
                    all_used_ba[d] = 0
            for d in new_diffs:
                if d <= max_diff:
                    all_used_ba[d] = 1
            row_maxes[ri] = new_row_max
            row_diffs[ri] = new_diffs
            static_diffs[ri] = _build_static_diffs(row, k, pair_idx)
            return True
        else:
            row[:] = saved_row

    return False

def _polish_phase(current_rows, k, n_rows, all_used_ba, row_diffs, static_diffs, row_maxes,
                  current_val, best_rows, best_val, deadline, pair_idx, max_diff):
    """Final polishing with only decreasing moves and tight tolerance."""
    T = 1.0
    T_min = 0.05
    cool = 0.99998
    no_improve = 0
    max_no_improve_polish = 15000

    while time.time() < deadline and no_improve < max_no_improve_polish:
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)

        old_val = current_rows[ri][pi]
        old_diffs = row_diffs[ri]

        # Only decreasing moves in polish
        min_allowed = current_rows[ri][pi - 1] + 1
        max_decrease = min(25, old_val - min_allowed)

        candidates = []
        for d in range(1, max_decrease + 1):
            candidates.append(old_val - d)

        if not candidates:
            no_improve += 1
            T = max(T_min, T * cool)
            continue

        random.shuffle(candidates)

        moved = False
        for new_val in candidates:
            success, new_diffs, cand_row_max, new_global_max = _try_move_fast(
                current_rows, ri, pi, new_val, static_diffs, row_diffs, all_used_ba,
                row_maxes, k, max_diff)

            if not success:
                continue

            delta = new_global_max - current_val

            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if random.random() < 0.15:
                    accept = True
            else:
                if T > T_min:
                    prob = math.exp(-delta / max(T, 0.03))
                    if random.random() < prob:
                        accept = True

            if accept:
                current_val = _apply_move_fast(
                    current_rows, ri, pi, new_val, new_diffs, cand_row_max,
                    row_diffs, all_used_ba, row_maxes, static_diffs, k, old_diffs,
                    pair_idx, max_diff)

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

    return best_rows, best_val

def _exact_descent_polish(current_rows, k, n_rows, all_used_ba, row_diffs, static_diffs,
                          row_maxes, current_val, best_rows, best_val, deadline,
                          pair_idx, max_diff):
    """
    Deterministic local solve: repeatedly scan every (row, position) and apply the
    first improving move found. Continue until a full pass makes no
    change or the deadline is reached. This is a tighter, exact local optimum
    search that complements the stochastic polish.
    """
    progress = True
    while progress and time.time() < deadline:
        progress = False
        for ri in range(n_rows):
            if time.time() >= deadline:
                break
            for pi in range(1, k + 1):
                if time.time() >= deadline:
                    break
                row = current_rows[ri]
                old_val = row[pi]
                old_diffs = row_diffs[ri]

                min_allowed = row[pi - 1] + 1
                max_allowed = (row[pi + 1] - 1) if pi < k else max_diff
                # Try decreasing values first (most likely to improve the max)
                for new_val in range(min_allowed, old_val):
                    success, new_diffs, cand_row_max, new_global_max = _try_move_fast(
                        current_rows, ri, pi, new_val, static_diffs, row_diffs,
                        all_used_ba, row_maxes, k, max_diff)
                    if not success:
                        continue
                    if new_global_max < current_val:
                        current_val = _apply_move_fast(
                            current_rows, ri, pi, new_val, new_diffs, cand_row_max,
                            row_diffs, all_used_ba, row_maxes, static_diffs, k,
                            old_diffs, pair_idx, max_diff)
                        if current_val < best_val:
                            best_val = current_val
                            best_rows = [r[:] for r in current_rows]
                        progress = True
                        break
                # Also try a few increasing values (may free up space for other rows)
                for new_val in range(old_val + 1, min(old_val + 15, max_allowed + 1)):
                    success, new_diffs, cand_row_max, new_global_max = _try_move_fast(
                        current_rows, ri, pi, new_val, static_diffs, row_diffs,
                        all_used_ba, row_maxes, k, max_diff)
                    if not success:
                        continue
                    if new_global_max < current_val:
                        current_val = _apply_move_fast(
                            current_rows, ri, pi, new_val, new_diffs, cand_row_max,
                            row_diffs, all_used_ba, row_maxes, static_diffs, k,
                            old_diffs, pair_idx, max_diff)
                        if current_val < best_val:
                            best_val = current_val
                            best_rows = [r[:] for r in current_rows]
                        progress = True
                        break

    return best_rows, best_val

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    # 1. Initialization
    rows = _greedy_init(n_rows, k)

    # Track used differences
    all_used_set = _compute_all_used_set(rows)

    best_rows = [r[:] for r in rows]
    best_val = _global_max(rows)

    row_maxes = [max(r) for r in rows]

    # Precompute pair indices and a reasonable max_diff bound for the bytearray.
    # The greedy init gives an upper bound; we allow some headroom for increases.
    greedy_max = best_val
    max_diff = greedy_max + 200  # headroom for random jumps / increases

    pair_idx = _pair_indices(k)

    # 2. Simulated Annealing - Phase 1 (exploration)
    current_rows = [r[:] for r in rows]
    current_val = best_val

    row_diffs = [_row_diffs_set(r) for r in current_rows]
    static_diffs = [_build_static_diffs(r, k, pair_idx) for r in current_rows]
    all_used_ba = _compute_all_used_ba(current_rows, max_diff)

    phase1_deadline = start_time + 55.0
    current_rows, current_val, all_used_ba, row_diffs, static_diffs, row_maxes, best_rows, best_val = _sa_phase(
        current_rows, k, n_rows, all_used_ba, row_diffs, static_diffs, row_maxes,
        current_val, best_rows, best_val, 250.0, 0.5, 0.99994, 30000, phase1_deadline,
        pair_idx, max_diff)

    # 3. Simulated Annealing - Phase 2 (exploitation with lower temp)
    if time.time() < deadline - 25:
        current_rows = [r[:] for r in best_rows]
        current_val = best_val
        all_used_ba = _compute_all_used_ba(current_rows, max_diff)
        row_diffs = [_row_diffs_set(r) for r in current_rows]
        static_diffs = [_build_static_diffs(r, k, pair_idx) for r in current_rows]
        row_maxes = [max(r) for r in current_rows]

        phase2_deadline = start_time + 90.0
        current_rows, current_val, all_used_ba, row_diffs, static_diffs, row_maxes, best_rows, best_val = _sa_phase(
            current_rows, k, n_rows, all_used_ba, row_diffs, static_diffs, row_maxes,
            current_val, best_rows, best_val, 50.0, 0.3, 0.99997, 20000, phase2_deadline,
            pair_idx, max_diff)

    # 4. Stochastic polish phase
    if time.time() < deadline - 15:
        current_rows = [r[:] for r in best_rows]
        current_val = best_val
        all_used_ba = _compute_all_used_ba(current_rows, max_diff)
        row_diffs = [_row_diffs_set(r) for r in current_rows]
        static_diffs = [_build_static_diffs(r, k, pair_idx) for r in current_rows]
        row_maxes = [max(r) for r in current_rows]

        polish_deadline = deadline - 8.0
        best_rows, best_val = _polish_phase(
            current_rows, k, n_rows, all_used_ba, row_diffs, static_diffs, row_maxes,
            current_val, best_rows, best_val, polish_deadline, pair_idx, max_diff)

    # 5. Exact descent polish (deterministic local solve) - extended budget
    if time.time() < deadline - 3.0:
        current_rows = [r[:] for r in best_rows]
        current_val = best_val
        all_used_ba = _compute_all_used_ba(current_rows, max_diff)
        row_diffs = [_row_diffs_set(r) for r in current_rows]
        static_diffs = [_build_static_diffs(r, k, pair_idx) for r in current_rows]
        row_maxes = [max(r) for r in current_rows]

        exact_deadline = deadline - 1.0
        best_rows, best_val = _exact_descent_polish(
            current_rows, k, n_rows, all_used_ba, row_diffs, static_diffs, row_maxes,
            current_val, best_rows, best_val, exact_deadline, pair_idx, max_diff)

    # Final validation and return
    return best_rows

# EVOLVE-BLOCK-END
