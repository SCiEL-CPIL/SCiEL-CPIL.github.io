"""Structural rewrite: simulated-annealing stochastic local search with row re-solve, multi-position moves, and parallel restarts."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _valid(rows, n_rows, k):
    """Check all rows are valid and all differences are globally distinct."""
    if len(rows) != n_rows:
        return False
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _max_entry(rows):
    return max(max(r) for r in rows)

def _row_diffs_set(row):
    s = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s

def _is_sidon(row):
    """Check if all pairwise differences in the row are distinct."""
    n = len(row)
    seen = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = row[j] - row[i]
            if d in seen:
                return False
            seen.add(d)
    return True

def _greedy_row_with_update(used, k, max_val=None, rng=None):
    """Build a single row greedily (optionally randomized), updating used as we go.
    If rng is None, deterministic smallest-first. Otherwise, pick from a small
    window of valid candidates with bias toward smaller values."""
    row = [0]
    x = 0
    limit = max_val if max_val is not None else 10000
    while len(row) < k + 1:
        if rng is None:
            x += 1
            if x > limit:
                return None, used
            new_diffs = [x - a for a in row]
            if any(d in used for d in new_diffs):
                continue
            if len(new_diffs) != len(set(new_diffs)):
                continue
            row.append(x)
            for d in new_diffs:
                used.add(d)
        else:
            # Find a small window of valid candidates starting from x+1
            valid_xs = []
            scan = x + 1
            while scan <= min(x + 40, limit) and len(valid_xs) < 6:
                new_diffs = [scan - a for a in row]
                if not any(d in used for d in new_diffs) and len(new_diffs) == len(set(new_diffs)):
                    valid_xs.append(scan)
                scan += 1
            if not valid_xs:
                # Fallback: scan further
                scan = x + 1
                while scan <= limit:
                    new_diffs = [scan - a for a in row]
                    if not any(d in used for d in new_diffs) and len(new_diffs) == len(set(new_diffs)):
                        valid_xs.append(scan)
                        break
                    scan += 1
                if not valid_xs:
                    return None, used
            # Weighted pick biased toward smaller
            weights = [1.0 / (i + 1) for i in range(len(valid_xs))]
            total = sum(weights)
            r = rng.random() * total
            cum = 0.0
            chosen = valid_xs[-1]
            for v, w in zip(valid_xs, weights):
                cum += w
                if r <= cum:
                    chosen = v
                    break
            row.append(chosen)
            for a in row[:-1]:
                used.add(chosen - a)
            x = chosen
    return row, used

def _build_greedy(n_rows, k, rng=None):
    """Fast greedy: each row takes smallest (or random-window) next mark with no diff collision."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row, used = _greedy_row_with_update(used, k, rng=rng)
        if row is None:
            # Deterministic fallback with no limit
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = [x - a for a in row]
                if not any(d in used for d in new) and len(new) == len(set(new)):
                    row.append(x)
                    used |= set(new)
        rows.append(row)
    return rows

def _resolve_row_given_others(ri, rows, k, max_target, rng=None):
    """Rebuild row ri greedily, avoiding only other rows' diffs.
    Returns new row (max <= max_target) or None."""
    other_used = set()
    for i, r in enumerate(rows):
        if i == ri:
            continue
        other_used |= _row_diffs_set(r)
    # Build fresh with a fresh used that only contains other rows' diffs
    fresh_used = set(other_used)
    row, _ = _greedy_row_with_update(fresh_used, k, max_val=max_target, rng=rng)
    return row

def _sa_move_single_pos(cur, cur_used, ri, pi, rng, k):
    """Try a single-position move on row ri at index pi.
    Returns (new_row, delta) where delta = new_max - old_max (can be negative, 0, or positive).
    If no valid move found, returns (None, 0)."""
    row = cur[ri]
    old_val = row[pi]
    lo = row[pi - 1] + 1
    hi = row[pi + 1] - 1 if pi < k else 10**6
    if hi < lo:
        return None, 0
    # Candidate range: within [lo, hi], but also allow some expansion
    span = max(1, hi - lo + 1)
    # Try a few random offsets
    candidates = []
    for _ in range(8):
        off = rng.randint(-span, span)
        nv = old_val + off
        if lo <= nv <= hi:
            candidates.append(nv)
    # Also try a few deterministic near moves
    for d in (1, -1, 2, -2, 3):
        nv = old_val + d
        if lo <= nv <= hi:
            candidates.append(nv)
    if not candidates:
        return None, 0
    # Pick a random candidate
    nv = rng.choice(candidates)
    if nv == old_val:
        return None, 0
    cand_row = row[:]
    cand_row[pi] = nv
    if not _is_sidon(cand_row):
        return None, 0
    new_diffs = _row_diffs_set(cand_row)
    old_diffs = _row_diffs_set(row)
    other_diffs = cur_used - old_diffs
    if new_diffs & other_diffs:
        return None, 0
    # Compute delta in objective (max entry)
    old_max = _max_entry(cur)
    new_max = max(max(cand_row), max(max(r) for i, r in enumerate(cur) if i != ri))
    delta = new_max - old_max
    return cand_row, delta

def _sa_move_row_swap(cur, cur_used, ri, rj, rng, k):
    """Swap the contents of rows ri and rj (as sets of differences).
    This is valid iff the union of differences is unchanged (which it is, since we
    just permute which row owns which differences). The max entry may change."""
    if ri == rj:
        return None, 0
    # Swap the rows
    old_ri = cur[ri][:]
    old_rj = cur[rj][:]
    new_ri = old_rj[:]
    new_rj = old_ri[:]
    # The used set is unchanged (same multiset of differences)
    # Check validity: each row must still be increasing and start at 0 (they are, since we swap whole rows)
    # Compute delta
    old_max = _max_entry(cur)
    new_max = max(max(new_ri), max(new_rj), max(max(r) for i, r in enumerate(cur) if i != ri and i != rj))
    delta = new_max - old_max
    return (new_ri, new_rj), delta

def _sa_move_resolve_row(cur, cur_used, ri, rng, k):
    """Re-resolve row ri from scratch, avoiding other rows' diffs.
    Returns (new_row, delta) or (None, 0)."""
    other_used = set()
    for i, r in enumerate(cur):
        if i == ri:
            continue
        other_used |= _row_diffs_set(r)
    fresh_used = set(other_used)
    # Try a randomized greedy with a modest target
    old_max = max(cur[ri])
    target = old_max + rng.randint(-10, 10)
    row, _ = _greedy_row_with_update(fresh_used, k, max_val=target, rng=rng)
    if row is None:
        # Try without limit
        row, _ = _greedy_row_with_update(fresh_used, k, rng=rng)
    if row is None:
        return None, 0
    new_diffs = _row_diffs_set(row)
    if new_diffs & other_used:
        return None, 0
    old_max_full = _max_entry(cur)
    new_max = max(max(row), max(max(r) for i, r in enumerate(cur) if i != ri))
    delta = new_max - old_max_full
    return row, delta

def _sa_run(n_rows, k, initial_rows, time_budget, seed):
    """Run a single SA trajectory starting from initial_rows.
    Returns the best valid rows seen."""
    rng = random.Random(seed)
    cur = [r[:] for r in initial_rows]
    cur_used = set()
    for r in cur:
        cur_used |= _row_diffs_set(r)
    cur_val = _max_entry(cur)
    best = [r[:] for r in cur]
    best_val = cur_val

    # SA schedule
    T0 = 12.0
    T_end = 0.05
    # We'll run until time_budget expires, with T decaying geometrically
    # Estimate steps per second roughly; we'll just loop and check time
    T = T0
    decay = 0.9995  # per-step decay factor (tuned for a few thousand steps)
    min_T = T_end

    steps = 0
    t0 = time.time()
    while time.time() - t0 < time_budget:
        steps += 1
        # Choose a move type
        r = rng.random()
        if r < 0.45:
            # Single position move
            ri = rng.randrange(n_rows)
            pi = rng.randrange(1, k + 1)
            new_row, delta = _sa_move_single_pos(cur, cur_used, ri, pi, rng, k)
            if new_row is None:
                T = max(min_T, T * decay)
                continue
            # Accept or reject
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                old_diffs = _row_diffs_set(cur[ri])
                new_diffs = _row_diffs_set(new_row)
                cur_used -= old_diffs
                cur_used |= new_diffs
                cur[ri] = new_row
                cur_val += delta
                if cur_val < best_val:
                    best_val = cur_val
                    best = [r2[:] for r2 in cur]
        elif r < 0.70:
            # Row swap
            if n_rows >= 2:
                ri = rng.randrange(n_rows)
                rj = rng.randrange(n_rows)
                if ri != rj:
                    result, delta = _sa_move_row_swap(cur, cur_used, ri, rj, rng, k)
                    if result is not None:
                        if delta <= 0 or rng.random() < math.exp(-delta / T):
                            cur[ri], cur[rj] = result
                            cur_val += delta
                            if cur_val < best_val:
                                best_val = cur_val
                                best = [r2[:] for r2 in cur]
        elif r < 0.85:
            # Re-resolve a row
            ri = rng.randrange(n_rows)
            new_row, delta = _sa_move_resolve_row(cur, cur_used, ri, rng, k)
            if new_row is not None:
                if delta <= 0 or rng.random() < math.exp(-delta / T):
                    old_diffs = _row_diffs_set(cur[ri])
                    new_diffs = _row_diffs_set(new_row)
                    cur_used -= old_diffs
                    cur_used |= new_diffs
                    cur[ri] = new_row
                    cur_val += delta
                    if cur_val < best_val:
                        best_val = cur_val
                        best = [r2[:] for r2 in cur]
        else:
            # Kick: random restart from best with a small perturbation
            cur = [r2[:] for r2 in best]
            cur_used = set()
            for r2 in cur:
                cur_used |= _row_diffs_set(r2)
            cur_val = best_val
            # Perturb a few positions
            for _ in range(rng.randint(1, 3)):
                ri = rng.randrange(n_rows)
                pi = rng.randrange(1, k + 1)
                row = cur[ri]
                lo = row[pi - 1] + 1
                hi = row[pi + 1] - 1 if pi < k else 10**6
                if hi >= lo:
                    nv = rng.randint(lo, hi)
                    cand_row = row[:]
                    cand_row[pi] = nv
                    if _is_sidon(cand_row):
                        new_diffs = _row_diffs_set(cand_row)
                        old_diffs = _row_diffs_set(row)
                        other_diffs = cur_used - old_diffs
                        if not (new_diffs & other_diffs):
                            cur_used -= old_diffs
                            cur_used |= new_diffs
                            cur[ri] = cand_row
                            cur_val = _max_entry(cur)

        T = max(min_T, T * decay)
        # Periodic re-anchoring if we've drifted too far
        if steps % 2000 == 0 and cur_val > best_val + 15:
            cur = [r2[:] for r2 in best]
            cur_used = set()
            for r2 in cur:
                cur_used |= _row_diffs_set(r2)
            cur_val = best_val

    return best, best_val

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    # Phase 1: diverse initializations
    inits = []
    inits.append(_build_greedy(n_rows, k, rng=None))
    for s in range(8):
        inits.append(_build_greedy(n_rows, k, rng=random.Random(1000 + s)))
    # Pick the best few as starting points for SA
    inits.sort(key=_max_entry)
    top_inits = inits[:4]

    # Phase 2: run multiple SA trajectories in parallel (sequentially, with time budget split)
    elapsed = time.time() - start_time
    remaining = deadline - time.time()
    n_sa = 4
    budget_per_sa = max(5.0, remaining / n_sa - 1.0)

    overall_best = None
    overall_best_val = float('inf')
    for i, init in enumerate(top_inits):
        if time.time() >= deadline - 2:
            break
        b, bv = _sa_run(n_rows, k, init, budget_per_sa, seed=42 + i * 7)
        if bv < overall_best_val:
            overall_best_val = bv
            overall_best = b

    # Phase 3: final polish with a short SA from the overall best
    if overall_best is None:
        overall_best = inits[0]
        overall_best_val = _max_entry(overall_best)
    elif time.time() < deadline - 5:
        b, bv = _sa_run(n_rows, k, overall_best, max(3.0, deadline - time.time() - 2), seed=999)
        if bv < overall_best_val:
            overall_best_val = bv
            overall_best = b

    # Final validation
    if not _valid(overall_best, n_rows, k):
        overall_best = _build_greedy(n_rows, k, rng=None)

    return overall_best
# EVOLVE-BLOCK-END
