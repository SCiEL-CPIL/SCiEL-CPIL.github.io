"""Construct difference triangle sets (n_rows, k) minimizing the largest entry.

Structural approach:
  * Pre-compute a catalog of valid Golomb rulers (rows of k+1 entries starting at 0)
    up to a scope bound.
  * Select n_rows mutually non-colliding rulers via weighted greedy independent set
    on the conflict graph (edges = overlapping difference sets).
  * Local refinement: shrink individual rows via targeted rebuilds.
"""
import time
import random

_DEADLINE = None
_TIME_LIMIT = 145.0  # safety margin before 150s kill


def _time_left():
    if _DEADLINE is None:
        return _TIME_LIMIT
    return max(0.0, _DEADLINE - time.time())


def _validate(rows, n_rows, k):
    if len(rows) != n_rows:
        return False
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        prev = -1
        for v in row:
            if v <= prev:
                return False
            prev = v
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True


def _scope(rows):
    return max(r[-1] for r in rows)


def _row_diffs(row):
    return {row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))}


def _all_diffs_except(rows, idx):
    used = set()
    for i, row in enumerate(rows):
        if i == idx:
            continue
        used |= _row_diffs(row)
    return used


def _construct_row_partial(k, used, max_val, rng, mode):
    """Build one full row (k+1 entries, starts at 0, increasing, all <= max_val)
    with no difference in `used`. Returns the row or None on failure."""
    row = [0]
    target = k + 1
    while len(row) < target:
        last = row[-1]
        candidates = []
        for cand in range(last + 1, max_val + 1):
            nd = {cand - a for a in row}
            if not (nd & used):
                candidates.append(cand)
        if not candidates:
            return None
        if mode == "small":
            x = candidates[0]
        elif mode == "flex":
            best = candidates[0]
            best_score = -1
            for cand in candidates[:12]:
                nd = {cand - a for a in row}
                cnt = 0
                for nxt in range(cand + 1, min(cand + 20, max_val) + 1):
                    nd2 = {nxt - a for a in row}
                    nd2.add(nxt - cand)
                    if not (nd2 & (used | nd)):
                        cnt += 1
                        if cnt >= 4:
                            break
                score = cnt * 100 - cand
                if score > best_score:
                    best_score = score
                    best = cand
            x = best
        else:  # "rand"
            pool = candidates[:min(8, len(candidates))]
            x = rng.choice(pool)
        row.append(x)
        used |= {x - a for a in row[:-1]}
    return row


def _generate_ruler_catalog(k, max_scope, rng, max_rules=5000):
    """Generate a diverse catalog of valid Golomb rulers with k+1 entries
    starting at 0, all entries <= max_scope. Returns list of (row, diff_set)."""
    catalog = []
    seen_diffs = set()
    
    # Generate rulers using different strategies
    modes = ["small", "flex", "rand"]
    per_mode = max_rules // len(modes)
    
    for mode in modes:
        count = 0
        attempts = 0
        while count < per_mode and attempts < per_mode * 10:
            attempts += 1
            if _time_left() < 5.0:
                break
            row = _construct_row_partial(k, set(), max_scope, rng, mode)
            if row is None:
                continue
            diffs = _row_diffs(row)
            # Use a hash of the diff set to avoid exact duplicates
            diff_key = frozenset(diffs)
            if diff_key in seen_diffs:
                continue
            seen_diffs.add(diff_key)
            catalog.append((row, diffs))
            count += 1
    
    return catalog


def _greedy_independent_set(catalog, n_rows, rng, weight_func=None):
    """Select n_rows mutually non-colliding rulers from catalog via weighted
    greedy independent set. weight_func(row, idx) -> priority (higher = pick first).
    Returns list of selected rows or None if cannot select n_rows."""
    if len(catalog) < n_rows:
        return None
    
    if weight_func is None:
        def weight_func(row, idx):
            return -row[-1]  # prefer smaller scope
    
    # Build conflict: two rulers conflict if their diff sets overlap
    selected = []
    selected_diffs = set()
    available = list(range(len(catalog)))
    
    # Randomize order for diversity
    rng.shuffle(available)
    
    # Sort by weight (descending priority)
    available.sort(key=lambda idx: weight_func(catalog[idx][0], idx), reverse=True)
    
    for idx in available:
        if len(selected) >= n_rows:
            break
        row, diffs = catalog[idx]
        if diffs & selected_diffs:
            continue
        selected.append(row)
        selected_diffs |= diffs
    
    if len(selected) < n_rows:
        return None
    return selected


def _greedy_independent_set_randomized(catalog, n_rows, rng, num_randoms=5):
    """Run multiple randomized greedy independent set passes, return best."""
    best = None
    best_scope = float('inf')
    
    for trial in range(num_randoms):
        if _time_left() < 3.0:
            break
        # Random weight: mix of scope preference and randomness
        def weight_func(row, idx, r=rng, t=trial):
            base = -row[-1]
            noise = r.uniform(-2, 2)
            return base + noise
        
        result = _greedy_independent_set(catalog, n_rows, rng, weight_func)
        if result is not None:
            s = _scope(result)
            if s < best_scope:
                best_scope = s
                best = result
    
    return best


def _iterative_tightening(rows, n_rows, k, rng, max_rounds=40):
    """Repeatedly rebuild each row under a tighter bound than its current scope."""
    rows = [list(r) for r in rows]
    for _ in range(max_rounds):
        if _time_left() < 2.0:
            break
        improved = False
        order = sorted(range(n_rows), key=lambda i: -rows[i][-1])
        for i in order:
            if _time_left() < 2.0:
                break
            cur_scope = rows[i][-1]
            for bound in range(cur_scope - 1, max(cur_scope - 12, rows[i][-2]), -1):
                if bound <= rows[i][-2]:
                    break
                other_used = _all_diffs_except(rows, i)
                new_row = _construct_row_partial(k, other_used, bound, rng, "flex")
                if new_row is not None:
                    rows[i] = new_row
                    improved = True
                    break
        if not improved:
            break
    return rows


def _local_search(rows, n_rows, k, rng, max_iters=400):
    """Collision-driven repair: repeatedly try to shrink the largest-scope row
    via rebuild under a tighter bound; if that fails, nudge an interior element
    to free up space, then retry."""
    rows = [list(r) for r in rows]
    for _ in range(max_iters):
        if _time_left() < 2.0:
            break
        i = max(range(n_rows), key=lambda j: rows[j][-1])
        cur_scope = rows[i][-1]
        found = False
        for bound in range(cur_scope - 1, max(cur_scope - 10, rows[i][-2]), -1):
            other_used = _all_diffs_except(rows, i)
            new_row = _construct_row_partial(k, other_used, bound, rng, "flex")
            if new_row is not None:
                rows[i] = new_row
                found = True
                break
        if found:
            continue
        row = rows[i]
        for pos in range(len(row) - 2, 0, -1):
            cur = row[pos]
            for new_val in range(cur - 1, row[pos - 1], -1):
                if new_val >= row[pos + 1]:
                    continue
                other_used = _all_diffs_except(rows, i)
                candidate_row = row[:]
                candidate_row[pos] = new_val
                new_d = _row_diffs(candidate_row)
                if new_d & other_used:
                    continue
                rows[i] = candidate_row
                found = True
                break
            if found:
                break
        if not found:
            break
    return rows


def construct(n_rows, k):
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT

    rng = random.Random(42)

    # Phase 1: Generate a catalog of candidate rulers
    # Start with a generous scope bound, then tighten
    # Estimate initial scope: for k=8, n_rows=4, known best is 134
    # Use a bound that's likely to have enough rulers
    initial_scope = 200  # generous initial bound for catalog generation
    
    catalog = _generate_ruler_catalog(k, initial_scope, rng, max_rules=3000)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 2: Greedy independent set selection from catalog
    if catalog:
        selected = _greedy_independent_set_randomized(catalog, n_rows, rng, num_randoms=10)
        if selected is not None:
            best_rows = selected
            best_scope = _scope(best_rows)
    
    # Phase 3: If catalog approach didn't work well, try direct construction
    if best_rows is None or best_scope > 150:
        # Direct parallel greedy as fallback
        used = set()
        direct_rows = []
        for _ in range(n_rows):
            row = _construct_row_partial(k, used, 10**9, rng, "flex")
            if row is None:
                break
            direct_rows.append(row)
            used |= _row_diffs(row)
        
        if len(direct_rows) == n_rows:
            ds = _scope(direct_rows)
            if ds < best_scope:
                best_scope = ds
                best_rows = direct_rows
    
    # Phase 4: Iterative tightening
    if best_rows is not None and _time_left() > 5.0:
        tightened = _iterative_tightening(best_rows, n_rows, k, rng, max_rounds=30)
        tc = _scope(tightened)
        if tc < best_scope:
            best_scope = tc
            best_rows = tightened
    
    # Phase 5: Local search refinement
    if best_rows is not None and _time_left() > 3.0:
        ls = _local_search(best_rows, n_rows, k, rng, max_iters=500)
        if _scope(ls) < best_scope:
            best_scope = _scope(ls)
            best_rows = ls
    
    # Phase 6: Try to generate a tighter catalog and re-select
    if best_rows is not None and _time_left() > 10.0:
        # Now we know a good scope, generate catalog up to best_scope - 1
        tight_scope = best_scope - 1
        if tight_scope > 50:
            tight_catalog = _generate_ruler_catalog(k, tight_scope, rng, max_rules=2000)
            if tight_catalog:
                tight_selected = _greedy_independent_set_randomized(
                    tight_catalog, n_rows, rng, num_randoms=5)
                if tight_selected is not None:
                    ts = _scope(tight_selected)
                    if ts < best_scope:
                        best_scope = ts
                        best_rows = tight_selected
                        # Tighten further
                        if _time_left() > 5.0:
                            tightened2 = _iterative_tightening(
                                best_rows, n_rows, k, rng, max_rounds=20)
                            tc2 = _scope(tightened2)
                            if tc2 < best_scope:
                                best_scope = tc2
                                best_rows = tightened2
    
    # Final validation and fallback
    if best_rows is None or not _validate(best_rows, n_rows, k):
        # Safe fallback: pure sequential greedy
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = _construct_row_partial(k, used, 10**9, rng, "flex")
            best_rows.append(row)
            used |= _row_diffs(row)
    
    return best_rows
