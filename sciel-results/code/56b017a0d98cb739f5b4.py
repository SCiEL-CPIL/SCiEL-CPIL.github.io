"""Difference triangle set: DLX exact-cover with branch-and-bound on scope."""
import random
import time

DEADLINE = 145.0  # seconds, safety margin before 150s kill


def _row_diffs_set(row):
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _scope(rows):
    return max(row[-1] for row in rows)


def _is_valid(rows, n_rows, k):
    if len(rows) != n_rows:
        return False
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True


def _greedy_bitmask(n_rows, k, scope):
    """Fast greedy: build rows one by one using smallest available differences."""
    used_mask = 0  # bitmask of used differences (scope < 64 assumed; else use set)
    use_set = scope >= 63
    used_set = set() if use_set else None
    rows = []
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            x = 1
            while True:
                new_diffs = [x - a for a in row]
                ok = True
                for d in new_diffs:
                    if d > scope:
                        ok = False
                        break
                    if use_set:
                        if d in used_set:
                            ok = False
                            break
                    else:
                        if (used_mask >> d) & 1:
                            ok = False
                            break
                if ok:
                    row.append(x)
                    for d in new_diffs:
                        if use_set:
                            used_set.add(d)
                        else:
                            used_mask |= (1 << d)
                    break
                x += 1
                if x > scope:
                    return None
        rows.append(row)
    return rows


def _generate_row_candidates(scope, k, limit=None):
    """Generate all valid rows (k+1 increasing ints starting at 0, max <= scope)
    with all pairwise differences distinct within the row.
    Returns list of (row_tuple, diff_mask, diff_set)."""
    candidates = []
    n = k + 1

    def backtrack(row, start, diff_set):
        if len(row) == n:
            mask = 0
            for d in diff_set:
                mask |= (1 << d)
            candidates.append((tuple(row), mask, diff_set))
            return
        # If we hit limit, stop
        if limit and len(candidates) >= limit:
            return
        # Choose next value
        # Lower bound: must be > row[-1]
        # Upper bound: scope
        lo = row[-1] + 1
        hi = scope
        # Pruning: remaining slots need at least (n - len(row)) - 1 more distinct values
        # so hi must be at least lo + (n - len(row) - 1)
        if hi < lo + (n - len(row) - 1):
            return
        for x in range(lo, hi + 1):
            new_diffs = [x - a for a in row]
            ok = True
            for d in new_diffs:
                if d in diff_set:
                    ok = False
                    break
            if not ok:
                continue
            new_diff_set = diff_set | set(new_diffs)
            row.append(x)
            backtrack(row, x, new_diff_set)
            row.pop()
            if limit and len(candidates) >= limit:
                return

    backtrack([0], 0, set())
    return candidates


def _dlx_solve(n_rows, k, scope, candidates, time_limit, start_time):
    """DLX exact cover: choose n_rows candidates (one per row) such that no diff
    is shared. Each candidate is a set of diff-values. We need a matching where
    no diff appears in more than one chosen candidate.

    This is a set packing / exact cover variant. We model it as:
    - Primary columns: diff values 1..scope (each must be covered at most once)
    - Secondary constraint: exactly n_rows candidates chosen.

    Actually, since rows are identical (any row can be any candidate), we just need
    to select n_rows candidates with disjoint diff-sets.

    We use backtracking with MRV heuristic on the diff-values.
    """
    # Precompute: for each diff d, list of candidate indices that contain d
    diff_to_cands = {}
    for ci, (row, mask, dset) in enumerate(candidates):
        for d in dset:
            if d not in diff_to_cands:
                diff_to_cands[d] = []
            diff_to_cands[d].append(ci)

    # Sort candidates by size of diff-set (smaller first for efficiency)
    # Actually, we want to pick candidates that "use up" rare diffs first.
    # Let's just do backtracking with MRV on the remaining required rows.

    best_found = None
    best_scope_found = scope

    # State: chosen_cands (list of candidate indices), used_diffs (set)
    def backtrack(chosen, used_diffs):
        nonlocal best_found, best_scope_found
        if time.time() - start_time > time_limit:
            return False
        if len(chosen) == n_rows:
            # All rows chosen, all diffs distinct
            rows = [list(candidates[ci][0]) for ci in chosen]
            s = _scope(rows)
            if s < best_scope_found:
                best_scope_found = s
                best_found = rows
            return True  # Found a valid solution, stop (we want minimal, but DLX finds first)

        # MRV: pick the diff-value with fewest available candidates? 
        # Actually, we need to pick a candidate. Let's pick the candidate with
        # the rarest diff (smallest candidate count among its diffs).
        
        # For efficiency, just try candidates in order, skipping those with conflicts
        # Better: pick the "most constrained" next candidate.
        
        # Simple approach: iterate candidates, skip if any diff in used_diffs
        # To improve, sort by minimum frequency of its diffs
        min_freq = scope + 1
        best_ci = -1
        for ci, (row, mask, dset) in enumerate(candidates):
            if any(d in used_diffs for d in dset):
                continue
            # Compute min frequency
            for d in dset:
                freq = len(diff_to_cands.get(d, []))
                if freq < min_freq:
                    min_freq = freq
                    best_ci = ci
                    break
        if best_ci == -1:
            # Try all valid candidates
            for ci, (row, mask, dset) in enumerate(candidates):
                if any(d in used_diffs for d in dset):
                    continue
                chosen.append(ci)
                new_used = used_diffs | set(dset)
                if backtrack(chosen, new_used):
                    return True
                chosen.pop()
                if time.time() - start_time > time_limit:
                    return False
            return False
        else:
            chosen.append(best_ci)
            dset = candidates[best_ci][2]
            new_used = used_diffs | set(dset)
            if backtrack(chosen, new_used):
                return True
            chosen.pop()
            if time.time() - start_time > time_limit:
                return False
            # Try next best
            min_freq2 = scope + 1
            best_ci2 = -1
            for ci, (row, mask, dset) in enumerate(candidates):
                if ci == best_ci:
                    continue
                if any(d in used_diffs for d in dset):
                    continue
                for d in dset:
                    freq = len(diff_to_cands.get(d, []))
                    if freq < min_freq2:
                        min_freq2 = freq
                        best_ci2 = ci
                        break
            if best_ci2 != -1:
                chosen.append(best_ci2)
                dset2 = candidates[best_ci2][2]
                new_used2 = used_diffs | set(dset2)
                if backtrack(chosen, new_used2):
                    return True
                chosen.pop()
            # Continue with remaining candidates
            for ci, (row, mask, dset) in enumerate(candidates):
                if ci in (best_ci, best_ci2):
                    continue
                if any(d in used_diffs for d in dset):
                    continue
                chosen.append(ci)
                new_used = used_diffs | set(dset)
                if backtrack(chosen, new_used):
                    return True
                chosen.pop()
                if time.time() - start_time > time_limit:
                    return False
            return False

    backtrack([], set())
    return best_found


def _local_search_refine(rows, n_rows, k, scope, time_limit, start_time):
    """Local search to reduce scope further."""
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)

    while time.time() - start_time < time_limit:
        improved = False
        for ri in range(n_rows):
            for ei in range(1, k + 1):
                old = rows[ri][ei]
                # Try to decrease the entry
                for new in range(old - 1, 0, -1):
                    if new <= rows[ri][ei - 1]:
                        break
                    if ei < k and new >= rows[ri][ei + 1]:
                        break
                    rows[ri][ei] = new
                    if _is_valid(rows, n_rows, k):
                        s = _scope(rows)
                        if s < best_scope:
                            best_scope = s
                            best_rows = [list(r) for r in rows]
                            improved = True
                        else:
                            rows[ri][ei] = old
                        break
                    rows[ri][ei] = old
        if not improved:
            break
    return best_rows, best_scope


def construct(n_rows, k):
    start_time = time.time()
    deadline = DEADLINE - start_time

    # Phase 1: Greedy initialization
    rows_greedy = _greedy_bitmask(n_rows, k, 200)
    if rows_greedy is None:
        # Fallback: simple greedy with larger scope
        used = set()
        rows_greedy = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows_greedy.append(row)
    best_rows = rows_greedy
    best_score = _scope(best_rows)

    # Phase 2: Try to improve with local search
    if time.time() - start_time < deadline * 0.3:
        rows_ls, score_ls = _local_search_refine(
            best_rows, n_rows, k, best_score, deadline * 0.25, start_time)
        if score_ls < best_score:
            best_score = score_ls
            best_rows = rows_ls

    # Phase 3: DLX exact cover with iterative deepening on scope
    # Try scopes from best_score - 1 down to the counting bound
    counting_bound = n_rows * (k + 1) * k // 2  # total diffs needed
    # Actually, the counting bound is ceil(sqrt(2 * total_diffs)) or similar
    # For n rows of k+1 entries, total diffs = n * k*(k+1)/2
    total_diffs = n_rows * k * (k + 1) // 2
    # Scope must be at least such that 1+2+...+scope >= total_diffs
    # scope*(scope+1)/2 >= total_diffs
    min_scope = int((2 * total_diffs) ** 0.5)
    if min_scope * (min_scope + 1) // 2 < total_diffs:
        min_scope += 1

    # Try to find a solution with scope < best_score using DLX
    if time.time() - start_time < deadline * 0.8:
        dlx_time = deadline * 0.7 - (time.time() - start_time)
        if dlx_time > 5:
            # Generate candidates for scope = best_score - 1
            target_scope = best_score - 1
            if target_scope >= min_scope:
                # Limit candidates to avoid explosion
                limit = 5000
                cands = _generate_row_candidates(target_scope, k, limit)
                if len(cands) >= n_rows:
                    result = _dlx_solve(n_rows, k, target_scope, cands, dlx_time, start_time)
                    if result is not None:
                        s = _scope(result)
                        if s < best_score:
                            best_score = s
                            best_rows = result

    # Phase 4: Final local search polish
    if time.time() - start_time < deadline * 0.95:
        polish_time = deadline * 0.9 - (time.time() - start_time)
        if polish_time > 1:
            rows_polish, score_polish = _local_search_refine(
                best_rows, n_rows, k, best_score, polish_time, start_time)
            if score_polish < best_score:
                best_score = score_polish
                best_rows = rows_polish

    return best_rows
