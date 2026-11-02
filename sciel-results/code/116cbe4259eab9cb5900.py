"""
Difference triangle set (7,5) – hybrid construction.

Strategy:
  1. Fast greedy baseline (original seed).
  2. Iterative-deepening backtracking over the scope bound M, building rows
     one at a time with strong forward-checking (difference-set intersection).
     This is the structural change: instead of "greedy + local repair", we
     directly search for the smallest M that admits a valid construction.
  3. Randomised restarts with different row-orderings and candidate biases
     to escape local minima within the time budget.
  4. Deadline guard: return the best valid result found before the wall-clock
     limit (with a safety margin).
"""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _greedy_fallback(n_rows, k):
    """Original greedy: always valid, used as safety net."""
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows


def _scope(rows):
    return max(max(r) for r in rows)


def _validate(rows, k):
    """Check that all rows are valid and all differences are distinct."""
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d <= 0:
                    return False
                if d in used:
                    return False
                used.add(d)
        # strict increase
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
    return True


def _backtrack_row(row, used, M, rng, time_limit, attempts_remaining):
    """
    Try to complete `row` (a list starting with [0]) to length k+1,
    using only values <= M, with no new difference colliding with `used`.
    Returns (completed_row, updated_used) on success, (None, None) on failure.
    """
    k_plus_1 = len(row) if row else None  # caller ensures row starts as [0]
    target_len = k_plus_1 if k_plus_1 else None
    # We'll use iterative deepening within this row: try smallest candidates first
    # but with randomization to escape traps.
    
    def bt(pos, row, used_local):
        if time.time() > time_limit:
            return None, None
        if pos == target_len:
            return list(row), used_local
        
        last = row[-1]
        # Candidate range: last+1 .. M
        # Try in increasing order with some randomization
        start = last + 1
        end = M
        
        if end < start:
            return None, None
        
        # Build candidate list, maybe shuffle for randomization
        candidates = list(range(start, end + 1))
        if rng.random() < 0.3:
            # Partial shuffle: pick a random subset to try first
            if len(candidates) > 5:
                # Try a few random ones first, then the rest in order
                random_subset = rng.sample(candidates, min(5, len(candidates)))
                remaining = [c for c in candidates if c not in set(random_subset)]
                candidates = random_subset + remaining
        
        for cand in candidates:
            new_diffs = {cand - a for a in row}
            if new_diffs & used_local:
                continue
            row.append(cand)
            new_used = used_local | new_diffs
            result_row, result_used = bt(pos + 1, row, new_used)
            if result_row is not None:
                return result_row, result_used
            row.pop()
        
        return None, None
    
    return bt(1, row, used)


def _construct_with_bound(n_rows, k, M, rng, time_limit):
    """
    Try to construct n_rows rows of k+1 entries, all <= M,
    with distinct differences across all rows.
    Returns list of rows on success, None on failure.
    """
    used = set()
    rows = []
    
    for r_idx in range(n_rows):
        row = [0]
        completed, new_used = _backtrack_row(row, used, M, rng, time_limit, n_rows - r_idx)
        if completed is None:
            return None
        rows.append(completed)
        used = new_used
    
    return rows


def _iterative_deepening(n_rows, k, start_M, max_M, rng, time_limit):
    """
    Iterative deepening on M (the scope bound).
    Try M = start_M, start_M+1, ... up to max_M or time_limit.
    Returns (rows, M) on success, (None, None) on failure.
    """
    for M in range(start_M, max_M + 1):
        if time.time() > time_limit:
            return None, None
        # Try a few randomised attempts at this M
        attempts = 3
        for _ in range(attempts):
            if time.time() > time_limit:
                break
            result = _construct_with_bound(n_rows, k, M, rng, time_limit)
            if result is not None:
                if _validate(result, k):
                    return result, M
    return None, None


def construct(n_rows, k):
    t0 = time.time()
    total_budget = 140.0  # safety margin before 150s kill
    deadline = t0 + total_budget
    
    # Step 1: Greedy baseline
    best_rows = _greedy_fallback(n_rows, k)
    best_M = _scope(best_rows)
    
    # Step 2: Iterative deepening backtracking
    # Lower bound: counting bound is roughly n_rows * k*(k+1)/2 differences needed,
    # max difference M must satisfy M(M+1)/2 >= n_rows * k*(k+1)/2
    num_diffs = n_rows * k * (k + 1) // 2
    # M(M+1)/2 >= num_diffs  =>  M >= ceil((-1 + sqrt(1 + 8*num_diffs))/2)
    import math
    lb = math.ceil((-1 + math.sqrt(1 + 8 * num_diffs)) / 2)
    lb = max(lb, k)  # at least k
    
    rng = random.Random(42)
    
    # Try iterative deepening from lb up to best_M - 1
    # If we find a solution with M < best_M, update best
    search_start = lb
    search_end = min(best_M - 1, lb + 50)  # limit range to avoid excessive time
    
    if search_start <= search_end:
        result, found_M = _iterative_deepening(
            n_rows, k, search_start, search_end, rng, deadline
        )
        if result is not None and found_M < best_M:
            best_rows = result
            best_M = found_M
    
    # Step 3: If time remains, try randomised greedy with local improvement
    if time.time() < deadline - 10.0:
        # Try several randomised greedy constructions
        for seed in range(20):
            if time.time() > deadline - 5.0:
                break
            rng2 = random.Random(seed)
            used, rows = set(), []
            success = True
            for r in range(n_rows):
                row = [0]
                x = 0
                while len(row) < k + 1:
                    # Try next few candidates, pick one that minimizes future conflict
                    found = False
                    window = 1 + (seed % 7)
                    candidates = []
                    for cand in range(x + 1, x + 1 + window):
                        new = {cand - a for a in row}
                        if not new & used:
                            candidates.append((sum(new), cand))
                    if candidates:
                        candidates.sort()
                        if rng2.random() < 0.7:
                            chosen = candidates[0][1]
                        else:
                            chosen = candidates[rng2.randrange(len(candidates))][1]
                        row.append(chosen)
                        used |= {chosen - a for a in row[:-1]}
                        x = chosen
                        found = True
                    else:
                        x += 1
                        new = {x - a for a in row}
                        if not new & used:
                            row.append(x)
                            used |= new
                            found = True
                        else:
                            continue
                    if not found:
                        success = False
                        break
                if not success:
                    break
                rows.append(row)
            
            if success and len(rows) == n_rows:
                sc = _scope(rows)
                if sc < best_M:
                    best_rows = rows
                    best_M = sc
    
    # Final validation
    if not _validate(best_rows, k):
        best_rows = _greedy_fallback(n_rows, k)
    
    return best_rows

# EVOLVE-BLOCK-END
