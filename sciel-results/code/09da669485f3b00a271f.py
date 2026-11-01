"""Backtracking search with constraint propagation for difference triangle set."""
import time
import random
import sys

_START = None

def construct(n_rows, k):
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + 145.0

    d = k * (k + 1) // 2
    total_diffs = n_rows * d
    m_min = 0
    while m_min * (m_min + 1) // 2 < total_diffs:
        m_min += 1
    lower_bound = m_min

    # Greedy initial solution
    def greedy_init():
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

    best_rows = greedy_init()
    best_scope = max(max(r) for r in best_rows)

    # Backtracking search
    # Strategy: build rows one at a time, choosing differences carefully
    # For each row, we need to pick k+1 positions starting at 0
    # such that all pairwise differences are unused

    def backtrack_search(max_scope):
        """Try to find a valid configuration with all entries <= max_scope."""
        used_diffs = set()
        rows = []

        def can_place_row(used, max_s):
            """Check if we can place a row of k+1 elements starting at 0 within max_s
            using only unused differences."""
            # Greedy check: try to build a row
            row = [0]
            available = set(range(1, max_s + 1)) - used
            for pos in range(1, k + 1):
                # Try to find next element
                found = False
                for x in range(row[-1] + 1, max_s + 1):
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used and new_diffs <= available:
                        row.append(x)
                        found = True
                        break
                if not found:
                    return None
            return row

        def search_row(idx, current_row, used):
            """Recursively build row idx. current_row has len(idx+1) elements."""
            if idx == k:
                # Row complete
                rows.append(current_row[:])
                if len(rows) == n_rows:
                    return True
                return search_row(0, [0], used)
            
            last = current_row[-1]
            # Try next positions
            for x in range(last + 1, max_scope + 1):
                new_diffs = {x - a for a in current_row}
                if new_diffs & used:
                    continue
                # Check if new_diffs are all distinct (they are by construction)
                current_row.append(x)
                used_copy = used | new_diffs
                if search_row(idx + 1, current_row, used_copy):
                    return True
                current_row.pop()
            return False

        # Start search
        if search_row(0, [0], used_diffs):
            return rows
        return None

    # Use backtracking to try to improve
    rng = random.Random(42)

    # Try decreasing scopes
    for target_scope in range(best_scope - 1, lower_bound - 1, -1):
        if time.time() > deadline:
            break
        result = backtrack_search(target_scope)
        if result is not None:
            best_rows = result
            best_scope = target_scope
            # Continue trying to go lower
            continue
        # If we couldn't find at this scope, break (likely infeasible)
        # But be careful: our search might be incomplete
        break

    # If backtracking didn't improve much, try randomized restarts with backtracking
    for restart in range(50):
        if time.time() > deadline:
            break
        # Try with some randomization in the search order
        used_diffs = set()
        rows = []
        found = False

        def search_rand(idx, current_row, used):
            nonlocal found
            if found:
                return True
            if idx == k:
                rows.append(current_row[:])
                if len(rows) == n_rows:
                    found = True
                    return True
                return search_rand(0, [0], used)

            last = current_row[-1]
            candidates = list(range(last + 1, best_scope + 1))
            rng.shuffle(candidates)
            for x in candidates:
                if found:
                    break
                new_diffs = {x - a for a in current_row}
                if new_diffs & used:
                    continue
                current_row.append(x)
                if search_rand(idx + 1, current_row, used | new_diffs):
                    return True
                current_row.pop()
            return False

        if search_rand(0, [0], used_diffs):
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                best_rows = rows
                best_scope = scope

    return best_rows
