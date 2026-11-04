import time
import sys
import random
from itertools import combinations

sys.setrecursionlimit(100000)


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    DEADLINE = 145.0  # 5s safety margin before 150s kill

    num_elements = k + 1
    total_diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * total_diffs_per_row

    # Lower bound on scope from counting: M(M-1)/2 >= total_diffs_needed
    # M(M-1) >= 2 * total_diffs_needed
    import math
    lb = int(math.ceil((1 + math.sqrt(1 + 8 * total_diffs_needed)) / 2))

    # Greedy fallback
    def greedy_solution():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = frozenset(x - a for a in row)
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    baseline = greedy_solution()
    best_solution = baseline
    best_scope = max(max(r) for r in baseline)

    # For each candidate scope M, try to find a valid assignment
    # using CSP with MRV (Minimum Remaining Values) heuristic.

    def solve_scope(M):
        """Try to find a valid DTS with all entries <= M."""
        # State: rows[r][c] = value, with rows[r][0] = 0 fixed.
        # We need to assign values to positions (r, c) for r in range(n_rows), c in range(1, num_elements).
        # Constraints:
        #   1. Strictly increasing within each row.
        #   2. All values <= M.
        #   3. All differences a[r][j] - a[r][i] for i<j are globally unique.

        # We'll use backtracking with MRV: at each step, pick the unassigned position
        # with the fewest legal values, and try them.

        # Precompute: for each position (r, c), what are the possible values?
        # Given the current partial assignment, the domain is [prev+1, M - remaining].

        # For efficiency, represent used differences as a set.

        used_diffs = set()
        rows = [[0] + [None] * (num_elements - 1) for _ in range(n_rows)]

        # Track for each row the last assigned value (for increasing constraint)
        # and the next position to fill.

        def get_domain(r, c, rows_state, used):
            """Get sorted list of legal values for position (r, c)."""
            # Must be > rows_state[r][c-1]
            lo = rows_state[r][c - 1] + 1
            # Must be <= M - (num_elements - 1 - c) to leave room for remaining
            hi = M - (num_elements - 1 - c)
            if lo > hi:
                return []
            # Check difference constraints with all previously assigned values in row r
            # For each candidate v, check that v - rows_state[r][i] is not in used for all i < c
            # Also need to check that rows_state[r][i] - v doesn't apply since v > all previous.
            candidates = []
            for v in range(lo, hi + 1):
                ok = True
                for i in range(c):
                    d = v - rows_state[r][i]
                    if d in used:
                        ok = False
                        break
                if ok:
                    candidates.append(v)
            return candidates

        # MRV: find unassigned position with smallest domain
        def find_mrv_position(rows_state, used):
            best_pos = None
            best_domain = None
            best_len = float('inf')
            for r in range(n_rows):
                # Find the first unassigned column in row r
                c = 1
                while c < num_elements and rows_state[r][c] is not None:
                    c += 1
                if c >= num_elements:
                    continue
                dom = get_domain(r, c, rows_state, used)
                if len(dom) < best_len:
                    best_len = len(dom)
                    best_pos = (r, c)
                    best_domain = dom
                    if best_len == 0:
                        return best_pos, dom
                    if best_len == 1:
                        break
            return best_pos, best_domain

        def backtrack(rows_state, used):
            if time.time() - start_time > DEADLINE:
                return None

            # Check if all positions are filled
            all_filled = True
            for r in range(n_rows):
                if any(rows_state[r][c] is None for c in range(1, num_elements)):
                    all_filled = False
                    break
            if all_filled:
                return [list(row) for row in rows_state]

            pos, dom = find_mrv_position(rows_state, used)
            if pos is None:
                return None  # Should not happen if all_filled check above works
            if not dom:
                return None

            r, c = pos
            for v in dom:
                # Try assigning v
                rows_state[r][c] = v
                # Add new differences
                new_diffs = set()
                conflict = False
                for i in range(c):
                    d = v - rows_state[r][i]
                    if d in used:
                        conflict = True
                        break
                    new_diffs.add(d)
                if conflict:
                    rows_state[r][c] = None
                    continue

                used.update(new_diffs)
                result = backtrack(rows_state, used)
                if result is not None:
                    return result
                used.difference_update(new_diffs)
                rows_state[r][c] = None

            return None

        result = backtrack(rows, used_diffs)
        return result

    # Try scopes from lower bound upward
    # Start from a reasonable lower bound; the prompt says counting bound is 105
    # but let's compute it properly
    search_start = max(lb, 100)
    search_end = max(best_scope, search_start + 20)

    for M in range(search_start, search_end + 1):
        if time.time() - start_time > DEADLINE:
            break
        sol = solve_scope(M)
        if sol is not None:
            # Validate
            all_diffs = set()
            valid = True
            for row in sol:
                if len(row) != num_elements or row[0] != 0:
                    valid = False
                    break
                for i in range(num_elements):
                    for j in range(i + 1, num_elements):
                        d = row[j] - row[i]
                        if d in all_diffs:
                            valid = False
                            break
                        all_diffs.add(d)
                if not valid:
                    break
            if valid and max(max(r) for r in sol) < best_scope:
                best_solution = sol
                best_scope = max(max(r) for r in sol)
                break
            elif valid:
                best_solution = sol
                best_scope = max(max(r) for r in sol)
                break

    return best_solution
