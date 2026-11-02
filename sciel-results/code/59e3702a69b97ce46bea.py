import time
import random
import math
import heapq
from itertools import combinations

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Constructs a Difference Triangle Set (DTS) with n_rows rows of k+1 elements.
    Uses a backtracking search with constraint propagation on a conflict graph.
    Each row is a Sidon set (all pairwise differences distinct within the row).
    All differences across all rows must be globally distinct.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s limit

    # Greedy fallback (always valid)
    def greedy_construct(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
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

    best_solution = greedy_construct(n_rows, k)
    best_scope = max(max(row) for row in best_solution)

    def _validate(rows, n_rows, k):
        all_diffs = set()
        for row in rows:
            if len(row) != k + 1 or row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True

    def _row_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def _global_max(rows):
        return max(max(row) for row in rows)

    # Backtracking search with constraint propagation
    # We build rows one at a time. For each row, we try to construct a Sidon set
    # that avoids all already-used differences, minimizing the maximum entry.

    def find_row(used_diffs, k, max_val, deadline):
        """
        Find a Sidon set of size k+1 starting at 0, with all elements <= max_val,
        such that no pairwise difference is in used_diffs.
        Returns the row (list) or None if not found within time.
        Uses backtracking with forward checking.
        """
        # We need to choose k elements from {1, ..., max_val}
        # such that all pairwise differences (including with 0) are not in used_diffs
        # and are all distinct from each other

        # Build list of candidate values for each position
        # Position 0 is always 0
        # For position i (1..k), we need a value > previous value
        # such that value - 0, value - row[j] for all j < i are not in used_diffs
        # and are all distinct from each other

        best_row = None
        best_max = max_val + 1

        def backtrack(pos, current_row, current_diffs, deadline_check):
            nonlocal best_row, best_max

            if time.time() > deadline_check:
                return None

            if pos == k + 1:
                row_max = current_row[-1]
                if row_max < best_max:
                    best_max = row_max
                    best_row = current_row[:]
                return current_row[:]

            # Lower bound: we need (k+1 - pos) more elements, each at least 1 apart
            # So minimum possible max is current_row[-1] + (k + 1 - pos)
            min_possible_max = current_row[-1] + (k + 1 - pos)
            if min_possible_max >= best_max:
                return None

            # Try candidate values for position pos
            # Must be > current_row[-1]
            # Must satisfy: val - 0 not in used_diffs, val - current_row[j] not in used_diffs for all j
            # And val - current_row[j] must not be in current_diffs (within-row uniqueness)
            
            low = current_row[-1] + 1
            high = min(max_val, best_max - 1)  # can't exceed best_max - 1 since we need more elements
            
            if low > high:
                return None

            # For efficiency, iterate through candidates in order
            for val in range(low, high + 1):
                if time.time() > deadline_check:
                    return None

                # Check if val is valid
                new_diffs = set()
                valid = True
                for j in range(pos):
                    d = val - current_row[j]
                    if d in used_diffs:
                        valid = False
                        break
                    if d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)

                if not valid:
                    continue

                # Forward checking: can we still complete?
                # The next value must be > val, and its difference with all current elements
                # must not be in used_diffs or new_diffs
                # Simple check: is there at least one valid next value?
                if pos + 1 < k + 1:
                    next_low = val + 1
                    found_next = False
                    for nv in range(next_low, min(max_val, best_max) + 1):
                        nd = set()
                        ok = True
                        for j in range(pos + 1):
                            dd = nv - current_row[j]
                            if dd in used_diffs or dd in new_diffs:
                                ok = False
                                break
                            if dd in nd:
                                ok = False
                                break
                            nd.add(dd)
                        if ok:
                            found_next = True
                            break
                    if not found_next:
                        continue

                # Try this value
                current_row.append(val)
                current_diffs.update(new_diffs)
                result = backtrack(pos + 1, current_row, current_diffs, deadline_check)
                current_row.pop()
                current_diffs -= new_diffs

                if result is not None:
                    return result

            return None

        result = backtrack(1, [0], set(), deadline)
        return result

    # Main backtracking: build rows one at a time
    # Try to find n_rows rows with minimal global max

    def solve_with_backtracking(n_rows, k, max_scope, deadline):
        """
        Try to find n_rows rows, each a Sidon set of size k+1 starting at 0,
        with all elements <= max_scope, and all differences globally distinct.
        Returns list of rows or None.
        """
        used_diffs = set()
        rows = []

        def backtrack_row(row_idx, deadline_check):
            if time.time() > deadline_check:
                return None

            if row_idx == n_rows:
                return rows[:]

            # Find a valid row
            row = find_row(used_diffs, k, max_scope, deadline_check)
            if row is None:
                return None

            rows.append(row)
            row_diffs = _row_diffs(row)
            used_diffs.update(row_diffs)

            result = backtrack_row(row_idx + 1, deadline_check)
            if result is not None:
                return result

            # Backtrack
            rows.pop()
            used_diffs -= row_diffs
            return None

        return backtrack_row(0, deadline)

    # Try increasing scopes
    # Lower bound: n_rows * k * (k+1) / 2
    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs

    # For (7,5): total_diffs = 7 * 5 * 6 / 2 = 105
    # Best known: 112, counting bound: 105

    # First, try the greedy solution to get an upper bound
    greedy_scope = best_scope

    # Now try backtracking for scopes from lb up to greedy_scope
    # But backtracking can be slow, so we limit it

    # Use a randomized approach: shuffle the order of row construction
    # and use multiple trials

    random.seed(42)

    for scope in range(lb, greedy_scope + 1):
        if time.time() > deadline - 10:
            break

        if scope < lb:
            continue

        # Try to solve with this scope
        # Use multiple random restarts with different row orderings
        found = None

        for trial in range(5):
            if time.time() > deadline - 5:
                break

            # Randomize: try different starting points
            # The key is that find_row already minimizes the max entry for each row
            # given the used differences, so the order matters less

            result = solve_with_backtracking(n_rows, k, scope, deadline - 5)
            if result is not None:
                found = result
                break

        if found is not None:
            if _validate(found, n_rows, k):
                actual_scope = _global_max(found)
                if actual_scope < best_scope:
                    best_scope = actual_scope
                    best_solution = found
                break  # Found the minimal scope

    # If backtracking didn't find a better solution, try a local search refinement
    # on the greedy solution to potentially reduce the scope

    def local_search_refine(rows, n_rows, k, deadline):
        """
        Refine rows using local search to reduce the global maximum.
        """
        current_rows = [r[:] for r in rows]
        current_val = _global_max(current_rows)
        all_used = set()
        row_diffs = []
        for row in current_rows:
            rd = _row_diffs(row)
            all_used |= rd
            row_diffs.append(rd)

        best_rows = [r[:] for r in current_rows]
        best_val = current_val

        T = 200.0
        T_min = 1.0
        cool = 0.99995
        no_improve = 0
        max_no_improve = 50000

        while time.time() < deadline and no_improve < max_no_improve:
            ri = random.randrange(n_rows)
            pi = random.randrange(1, k + 1)

            old_val = current_rows[ri][pi]
            min_allowed = current_rows[ri][pi - 1] + 1
            max_allowed = (current_rows[ri][pi + 1] - 1) if pi < k else current_val

            # Try to decrease
            candidates = []
            for d in range(1, min(50, old_val - min_allowed + 1)):
                candidates.append(old_val - d)

            # Some random exploration
            if random.random() < 0.2:
                lo = min_allowed
                hi = max_allowed
                if lo <= hi:
                    candidates.append(random.randint(lo, hi))

            if not candidates:
                no_improve += 1
                T = max(T_min, T * cool)
                continue

            random.shuffle(candidates)
            moved = False

            for new_val in candidates:
                if new_val <= min_allowed or new_val > max_allowed:
                    continue

                cand_row = current_rows[ri][:]
                cand_row[pi] = new_val

                # Check Sidon property
                new_diffs = set()
                valid = True
                for i in range(len(cand_row)):
                    for j in range(i + 1, len(cand_row)):
                        d = cand_row[j] - cand_row[i]
                        if d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                if not valid:
                    continue

                # Check cross-row conflicts
                for d in new_diffs:
                    if d in all_used and d not in row_diffs[ri]:
                        valid = False
                        break
                if not valid:
                    continue

                # Compute new global max
                cand_max = max(cand_row)
                old_row_max = max(current_rows[ri])
                row_maxes = [max(r) for r in current_rows]
                row_maxes[ri] = cand_max
                new_global = max(row_maxes)
                delta = new_global - current_val

                accept = False
                if delta < 0:
                    accept = True
                elif delta == 0:
                    accept = random.random() < 0.1
                else:
                    if T > T_min:
                        p = math.exp(-delta / max(T, 0.1))
                        accept = random.random() < p

                if accept:
                    all_used -= row_diffs[ri]
                    all_used |= new_diffs
                    current_rows[ri] = cand_row
                    row_diffs[ri] = new_diffs
                    current_val = new_global

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

    # Apply local search refinement
    if time.time() < deadline - 5:
        refined_rows, refined_scope = local_search_refine(
            [r[:] for r in best_solution], n_rows, k, deadline
        )
        if refined_scope < best_scope:
            best_scope = refined_scope
            best_solution = refined_rows

    return best_solution

# EVOLVE-BLOCK-END
