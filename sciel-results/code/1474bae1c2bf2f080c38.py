import time
import random
import sys
from itertools import combinations


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all pairwise differences are distinct across all rows.
    Minimizes the largest entry.
    """
    deadline = time.time() + 140.0  # 10s safety margin for 150s limit

    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row

    # Lower bound on M: need total_diffs_needed distinct differences in {1..M}
    M_lower = total_diffs_needed
    # Also, each row has max entry >= k, and differences go up to max_entry
    # A tighter lower bound: the largest difference in a row is at most M,
    # and we need diffs_per_row distinct values per row, all <= M.
    # So M >= diffs_per_row is trivial; M >= total_diffs_needed is the packing bound.

    # Upper bound from greedy
    greedy_rows = _greedy_construct(n_rows, k)
    best_M = _max_entry(greedy_rows)
    best_rows = greedy_rows

    # Try to improve by searching M from M_lower upward
    # But first, let's try a smarter approach: enumerate valid difference sets
    # and use set packing + local search.

    # For (7,5): diffs_per_row = 15, total = 105. M_lower = 105.
    # Best known is 112. So we try M in [105, best_M-1].

    # Strategy: For each M, enumerate all valid difference sets (subsets of {1..M}
    # of size diffs_per_row that can be realized as pairwise differences of a
    # strictly increasing sequence of length k+1 starting at 0).
    # Then find n_rows disjoint such sets.

    # However, enumerating all valid difference sets for M=112, k=5 is:
    # C(112, 15) which is way too large. We need a smarter enumeration.

    # Alternative: generate candidate rows directly (as before) but use a
    # DIFFERENT search method: simulated annealing over row assignments.

    # NEW STRATEGY: Represent the solution as a list of n_rows rows.
    # Use simulated annealing where moves are:
    #   1. Regenerate one row (random valid row compatible with used diffs)
    #   2. Swap two rows
    #   3. Perturb one row (change one element)
    # Objective: minimize max_entry, with penalty for invalidity.

    # Actually, let's try a cleaner structural approach:
    # Enumerate candidate rows (valid difference sets that fit in [0, M]),
    # then use a maximum-bipartite-matching or greedy set-packing with
    # backtracking limited to the last few rows.

    # For efficiency, let's do:
    # Phase 1: Greedy to get a good initial solution (already have it).
    # Phase 2: Local search (simulated annealing) to reduce max entry.
    # Phase 3: Try to fit into smaller M by constrained search.

    # Let's implement a local search that tries to reduce the max entry
    # by regenerating rows that use large values.

    rows = list(greedy_rows)
    used_diffs = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used_diffs.add(row[j] - row[i])

    current_M = _max_entry(rows)
    best_local_M = current_M
    best_local_rows = [list(r) for r in rows]

    # Simulated annealing: try to reduce max entry by regenerating rows
    # that contribute to the max.
    temperature = 10.0
    cooling = 0.95
    steps_per_temp = 50

    for temp_idx in range(200):
        if time.time() > deadline - 10:
            break
        temperature *= cooling
        if temperature < 0.01:
            break

        for step in range(steps_per_temp):
            if time.time() > deadline - 10:
                    break

            # Pick a row to regenerate: prefer rows with large max entry
            max_entries = [max(r) for r in rows]
            # Weight selection probability by max entry
            weights = [me + 1 for me in max_entries]
            total_w = sum(weights)
            r = random.random() * total_w
            cum = 0
            chosen = 0
            for idx, w in enumerate(weights):
                cum += w
                if r <= cum:
                    chosen = idx
                    break

            old_row = rows[chosen]
            old_diffs = set()
            for i in range(len(old_row)):
                for j in range(i + 1, len(old_row)):
                    old_diffs.add(old_row[j] - old_row[i])

            # Remove old row's diffs
            available_diffs = used_diffs - old_diffs

            # Try to generate a new row with smaller max entry
            target_max = min(max_entries[chosen] - 1, current_M - 1)
            if target_max < k:
                target_max = current_M

            new_row = _generate_row_with_max_limit(k, available_diffs, target_max)
            if new_row is None:
                # Try without strict limit
                new_row = _generate_row_with_max_limit(k, available_diffs, current_M)

            if new_row is not None:
                new_diffs = set()
                for i in range(len(new_row)):
                    for j in range(i + 1, len(new_row)):
                        new_diffs.add(new_row[j] - new_row[i])

                new_M = _max_entry(rows[:chosen] + [new_row] + rows[chosen+1:])
                old_M = current_M

                # Accept if better, or with probability exp(-delta/T)
                delta = new_M - old_M
                if delta <= 0 or random.random() < _exp_neg(delta / max(temperature, 0.01)):
                    rows[chosen] = new_row
                    used_diffs = available_diffs | new_diffs
                    current_M = _max_entry(rows)
                    if current_M < best_local_M:
                        best_local_M = current_M
                        best_local_rows = [list(r) for r in rows]

    if best_local_M < best_M:
        best_M = best_local_M
        best_rows = best_local_rows

    # Now try to squeeze into even smaller M using constrained backtracking
    # on the last few rows, using the difference-set approach.
    for M_try in range(max(M_lower, best_M - 5), best_M):
        if time.time() > deadline - 5:
            break

        rows_try = _solve_constrained(n_rows, k, M_try, deadline)
        if rows_try is not None:
            best_rows = rows_try
            best_M = M_try
            break

    return best_rows


def _generate_row_with_max_limit(k, available_diffs, max_limit):
    """
    Generate a row of k+1 increasing integers starting at 0, with max <= max_limit,
    such that all pairwise differences are in available_diffs (i.e., not used).
    """
    # Try multiple random attempts
    for _ in range(200):
        # Pick k values from 1..max_limit
        if max_limit < k:
            return None
        vals = random.sample(range(1, max_limit + 1), k)
        vals.sort()
        row = [0] + vals

        # Check validity
        d_set = set()
        valid = True
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in d_set:
                    valid = False
                    break
                if d not in available_diffs:
                    valid = False
                    break
                d_set.add(d)

        if valid:
            return row

    return None


def _exp_neg(x):
    """Safe exp(-x) for large x."""
    if x > 50:
        return 0.0
    import math
    return math.exp(-x)


def _solve_constrained(n_rows, k, M, deadline):
    """
    Try to find n_rows rows with max entry <= M using backtracking.
    Uses the difference-set approach: precompute which differences are available.
    """
    used_diffs = set()
    rows = []

    def backtrack(idx):
        if time.time() > deadline - 2:
            return False
        if idx == n_rows:
            return True

        # Try to find a valid row
        for attempt in range(500):
            if time.time() > deadline - 2:
                return False

            if M < k:
                return False

            vals = random.sample(range(1, M + 1), k)
            vals.sort()
            row = [0] + vals

            d_set = set()
            valid = True
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in d_set or d in used_diffs:
                        valid = False
                        break
                    d_set.add(d)

            if valid:
                rows.append(row)
                used_diffs.update(d_set)
                if backtrack(idx + 1):
                    return True
                rows.pop()
                used_diffs.difference_update(d_set)

        return False

    # Multiple restarts
    for restart in range(100):
        if time.time() > deadline - 2:
            break
        random.seed(restart * 7919 + M)
        used_diffs.clear()
        rows.clear()
        if backtrack(0):
            return rows

    return None


def _greedy_construct(n_rows, k):
    """
    Greedy construction: pick the smallest possible next value that doesn't conflict.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used and len(new) == len(row):
                row.append(x)
                used |= new
        rows.append(row)
    return rows


def _max_entry(rows):
    return max(max(row) for row in rows)


def _is_valid(rows, k):
    """
    Check if rows are valid.
    """
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        if not all(row[i] < row[i+1] for i in range(k)):
            return False

        d_set = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in d_set:
                    return False
                if d in used:
                    return False
                d_set.add(d)
        used |= d_set
    return True
