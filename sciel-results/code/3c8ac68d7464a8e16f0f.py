import time
import random
import sys
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row

    def get_row_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        used = set()
        for row in rows:
            n = len(row)
            for i in range(n):
                ri = row[i]
                for j in range(i + 1, n):
                    d = row[j] - ri
                    if d in used:
                        return False
                    used.add(d)
        return True

    # Greedy fallback: always available
    def greedy_solve():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            for _ in range(k):
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        row.append(x)
                        used.update(new_diffs)
                        break
                    x += 1
            rows.append(row)
        return rows

    greedy_rows = greedy_solve()
    greedy_M = max(max(r) for r in greedy_rows)
    best_solution = greedy_rows
    best_M = greedy_M

    # Strategy: For each candidate M, generate a pool of valid rows (all C(M, k) subsets
    # of {1..M} prepended with 0), then use randomized set-packing to select n_rows
    # rows with disjoint difference sets.
    #
    # Key insight: We don't need ALL C(M,k) rows. We can:
    #   1. Generate a random sample of candidate rows
    #   2. Filter to those whose diff-set doesn't conflict with already-selected rows
    #   3. Use greedy + randomized restarts to find a packing of size n_rows

    def try_set_packing(M, time_budget):
        """Try to find n_rows rows with max entry <= M using set-packing from a pool."""
        end_time = start_time + time_budget
        max_attempts = 200  # randomized restarts

        for attempt in range(max_attempts):
            if time.time() > end_time:
                break

            # Generate a pool of candidate rows
            # For M ~ 105-120, C(M, k) is large, so sample
            # We want diversity: sample from different "shapes" of rows
            pool_size = min(5000, 1000 + attempt * 100)  # grow pool slightly each restart
            pool = []
            seen_diffsets = set()

            # Generate random rows
            for _ in range(pool_size * 3):  # oversample to get unique diffsets
                if time.time() > end_time:
                    break
                # Random increasing sequence of k elements from 1..M
                if M < k:
                    continue
                elems = sorted(random.sample(range(1, M + 1), k))
                row = [0] + elems
                if row[-1] > M:
                    continue
                dset = get_row_diffs(row)
                # Check internal consistency (always true for a row)
                key = frozenset(dset)
                if key not in seen_diffsets:
                    seen_diffsets.add(key)
                    pool.append((row, dset))
                if len(pool) >= pool_size:
                    break

            if len(pool) < n_rows:
                continue

            # Randomized greedy set-packing with look-ahead
            selected = []
            used_diffs = set()
            available = list(pool)
            random.shuffle(available)

            success = True
            for _ in range(n_rows):
                if time.time() > end_time:
                    success = False
                    break

                # Find candidates that don't conflict
                candidates = []
                for idx, (row, dset) in enumerate(available):
                    if not (dset & used_diffs):
                        candidates.append(idx)

                if not candidates:
                    success = False
                    break

                # Score candidates: prefer rows with "spread out" differences
                # to leave more room for future rows
                def score_candidate(idx):
                    row, dset = available[idx]
                    # Prefer rows where the max difference is smaller (more compact)
                    # but also prefer rows that use "rare" differences
                    # Heuristic: minimize max diff, break ties randomly
                    return max(dset)

                # Take top few and pick randomly among them for diversity
                candidates_sorted = sorted(candidates, key=score_candidate)
                top_n = min(10, len(candidates_sorted))
                pick_idx = random.choice(candidates_sorted[:top_n])

                row, dset = available[pick_idx]
                selected.append(row)
                used_diffs.update(dset)
                # Remove this row and any conflicting rows from available
                available = [av for i, av in enumerate(available) if i != pick_idx]
                # Optionally prune conflicting rows early
                available = [av for av in available if not (av[1] & used_diffs)]

            if success and len(selected) == n_rows:
                return selected

        return None

    # Try M values from lower bound upward
    # Lower bound: total_diffs (counting bound)
    # Upper bound: greedy_M - 1 (we already have greedy)
    lower_bound = total_diffs
    # Also consider that M must be at least k (trivial)
    lower_bound = max(lower_bound, k)

    # Give more time to smaller M values (harder to find solutions)
    # Total search time available: deadline - time elapsed - 5s buffer
    remaining = deadline - time.time() - 5.0
    if remaining <= 0:
        return best_solution

    # Try M from lower_bound up to min(greedy_M - 1, lower_bound + 30)
    # Cap the range to avoid spending all time on infeasible small M
    upper_try = min(greedy_M - 1, lower_bound + 25)

    # Allocate time proportionally: earlier M gets more time
    num_M_values = max(1, upper_try - lower_bound + 1)
    time_per_M = remaining / num_M_values
    # Cap time per M to avoid one M eating everything
    time_per_M = min(time_per_M, 15.0)

    for M in range(lower_bound, upper_try + 1):
        if time.time() > deadline - 5.0:
            break

        # Allocate time for this M
        alloc = min(time_per_M, deadline - time.time() - 3.0)
        if alloc < 0.5:
            break

        sol = try_set_packing(M, alloc)
        if sol is not None:
            # Verify validity
            if is_valid(sol):
                actual_M = max(max(r) for r in sol)
                if actual_M <= M and actual_M < best_M:
                    best_solution = sol
                    best_M = actual_M
                    break  # Found smallest M, stop

    # Final validation
    if not is_valid(best_solution):
        best_solution = greedy_solve()

    return best_solution


if __name__ == "__main__":
    rows = construct(7, 5)
    max_val = max(max(r) for r in rows)
    print(f"Max value: {max_val}")
    for r in rows:
        print(r)
    # Validate
    used = set()
    valid = True
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    valid = False
                    print(f"Conflict: {d} in row {row}")
                used.add(d)
    print(f"Valid: {valid}")
    print(f"Total diffs: {len(used)} (expected {7 * 15})")
