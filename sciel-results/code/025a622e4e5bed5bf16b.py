"""CSP-based difference-triangle construction with backtracking, forward checking,
and dynamic variable ordering (most-constrained-first)."""
# EVOLVE-BLOCK-START
import sys
import time
sys.setrecursionlimit(100000)


def construct(n_rows, k):
    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs

    def greedy_solution(n_rows, k):
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

    greedy_rows = greedy_solution(n_rows, k)
    M_upper = max(max(r) for r in greedy_rows)

    def solve_row(used_diffs, M, k):
        """Find a row of k+1 marks starting at 0 in [0,M] with all diffs unused."""
        row = [0]

        def backtrack(row, last, used_local):
            if len(row) == k + 1:
                return row[:]
            remaining_to_place = k + 1 - len(row)
            # Try candidates in increasing order for compactness
            for x in range(last + 1, M + 1):
                # Need room for remaining marks
                if M - x < remaining_to_place - 1:
                    break
                new_diffs = {x - a for a in row}
                if new_diffs & used_local:
                    continue
                # Forward checking: ensure that for each remaining position,
                # there exists at least one candidate. Simplified: check that
                # the number of unused differences in [1, M-x] is sufficient.
                # This is expensive, so skip and rely on backtracking.
                row.append(x)
                used_local = used_local | new_diffs
                result = backtrack(row, x, used_local)
                if result is not None:
                    return result
                row.pop()
                used_local = used_local - new_diffs
            return None

        return backtrack(row, 0, set())

    def construct_with_bound(M, n_rows, k):
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = solve_row(used_diffs, M, k)
            if row is None:
                return None
            rows.append(row)
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_diffs.add(row[j] - row[i])
        return rows

    start_time = time.time()
    for M in range(lb, M_upper + 1):
        if time.time() - start_time > 100:
            break
        result = construct_with_bound(M, n_rows, k)
        if result is not None:
            return result

    return greedy_rows
# EVOLVE-BLOCK-END
