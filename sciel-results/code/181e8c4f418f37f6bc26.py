import time
import random
import sys
from array import array


def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]

    deadline = time.time() + 118.0  # safety margin before 120s
    m = k + 1  # entries per row
    needed = n_rows

    # ---------- helpers ----------

    def row_diffs(row):
        """Return the set of all positive differences in a row."""
        s = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                s.add(row[j] - row[i])
        return s

    def greedy_row(used, start=0, skip_prob=0.0, rng=None):
        """Build one row greedily avoiding differences in 'used'."""
        row = [0]
        x = start
        while len(row) < m:
            x += 1
            if skip_prob > 0 and rng is not None and rng.random() < skip_prob:
                continue
            ok = True
            for a in row:
                if (x - a) in used:
                    ok = False
                    break
            if ok:
                for a in row:
                    used.add(x - a)
                row.append(x)
        return row

    def compact_rows(rows):
        """Right-to-left compaction: reduce each element to minimum feasible value."""
        all_diffs = set()
        for row in rows:
            all_diffs |= row_diffs(row)

        improved = True
        max_rounds = 50
        rounds = 0
        while improved and rounds < max_rounds:
            improved = False
            rounds += 1
            for r_idx in range(len(rows)):
                row = rows[r_idx]
                for pos in range(len(row) - 1, 0, -1):
                    cur = row[pos]
                    if cur <= row[pos - 1] + 1:
                        continue
                    for nv in range(row[pos - 1] + 1, cur):
                        old_d = set()
                        for i in range(pos):
                            old_d.add(cur - row[i])
                        for j in range(pos + 1, len(row)):
                            old_d.add(row[j] - cur)
                        new_d = set()
                        for i in range(pos):
                            new_d.add(nv - row[i])
                        for j in range(pos + 1, len(row)):
                            new_d.add(row[j] - nv)
                        conflict = False
                        for d in new_d:
                            if d in old_d:
                                continue
                            if d in all_diffs:
                                conflict = True
                                break
                        if not conflict:
                            for d in old_d:
                                all_diffs.discard(d)
                            for d in new_d:
                                all_diffs.add(d)
                            row[pos] = nv
                            improved = True
                            break
        return rows

    # ---------- backtracking search ----------

    def backtrack_search(M, max_rows_to_find, time_limit):
        """
        Try to find 'max_rows_to_find' rows with max entry <= M.
        Uses backtracking with forward checking.
        Returns list of rows found, or None if timeout/no solution.
        """
        start_time = time.time()
        used_diffs = set()
        rows_found = []

        # Precompute: for a given partial row (list of ints starting with 0),
        # and a current max M, we try to extend to full length m.
        # We build rows one at a time.

        def build_row(row, pos, next_val, used_diffs):
            """
            Recursively build a row of length m.
            row: current partial row (list)
            pos: next position to fill (index in row)
            next_val: minimum value to try at pos
            used_diffs: set of differences already used globally
            Returns True if row completed, False otherwise.
            """
            if time.time() - start_time > time_limit:
                return False

            if pos == m:
                return True

            # Try values from next_val to M
            for v in range(next_val, M + 1):
                # Check if adding v creates any conflict
                ok = True
                new_diffs = []
                for i in range(pos):
                    d = v - row[i]
                    if d in used_diffs:
                        ok = False
                        break
                    new_diffs.append(d)
                if not ok:
                    continue

                # Also check: for future positions, the minimum possible value
                # is v+1, so differences to future elements will be at least 1.
                # We just need to ensure no immediate conflict.

                row.append(v)
                for d in new_diffs:
                    used_diffs.add(d)

                if build_row(row, pos + 1, v + 1, used_diffs):
                    return True

                # Backtrack
                for d in new_diffs:
                    used_diffs.discard(d)
                row.pop()

            return False

        # Build rows one by one
        for r_idx in range(max_rows_to_find):
            if time.time() - start_time > time_limit:
                return None

            row = [0]
            # First element is 0. Try to build the rest.
            if not build_row(row, 1, 1, used_diffs):
                return None

            rows_found.append(row)

        return rows_found

    # ---------- main ----------

    # Baseline: deterministic greedy
    u = set()
    base_rows = []
    for _ in range(needed):
        base_rows.append(greedy_row(u, 0, 0.0, None))
    base_rows = compact_rows(base_rows)
    best_scope = max(max(r) for r in base_rows)
    best_rows = [list(r) for r in base_rows]

    rng = random.Random(42)

    # Try randomized greedy to get a better initial upper bound
    for trial in range(20):
        if time.time() > deadline:
            break
        u = set()
        trial_rows = []
        for _ in range(needed):
            skip = rng.choice([0.0, 0.05, 0.1, 0.15])
            row = greedy_row(u, 0, skip, rng)
            trial_rows.append(row)
        trial_rows = compact_rows(trial_rows)
        scope = max(max(r) for r in trial_rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in trial_rows]

    # Iterative deepening backtracking: try M from lower bound up to best_scope - 1
    # Lower bound: counting argument. Total distinct differences needed = n_rows * k*(k+1)/2
    # Max possible differences in [1, M] = M*(M+1)/2
    # So M*(M+1)/2 >= n_rows * k*(k+1)/2  =>  M >= sqrt(n_rows * k*(k+1))
    import math
    lb = int(math.ceil(math.sqrt(needed * k * (k + 1))))
    # Also, each row has k+1 entries starting at 0, so max entry >= k (trivial)
    lb = max(lb, k)

    # Try M values from best_scope - 1 down to lb
    # We want the smallest M that works. Try from lb upward, but that could be slow.
    # Better: try from best_scope - 1 downward to find if we can improve.
    # Actually, we want to find the minimum M. Let's try from lb upward with a time limit.
    # But that could take forever. Instead, let's try a few M values near best_scope.

    # Try M = best_scope - 1, best_scope - 2, ... down to max(lb, best_scope - 10)
    for M in range(best_scope - 1, max(lb, best_scope - 15) - 1, -1):
        if time.time() > deadline:
            break
        time_left = deadline - time.time()
        if time_left < 1.0:
            break
        bnb_time = min(time_left - 0.5, 5.0)  # limit per M attempt
        result = backtrack_search(M, needed, bnb_time)
        if result is not None:
            # Found a valid set with scope M
            result = compact_rows(result)
            actual_scope = max(max(r) for r in result)
            if actual_scope < best_scope:
                best_scope = actual_scope
                best_rows = [list(r) for r in result]
            # Try to go even lower
            break

    # Final compaction on best
    best_rows = compact_rows(best_rows)
    best_scope = max(max(r) for r in best_rows)

    return best_rows


if __name__ == "__main__":
    sol = construct(4, 7)
    used = set()
    valid = True
    for row in sol:
        if len(row) != 8 or row[0] != 0:
            valid = False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    valid = False
                used.add(d)
    print("n_rows=4 k=7 valid:", valid, "scope:", max(max(r) for r in sol))
