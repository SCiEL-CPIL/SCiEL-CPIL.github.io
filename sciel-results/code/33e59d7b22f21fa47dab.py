"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(n_rows, k):
    start = time.time()
    def now():
        return time.time()
    deadline = start + 145  # safety margin before 150s limit

    # Each row has k+1 elements, all differences positive.
    diffs_per_row = k * (k + 1) // 2

    def diff_set_of_row(row):
        s = set()
        n = len(row)
        for i in range(n):
            ai = row[i]
            for j in range(i + 1, n):
                s.add(row[j] - ai)
        return s

    def validate(rows):
        used = set()
        for r in rows:
            for d in diff_set_of_row(r):
                if d in used:
                    return False
                used.add(d)
            # also check strictly increasing and start at 0
        for r in rows:
            if not r or r[0] != 0:
                return False
            for a, b in zip(r, r[1:]):
                if b <= a:
                    return False
        return True

    def max_val(rows):
        return max(max(r) for r in rows)

    # ---------- Greedy row builder (baseline) ----------
    def greedy_row(used, temp=0.0, scope=None):
        """Build one row greedily. scope is an upper bound on the max element."""
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if scope is not None and x > scope:
                return None, used
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                if temp > 0 and random.random() < temp and x < 300:
                    continue
                row.append(x)
                used = used | new_diffs
        return row, used

    # ---------- Randomized backtracking row builder with scope bound ----------
    def backtrack_row(used, scope, time_limit=None, rng=None, max_restarts=5):
        """
        Build one row of size k+1 with max <= scope, all diffs disjoint from used.
        Returns (row, used_set) or (None, None). Uses randomized DFS with multiple restarts.
        """
        if rng is None:
            rng = random
        if time_limit is not None and now() > time_limit:
            return None, None

        target_len = k + 1

        for restart in range(max_restarts):
            if time_limit is not None and now() > time_limit:
                return None, None

            row = [0]
            cur_used = set(used)

            def dfs(row, cur_used):
                if time_limit is not None and now() > time_limit:
                    return None
                if len(row) == target_len:
                    return row, cur_used

                last = row[-1]
                rem = target_len - len(row)

                # Minimum possible next value
                min_next = last + 1
                # Maximum allowed: scope - (rem - 1) so we can still add rem-1 more
                max_next = scope - (rem - 1)

                if min_next > max_next:
                    return None

                # Generate candidate list and optionally shuffle for randomization
                candidates = list(range(min_next, max_next + 1))
                if restart > 0 and len(candidates) > 1:
                    # Partial shuffle: bias toward smaller values but add randomness
                    # Use a weighted approach: earlier candidates more likely but allow jumps
                    shuffle_point = min(len(candidates), 10 + restart * 5)
                    # Shuffle a portion to break determinism
                    candidates[:shuffle_point] = rng.sample(candidates[:shuffle_point], shuffle_point)

                for x in candidates:
                    if time_limit is not None and now() > time_limit:
                        return None

                    # Check conflicts
                    conflict = False
                    for a in row:
                        d = x - a
                        if d in cur_used:
                            conflict = True
                            break
                    if conflict:
                        continue

                    new_row = row + [x]
                    new_cur = cur_used.copy()
                    for a in row:
                        new_cur.add(x - a)

                    res = dfs(new_row, new_cur)
                    if res is not None:
                        return res
                return None

            res = dfs(row, cur_used)
            if res is not None:
                return res

        return None, None

    # ---------- Initial construction ----------
    random.seed(1234)
    # Start with pure greedy
    used = set()
    rows = []
    for _ in range(n_rows):
        row, used = greedy_row(used, temp=0.0)
        rows.append(row)
    best_rows = [list(r) for r in rows]
    best_max = max_val(best_rows)

    # ---------- Multi-temperature restarts for better initial ----------
    temps = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    attempts = 0
    while now() < deadline - 20 and attempts < 200:
        attempts += 1
        temp = temps[attempts % len(temps)]
        used = set()
        trial_rows = []
        ok = True
        for _ in range(n_rows):
            if now() > deadline - 20:
                ok = False
                break
            row, used = greedy_row(used, temp=temp)
            if row is None:
                ok = False
                break
            trial_rows.append(row)
        if not ok:
            continue
        mv = max_val(trial_rows)
        if mv < best_max:
            best_max = mv
            best_rows = [list(r) for r in trial_rows]

    # ---------- Backtracking improvement: replace rows with smaller ones ----------
    def try_replace_row(i, scope_target, time_limit=None, max_attempts=3):
        """Try to find a valid replacement for row i with max <= scope_target,
        given other rows fixed. Returns new row or None."""
        other_rows = [r for j, r in enumerate(best_rows) if j != i]
        fixed_used = set()
        for r in other_rows:
            fixed_used |= diff_set_of_row(r)
        # Use backtracking with the target scope, multiple attempts
        scope = scope_target
        for attempt in range(max_attempts):
            if time_limit is not None and now() > time_limit:
                return None
            res, _ = backtrack_row(fixed_used, scope, time_limit=time_limit, max_restarts=3)
            if res is not None:
                new_row = res
                # Validate full set
                new_rows = [r if j != i else new_row for j, r in enumerate(best_rows)]
                if validate(new_rows):
                    return new_row
        return None

    # Iteratively try to lower the scope
    # Determine current per-row maxes; try to reduce the bottleneck row.
    improved = True
    while improved and now() < deadline - 15:
        improved = False
        # Sort rows by their max descending; try to improve the largest first
        order = sorted(range(n_rows), key=lambda i: best_rows[i][-1], reverse=True)
        for i in order:
            if now() > deadline - 15:
                break
            cur_max_row = best_rows[i][-1]
            # Try to find a replacement with strictly smaller max
            target = cur_max_row - 1
            new_row = try_replace_row(i, target, time_limit=deadline - 15, max_attempts=3)
            if new_row is not None:
                new_rows = [r if j != i else new_row for j, r in enumerate(best_rows)]
                mv = max_val(new_rows)
                if mv < best_max:
                    best_max = mv
                    best_rows = new_rows
                    improved = True

    # ---------- Pair-swap refinement ----------
    if now() < deadline - 25:
        pairs = [(i, j) for i in range(n_rows) for j in range(i + 1, n_rows)]
        random.shuffle(pairs)
        for (i, j) in pairs:
            if now() > deadline - 15:
                break
            other_rows = [r for x, r in enumerate(best_rows) if x != i and x != j]
            fixed_used = set()
            for r in other_rows:
                fixed_used |= diff_set_of_row(r)
            # Build two rows with backtracking, trying to keep max low
            for scope in range(best_max - 1, best_max - 10, -1):
                if now() > deadline - 15:
                    break
                for attempt in range(2):
                    if now() > deadline - 15:
                        break
                    r1, used_after1 = backtrack_row(fixed_used, scope, time_limit=deadline - 15, max_restarts=3)
                    if r1 is None:
                        continue
                    r2, _ = backtrack_row(used_after1, scope, time_limit=deadline - 15, max_restarts=3)
                    if r2 is None:
                        continue
                    new_rows = [r if x not in (i, j) else (r1 if x == i else r2)
                                for x, r in enumerate(best_rows)]
                    if validate(new_rows):
                        mv = max_val(new_rows)
                        if mv < best_max:
                            best_max = mv
                            best_rows = new_rows
                        break
                else:
                    continue
                break

    # ---------- Final polish: single-row backtracking with tighter scopes ----------
    while now() < deadline - 5:
        any_improve = False
        order = sorted(range(n_rows), key=lambda i: best_rows[i][-1], reverse=True)
        for i in order:
            if now() > deadline - 5:
                break
            cur_max_row = best_rows[i][-1]
            # Try a few scopes below current row max
            for target in range(min(cur_max_row - 1, best_max - 1), max(cur_max_row - 8, 0) - 1, -1):
                if now() > deadline - 5:
                    break
                new_row = try_replace_row(i, target, time_limit=deadline - 5, max_attempts=2)
                if new_row is not None:
                    new_rows = [r if j != i else new_row for j, r in enumerate(best_rows)]
                    mv = max_val(new_rows)
                    if mv < best_max:
                        best_max = mv
                        best_rows = new_rows
                        any_improve = True
                        break
        if not any_improve:
            break

    return best_rows
# EVOLVE-BLOCK-END
