# EVOLVE-BLOCK-START
import time
import random
import sys

def construct(n_rows, k):
    start = time.time()
    deadline = start + 140  # safety margin before 150s limit

    def now():
        return time.time()

    def time_left():
        return deadline - now()

    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2

    sys.setrecursionlimit(100000)

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
            if not r or r[0] != 0:
                return False
            for a, b in zip(r, r[1:]):
                if b <= a:
                    return False
            # Check for internal duplicates first
            row_diffs = diff_set_of_row(r)
            # diff_set_of_row returns a set, so it hides duplicates.
            # We must check the list of differences explicitly.
            diffs_list = []
            n = len(r)
            for i in range(n):
                ai = r[i]
                for j in range(i + 1, n):
                    d = r[j] - ai
                    if d in row_diffs:
                        # This check is redundant if we just iterate, 
                        # but we need to ensure all are distinct.
                        pass
                    diffs_list.append(d)
            if len(diffs_list) != len(set(diffs_list)):
                return False
            
            for d in row_diffs:
                if d in used:
                    return False
                used.add(d)
        return True

    def max_val(rows):
        return max(max(r) for r in rows)

    # ---------- Bitset backtracking row builder ----------
    # used_bits: an integer whose bit d is 1 if difference d is already used.
    # We need differences up to scope, so we mask with (1 << (scope+1)) - 1.

    def build_row_backtrack(used_bits, scope, time_limit=None, rng=None, max_nodes=None):
        """Find a strictly increasing row of length row_len starting at 0,
        all <= scope, with no difference colliding with used_bits (and none
        internal to the row). Returns row or None."""
        if rng is None:
            rng = random
        mask = (1 << (scope + 1)) - 1
        nodes = [0]
        target = row_len

        def conflict(row, x, cur_used):
            # check x - a for a in row against used_bits (external),
            # cur_used (internal to current row prefix), and each other (seen)
            seen = 0
            for a in row:
                d = x - a
                if d > scope:
                    return False  # can't happen if x <= scope and a>=0
                bit = 1 << d
                if (used_bits & bit) or (seen & bit) or (cur_used & bit):
                    return True
                seen |= bit
            return False

        def dfs(row, used, depth):
            if max_nodes is not None:
                nodes[0] += 1
                if nodes[0] > max_nodes:
                    return None
            if time_limit is not None and now() > time_limit:
                return None
            if len(row) == target:
                return list(row)

            last = row[-1]
            rem = target - len(row)
            min_next = last + 1
            max_next = scope - (rem - 1)
            if min_next > max_next:
                return None

            # Candidate ordering: prefer smaller values to keep scope tight,
            # with a bit of randomization for diversity across restarts.
            cands = list(range(min_next, max_next + 1))
            r = rng.random()
            if r < 0.4:
                # mostly increasing, shuffle tail
                if len(cands) > 4:
                    split = min(3, len(cands))
                    head = cands[:split]
                    tail = cands[split:]
                    rng.shuffle(tail)
                    cands = head + tail
            elif r < 0.7:
                rng.shuffle(cands)
            # else: keep increasing

            for x in cands:
                if conflict(row, x, used):
                    continue
                # compute new used bits
                new_used = used
                for a in row:
                    d = x - a
                    new_used |= (1 << d)
                row.append(x)
                res = dfs(row, new_used, depth + 1)
                row.pop()
                if res is not None:
                    return res
            return None

        res = dfs([0], used_bits & mask, 0)
        return res

    def build_row_greedy(used_bits, scope=None, temp=0.0, rng=None):
        """Fast greedy row builder. Returns row or None."""
        if rng is None:
            rng = random
        row = [0]
        x = 0
        while len(row) < row_len:
            x += 1
            if scope is not None and x > scope:
                return None
            new_diffs = [x - a for a in row]
            conflict = False
            seen = 0
            for d in new_diffs:
                if d > 5000:
                    continue
                bit = 1 << d
                if (used_bits & bit) or (seen & bit):
                    conflict = True
                    break
                seen |= bit
            if conflict:
                continue
            if temp > 0 and rng.random() < temp and x < 500:
                continue
            row.append(x)
            for d in new_diffs:
                used_bits |= (1 << d)
        return row

    # ---------- Initial construction ----------
    rng = random
    rng.seed(42)

    # Greedy initial solution with a generous scope
    used_bits = 0
    rows = []
    for _ in range(n_rows):
        row = build_row_greedy(used_bits, scope=10000, temp=0.0, rng=rng)
        if row is None:
            # fallback
            row = list(range(row_len))
        rows.append(row)
        for d in diff_set_of_row(row):
            used_bits |= (1 << d)

    best_rows = [list(r) for r in rows]
    best_max = max_val(best_rows)

    # ---------- Multi-temperature greedy restarts ----------
    temps = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
    attempts = 0
    max_attempts = 150
    while time_left() > 40 and attempts < max_attempts:
        attempts += 1
        temp = temps[attempts % len(temps)]
        used_bits = 0
        trial_rows = []
        ok = True
        for _ in range(n_rows):
            if time_left() < 40:
                ok = False
                break
            row = build_row_greedy(used_bits, scope=best_max + 20, temp=temp, rng=rng)
            if row is None:
                ok = False
                break
            trial_rows.append(row)
            for d in diff_set_of_row(row):
                used_bits |= (1 << d)
        if not ok:
            continue
        mv = max_val(trial_rows)
        if mv < best_max:
            best_max = mv
            best_rows = [list(r) for r in trial_rows]

    # ---------- Compute used bits for all rows except index i ----------
    def used_bits_except(i):
        ub = 0
        for j, r in enumerate(best_rows):
            if j == i:
                continue
            for d in diff_set_of_row(r):
                ub |= (1 << d)
        return ub

    # ---------- Row replacement to reduce scope ----------
    def try_replace_row(i, scope_target, time_limit=None, rng=None, max_nodes=None):
        """Find a replacement for row i with max <= scope_target, disjoint from
        the differences of all other rows."""
        if rng is None:
            rng = random
        fixed_used = used_bits_except(i)
        for scope in range(scope_target, min(best_max, scope_target + 4) + 1):
            if time_limit is not None and now() > time_limit:
                return None
            res = build_row_backtrack(fixed_used, scope, time_limit=time_limit,
                                      rng=rng, max_nodes=max_nodes)
            if res is not None:
                new_rows = [r if j != i else res for j, r in enumerate(best_rows)]
                if validate(new_rows):
                    return res
        return None

    # Iterative row replacement to reduce scope
    improved = True
    while improved and time_left() > 30:
        improved = False
        order = sorted(range(n_rows), key=lambda i: best_rows[i][-1], reverse=True)
        for i in order:
            if time_left() < 30:
                break
            cur_max_row = best_rows[i][-1]
            target = cur_max_row - 1
            if target < k:
                continue
            new_row = try_replace_row(i, target, time_limit=deadline - 30, rng=rng,
                                      max_nodes=200000)
            if new_row is not None:
                new_rows = [r if j != i else new_row for j, r in enumerate(best_rows)]
                mv = max_val(new_rows)
                if mv < best_max:
                    best_max = mv
                    best_rows = new_rows
                    improved = True

    # ---------- Pair-swap refinement ----------
    if time_left() > 40:
        pairs = [(i, j) for i in range(n_rows) for j in range(i + 1, n_rows)]
        rng.shuffle(pairs)
        for (i, j) in pairs:
            if time_left() < 30:
                break
            fixed_used = 0
            for x, r in enumerate(best_rows):
                if x == i or x == j:
                    continue
                for d in diff_set_of_row(r):
                    fixed_used |= (1 << d)

            for scope in range(best_max - 1, max(best_max - 8, k) - 1, -1):
                if time_left() < 30:
                    break
                r1 = build_row_backtrack(fixed_used, scope, time_limit=deadline - 30,
                                         rng=rng, max_nodes=100000)
                if r1 is None:
                    continue
                ub1 = fixed_used
                for d in diff_set_of_row(r1):
                    ub1 |= (1 << d)
                r2 = build_row_backtrack(ub1, scope, time_limit=deadline - 30,
                                         rng=rng, max_nodes=100000)
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

    # ---------- Final polish: aggressive single-row backtracking ----------
    while time_left() > 12:
        any_improve = False
        order = sorted(range(n_rows), key=lambda i: best_rows[i][-1], reverse=True)
        for i in order:
            if time_left() < 12:
                break
            cur_max_row = best_rows[i][-1]
            for target in range(min(cur_max_row - 1, best_max - 1),
                                max(cur_max_row - 6, k) - 1, -1):
                if time_left() < 12:
                    break
                new_row = try_replace_row(i, target, time_limit=deadline - 12, rng=rng,
                                          max_nodes=150000)
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

    # ---------- Final validation and cleanup ----------
    for i in range(n_rows):
        r = best_rows[i]
        if not r or r[0] != 0:
            best_rows[i] = list(range(row_len))
        else:
            for a, b in zip(r, r[1:]):
                if b <= a:
                    best_rows[i] = list(range(row_len))
                    break

    return best_rows
# EVOLVE-BLOCK-END
