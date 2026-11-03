"""Difference triangle set (4,7): CSP over difference values with backtracking + forward checking."""
import random
import sys
sys.setrecursionlimit(100000)

def construct(n_rows, k):
    # We need n_rows rows, each a list of k+1 increasing integers starting at 0,
    # such that ALL positive differences a_j - a_i (i<j) across ALL rows are distinct.
    # Minimize the largest entry (the scope).
    #
    # Approach: backtracking row-builder with global difference tracking,
    # forward checking, and randomized restarts to escape local minima.
    # We search for the smallest feasible scope M via iterative deepening,
    # but to respect the time budget we run a bounded randomized search that
    # returns the best valid configuration found, falling back to greedy.

    num_diffs_per_row = k * (k + 1) // 2

    # ---- Greedy baseline (always valid, gives an upper bound) ----
    def greedy():
        used, rows = set(), []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    best = greedy()
    best_scope = max(max(r) for r in best)

    # ---- Backtracking row-builder with forward checking ----
    # A row is built mark by mark. State: current row marks, global used set.
    # Forward check: before choosing next mark x, ensure all (x - a) for a in row
    # are unused globally AND that the remaining "room" can still fit the rest.

    def try_build_row(used, k, scope, rng, depth=0):
        """Try to build ONE row (k+1 marks, starting at 0) within scope,
        given the global used difference set. Returns (row, new_used) or None."""
        row = [0]
        # We'll do backtracking over the remaining k marks.
        # Use a recursive search with ordering by smallest feasible x.
        def rec(row, used_local):
            if len(row) == k + 1:
                return row, used_local
            last = row[-1]
            # Candidate next marks: must be > last, <= scope, and differences to all row marks unused.
            # To keep search tractable, try a limited window of smallest candidates first,
            # but also occasionally jump (guided by rng) to diversify.
            start = last + 1
            # Upper bound on next mark: we still need (k+1 - len(row)) - 1 more marks after this,
            # each at least 1 apart, and the final mark <= scope.
            remaining_after = (k + 1) - len(row) - 1
            max_x = scope - remaining_after
            if max_x < start:
                return None
            # Collect feasible candidates (differences to all current marks unused).
            candidates = []
            for x in range(start, max_x + 1):
                ok = True
                for a in row:
                    d = x - a
                    if d in used_local:
                        ok = False
                        break
                if ok:
                    candidates.append(x)
            if not candidates:
                return None
            # Order: mostly smallest-first, with a bit of randomization to diversify restarts.
            # Shuffle a small tail to allow jumps.
            if len(candidates) > 1 and rng.random() < 0.3:
                # Pick a random candidate from the first few, or a random one from a wider set
                pool = candidates[:max(1, len(candidates) // 4)]
                x = rng.choice(pool)
            else:
                x = candidates[0]
            new_used = used_local | {x - a for a in row}
            res = rec(row + [x], new_used)
            if res is not None:
                return res
            # If the chosen candidate failed, try the next candidates (limited backtracking).
            # To bound cost, only backtrack over a small number of alternatives.
            for x in candidates[1:6]:
                new_used = used_local | {x - a for a in row}
                res = rec(row + [x], new_used)
                if res is not None:
                    return res
            return None
        return rec(row, used)

    # ---- Full backtracking for all rows ----
    def build_all(scope, seed, max_nodes=200000):
        rng = random.Random(seed)
        used = set()
        rows = []
        nodes = [0]

        def rec_row(idx, used_local, rows_local):
            if nodes[0] > max_nodes:
                return None
            nodes[0] += 1
            if idx == n_rows:
                return rows_local
            res = try_build_row(used_local, k, scope, rng)
            if res is None:
                return None
            row, new_used = res
            out = rec_row(idx + 1, new_used, rows_local + [row])
            if out is not None:
                return out
            return None

        return rec_row(0, used, rows)

    # ---- Iterative deepening / randomized search to beat the greedy scope ----
    # We try to find a valid configuration with scope < best_scope.
    # We do a bounded number of randomized restarts at decreasing scopes.
    import time
    t0 = time.time()
    time_budget = 100  # leave margin under 120s

    # Candidate scopes to try: from best_scope-1 downward, but sample a few.
    # For (4,7) best_scope is greedy's value; known optimum is 134.
    # We try scopes in a smart order: first just below greedy, then down toward the bound.
    scopes_to_try = []
    for s in range(best_scope - 1, max(0, best_scope - 40), -1):
        scopes_to_try.append(s)
    # Also include a few lower scopes in case greedy is far from optimal.
    for s in range(max(0, best_scope - 40), max(0, best_scope - 80), -1):
        scopes_to_try.append(s)
    # De-duplicate preserving order
    seen = set()
    scopes_to_try = [s for s in scopes_to_try if not (s in seen or seen.add(s))]

    found = None
    for scope in scopes_to_try:
        if time.time() - t0 > time_budget:
            break
        # A few restarts per scope
        for seed in range(8):
            if time.time() - t0 > time_budget:
                break
            res = build_all(scope, seed, max_nodes=150000)
            if res is not None:
                found = res
                best = res
                best_scope = scope
                break
        if found is not None:
            break

    # ---- Local refinement: try to shave the scope by re-solving with a tighter bound ----
    # (Already handled by the loop above; keep the best.)

    # Validate the returned best (defensive).
    def validate(rows):
        all_diffs = set()
        for r in rows:
            if len(r) != k + 1:
                return False
            if r[0] != 0:
                return False
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d <= 0:
                        return False
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True

    if not validate(best):
        # Fallback to a fresh greedy (shouldn't happen).
        best = greedy()

    return best
