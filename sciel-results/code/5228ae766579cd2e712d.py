"""Difference triangle set (4,7): CSP over difference values with backtracking + forward checking."""
import random
import sys
import time
sys.setrecursionlimit(100000)

def construct(n_rows, k):
    # We need n_rows rows, each a list of k+1 increasing integers starting at 0,
    # such that ALL positive differences a_j - a_i (i<j) across ALL rows are distinct.
    # Minimize the largest entry (the scope).
    #
    # Approach: backtracking row-builder with global difference tracking,
    # forward checking, and randomized restarts to escape local minima.

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
    def try_build_row(used, k, scope, rng, strategy="smallest"):
        """Try to build ONE row (k+1 marks, starting at 0) within scope,
        given the global used difference set. Returns (row, new_used) or None."""
        row = [0]
        def rec(row, used_local):
            if len(row) == k + 1:
                return row, used_local
            last = row[-1]
            remaining_marks = (k + 1) - len(row)
            # We need remaining_marks more marks (including the next one)
            # The next mark must leave room for (remaining_marks - 1) more marks
            remaining_after = remaining_marks - 1
            max_x = scope - remaining_after
            if max_x < last + 1:
                return None
            
            # Collect feasible candidates
            candidates = []
            for x in range(last + 1, max_x + 1):
                ok = True
                for a in row:
                    d = x - a
                    if d in used_local:
                        ok = False
                        break
                if ok:
                    # Forward check: can we still fit remaining marks?
                    # The gap from x to scope must accommodate (remaining_after) marks
                    # each at least 1 apart, so we need scope - x >= remaining_after
                    # (already ensured by max_x)
                    # Additional check: the differences from x to future marks must be available
                    # This is hard to check exactly, so we use a heuristic:
                    # count how many values in [x+1, scope] have all diffs to row+[x] unused
                    if remaining_after > 0:
                        # Quick feasibility: at least remaining_after values must exist
                        # in (x, scope] that don't conflict
                        feasible_count = 0
                        for y in range(x + 1, scope + 1):
                            if feasible_count >= remaining_after:
                                break
                            conflict = False
                            for a in row:
                                if (y - a) in used_local or (y - x) in used_local:
                                    conflict = True
                                    break
                            if not conflict:
                                feasible_count += 1
                        if feasible_count < remaining_after:
                            continue
                    candidates.append(x)
            
            if not candidates:
                return None
            
            # Choose candidate based on strategy
            if strategy == "smallest":
                x = candidates[0]
            elif strategy == "largest":
                x = candidates[-1]
            elif strategy == "random":
                x = rng.choice(candidates)
            elif strategy == "tightest":
                # Pick candidate that leaves the most "flexibility" for future marks
                # i.e., the one with the most available future positions
                best_x = candidates[0]
                best_score = -1
                for cx in candidates[:20]:  # limit to first 20 for speed
                    score = 0
                    for y in range(cx + 1, scope + 1):
                        conflict = False
                        for a in row:
                            if (y - a) in used_local or (y - cx) in used_local:
                                conflict = True
                                break
                        if not conflict:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_x = cx
                x = best_x
            else:
                x = candidates[0]
            
            new_used = used_local | {x - a for a in row}
            res = rec(row + [x], new_used)
            if res is not None:
                return res
            
            # Backtrack: try other candidates
            for x in candidates[1:10]:
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
        
        # Choose strategy based on seed for diversity
        strategies = ["smallest", "tightest", "random", "largest"]
        strategy = strategies[seed % len(strategies)]
        
        def rec_row(idx, used_local, rows_local):
            if nodes[0] > max_nodes:
                return None
            nodes[0] += 1
            if idx == n_rows:
                return rows_local
            res = try_build_row(used_local, k, scope, rng, strategy)
            if res is None:
                return None
            row, new_used = res
            out = rec_row(idx + 1, new_used, rows_local + [row])
            if out is not None:
                return out
            return None

        return rec_row(0, used, rows)

    # ---- Randomized search to beat the greedy scope ----
    t0 = time.time()
    time_budget = 105  # leave margin under 120s

    # Try scopes from best_scope-1 down to a reasonable lower bound
    # For (4,7), known optimum is 134, counting bound is 112
    lower_bound = max(0, best_scope - 60)
    
    found = None
    # First, try scopes near the greedy solution with multiple restarts
    for scope in range(best_scope - 1, lower_bound - 1, -1):
        if time.time() - t0 > time_budget:
            break
        # Number of restarts decreases as we go lower (harder scopes)
        num_restarts = min(12, max(4, 20 - (best_scope - scope) // 3))
        for seed in range(num_restarts):
            if time.time() - t0 > time_budget:
                break
            res = build_all(scope, seed, max_nodes=100000)
            if res is not None:
                found = res
                best = res
                best_scope = scope
                break
        if found is not None:
            # Try to improve further
            continue
    
    # If we found something, try a few more targeted searches at the best scope
    if found is not None and time.time() - t0 < time_budget - 10:
        for seed in range(10, 20):
            if time.time() - t0 > time_budget:
                break
            res = build_all(best_scope - 1, seed, max_nodes=80000)
            if res is not None:
                best = res
                best_scope = best_scope - 1
                break

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
