"""Difference triangle set (4,7) — bitset-accelerated constructive search."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    deadline = time.time() + 145.0  # 145s hard cap to stay under 150s wall-clock
    total_diffs = n_rows * (k * (k + 1) // 2)

    # Bitset size: we need to track differences up to some max.
    # Conservative upper bound: greedy ruler for n_rows rows of k+1 marks.
    # Each row's max is at most k*(k+1)//2 * n_rows roughly, but let's be safe.
    # For n_rows=4, k=7: total diffs = 4*28 = 112. Max diff in greedy is ~134.
    # We'll use a bitset of size 512 to be safe (covers up to 511).
    MAX_DIFF = 512
    bitset_mask = (1 << MAX_DIFF) - 1

    def greedy_row(used_bits, row_start=0):
        """Build one row greedily given current used difference bits."""
        row = [0]
        cur = 0
        while len(row) < k + 1:
            cur += 1
            # Check all differences cur - a for a in row
            ok = True
            for a in row:
                d = cur - a
                if d >= MAX_DIFF:
                    return None  # overflow, shouldn't happen with MAX_DIFF=512
                if (used_bits >> d) & 1:
                    ok = False
                    break
            if ok:
                row.append(cur)
                for a in row[:-1]:
                    d = cur - a
                    used_bits |= (1 << d)
        return row, used_bits

    def evaluate_scope(rows):
        """Return the largest entry across all rows."""
        return max(max(r) for r in rows)

    def validate(rows):
        """Check all positive differences are distinct across all rows."""
        seen = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    # --- Strategy 1: Pure greedy baseline ---
    def pure_greedy():
        used_bits = 0
        rows = []
        for _ in range(n_rows):
            row, used_bits = greedy_row(used_bits)
            if row is None:
                return None
            rows.append(row)
        return rows

    best_rows = pure_greedy()
    if best_rows is None:
        # Fallback: very simple construction
        rows = []
        for i in range(n_rows):
            rows.append([0] + [j * (i + 1) for j in range(1, k + 1)])
        return rows

    best_scope = evaluate_scope(best_rows)

    # --- Strategy 2: Randomized greedy with restarts ---
    # At each step, instead of always taking the smallest valid x,
    # occasionally take a slightly larger valid x to escape local optima.
    def randomized_greedy(seed, max_skip=3):
        rng = random.Random(seed)
        used_bits = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            cur = 0
            while len(row) < k + 1:
                # Find all valid candidates in a window
                candidates = []
                for delta in range(1, max_skip + 2):
                    x = cur + delta
                    ok = True
                    for a in row:
                        d = x - a
                        if d >= MAX_DIFF:
                            ok = False
                            break
                        if (used_bits >> d) & 1:
                            ok = False
                            break
                    if ok:
                        candidates.append(x)
                if not candidates:
                    # Expand search further
                    found = False
                    for delta in range(max_skip + 2, max_skip + 50):
                        x = cur + delta
                        ok = True
                        for a in row:
                            d = x - a
                            if d >= MAX_DIFF:
                                break
                            if (used_bits >> d) & 1:
                                ok = False
                                break
                        if ok:
                            candidates.append(x)
                            found = True
                            break
                    if not found:
                        return None
                # Pick: mostly smallest, sometimes random from top candidates
                if len(candidates) > 1 and rng.random() < 0.3:
                    # Pick randomly among first few
                    pick_idx = min(len(candidates) - 1, rng.randint(0, 2))
                    x = candidates[pick_idx]
                else:
                    x = candidates[0]
                row.append(x)
                for a in row[:-1]:
                    d = x - a
                    used_bits |= (1 << d)
                cur = x
            rows.append(row)
        return rows

    # Run randomized greedy with multiple seeds
    n_restarts = 200
    for seed in range(n_restarts):
        if time.time() > deadline - 10:
            break
        rows = randomized_greedy(seed)
        if rows is None:
            continue
        if not validate(rows):
            continue
        scope = evaluate_scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = rows

    # --- Strategy 3: Local search / hill climbing on the best solution ---
    # Try to reduce the scope by adjusting individual entries.
    def local_search(rows, max_iters=5000):
        """Try to reduce the maximum entry by local moves."""
        rows = [list(r) for r in rows]
        current_scope = evaluate_scope(rows)
        best_local = list(rows)
        best_local_scope = current_scope
        rng = random.Random(42)

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            # Pick a random row and a random position to try moving
            ri = rng.randint(0, n_rows - 1)
            # Try to reduce the last element or a random element
            pi = rng.randint(1, len(rows[ri]) - 1)
            old_val = rows[ri][pi]
            # Try reducing by 1, 2, or 3
            for reduction in [1, 2, 3]:
                new_val = old_val - reduction
                if new_val <= rows[ri][pi - 1]:
                    continue
                # Check if this move is valid
                rows[ri][pi] = new_val
                if validate(rows):
                    new_scope = evaluate_scope(rows)
                    if new_scope < best_local_scope:
                        best_local_scope = new_scope
                        best_local = [list(r) for r in rows]
                    # Keep the improvement
                    break
                else:
                    rows[ri][pi] = old_val

        return best_local, best_local_scope

    improved_rows, improved_scope = local_search(best_rows)
    if improved_scope < best_scope:
        best_scope = improved_scope
        best_rows = improved_rows

    # --- Strategy 4: Backtracking with bitset for small cases ---
    # For n_rows <= 4 and k <= 7, we can do targeted backtracking.
    if n_rows <= 4 and k <= 7:
        def backtrack_row(used_bits, row, remaining, max_val):
            """Backtrack to build a row with max entry <= max_val."""
            if len(row) == k + 1:
                return row, used_bits
            start = row[-1] + 1
            for x in range(start, min(max_val + 1, MAX_DIFF)):
                ok = True
                for a in row:
                    d = x - a
                    if d >= MAX_DIFF or ((used_bits >> d) & 1):
                        ok = False
                        break
                if ok:
                    new_used = used_bits
                    for a in row:
                        d = x - a
                        new_used |= (1 << d)
                    result = backtrack_row(new_used, row + [x], remaining - 1, max_val)
                    if result is not None:
                        return result
            return None

        # Try to find a solution with scope < best_scope
        target_scope = best_scope - 1
        while target_scope >= 50:
            if time.time() > deadline - 20:
                break
            used_bits = 0
            rows = []
            success = True
            for _ in range(n_rows):
                result = backtrack_row(used_bits, [0], k, target_scope)
                if result is None:
                    success = False
                    break
                row, used_bits = result
                rows.append(row)
            if success and validate(rows):
                scope = evaluate_scope(rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = rows
                break
            target_scope -= 1

    return best_rows
# EVOLVE-BLOCK-END
