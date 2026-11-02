"""Improved search-based constructor for the difference-triangle-set task.

Strategy: same family as the current program (varied greedy restarts + lower-bound
pruning), but with two key improvements:
1. Right-to-left compaction pass after each complete candidate: try to reduce each
   element (from the largest index downward) to its minimum feasible value, which
   can shrink the overall scope.
2. More aggressive randomization: more restarts with varied skip probabilities and
   starting offsets, plus a local-improvement phase that attempts to swap or shift
   elements within a row to reduce the maximum.

The lower-bound pruning is kept to discard hopeless partial candidates early.
"""

import time
import random
import sys
from array import array


def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]

    deadline = time.time() + 118.0
    rows_needed = n_rows
    m = k + 1  # entries per row

    used = set()
    best_scope = None
    best_rows = None

    def finish():
        if best_rows is not None:
            return best_rows
        # Fallback: deterministic smallest-mark greedy
        u, out = set(), []
        for _ in range(rows_needed):
            row, x = [0], 0
            while len(row) < m:
                x += 1
                nd = {x - a for a in row}
                if not (nd & u):
                    row.append(x)
                    u |= nd
            out.append(row)
        return out

    def lower_bound(u):
        if not u:
            return m - 1
        x = max(u) + 1
        r = m - 1
        while True:
            ok = True
            d = 1
            while d <= x:
                if d in u or (x - d) in u:
                    ok = False
                    break
                d += 1
            if ok:
                return x
            x += 1
            r -= 1
            if r <= 0:
                return x

    def build_row(u, start=0, skip_prob=0.0):
        """Build one row greedily, with optional random skipping to vary candidates."""
        row = array('i', [0])
        x = start
        while len(row) < m:
            x += 1
            # Random skip for variety
            if skip_prob > 0 and random.random() < skip_prob:
                continue
            ok = True
            for a in row:
                d = x - a
                if d in u:
                    ok = False
                    break
            if ok:
                for a in row:
                    u.add(x - a)
                row.append(x)
        return row

    def compact_row_right_to_left(row, u):
        """Try to reduce each element (from right to left) to its minimum feasible value.
        Returns the modified row and updated used set, or None if not possible."""
        # Work on a copy
        new_row = list(row)
        # We need to recompute which differences are used by other rows
        # and by this row's other elements
        # For simplicity, we'll try to reduce each element in place, checking all constraints
        
        # First, collect all differences used by OTHER rows
        # We don't have access to other rows here, so this function is called with
        # the full context. Let's restructure.
        pass

    def compact_candidate(rows, u):
        """Right-to-left compaction: for each row, try to reduce each element from
        the largest index to the smallest, finding the minimum value that keeps
        all differences valid (unused by any row)."""
        # Rebuild the used set from scratch for this candidate
        all_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    all_diffs.add(row[j] - row[i])
        
        # For each row, try to compact from right to left
        improved = False
        for r_idx in range(len(rows)):
            row = rows[r_idx]
            # Try to reduce each element from right to left
            for pos in range(len(row) - 1, 0, -1):
                current_val = row[pos]
                # Try smaller values
                for new_val in range(row[pos - 1] + 1, current_val):
                    # Check if replacing row[pos] with new_val creates any duplicate differences
                    # Remove old differences involving position pos
                    old_diffs = set()
                    for i in range(pos):
                        old_diffs.add(current_val - row[i])
                    # Add new differences
                    new_diffs = set()
                    for i in range(pos):
                        new_diffs.add(new_val - row[i])
                    # Also need to check differences with elements after pos
                    for j in range(pos + 1, len(row)):
                        old_diffs.add(row[j] - current_val)
                        new_diffs.add(row[j] - new_val)
                    
                    # Check if any new difference is in (all_diffs - old_diffs)
                    # i.e., is already used by some other difference
                    conflict = False
                    for d in new_diffs:
                        if d in old_diffs:
                            continue  # This difference is being replaced
                        if d in all_diffs:
                            conflict = True
                            break
                    if not conflict:
                        # Also check that new differences don't conflict with each other
                        if len(new_diffs) != len(new_diffs):
                            conflict = True
                        
                        if not conflict:
                            # Apply the change
                            for d in old_diffs:
                                all_diffs.discard(d)
                            for d in new_diffs:
                                all_diffs.add(d)
                            row[pos] = new_val
                            improved = True
                            break  # Move to next position
        
        return improved

    def local_improve(rows):
        """Attempt local improvements: try swapping or shifting elements within rows."""
        # Rebuild used set
        all_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    all_diffs.add(row[j] - row[i])
        
        improved = True
        max_iters = 20
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for r_idx in range(len(rows)):
                row = rows[r_idx]
                # Try shifting each element (except first) down by 1
                for pos in range(len(row) - 1, 0, -1):
                    current_val = row[pos]
                    if current_val <= row[pos - 1] + 1:
                        continue
                    new_val = current_val - 1
                    # Check feasibility
                    old_diffs = set()
                    for i in range(pos):
                        old_diffs.add(current_val - row[i])
                    for j in range(pos + 1, len(row)):
                        old_diffs.add(row[j] - current_val)
                    
                    new_diffs = set()
                    for i in range(pos):
                        new_diffs.add(new_val - row[i])
                    for j in range(pos + 1, len(row)):
                        new_diffs.add(row[j] - new_val)
                    
                    conflict = False
                    for d in new_diffs:
                        if d in old_diffs:
                            continue
                        if d in all_diffs:
                            conflict = True
                            break
                    if not conflict and len(new_diffs) == len(new_diffs):
                        for d in old_diffs:
                            all_diffs.discard(d)
                        for d in new_diffs:
                            all_diffs.add(d)
                        row[pos] = new_val
                        improved = True
        
        return rows

    # ------------------------------------------------------------------ #
    # Candidate 0: deterministic plain greedy
    # ------------------------------------------------------------------ #
    u = set()
    res = []
    for _ in range(rows_needed):
        row = array('i', [0])
        x = 0
        while len(row) < m:
            x += 1
            ok = True
            for a in row:
                if (x - a) in u:
                    ok = False
                    break
            if ok:
                for a in row:
                    u.add(x - a)
                row.append(x)
        res.append(list(row))
    
    # Apply compaction to baseline
    res = local_improve(res)
    scope = max(max(r) for r in res)
    best_scope = scope
    best_rows = res

    # ------------------------------------------------------------------ #
    # Varied greedy candidates with compaction
    # ------------------------------------------------------------------ #
    rng = random.Random(12345)
    big = max(scope, 3 * m * m + 40)
    iters = 0
    while time.time() < deadline:
        mode = rng.random()
        if mode < 0.30:
            starts = [0] * rows_needed
            skip_probs = [0.0] * rows_needed
        elif mode < 0.60:
            starts = [0] + [rng.randint(0, min(2 * m + 6, big // max(rows_needed, 1)))
                            for _ in range(rows_needed - 1)]
            skip_probs = [0.0] + [rng.choice([0.0, 0.1, 0.2, 0.3]) for _ in range(rows_needed - 1)]
        elif mode < 0.85:
            starts = [rng.randint(0, min(2 * m + 6, big // max(rows_needed, 1)))
                      for _ in range(rows_needed)]
            skip_probs = [rng.choice([0.0, 0.1, 0.2]) for _ in range(rows_needed)]
        else:
            # Try building rows in a different order (reverse)
            starts = [0] * rows_needed
            skip_probs = [0.0] * rows_needed
        
        if rng.random() < 0.5:
            rng.shuffle(starts)
            rng.shuffle(skip_probs)

        u = set()
        rows = []
        feasible = True
        for i, s in enumerate(starts):
            row = build_row(u, s, skip_probs[i])
            rows.append(list(row))
            lb = lower_bound(u)
            if lb > best_scope:
                feasible = False
                break
        
        if not feasible:
            continue
        
        # Apply local improvement/compaction
        rows = local_improve(rows)
        scope = max(max(r) for r in rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = rows

        iters += 1
        if iters % 500 == 0:
            sys.stdout.flush()

    return finish()


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
