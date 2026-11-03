import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs + 1  # counting lower bound on scope

    # Bitmask size: we'll track differences up to some max
    # For (4,7): total_diffs = 4*7*8//2 = 112, so scope >= 113
    # We'll use a generous max_diff
    max_diff = 500  # generous; we can extend if needed

    def make_bitset(size):
        """Create a bitset as a Python int (bits 0..size-1)."""
        return 0

    def bitset_test(bs, d):
        return (bs >> d) & 1

    def bitset_set(bs, d):
        return bs | (1 << d)

    def bitset_clear(bs, d):
        return bs & ~(1 << d)

    def validate_rows(rows):
        """Check all differences are distinct across all rows."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def get_scope(rows):
        return max(max(r) for r in rows)

    def build_rows_csp(seed, max_scope_hint=None):
        """
        Build rows using backtracking with forward checking.
        Each row is built element by element, checking against ALL used differences.
        """
        rng = random.Random(seed)
        used = set()  # all differences used so far
        rows = []

        if max_scope_hint is None:
            max_scope_hint = lb + 80

        for row_idx in range(n_rows):
            row = [0]
            # Build remaining k elements
            success = _build_row_backtrack(
                row, k + 1, used, max_scope_hint, rng, row_idx
            )
            if not success:
                return None
            rows.append(row)

        return rows

    def _build_row_backtrack(row, target_len, used, max_scope, rng, row_idx, depth=0):
        """
        Recursively build the row, trying to keep values small.
        Uses forward checking: for each candidate, verify no difference conflicts.
        """
        if len(row) == target_len:
            return True

        if depth > 200:  # safety
            return False

        last = row[-1]

        # Generate candidate values, biased toward smaller
        # Try a window of candidates
        candidates = []
        # Primary range: compact
        primary_end = min(last + 40, max_scope)
        candidates.extend(range(last + 1, primary_end + 1))

        # Add some spread candidates
        spread_end = min(last + 80, max_scope)
        candidates.extend(range(primary_end + 1, spread_end + 1))

        # Shuffle with bias: keep small ones more likely first
        # Use a weighted shuffle
        if rng.random() < 0.5:
            rng.shuffle(candidates)
        else:
            # Partial shuffle: keep first half mostly ordered
            half = len(candidates) // 2
            if half > 0:
                rng.shuffle(candidates[half:])

        for x in candidates:
            if x > max_scope:
                break
            # Check: for each a in row, x - a must not be in used
            valid = True
            new_diffs = []
            for a in row:
                d = x - a
                if d in used:
                    valid = False
                    break
                new_diffs.append(d)

            if not valid:
                continue

            # Forward check: can we still complete the row?
            # Heuristic: check that for the next few elements, there's room
            # Simple check: ensure x + (target_len - len(row) - 1) <= max_scope
            if x + (target_len - len(row) - 1) > max_scope:
                continue

            # Accept
            row.append(x)
            used.update(new_diffs)

            if _build_row_backtrack(row, target_len, used, max_scope, rng, row_idx, depth + 1):
                return True

            # Backtrack
            row.pop()
            for d in new_diffs:
                used.discard(d)

        # If we exhausted candidates in the primary window, try wider
        wide_start = max_scope
        for x in range(wide_start + 1, max_scope + 200):
            valid = True
            new_diffs = []
            for a in row:
                d = x - a
                if d in used:
                    valid = False
                    break
                new_diffs.append(d)
            if not valid:
                continue
            row.append(x)
            used.update(new_diffs)
            if _build_row_backtrack(row, target_len, used, max_scope, rng, row_idx, depth + 1):
                return True
            row.pop()
            for d in new_diffs:
                used.discard(d)

        return False

    def compact_rows(rows, seed, max_iters=100000):
        """
        Aggressively compact: try to move every element leftward.
        Uses a bitmask for fast difference checking.
        """
        rng = random.Random(seed)
        rows = [list(r) for r in rows]

        # Build the set of all differences
        all_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    all_diffs.add(row[j] - row[i])

        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1

            # Try each element in each row (except 0)
            for r_idx in range(n_rows):
                row = rows[r_idx]
                for e_idx in range(1, len(row)):
                    old_val = row[e_idx]
                    if old_val <= row[e_idx - 1] + 1:
                        continue

                    # Compute differences involving this element
                    # These will be removed
                    old_diffs = set()
                    for j in range(len(row)):
                        if j == e_idx:
                            continue
                        if j < e_idx:
                            old_diffs.add(old_val - row[j])
                        else:
                            old_diffs.add(row[j] - old_val)

                    # Try new values from old_val-1 down to row[e_idx-1]+1
                    prev_val = row[e_idx - 1]
                    next_val = row[e_idx + 1] if e_idx + 1 < len(row) else float('inf')

                    for new_val in range(old_val - 1, prev_val, -1):
                        if new_val >= next_val:
                            continue

                        # New differences
                        new_diffs = set()
                        for j in range(len(row)):
                            if j == e_idx:
                                continue
                            if j < e_idx:
                                new_diffs.add(new_val - row[j])
                            else:
                                new_diffs.add(row[j] - new_val)

                        # Check: new_diffs must not conflict with (all_diffs - old_diffs)
                        valid = True
                        for d in new_diffs:
                            if d in all_diffs and d not in old_diffs:
                                valid = False
                                break

                        if valid:
                            # Apply
                            row[e_idx] = new_val
                            all_diffs = (all_diffs - old_diffs) | new_diffs
                            improved = True
                            break

        return rows

    def local_search(rows, seed, max_iters=20000):
        """
        Local search: try to reduce scope by moving elements.
        """
        rng = random.Random(seed)
        rows = [list(r) for r in rows]
        scope = get_scope(rows)

        all_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    all_diffs.add(row[j] - row[i])

        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break

            # Pick a random row and element
            r_idx = rng.randint(0, n_rows - 1)
            row = rows[r_idx]
            e_idx = rng.randint(1, len(row) - 1)
            old_val = row[e_idx]

            if old_val <= row[e_idx - 1] + 1:
                continue

            prev_val = row[e_idx - 1]
            next_val = row[e_idx + 1] if e_idx + 1 < len(row) else float('inf')

            # Try to reduce
            best_new = None
            for new_val in range(old_val - 1, prev_val, -1):
                if new_val >= next_val:
                    continue

                old_diffs = set()
                for j in range(len(row)):
                    if j == e_idx:
                        continue
                    if j < e_idx:
                        old_diffs.add(old_val - row[j])
                    else:
                        old_diffs.add(row[j] - old_val)

                new_diffs = set()
                for j in range(len(row)):
                    if j == e_idx:
                        continue
                    if j < e_idx:
                        new_diffs.add(new_val - row[j])
                    else:
                        new_diffs.add(row[j] - new_val)

                valid = True
                for d in new_diffs:
                    if d in all_diffs and d not in old_diffs:
                        valid = False
                        break

                if valid:
                    best_new = new_val
                    break

            if best_new is not None:
                row[e_idx] = best_new
                old_diffs = set()
                for j in range(len(row)):
                    if j == e_idx:
                        continue
                    if j < e_idx:
                        old_diffs.add(old_val - row[j])
                    else:
                        old_diffs.add(row[j] - old_val)
                new_diffs = set()
                for j in range(len(row)):
                    if j == e_idx:
                        continue
                    if j < e_idx:
                        new_diffs.add(best_new - row[j])
                    else:
                        new_diffs.add(row[j] - best_new)
                all_diffs = (all_diffs - old_diffs) | new_diffs

                new_scope = get_scope(rows)
                if new_scope < scope:
                    scope = new_scope

        return rows, scope

    best_rows = None
    best_scope = float('inf')

    # Phase 1: CSP-based construction with multiple seeds
    # Try with different scope hints
    for attempt in range(2000):
        if time.time() > deadline - 10:
            break

        seed = attempt * 7919 + 13
        # Vary the scope hint
        scope_hint = lb + 30 + (attempt % 50)

        rows = build_rows_csp(seed, max_scope_hint=scope_hint)
        if rows is not None:
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                if best_scope <= lb + 5:
                    break

    if best_rows is None:
        # Fallback: simple greedy
        for attempt in range(500):
            if time.time() > deadline - 10:
                break
            rng = random.Random(attempt * 104729 + 3)
            used = set()
            rows = []
            success = True
            for _ in range(n_rows):
                row = [0]
                while len(row) < k + 1:
                    last = row[-1]
                    found = False
                    for x in range(last + 1, last + 200):
                        new_diffs = [x - a for a in row]
                        if any(d in used for d in new_diffs):
                            continue
                        row.append(x)
                        used.update(new_diffs)
                        found = True
                        break
                    if not found:
                        success = False
                        break
                if not success:
                    break
                rows.append(row)
            if success:
                scope = get_scope(rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = rows

    if best_rows is None:
        # Last resort: return a valid but large construction
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last = row[-1]
                for x in range(last + 1, last + 1000):
                    new_diffs = [x - a for a in row]
                    if any(d in used for d in new_diffs):
                        continue
                    row.append(x)
                    used.update(new_diffs)
                    break
            rows.append(row)
        return rows

    # Phase 2: Compaction
    if time.time() < deadline - 15:
        compacted = compact_rows(best_rows, seed=42, max_iters=50000)
        scope = get_scope(compacted)
        if scope < best_scope:
            best_scope = scope
            best_rows = compacted

    # Phase 3: Local search
    if time.time() < deadline - 10:
        for ls_seed in range(10):
            if time.time() > deadline - 5:
                break
            improved, imp_scope = local_search(best_rows, 100000 + ls_seed * 31, max_iters=10000)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved

    # Phase 4: Final compaction
    if time.time() < deadline - 5:
        compacted = compact_rows(best_rows, seed=999, max_iters=30000)
        scope = get_scope(compacted)
        if scope < best_scope:
            best_scope = scope
            best_rows = compacted

    # Validate
    if not validate_rows(best_rows):
        # Shouldn't happen, but safety
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last = row[-1]
                for x in range(last + 1, last + 2000):
                    new_diffs = [x - a for a in row]
                    if any(d in used for d in new_diffs):
                        continue
                    row.append(x)
                    used.update(new_diffs)
                    break
            rows.append(row)
        return rows

    return best_rows


# Quick self-test
if __name__ == "__main__":
    result = construct(4, 7)
    if result:
        print(f"Scope: {max(max(r) for r in result)}")
        print(f"Rows: {result}")
        # Validate
        used = set()
        valid = True
        for row in result:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        valid = False
                        print(f"Duplicate diff {d}")
                    used.add(d)
        print(f"Valid: {valid}")
    else:
        print("No valid construction found")
