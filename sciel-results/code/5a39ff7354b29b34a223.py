import time
import random
import copy

def construct(n_rows, k):
    """
    Constructs a difference triangle set with n_rows rows and k+1 entries per row.
    Uses randomized DFS for initial valid solution + Simulated Annealing to minimize scope.
    Bit-parallel difference tracking for speed.
    """
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s kill

    # Total number of differences per row: C(k+1, 2)
    n_diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * n_diffs_per_row

    # Maximum difference we'll ever consider (for bitmask sizing)
    # Scope is at most ~200, so bitmask up to 200 is fine
    MAX_DIFF = 300

    # Bit-parallel helpers
    def diff_bit(d):
        return 1 << d

    # Greedy initialization (produces a valid solution, but scope may be large)
    def greedy_init():
        used_mask = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs_mask = 0
                    valid = True
                    for a in row:
                        d = x - a
                        if d > MAX_DIFF:
                            valid = False
                            break
                        bit = diff_bit(d)
                        if used_mask & bit:
                            valid = False
                            break
                        if new_diffs_mask & bit:
                            valid = False
                            break
                        new_diffs_mask |= bit
                    if valid:
                        row.append(x)
                        used_mask |= new_diffs_mask
                        break
                    x += 1
            rows.append(row)
        return rows

    # Randomized DFS to find a valid solution under target_max
    def dfs_find(target_max, seed):
        random.seed(seed)
        used_mask = 0
        rows = []

        def backtrack(row_idx, current_row):
            nonlocal used_mask, rows
            if time.time() - start_time > time_limit - 5:
                return False
            if row_idx == n_rows:
                return True

            if len(current_row) == k + 1:
                rows.append(current_row[:])
                return backtrack(row_idx + 1, [0])

            slots_left = (k + 1) - len(current_row)
            min_next = current_row[-1] + 1
            if min_next + slots_left - 1 > target_max:
                return False

            # Generate candidates: try smallest first with some randomness
            candidates = list(range(min_next, target_max + 1))
            # Shuffle a bit for diversity, but bias toward small
            # Use a weighted approach: keep small values more likely
            if len(candidates) > 20:
                # Take a random subset plus the smallest few
                small = candidates[:min(10, len(candidates))]
                rest = candidates[10:]
                random.shuffle(rest)
                candidates = small + rest[:min(30, len(rest))]
            else:
                random.shuffle(candidates)
                candidates.sort(key=lambda x: x if random.random() < 0.7 else -x)

            for x in candidates:
                new_diffs_mask = 0
                valid = True
                for a in current_row:
                    d = x - a
                    if d > MAX_DIFF:
                        valid = False
                        break
                    bit = diff_bit(d)
                    if used_mask & bit:
                        valid = False
                        break
                    if new_diffs_mask & bit:
                        valid = False
                        break
                    new_diffs_mask |= bit
                if valid:
                    current_row.append(x)
                    old_mask = used_mask
                    used_mask |= new_diffs_mask
                    if backtrack(row_idx, current_row):
                        return True
                    used_mask = old_mask
                    current_row.pop()

            return False

        success = backtrack(0, [0])
        if success:
            return rows
        return None

    # Build difference count structure for SA
    # We'll track: for each difference value, how many times it appears
    # And a "conflict" count = sum over all d of max(0, count[d] - 1)
    # For speed, we use a list indexed by difference value

    def build_sa_state(rows):
        """Build difference counts array and conflict count."""
        counts = [0] * (MAX_DIFF + 1)
        for row in rows:
            for i in range(k + 1):
                for j in range(i + 1, k + 1):
                    d = row[j] - row[i]
                    if d <= MAX_DIFF:
                        counts[d] += 1
        conflicts = 0
        for c in counts:
            if c > 1:
                conflicts += c - 1
        scope = max(max(r) for r in rows)
        return counts, conflicts, scope

    # Phase 1: Try to find a good initial solution via randomized DFS
    # Start with greedy, then try DFS with progressively tighter bounds
    rows = greedy_init()
    best_rows = copy.deepcopy(rows)
    best_scope = max(max(r) for r in rows)

    # Try DFS with target = best_scope - 1, then -2, etc.
    # But limit the time spent here
    dfs_deadline = start_time + 20.0  # Spend at most 20s on DFS
    for target in range(best_scope - 1, max(50, best_scope - 30), -1):
        if time.time() > dfs_deadline:
            break
        result = dfs_find(target, seed=target)
        if result:
            s = max(max(r) for r in result)
            if s < best_scope:
                best_scope = s
                best_rows = copy.deepcopy(result)
                rows = copy.deepcopy(result)
                break

    # Phase 2: Simulated Annealing to minimize scope
    counts, conflicts, current_scope = build_sa_state(rows)

    # SA Parameters
    T = 5.0
    cooling = 0.9999995
    min_T = 0.005
    penalty = 500

    sa_deadline = start_time + time_limit - 5.0  # Leave 5s margin

    while time.time() < sa_deadline:
        # Pick a random move
        r_idx = random.randint(0, n_rows - 1)
        j_idx = random.randint(1, k)  # Never touch index 0

        old_val = rows[r_idx][j_idx]

        # Try a random step, biased toward smaller values
        if old_val > current_scope * 0.7:
            delta = random.randint(-8, 3)
        else:
            delta = random.randint(-5, 5)

        new_val = old_val + delta

        # Check bounds and ordering
        if new_val <= 0:
            continue
        if j_idx > 0 and new_val <= rows[r_idx][j_idx - 1]:
            continue
        if j_idx < k and new_val >= rows[r_idx][j_idx + 1]:
            continue

        row = rows[r_idx]

        # Calculate changes in conflicts using the counts array
        # Diffs involving position j_idx with all other positions in the row
        old_diffs = []
        new_diffs = []
        for i in range(k + 1):
            if i == j_idx:
                continue
            old_d = old_val - row[i] if old_val > row[i] else row[i] - old_val
            new_d = new_val - row[i] if new_val > row[i] else row[i] - new_val
            old_diffs.append(old_d)
            new_diffs.append(new_d)

        # Apply changes to counts
        # First decrement old diffs
        for d in old_diffs:
            if d <= MAX_DIFF:
                counts[d] -= 1
        # Increment new diffs
        for d in new_diffs:
            if d <= MAX_DIFF:
                counts[d] += 1

        # Recalculate conflicts (fast: iterate over counts array)
        new_conflicts = 0
        for c in counts:
            if c > 1:
                new_conflicts += c - 1

        # Calculate new scope
        if new_val > current_scope:
            new_scope = new_val
        elif old_val == current_scope:
            new_scope = max(max(r) for r in rows)
        else:
            new_scope = current_scope

        delta_scope = new_scope - current_scope
        delta_conflicts = new_conflicts - conflicts
        delta_cost = delta_scope + penalty * delta_conflicts

        # Acceptance
        accept = False
        if delta_cost < 0:
            accept = True
        else:
            if delta_cost < 500:
                prob = 2.71828 ** (-delta_cost / max(T, min_T))
                if random.random() < prob:
                    accept = True

        if accept:
            rows[r_idx][j_idx] = new_val
            current_scope = new_scope
            conflicts = new_conflicts

            if conflicts == 0 and current_scope < best_scope:
                best_scope = current_scope
                best_rows = copy.deepcopy(rows)
        else:
            # Rollback
            for d in old_diffs:
                if d <= MAX_DIFF:
                    counts[d] += 1
            for d in new_diffs:
                if d <= MAX_DIFF:
                    counts[d] -= 1

        T *= cooling
        if T < min_T:
            T = min_T

    # Final validation: ensure best_rows is valid
    if best_conflicts_check(best_rows):
        return best_rows
    else:
        # Fallback: return greedy (which is valid by construction)
        return greedy_init()

def best_conflicts_check(rows):
    """Validate that all differences are distinct across all rows."""
    seen = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True
