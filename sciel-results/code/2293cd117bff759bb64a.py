"""Difference triangle set (n_rows, k) constructor."""
import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # Each row has k+1 entries: a[0]=0 < a[1] < ... < a[k]
    # Differences: for each row, all a[j]-a[i] for i<j must be distinct across all rows.
    # Total differences per row: k*(k+1)/2
    # Total differences: n_rows * k*(k+1)/2
    total_diffs = n_rows * k * (k + 1) // 2

    def is_valid(rows):
        """Check if all differences are distinct."""
        seen = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    def scope(rows):
        """Return the largest entry across all rows."""
        return max(max(r) for r in rows)

    def count_collisions(rows):
        """Count how many differences are duplicated."""
        from collections import Counter
        cnt = Counter()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    cnt[row[j] - row[i]] += 1
        return sum(c - 1 for c in cnt.values() if c > 1)

    def greedy_construct(seed_val=0):
        """Greedy construction with optional randomization."""
        rng = random.Random(seed_val)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    def randomized_greedy(seed_val):
        """Randomized greedy: at each step, pick from a random subset of valid candidates."""
        rng = random.Random(seed_val)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Try a window of candidates
                best_x = None
                candidates = []
                for _try in range(min(50, 200)):
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        candidates.append(x)
                        if len(candidates) >= 5:
                            break
                if not candidates:
                    # Fallback: keep searching
                    while True:
                        x += 1
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            candidates.append(x)
                            break
                # Pick randomly from candidates (bias toward smaller)
                if len(candidates) > 1 and rng.random() < 0.3:
                    chosen = rng.choice(candidates)
                else:
                    chosen = candidates[0]
                row.append(chosen)
                used |= {chosen - a for a in row[:-1]}
            rows.append(row)
        return rows

    def local_search(rows, seed_val):
        """Local search: try to reduce scope by adjusting entries."""
        rng = random.Random(seed_val)
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])

        best_scope = scope(rows)
        best_rows = [list(r) for r in rows]
        current = [list(r) for r in rows]
        current_scope = best_scope

        for iteration in range(2000):
            if time.time() > deadline:
                break

            # Pick a random row and entry to modify
            r_idx = rng.randrange(n_rows)
            # Only modify non-zero entries
            if k == 0:
                break
            e_idx = rng.randrange(1, k + 1)

            old_val = current[r_idx][e_idx]
            # Try to move it to a smaller or different value
            direction = -1 if rng.random() < 0.7 else 1
            step = rng.randint(1, max(1, current_scope // 10))
            new_val = old_val + direction * step

            # Check constraints
            if new_val <= 0 or new_val >= current_scope + 1:
                continue
            # Must maintain strict increase within the row
            if e_idx > 0 and new_val <= current[r_idx][e_idx - 1]:
                continue
            if e_idx < k and new_val >= current[r_idx][e_idx + 1]:
                continue

            # Check all differences involving this entry
            row = current[r_idx]
            # Remove old differences
            old_diffs = set()
            for i in range(len(row)):
                if i != e_idx:
                    d = abs(row[e_idx] - row[i])
                    old_diffs.add(d)

            # Tentatively change
            row[e_idx] = new_val
            new_diffs = set()
            for i in range(len(row)):
                if i != e_idx:
                    d = abs(row[e_idx] - row[i])
                    new_diffs.add(d)

            # Check if any new diff is already used by other entries
            conflict = False
            for d in new_diffs:
                if d in used and d not in old_diffs:
                    # This diff was already used by something else
                    conflict = True
                    break

            if not conflict:
                # Update used set
                used -= old_diffs
                used |= new_diffs
                new_scope = max(max(r) for r in current)
                if new_scope < current_scope:
                    current_scope = new_scope
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [list(r) for r in current]
            else:
                # Revert
                row[e_idx] = old_val

        return best_rows, best_scope

    def simulated_annealing(rows, seed_val):
        """Simulated annealing to reduce scope."""
        rng = random.Random(seed_val)
        current = [list(r) for r in rows]
        current_scope = scope(current)
        best = [list(r) for r in current]
        best_scope = current_scope

        temperature = current_scope * 0.5
        cooling = 0.995

        for iteration in range(5000):
            if time.time() > deadline:
                break
            if temperature < 0.1:
                break

            r_idx = rng.randrange(n_rows)
            if k == 0:
                break
            e_idx = rng.randrange(1, k + 1)
            old_val = current[r_idx][e_idx]
            new_val = old_val + rng.randint(-max(1, current_scope // 20), max(1, current_scope // 20))

            if new_val <= 0:
                continue
            row = current[r_idx]
            if e_idx > 0 and new_val <= row[e_idx - 1]:
                continue
            if e_idx < k and new_val >= row[e_idx + 1]:
                continue

            # Compute change in scope
            old_max = max(row)
            row[e_idx] = new_val
            new_max = max(row)
            delta_scope = new_max - old_max if e_idx == k or new_val > old_max else 0
            # More careful:
            actual_new_scope = max(max(r) for r in current)
            delta = actual_new_scope - current_scope

            # Check validity (all diffs distinct)
            valid = True
            used_diffs = set()
            for ri in range(n_rows):
                for ii in range(len(current[ri])):
                    for jj in range(ii + 1, len(current[ri])):
                        d = current[ri][jj] - current[ri][ii]
                        if d in used_diffs:
                            valid = False
                            break
                        used_diffs.add(d)
                    if not valid:
                        break
                if not valid:
                    break

            if valid:
                if delta <= 0 or rng.random() < (10 ** (-delta / max(temperature, 0.01))):
                    current_scope = actual_new_scope
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best = [list(r) for r in current]
                else:
                    row[e_idx] = old_val
            else:
                row[e_idx] = old_val

            temperature *= cooling

        return best, best_scope

    # Phase 1: Get initial solutions from multiple strategies
    candidates = []

    # Pure greedy
    rows_greedy = greedy_construct(0)
    candidates.append(rows_greedy)

    # Randomized greedy with different seeds
    for seed in range(1, 11):
        if time.time() > deadline - 5:
            break
        rows = randomized_greedy(seed)
        candidates.append(rows)

    # Pick the best initial solution
    best_initial = min(candidates, key=scope)
    best_scope_val = scope(best_initial)

    # Phase 2: Local search
    if time.time() < deadline - 3:
        ls_result, ls_scope = local_search(best_initial, seed_val=42)
        if ls_scope < best_scope_val:
            best_initial = ls_result
            best_scope_val = ls_scope

    # Phase 3: Try simulated annealing on best result
    if time.time() < deadline - 2:
        sa_result, sa_scope = simulated_annealing(best_initial, seed_val=123)
        if sa_scope < best_scope_val:
            best_initial = sa_result
            best_scope_val = sa_scope

    # Final validation
    if not is_valid(best_initial):
        # Fallback to greedy
        best_initial = greedy_construct(0)

    return best_initial
