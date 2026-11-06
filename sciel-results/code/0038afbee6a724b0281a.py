"""Optimized Sidon set construction using bitset-based greedy set packing and MRV backtracking."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Bitmask for differences 1..30
    # bit d-1 represents difference d
    FULL_MASK = (1 << 30) - 1  # bits 0..29 for diffs 1..30

    def diff_mask(a):
        """Return bitmask of all positive differences in sorted set a."""
        mask = 0
        n = len(a)
        for i in range(n):
            ai = a[i]
            for j in range(i + 1, n):
                d = a[j] - ai
                if 1 <= d <= 30:
                    mask |= (1 << (d - 1))
        return mask

    def is_sidon(a):
        """Check if sorted list a is a Sidon set (all positive differences distinct)."""
        diffs = set()
        n = len(a)
        for i in range(n):
            ai = a[i]
            for j in range(i + 1, n):
                d = a[j] - ai
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def check_and_mask(a):
        """Check Sidon and return (ok, mask, max_val)."""
        diffs = set()
        n = len(a)
        mask = 0
        for i in range(n):
            ai = a[i]
            for j in range(i + 1, n):
                d = a[j] - ai
                if d in diffs:
                    return False, 0, 0
                diffs.add(d)
                if 1 <= d <= 30:
                    mask |= (1 << (d - 1))
        return True, mask, max(a) if a else 0

    # --- Known candidates with small diameter ---
    known_candidates = [
        [0, 1, 3, 7, 12, 21, 25],
        [0, 1, 3, 7, 12, 21, 28],
        [0, 1, 3, 7, 12, 25, 28],
        [0, 2, 5, 10, 18, 23, 28],
        [0, 1, 4, 10, 18, 23, 28],
        [0, 1, 2, 7, 12, 22, 28],
        [0, 1, 3, 8, 12, 22, 26],
        [0, 1, 4, 10, 12, 28, 32, 45, 63],
        [0, 1, 4, 10, 12, 28, 32, 45],
        [0, 1, 4, 10, 12, 28, 32, 45, 47],
        [0, 1, 4, 10, 12, 28, 32, 45, 49],
        [0, 1, 3, 7, 12, 21, 25, 28],
        [0, 1, 3, 7, 12, 21, 25, 26],
        [0, 1, 3, 7, 12, 21, 25, 27],
    ]

    best = None
    best_max = 10**6

    for c in known_candidates:
        c = sorted(set(c))
        if not c or len(c) > 32:
            continue
        ok, mask, mx = check_and_mask(c)
        if ok and mask == FULL_MASK:
            if mx < best_max:
                best = c
                best_max = mx

    if best is not None:
        return best

    # --- Bitset-based greedy set packing ---
    # Candidate pool: integers 1..100
    def greedy_set_packing(max_candidate=100):
        """Greedy: start with {0}, iteratively add element maximizing new coverage / diameter cost."""
        a = [0]
        covered_mask = 0
        remaining = FULL_MASK

        for _ in range(31):  # at most 31 more elements
            if remaining == 0:
                break
            if time.time() > deadline - 5:
                break

            best_x = None
            best_score = -1
            best_new_mask = 0
            best_new_max = 0

            for x in range(1, max_candidate + 1):
                if x in a:
                    continue
                trial = a + [x]
                trial.sort()
                if len(trial) > 32:
                    continue

                # Check Sidon
                diffs = set()
                ok = True
                new_mask = 0
                for i in range(len(trial)):
                    ti = trial[i]
                    for j in range(i + 1, len(trial)):
                        d = trial[j] - ti
                        if d in diffs:
                            ok = False
                            break
                        diffs.add(d)
                        if 1 <= d <= 30:
                            new_mask |= (1 << (d - 1))
                    if not ok:
                        break

                if not ok:
                    continue

                new_coverage = new_mask & remaining
                new_bits = bin(new_coverage).count('1')
                if new_bits == 0:
                    continue

                new_max = max(trial)
                # Score: coverage gained / diameter increase
                diameter_cost = new_max - max(a) if max(a) > 0 else new_max
                if diameter_cost <= 0:
                    diameter_cost = 1
                score = new_bits / diameter_cost

                if score > best_score:
                    best_score = score
                    best_x = x
                    best_new_mask = new_mask
                    best_new_max = new_max

            if best_x is None:
                # No progress possible with current max_candidate, expand
                break

            a.append(best_x)
            a.sort()
            covered_mask = best_new_mask
            remaining = FULL_MASK & ~covered_mask

        # Verify
        ok, mask, mx = check_and_mask(a)
        if ok and mask == FULL_MASK and mx < best_max:
            best = a
            best_max = mx
        return a

    # Run greedy with different candidate limits
    for limit in [50, 70, 90, 100, 120]:
        if time.time() > deadline - 10:
            break
        result = greedy_set_packing(limit)
        if result:
            ok, mask, mx = check_and_mask(result)
            if ok and mask == FULL_MASK and mx < best_max:
                best = result
                best_max = mx

    if best is None:
        # Fallback: simple greedy
        a = [0]
        covered = set()
        for d in range(1, 31):
            if d in covered:
                continue
            candidates = sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a))
            found = False
            for x in candidates:
                trial = sorted(a + [x])
                if is_sidon(trial):
                    a = trial
                    covered = set(trial[j] - trial[i] for i in range(len(trial)) for j in range(i+1, len(trial)))
                    found = True
                    break
            if not found:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if is_sidon(trial):
                        a = trial
                        covered = set(trial[j] - trial[i] for i in range(len(trial)) for j in range(i+1, len(trial)))
                        break
                    y += 1
        if best is None or max(a) < best_max:
            best = a
            best_max = max(a)

    # --- MRV Backtracking refinement ---
    # Try to find a solution with smaller diameter by backtracking
    if time.time() < deadline - 15:
        target_max = best_max - 1
        # Try progressively smaller targets
        for target in range(target_max, max(35, target_max - 20), -1):
            if time.time() > deadline - 5:
                break
            if not _mrv_backtrack(target, deadline):
                continue
            # If found, update
            ok, mask, mx = check_and_mask(best)
            if ok and mask == FULL_MASK and mx < best_max:
                best_max = mx

    # --- Local search: try to reduce max element ---
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            cur = list(best)
            max_val = max(cur)

            # Try replacing largest element with smaller unused value
            for new_val in range(1, max_val):
                if new_val in cur:
                    continue
                trial = [v for v in cur if v != max_val] + [new_val]
                trial.sort()
                if len(set(trial)) != len(trial):
                    continue
                ok, mask, mx = check_and_mask(trial)
                if ok and mask == FULL_MASK and mx < best_max:
                    best = trial
                    best_max = mx
                    improved = True
                    break

            # Try removing largest element if we have >7 elements
            if not improved and len(cur) > 7:
                trial = [v for v in cur if v != max_val]
                trial.sort()
                ok, mask, mx = check_and_mask(trial)
                if ok and mask == FULL_MASK and mx < best_max:
                    best = trial
                    best_max = mx
                    improved = True

    return best


def _mrv_backtrack(target_max, deadline):
    """MRV backtracking to find Sidon set with max <= target_max covering all diffs 1..30."""
    global best, best_max
    start_time = time.time()
    FULL_MASK = (1 << 30) - 1

    # Candidate pool: 0..target_max
    # We must include 0. Try to build a set from {1, ..., target_max}
    
    def backtrack(current_set, covered_mask, candidates):
        if time.time() > deadline - 3:
            return False
        if covered_mask == FULL_MASK:
            # Found valid solution
            global best, best_max
            trial = sorted(current_set)
            ok, mask, mx = check_and_mask(trial)
            if ok and mask == FULL_MASK and mx < best_max:
                best = trial
                best_max = mx
            return True

        if not candidates:
            return False

        # MRV: pick the uncovered difference that has fewest candidate realizations
        remaining = FULL_MASK & ~covered_mask
        # For each uncovered diff d, count how many pairs (a, b) in current_set + {x} 
        # where x in candidates could realize d
        # Actually, we need: for each candidate x, what new diffs does it add?
        # MRV: find the diff d in remaining that is "hardest" to cover
        
        # Simpler MRV: among uncovered diffs, find one with fewest ways to be realized
        # A diff d can be realized by adding x if there exists a in current_set such that |x-a|=d
        # i.e., x = a+d or x = a-d for some a in current_set
        
        diff_options = {}
        for d in range(1, 31):
            bit = 1 << (d-1)
            if not (remaining & bit):
                continue
            # Count candidates that could realize this diff
            count = 0
            for x in candidates:
                for a in current_set:
                    if abs(x - a) == d:
                        count += 1
                        break
            diff_options[d] = count

        if not diff_options:
            return False

        # Pick the diff with minimum options (MRV)
        min_diff = min(diff_options, key=lambda d: diff_options[d])
        if diff_options[min_diff] == 0:
            return False

        # Filter candidates: only those that could realize min_diff
        filtered = [x for x in candidates if any(abs(x - a) == min_diff for a in current_set)]
        if not filtered:
            return False

        # Try candidates in order of how many new diffs they cover
        def coverage_score(x):
            new_mask = 0
            for a in current_set:
                d = abs(x - a)
                if 1 <= d <= 30:
                    new_mask |= (1 << (d-1))
            return bin(new_mask & remaining).count('1')

        filtered.sort(key=coverage_score, reverse=True)

        for x in filtered:
            # Check Sidon condition
            trial = list(current_set) + [x]
            diffs = set()
            ok = True
            new_mask = 0
            for i in range(len(trial)):
                ti = trial[i]
                for j in range(i+1, len(trial)):
                    d = trial[j] - ti
                    if d in diffs:
                        ok = False
                        break
                    diffs.add(d)
                    if 1 <= d <= 30:
                        new_mask |= (1 << (d-1))
            if not ok:
                continue

            new_covered = covered_mask | new_mask
            new_candidates = [c for c in candidates if c != x]
            
            if backtrack(trial, new_covered, new_candidates):
                return True

        return False

    # Initial call: start with {0}, candidates are 1..target_max
    initial_candidates = list(range(1, target_max + 1))
    # Limit search: try at most 32 elements total (including 0)
    # To keep search tractable, limit initial candidates
    if len(initial_candidates) > 60:
        # Focus on smaller candidates first
        initial_candidates = list(range(1, 61))

    result = backtrack([0], 0, initial_candidates)
    return result
# EVOLVE-BLOCK-END
