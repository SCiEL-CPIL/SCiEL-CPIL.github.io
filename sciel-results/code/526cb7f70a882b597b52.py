import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s kill

    # Bitmask for differences 1..30 (bit d-1 represents difference d)
    TARGET_MASK = (1 << 30) - 1

    def solve_for_D(D):
        """Find a Sidon set in [0,D] containing 0, covering all diffs 1..30."""
        # Precompute for each candidate x in 1..D:
        #   - which diffs 1..30 it covers (as bitmask)
        #   - for each existing element y, the diff |x-y|
        # We'll use backtracking with bitsets for speed.

        # For a given current set S (sorted), adding x is valid iff:
        #   for all y in S: (x - y) not in used_diffs_set
        #   and (x - y) not equal to (x' - y') for any pair already in S
        # Actually simpler: all pairwise differences must be distinct.
        # When adding x to S, new diffs are {x - y : y in S}.
        # These must not collide with any existing diff in S.

        # Strategy: DFS adding elements in increasing order.
        # State: current elements list, used_diffs bitmask (for 1..30), 
        #        covered_mask (which of 1..30 are covered)
        #        Also track ALL used diffs (not just 1..30) for collision check.

        # For efficiency, precompute diff[x][y] = x - y for 0 <= y < x <= D
        # But D can be up to ~200, so D^2 is manageable.

        # We'll use a different approach: track used diffs as a set of integers.
        # But for speed with bitmasks for 1..30, let's do:

        # Precompute: for each x in 1..D, and each potential y in 0..x-1,
        # the difference d = x - y. If d <= 30, it affects the mask.

        # DFS:
        # elements: list (sorted)
        # used_small: bitmask of which diffs in 1..30 are already used
        # all_used: frozenset of all used differences (for collision checking)
        #   -> too expensive. Instead, use a boolean array or set.

        # Better: use a set for all used diffs, and bitmask for 1..30 coverage.

        best_result = [None]

        def dfs(elements, used_diffs_set, covered_mask, last_elem):
            if time.time() > deadline:
                return False

            if covered_mask == TARGET_MASK:
                best_result[0] = elements[:]
                return True

            # Pruning: can we still cover remaining diffs?
            remaining = TARGET_MASK ^ covered_mask
            # Max new diffs from adding one element = len(elements)
            # We need at least ceil(popcount(remaining) / len(elements)) more elements
            # But we also have Sidon constraint limiting how many we can add.
            # Simple: if no candidate works, return False.

            # Try candidates from last_elem+1 to D
            for x in range(last_elem + 1, D + 1):
                # Check if adding x creates any collision
                new_diffs = []
                valid = True
                for y in elements:
                    d = x - y  # x > y since sorted
                    if d in used_diffs_set:
                        valid = False
                        break
                    new_diffs.append(d)

                if not valid:
                    continue

                # Update covered mask
                new_mask = covered_mask
                for d in new_diffs:
                    if 1 <= d <= 30:
                        new_mask |= (1 << (d - 1))

                # Quick prune: if this doesn't add any new covered diff and we're deep, skip?
                # No, it might enable future additions.

                new_used = used_diffs_set | set(new_diffs)
                new_elements = elements + [x]

                if dfs(new_elements, new_used, new_mask, x):
                    return True

            return False

        # Initial state: elements = [0], used_diffs = empty, covered_mask = 0
        success = dfs([0], set(), 0, 0)
        if success and best_result[0] is not None:
            return best_result[0]
        return None

    # Try increasing D from a lower bound
    # Lower bound: need 30 distinct positive diffs, Sidon set of size k has k(k-1)/2 diffs
    # So k(k-1)/2 >= 30 => k >= 9 (since 9*8/2=36)
    # With k=9 in [0,D], max diff is D, need to cover 1..30
    # D must be at least 30 (to have diff 30), but practically larger.
    # Known: smallest Golomb ruler covering 1..30 has diameter around 70-80.

    for D in range(50, 200):
        if time.time() > deadline:
            break
        result = solve_for_D(D)
        if result is not None:
            return result

    # Fallback: greedy construction
    a = [0]
    all_diffs = set()
    for d_target in range(1, 31):
        if d_target in all_diffs:
            continue
        found = False
        for x in range(1, 500):
            if x in a:
                continue
            new_diffs = [x - y for y in a]
            if any(nd in all_diffs for nd in new_diffs):
                continue
            # Check all pairwise diffs distinct
            trial = sorted(a + [x])
            trial_diffs = set()
            ok = True
            for i in range(len(trial)):
                for j in range(i + 1, len(trial)):
                    dd = trial[j] - trial[i]
                    if dd in trial_diffs:
                        ok = False
                        break
                    trial_diffs.add(dd)
                if not ok:
                    break
            if ok:
                a = trial
                all_diffs = trial_diffs
                found = True
                break
        if not found:
            # Add two elements to force diff d_target
            y = 2 * max(a) + d_target + 1
            while True:
                trial = sorted(a + [y, y + d_target])
                trial_diffs = set()
                ok = True
                for i in range(len(trial)):
                    for j in range(i + 1, len(trial)):
                        dd = trial[j] - trial[i]
                        if dd in trial_diffs:
                            ok = False
                            break
                        trial_diffs.add(dd)
                    if not ok:
                        break
                if ok:
                    a = trial
                    all_diffs = trial_diffs
                    break
                y += 1
    return a
