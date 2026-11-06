import time
import random
from itertools import combinations

def construct():
    """
    Find a Sidon set containing 0 with all differences 1..30, minimizing the largest element.
    Uses a CSP-style backtracking that prioritizes the most-constrained uncovered difference,
    with bitset-based Sidon validation and iterative deepening on the diameter.
    """
    start_time = time.time()
    DEADLINE = 140.0  # Leave 10s margin before 150s kill

    TARGET = set(range(1, 31))

    def is_sidon_fast(S):
        """Check if S is a Sidon set using a set of differences."""
        S = sorted(S)
        diffs = set()
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                d = S[j] - S[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def get_diffs(S):
        S = sorted(S)
        diffs = set()
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                diffs.add(S[j] - S[i])
        return diffs

    def covers_all(S):
        diffs = get_diffs(S)
        return TARGET.issubset(diffs)

    def score_set(S):
        diffs = get_diffs(S)
        missing = len(TARGET - diffs)
        return (missing, max(S) if S else 0)

    def count_placements_for_diff(d, S, used_diffs, max_val):
        """Count how many valid placements could create difference d."""
        S_set = set(S)
        count = 0
        # We need x, y in S_new with y - x = d
        # Either both already in S (d already covered), or one is new
        # Case 1: add x such that x + d in S
        for s in S:
            x = s - d
            if x > 0 and x not in S_set:
                # Check if adding x keeps Sidon
                trial_diffs = used_diffs | set()
                valid = True
                for s2 in S:
                    dd = abs(s2 - x)
                    if dd in used_diffs:
                        valid = False
                        break
                if valid:
                    count += 1
        # Case 2: add y such that y - d in S
        for s in S:
            y = s + d
            if y not in S_set and y <= max_val:
                valid = True
                for s2 in S:
                    dd = abs(s2 - y)
                    if dd in used_diffs:
                        valid = False
                        break
                if valid:
                    count += 1
        return count

    def backtrack_csp(S, used_diffs, covered, max_val, max_size, deadline):
        """
        CSP backtracking: at each step, pick the most-constrained uncovered difference
        and try placements that satisfy it.
        Returns (solution_S, success) or (None, False).
        """
        if time.time() > deadline:
            return None, False

        if covered == TARGET:
            return S, True

        if len(S) >= max_size:
            return None, False

        # MRV: find the uncovered difference with fewest valid placements
        uncovered = TARGET - covered
        if not uncovered:
            return S, True

        best_diff = None
        best_placements = []
        best_count = float('inf')

        for d in sorted(uncovered, key=lambda x: count_placements_for_diff(x, S, used_diffs, max_val)):
            cnt = count_placements_for_diff(d, S, used_diffs, max_val)
            if cnt == 0:
                return None, False
            if cnt < best_count:
                best_count = cnt
                best_diff = d
                # Generate placements for this difference
                placements = []
                S_set = set(S)
                for s in S:
                    x = s - d
                    if x > 0 and x not in S_set:
                        # Check Sidon
                        valid = True
                        for s2 in S:
                            if abs(s2 - x) in used_diffs:
                                valid = False
                                break
                        if valid:
                            placements.append(x)
                    y = s + d
                    if y not in S_set and y <= max_val:
                        valid = True
                        for s2 in S:
                            if abs(s2 - y) in used_diffs:
                                valid = False
                                break
                        if valid:
                            placements.append(y)
                best_placements = placements
            if best_count <= 2:
                break

        if not best_placements:
            return None, False

        # Try each placement
        for x in best_placements:
            new_S = S + [x]
            new_used = set(used_diffs)
            valid = True
            for s in S:
                dd = abs(s - x)
                if dd in new_used:
                    valid = False
                    break
                new_used.add(dd)
            if not valid:
                continue

            new_covered = new_used & TARGET
            sol, ok = backtrack_csp(new_S, new_used, new_covered, max_val, max_size, deadline)
            if ok:
                return sol, True

        return None, False

    def greedy_init(max_val):
        """Greedy construction to get a starting point."""
        S = [0]
        used_diffs = set()
        covered = set()
        target = TARGET

        for step in range(31):
            if time.time() > DEADLINE - 20:
                break
            if covered == target:
                break

            best_x = None
            best_gain = 0

            # Candidate values: focus on creating missing differences
            missing = target - covered
            candidates = set()
            for s in S:
                for d in missing:
                    candidates.add(s + d)
                    if s - d > 0:
                        candidates.add(s - d)
            # Add some random candidates
            for _ in range(100):
                candidates.add(random.randint(1, max_val))

            candidates = [c for c in candidates if c > 0 and c not in set(S) and c <= max_val]
            candidates.sort()

            for x in candidates[:300]:
                # Check Sidon
                valid = True
                new_diffs = set()
                for s in S:
                    dd = abs(s - x)
                    if dd in used_diffs or dd in new_diffs:
                        valid = False
                        break
                    new_diffs.add(dd)
                if not valid:
                    continue
                gain = len(new_diffs & target)
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    if best_gain >= len(missing):
                        break

            if best_x is None:
                # Try any valid small value
                for x in range(1, max_val + 1):
                    if x in set(S):
                        continue
                    valid = True
                    new_diffs = set()
                    for s in S:
                        dd = abs(s - x)
                        if dd in used_diffs or dd in new_diffs:
                            valid = False
                            break
                        new_diffs.add(dd)
                    if valid:
                        best_x = x
                        break
                if best_x is None:
                    break

            S.append(best_x)
            for s in S[:-1]:
                used_diffs.add(abs(s - best_x))
            covered = used_diffs & target

        return S

    def local_refine(S, deadline):
        """Try to reduce max(S) by replacing elements with smaller values."""
        S = sorted(S)
        best = S[:]
        best_max = max(S)

        while time.time() < deadline:
            improved = False
            for i in range(len(S)):
                if S[i] == 0:
                    continue
                old_val = S[i]
                # Try smaller values
                for new_val in range(1, old_val):
                    if new_val in S:
                        continue
                    trial = S[:]
                    trial[i] = new_val
                    trial.sort()
                    if not is_sidon_fast(trial):
                        continue
                    if not covers_all(trial):
                        continue
                    if max(trial) < best_max:
                        S = trial
                        best = trial
                        best_max = max(S)
                        improved = True
                        break
            if not improved:
                break

        return best

    # Main search: iterative deepening on max_val
    best_S = None
    best_score = (30, 10**9)

    # First, get a valid solution with greedy
    random.seed(42)
    S0 = greedy_init(1000000)
    if S0 and covers_all(S0) and is_sidon_fast(S0):
        s0 = score_set(S0)
        if s0 < best_score:
            best_score = s0
            best_S = S0

    # Try CSP backtracking with increasing max_val bounds
    # Start with a reasonable bound and decrease
    max_val_candidates = [200, 150, 120, 100, 90, 80, 70, 60, 50]

    for max_val in max_val_candidates:
        if time.time() > DEADLINE - 30:
            break

        # Try CSP backtracking
        S_init = [0]
        used_diffs = set()
        covered = set()
        sol, ok = backtrack_csp(S_init, used_diffs, covered, max_val, 32, DEADLINE - 10)

        if ok and sol and covers_all(sol) and is_sidon_fast(sol):
            s = score_set(sol)
            if s < best_score:
                best_score = s
                best_S = sol
                # Found a solution; try to improve with local refinement
                if time.time() < DEADLINE - 10:
                    refined = local_refine(sol, DEADLINE - 10)
                    if refined and max(refined) < max(sol):
                        rs = score_set(refined)
                        if rs < best_score:
                            best_score = rs
                            best_S = refined
                break

    # If we have a best_S, try local refinement to reduce diameter
    if best_S and time.time() < DEADLINE - 5:
        refined = local_refine(best_S, DEADLINE - 5)
        if refined:
            rs = score_set(refined)
            if rs < best_score:
                best_score = rs
                best_S = refined

    # Final check
    if best_S is None or not covers_all(best_S) or not is_sidon_fast(best_S):
        # Fallback: simple construction
        best_S = [0]
        covered = set()
        for d in range(1, 31):
            if d in covered:
                continue
            for x in range(1, 1000000):
                if x in best_S:
                    continue
                trial = sorted(best_S + [x])
                if is_sidon_fast(trial):
                    trial_diffs = get_diffs(trial)
                    if d in trial_diffs:
                        best_S = trial
                        covered = trial_diffs & TARGET
                        break
            else:
                y = 2 * max(best_S) + d + 1
                best_S = sorted(best_S + [y, y + d])
                covered = get_diffs(best_S) & TARGET

    return best_S
