"""Sidon set containing 0 with differences 1..30, minimizing the largest element."""
import time
from collections import deque

TIME_LIMIT = 110  # wall-clock seconds, safety margin under 120s

def construct():
    t0 = time.time()
    DEADLINE = t0 + TIME_LIMIT
    FULL = (1 << 31) - 1  # bits 1..30

    # ---------- Phase 1: BFS over difference-sets ----------
    # state: (used_mask, diameter, size)
    # start: {0} -> used=0, diameter=0, size=1
    # goal: used == FULL
    # We want minimum diameter, then minimum size.

    # BFS level by level by diameter is expensive; instead use Dijkstra-like
    # BFS where we track best (diameter, size) per used_mask.
    # But the number of used_masks is 2^30 which is too large.
    # Instead: BFS by number of elements (size), and for each size find
    # the minimum diameter achievable.

    # Better approach: iterative deepening on diameter.
    # For a given diameter D, can we find a Sidon set in [0,D] with 0
    # that realizes all differences 1..30?
    # This is a constraint satisfaction / backtracking problem.
    # But we want to minimize D, so we binary-search or increment D.

    # Actually, let's think about what's achievable.
    # A Sidon set of size n in [0, M] has at most n(n-1)/2 distinct positive differences.
    # We need 30 differences, so n(n-1)/2 >= 30 -> n >= 9 (since 8*7/2=28 < 30, 9*8/2=36).
    # So minimum size is 9.
    # For a perfect difference set of order k (size k^2+k+1), the diameter is k^2+k.
    # For size 9, that's not a perfect difference set.
    # Let's try to find the minimum diameter for size 9, 10, ...

    # Strategy: for each size n from 9 upward, try to find the minimum diameter
    # via backtracking. Return the best (diameter, set) found.

    # Backtracking: build the set incrementally, always adding the smallest
    # candidate that keeps the Sidon property and helps realize missing diffs.

    best_solution = None  # (diameter, frozenset)
    best_diameter = float('inf')

    def check_time():
        return time.time() < DEADLINE

    def try_find(n, max_diam, t0):
        """Try to find a Sidon set of size n containing 0 with all diffs 1..30
        and diameter <= max_diam. Return the set or None."""
        # We build the set element by element.
        # Start with {0}. At each step, try adding a new element x.
        # x must be in [1, max_diam].
        # Maintain: current set S, used differences, forbidden differences.

        # Use recursive backtracking with pruning.
        # To keep it fast, we use a heuristic: always try to add elements
        # that realize the most missing differences.

        # Actually, for small n and moderate max_diam, a simple DFS with
        # good pruning should work.

        # Let's use iterative approach:
        # S = [0]
        # At each step, find all valid next elements, sort by some heuristic,
        # and try them.

        # For efficiency, precompute nothing; just check on the fly.

        S = [0]
        used = set()  # differences realized
        forbidden = set()  # differences that cannot be realized without conflict

        # Actually, let's just do a clean backtracking.

        def backtrack(S, used, conf):
            nonlocal best_solution, best_diameter
            if not check_time():
                return None
            if len(S) == n:
                if used == set(range(1, 31)):
                    diam = max(S)
                    if diam < best_diameter:
                        best_diameter = diam
                        best_solution = frozenset(S)
                    return S
                return None

            # Find missing differences
            missing = set(range(1, 31)) - used

            # Generate candidate elements
            # A new element x can be added if:
            #   for all y in S, |x-y| not in used and |x-y| not in conf
            #   (and x not in S, x >= 0, x <= max_diam)
            #
            # But we also need x to help realize some missing difference.
            # Actually, x might not directly realize a missing diff, but
            # combined with future elements it might. However, for efficiency,
            # let's require that x realizes at least one missing difference
            # OR that x is small (to keep diameter low).

            # Let's just try all valid x and let the search figure it out.
            # But that's too many. Let's be smarter.

            # Heuristic: for each missing difference d, the candidates are
            #   x = y + d for y in S  (x > y)
            #   x = y - d for y in S  (x < y, so x = y - d >= 0)
            # Union these, filter by validity, and try them.

            candidates = set()
            for d in missing:
                for y in S:
                    c1 = y + d
                    if c1 <= max_diam and c1 not in S:
                        candidates.add(c1)
                    c2 = y - d
                    if c2 >= 0 and c2 not in S:
                        candidates.add(c2)

            # Also try small elements that might help later
            # (e.g., elements that don't realize a missing diff now but
            #  create new opportunities)
            # For now, skip this to keep the search focused.

            # Filter candidates: check Sidon property
            valid_candidates = []
            for x in candidates:
                if x in S:
                    continue
                ok = True
                new_diffs = []
                for y in S:
                    diff = abs(x - y)
                    if diff in used or diff in conf:
                        ok = False
                        break
                    new_diffs.append(diff)
                if ok:
                    valid_candidates.append((x, new_diffs))

            # Sort by number of new differences realized (more is better)
            # and by value (smaller is better for diameter)
            valid_candidates.sort(key=lambda t: (-len(t[1]), t[0]))

            for x, new_diffs in valid_candidates:
                if not check_time():
                    return None
                new_used = used | set(new_diffs)
                # Compute new forbidden differences
                new_conf = set(conf)
                S_new = S + [x]
                for i in range(len(S_new)):
                    for j in range(i + 1, len(S_new)):
                        d = S_new[j] - S_new[i]
                        if d in new_used:
                            # This difference is already used; check if it's
                            # used by a different pair
                            # Count how many pairs give this difference
                            cnt = 0
                            for a in range(len(S_new)):
                                for b in range(a + 1, len(S_new)):
                                    if S_new[b] - S_new[a] == d:
                                        cnt += 1
                            if cnt > 1:
                                # Conflict! This shouldn't happen if we checked
                                # properly. Actually, the check above should
                                # prevent this. Let me re-examine.
                                # The check "diff in used" means this difference
                                # was already realized by an existing pair.
                                # So adding x creates a second pair with the
                                # same difference -> conflict.
                                # So we should NOT add x.
                                # But we already filtered for this.
                                # So this branch shouldn't be reached.
                                pass
                # Recompute forbidden properly
                new_conf = set()
                for i in range(len(S_new)):
                    for j in range(i + 1, len(S_new)):
                        d = S_new[j] - S_new[i]
                        if d in new_used:
                            # Check if d is used by more than one pair
                            cnt = 0
                            for a in range(len(S_new)):
                                for b in range(a + 1, len(S_new)):
                                    if S_new[b] - S_new[a] == d:
                                        cnt += 1
                            if cnt > 1:
                                # This is a conflict, should not happen
                                pass
                        else:
                            # d is not in new_used, but might be forbidden
                            # if there's another pair with the same difference
                            # Actually, forbidden means: if we add a future
                            # element z such that |z - w| = d for some w in S_new,
                            # it would create a duplicate.
                            # For now, let's just track used and check on the fly.
                            pass

                # Simplified: just check validity when adding
                result = backtrack(S_new, new_used, new_conf)
                if result is not None:
                    return result
            return None

        return backtrack([0], set(), set())

    # Try increasing diameters
    # Lower bound: we need 30 differences, min size 9.
    # For size 9 in [0, D], max differences = 9*8/2 = 36.
    # We need 30 specific ones. D must be at least 30 (to realize diff 30).
    # Let's try D from 30 upward.

    for n in range(9, 33):  # size 9 to 32
        if not check_time():
            break
        for D in range(30, 500):  # diameter from 30 to 499
            if not check_time():
                break
            if best_solution is not None and D >= best_diameter:
                break
            # Try to find a solution of size n with diameter D
            # Use a simpler, more efficient backtracking

            S = [0]
            used = set()
            target = set(range(1, 31))

            def dfs(S, used):
                nonlocal best_solution, best_diameter
                if not check_time():
                    return None
                if len(S) == n:
                    if used == target:
                        diam = max(S)
                        if diam < best_diameter:
                            best_diameter = diam
                            best_solution = frozenset(S)
                        return S
                    return None

                missing = target - used
                if not missing:
                    # All differences realized, but we need exactly n elements
                    # We can stop early if we have fewer than n, but the
                    # problem says "at most 32", so fewer is fine.
                    # Actually, the task says "at most 32 distinct integers",
                    # so we can return as soon as all diffs are realized.
                    diam = max(S)
                    if diam < best_diameter:
                        best_diameter = diam
                        best_solution = frozenset(S)
                    return S

                # Generate candidates
                candidates = set()
                for d in missing:
                    for y in S:
                        c1 = y + d
                        if c1 <= D and c1 not in S:
                            candidates.add(c1)
                        c2 = y - d
                        if c2 >= 0 and c2 not in S:
                            candidates.add(c2)

                # Filter and sort
                valid = []
                for x in candidates:
                    new_diffs = []
                    ok = True
                    for y in S:
                        diff = abs(x - y)
                        if diff in used:
                            ok = False
                            break
                        new_diffs.append(diff)
                    if ok:
                        valid.append((x, new_diffs))

                valid.sort(key=lambda t: (-len(t[1]), t[0]))

                for x, new_diffs in valid:
                    if not check_time():
                        return None
                    result = dfs(S + [x], used | set(new_diffs))
                    if result is not None:
                        return result
                return None

            dfs([0], set())
            if best_solution is not None:
                # We found a solution; try to improve with smaller D
                # but break the D loop if D >= best_diameter
                pass

    # Return the best solution found
    if best_solution is not None:
        return sorted(best_solution)

    # Fallback: use the greedy approach from the seed
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(diffs(trial))) == len(diffs(trial)):
                a = trial
                break
        else:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
    return a
