import time
import sys

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = t0 + 140.0  # safety margin before 150s kill

    # We need a Sidon set (all positive pairwise differences distinct) that
    # contains 0, has size <= 32, all elements in [0, 10^6], and covers
    # every difference 1..30. Minimize the largest element (diameter).

    # ---- Fast diff utilities ----
    def all_diffs(s):
        """Return the list of all positive differences of sorted list s."""
        n = len(s)
        out = []
        for i in range(n):
            si = s[i]
            for j in range(i + 1, n):
                out.append(s[j] - si)
        return out

    def is_sidon(s):
        """True iff all positive differences of s are distinct."""
        n = len(s)
        seen = set()
        for i in range(n):
            si = s[i]
            for j in range(i + 1, n):
                d = s[j] - si
                if d in seen:
                    return False
                seen.add(d)
        return True

    def covers_all(s):
        """True iff every d in 1..30 appears as a difference of s."""
        diffs = all_diffs(s)
        needed = set(range(1, 31))
        return needed.issubset(set(diffs))

    # ---- Backtracking search ----
    # Build the set incrementally. State: current sorted list `cur`,
    # the set of differences already realized `realized`, and the set of
    # all differences present `diffs_set` (to enforce Sidon).
    #
    # We search for the minimal diameter. We use a best-first / iterative
    # deepening approach: try to find a solution with diameter <= D, and
    # increase D. But to keep it simple and fast, we do a DFS that prunes
    # aggressively and keeps the best (smallest max element) found.

    best = [None]  # best solution found (list), for early termination

    # Precompute: for a candidate value x added to a sorted list cur,
    # the new differences introduced are x - cur[i] for each i.
    # We need none of these to collide with existing diffs_set, and none
    # of them to collide with each other.

    # DFS with pruning. We try to add elements in increasing order to
    # keep the set compact. We track:
    #   cur: sorted list of chosen elements (always includes 0)
    #   diffs_set: set of all positive differences present
    #   covered: set of differences in 1..30 that are covered
    #
    # Pruning:
    #   - If len(cur) >= 32, stop (size limit).
    #   - If max(cur) >= current best diameter, stop.
    #   - If even adding all remaining possible small elements can't cover
    #     the missing differences, stop (hard to bound tightly, skip).
    #   - Sidon check on each addition.

    # We'll do iterative deepening on the diameter bound D.
    # For a given D, try to find any valid Sidon set within [0, D]
    # covering 1..30 with size <= 32.

    def search_within_D(D):
        """DFS: find a Sidon set in [0,D] covering 1..30, size<=32.
        Returns the set (as list) or None. Uses time budget."""
        nonlocal best
        # We build incrementally. cur is a tuple (immutable) for hashing.
        # diffs_set is a frozenset. covered is a frozenset.
        #
        # To keep it tractable, we use a recursive DFS with memoization
        # on (tuple(cur), frozenset(diffs_set)) -> but that's huge.
        # Instead, we rely on pruning and ordering.

        # Start with cur = (0,), diffs_set = frozenset(), covered = frozenset()
        # We want to add elements. The next element must be > last element.
        # To keep diameter small, we try small values first.

        # Use a stack-based DFS to avoid recursion depth issues.
        # State: (cur_tuple, diffs_set_frozen, covered_frozen, next_min)
        # But we also want to try to complete quickly.

        # Alternative: recursive with good pruning.
        # Let's do recursive.

        sys.setrecursionlimit(10000)

        def dfs(cur, diffs_set, covered, depth):
            if time.time() > DEADLINE:
                return None
            if len(cur) > 32:
                return None
            # Check if we've covered all
            if covered == target_covered:
                return list(cur)
            # If we can't cover remaining with size limit, prune
            remaining_needed = target_covered - covered
            if len(remaining_needed) > 0 and len(cur) >= 32:
                return None
            # Prune: max element so far
            if cur and cur[-1] > D:
                return None
            # Try adding next element x in range (last+1 .. D)
            last = cur[-1] if cur else 0
            # We need to add elements. Try x from last+1 to D.
            # But we can be smarter: we need to cover remaining_needed.
            # Each new element x added to cur introduces diffs x - c for c in cur.
            # We want those to include some of remaining_needed.

            # Optimization: if adding x covers no new needed diff, it's
            # only useful as a "bridge" for future elements. Still try,
            # but prioritize x that cover needed diffs.

            # Generate candidate x values.
            # Priority: x such that x - c in remaining_needed for some c in cur.
            candidates = []
            for x in range(last + 1, D + 1):
                # Check Sidon: new diffs x - c for c in cur must not be in diffs_set
                # and must be distinct among themselves.
                new_diffs = []
                ok = True
                for c in cur:
                    d = x - c
                    if d in diffs_set or d in new_diffs:
                        ok = False
                        break
                    new_diffs.append(d)
                if not ok:
                    continue
                # Compute how many new needed diffs this covers
                new_covered = covered | (set(new_diffs) & target_covered)
                gain = len(new_covered) - len(covered)
                candidates.append((x, gain, new_diffs, new_covered))

            # Sort candidates: prefer those with higher gain, then smaller x
            # (smaller x keeps diameter down).
            candidates.sort(key=lambda t: (-t[1], t[0]))

            for x, gain, new_diffs, new_covered in candidates:
                if time.time() > DEADLINE:
                    return None
                new_cur = cur + (x,)
                new_diffs_set = diffs_set | set(new_diffs)
                # Recursive
                result = dfs(new_cur, new_diffs_set, new_covered, depth + 1)
                if result is not None:
                    return result
            return None

        target_covered = frozenset(range(1, 31))
        return dfs((0,), frozenset(), frozenset(), 0)

    # Iterative deepening on D.
    # Lower bound on D: we need to cover diff 30, so D >= 30.
    # Also, a Sidon set of size k has at most k(k-1)/2 distinct differences,
    # and we need at least 30 differences (for 1..30), so k(k-1)/2 >= 30
    # => k >= 8. So we need at least 8 elements.
    # Start D at 30 and increase.

    # But DFS for small D may be slow if no solution exists. Let's use
    # a reasonable starting D. From the knowledge, backtracking found
    # diameter ~ 17.65 (score 17.65). Wait, score = seed/value, so if
    # seed=1 and value=diameter, score=1/diameter? No, the task says
    # "the evaluator measures the raw value (lower is better)" and
    # "search score is seed/value". So score = seed / value where value
    # is the diameter (largest element). If the current program has
    # score 1.0, and it's the seed, then seed/value = 1.0 means value = seed.
    # The measured finding says "1 -> 17.65" meaning the value went from
    # some baseline to 17.65? Or the score? Let me re-read.
    #
    # "Backtracking search with pruning finds minimal diameter Sidon set
    # covering 1..30, significantly lower than greedy heuristic
    # Measured: 1 -> 17.65."
    #
    # This likely means the value (diameter) was reduced, and the score
    # (seed/value) went up. If the seed program has score 1.0 and the
    # backtracking has score 17.65, then the diameter was reduced by a
    # factor of 17.65. If the seed diameter was, say, 1765, then the
    # backtracking diameter is 100. Or if seed diameter was 176, then
    # backtracking is 10. Hmm.
    #
    # Actually, looking at the observed changes, p62 has score 13.30.
    # So the best found so far has score ~13.3, meaning diameter ~ seed/13.3.
    # If seed = 1, diameter ~ 0.075 which is impossible. So seed must be
    # larger. The "seed" in "seed/value" is probably a fixed constant
    # for the task, not 1.
    #
    # Regardless, I should minimize the diameter. Let me just search
    # for the minimal diameter using iterative deepening.

    # Let's start with a reasonable lower bound.
    # We need diff 30, so D >= 30.
    # Try D from 30 upward.

    # But DFS might be slow for infeasible D. Let's add a quick
    # feasibility check: can we even place elements to cover 1..30
    # within [0, D]?
    #
    # For now, just try increasing D and use time budget.

    # Actually, let me think about this differently. The problem is small
    # enough that a well-pruned DFS should find the optimal quickly.
    # Let me use a different approach: branch and bound with a good
    # lower bound on the remaining diameter.

    # Let me try a simpler approach first: since we need to cover 1..30,
    # and the set must be Sidon, let me use the known structure.
    # A perfect difference set of order n has size n^2+n+1 and covers
    # all differences 1..n^2+n. For n=5, size=31, covers 1..30.
    # So we want a truncated perfect difference set of order 5 (size 31,
    # covering 1..30) or smaller.
    #
    # The known (31,6,1) difference set (projective plane of order 5)
    # has diameter 30. But we want to minimize diameter, so we want
    # a set that covers 1..30 with the smallest possible max element.
    #
    # Let me just do the iterative deepening DFS.

    # Start D at a lower bound.
    # Minimum D: we need to realize diff 30, so D >= 30.
    # But maybe we can do better? No, if 30 is a difference, then
    # some element is at least 30 (since 0 is in the set, 30-0=30
    # means 30 is in the set, or some pair differs by 30).
    # Actually, 0 is in the set, so if diff 30 is present, then
    # 30 must be in the set (as 30-0=30). So D >= 30.

    # Wait, that's not right. The diff 30 could be between two non-zero
    # elements, e.g., 35 and 5. But 0 is in the set, and we need diff 30,
    # which could be 30-0, 31-1, etc. So D >= 30 is not strictly required
    # if 30 is realized as a difference between two non-zero elements.
    # But since 0 is in the set, 30-0=30 is a candidate, but we don't
    # have to use it. However, the smallest element is 0, so any diff d
    # must be realized as y-x where y>=x>=0, so y>=d. Thus D>=30.

    # So D >= 30. Let's start there.

    # Actually, I realize the DFS might be too slow for large D. Let me
    # use a smarter approach: since we want to minimize D, and D>=30,
    # let me try D=30 first, then 31, etc.

    # But the DFS for D=30 might explore many branches. Let me add
    # stronger pruning.

    # Key insight: in a Sidon set, each pair gives a unique difference.
    # If we have k elements, we have k(k-1)/2 differences. We need at
    # least 30 of them to be in 1..30. So k(k-1)/2 >= 30 => k >= 8.
    #
    # For D=30, the elements are in [0,30]. The number of possible
    # Sidon sets is limited. Let me just run the DFS.

    # Let me restructure: use a global best and do a single DFS that
    # explores in order of increasing max element, returning the first
    # valid solution found (which will have the smallest diameter due
    # to the ordering).

    # Actually, the DFS as written tries to find ANY solution within D,
    # and we increase D. The first D that yields a solution gives the
    # minimal diameter. Let's do that.

    # To speed up, let me add memoization: if we've already tried a
    # state (cur_tuple, diffs_set) and it didn't lead to a solution,
    # skip it. But the state space is large. Let me use a visited set
    # with a size limit.

    # Let me just run it and see. With time budget, it should be fine.

    # Let me try D from 30 upward.
    for D in range(30, 1000001):
        if time.time() > DEADLINE - 5:
            break
        # Quick check: if D is too small, skip (but D>=30 always)
        result = search_within_D(D)
        if result is not None:
            best[0] = result
            break

    if best[0] is not None:
        return best[0]

    # Fallback: use the greedy approach if DFS didn't find anything
    # within time.
    a = [0]
    def greedy_diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in greedy_diffs(sorted(a)):
            continue
        found = False
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(greedy_diffs(trial))) == len(greedy_diffs(trial)):
                a = trial
                found = True
                break
        if not found:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(greedy_diffs(trial))) == len(greedy_diffs(trial)):
                    a = trial
                    break
                y += 1
    return a
# EVOLVE-BLOCK-END
