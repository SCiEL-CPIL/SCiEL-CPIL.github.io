import time
import sys

def construct():
    start_time = time.time()
    deadline = 145.0

    NEED = set(range(1, 31))
    MAXN = 32

    def check_solution(A):
        """Verify A is a valid Sidon set containing 0, size <= 32, all diffs 1..30 present."""
        if not A or 0 not in A:
            return False
        A = sorted(set(A))
        if len(A) > MAXN:
            return False
        if A[-1] > 10**6:
            return False
        diffs = set()
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                d = A[j] - A[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return NEED.issubset(diffs)

    def dpll_solve(M, max_elems=MAXN):
        """
        DPLL-style backtracking to find a Sidon set A subset [0,M] with 0,
        size <= max_elems, covering all diffs 1..30.
        Returns list A or None.
        """
        # State: A (sorted list), diff_count (dict d->count), diff_covered (set of d in 1..30 seen)
        # We build A in increasing order.
        
        A = [0]
        diff_count = {}
        covered = set()
        missing = set(NEED)

        # Precompute: for each potential x, which diffs it would create with current A
        # But A changes, so we compute on the fly.

        def get_candidates(start_val, A, diff_count, missing):
            """
            Generate candidate values for next element.
            Priority: candidates that cover missing diffs.
            """
            if not missing:
                return []

            # For each missing d, and each a in A:
            #   x = a + d is a candidate (if x <= M, x >= start_val, x not in A)
            #   x = a - d is a candidate (if x > 0, x >= start_val, x not in A)
            # Also consider x = start_val if it doesn't create conflicts.

            A_set = set(A)
            max_a = A[-1]
            cand_set = set()

            # Candidates from covering missing diffs
            for d in missing:
                for a in A:
                    x1 = a + d
                    if x1 <= M and x1 >= start_val and x1 not in A_set:
                        cand_set.add(x1)
                    x2 = a - d
                    if x2 > 0 and x2 >= start_val and x2 not in A_set:
                        cand_set.add(x2)

            # Also try start_val itself as a "neutral" candidate
            if start_val <= M and start_val not in A_set:
                cand_set.add(start_val)

            # Also try a few values near M to explore high-end
            for x in [M, M-1, M-2]:
                if x >= start_val and x not in A_set:
                    cand_set.add(x)

            # Filter and score candidates
            candidates = []
            for x in cand_set:
                if x < start_val or x > M or x in A_set:
                    continue

                # Check Sidon property: for each a in A, d = x - a must not be in diff_count
                ok = True
                new_diffs = []
                for a in A:
                    if x > a:
                        d = x - a
                        if d in diff_count:
                            ok = False
                            break
                        new_diffs.append(d)
                    elif x < a:
                        d = a - x
                        if d in diff_count:
                            ok = False
                            break
                        new_diffs.append(d)

                if not ok:
                    continue

                # Score: how many missing diffs does x cover?
                covered_now = len(missing.intersection(new_diffs))
                candidates.append((covered_now, x))

            # Sort: high coverage first, then by value (prefer smaller to keep diameter down)
            candidates.sort(key=lambda t: (-t[0], t[1]))
            return candidates

        def backtrack(start_val, A, diff_count, missing, depth):
            # Time check
            if time.time() > deadline - 2:
                return None

            if not missing:
                return list(A)

            if len(A) >= max_elems:
                return None

            # Pruning: max possible diff is M. If max(missing) > M, fail.
            if missing and max(missing) > M:
                return None

            # Pruning: with remaining slots, can we cover all missing?
            # Each new element can create at most (len(A) + remaining - 1) new diffs
            # Actually each new element x creates diffs with all existing A elements.
            # With r remaining elements, max new diffs = sum over k of (len(A) + k)
            # where k = 0, 1, ..., r-1
            # = r * len(A) + r*(r-1)//2
            remaining = max_elems - len(A)
            max_new_diffs = remaining * len(A) + remaining * (remaining - 1) // 2
            if len(missing) > max_new_diffs:
                return None

            candidates = get_candidates(start_val, A, diff_count, missing)
            if not candidates:
                return None

            # Limit candidates to avoid explosion
            # Prefer high-coverage candidates
            high_cov = [c for c in candidates if c[0] > 0]
            low_cov = [c for c in candidates if c[0] == 0]

            # Try high-coverage first (all of them, up to a limit)
            try_list = high_cov[:20] + low_cov[:5]

            for covered_count, x in try_list:
                new_A = A + [x]
                new_diff_count = dict(diff_count)
                new_missing = set(missing)

                valid = True
                for a in A:
                    d = abs(x - a)
                    if d in new_diff_count:
                        valid = False
                        break
                    new_diff_count[d] = new_diff_count.get(d, 0) + 1
                    if d in new_missing:
                        new_missing.discard(d)

                if not valid:
                    continue

                res = backtrack(x + 1, new_A, new_diff_count, new_missing, depth + 1)
                if res is not None:
                    return res

            return None

        return backtrack(1, [0], {}, set(NEED), 0)

    # First, find any solution with a reasonable upper bound
    # Lower bound: 30 (need diff 30)
    # Upper bound: start with 100, expand if needed

    best_sol = None
    best_max = float('inf')

    # Try increasing M values to find minimal
    # Start with a known-good upper bound
    hi = 100
    while time.time() < deadline - 10:
        sol = dpll_solve(hi)
        if sol is not None:
            best_sol = sol
            best_max = max(sol)
            break
        hi *= 2
        if hi > 10000:
            break

    if best_sol is None:
        # Fallback: greedy
        A = [0]
        diffs = set()
        for x in range(1, 10000):
            ok = True
            new_diffs = []
            for a in A:
                d = abs(x - a)
                if d in diffs:
                    ok = False
                    break
                new_diffs.append(d)
            if not ok:
                continue
            A.append(x)
            diffs.update(new_diffs)
            if NEED.issubset(diffs):
                best_sol = A
                best_max = max(A)
                break

    if best_sol is None:
        return [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]

    # Now try to reduce the maximum element
    # Binary search between lower bound and best_max
    lo = 30
    hi = best_max - 1

    while lo <= hi and time.time() < deadline - 5:
        mid = (lo + hi) // 2
        sol = dpll_solve(mid)
        if sol is not None:
            if max(sol) < best_max:
                best_sol = sol
                best_max = max(sol)
            hi = mid - 1
        else:
            lo = mid + 1

    return best_sol
