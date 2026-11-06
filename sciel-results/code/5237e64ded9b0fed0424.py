import time
import random

def construct():
    """
    Find a Sidon set containing 0 with all differences 1..30, minimizing the largest element.
    
    Structural approach: Model as a set-packing / maximum-coverage problem.
    Each candidate element x generates a "difference signature" relative to the current set.
    We use iterative deepening on the diameter bound with a backtracking search that
    processes required differences in MRV order, using bitset-based conflict detection
    for O(1) Sidon validity checks.
    """
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill

    TARGET = frozenset(range(1, 31))
    TARGET_LIST = list(range(1, 31))
    N_TARGET = 30

    def is_sidon(S):
        """Check if sorted list S is a Sidon set (all pairwise differences distinct)."""
        S = sorted(S)
        diffs = set()
        n = len(S)
        for i in range(n):
            for j in range(i + 1, n):
                d = S[j] - S[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def get_diffs(S):
        S = sorted(S)
        diffs = set()
        n = len(S)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(S[j] - S[i])
        return diffs

    def covers_all(S):
        diffs = get_diffs(S)
        return TARGET.issubset(diffs)

    def validate(S):
        """Full validation: Sidon + covers 1..30 + includes 0 + at most 32 elements + range."""
        if not S or 0 not in S:
            return False
        if len(S) > 32:
            return False
        if max(S) > 10**6:
            return False
        if len(set(S)) != len(S):
            return False
        if not is_sidon(S):
            return False
        diffs = get_diffs(S)
        return TARGET.issubset(diffs)

    def greedy_construct(max_val, deadline):
        """
        Greedy: start with {0}, repeatedly add the element that covers the most
        missing differences while maintaining Sidon property.
        Uses bitset for fast difference conflict checking.
        """
        S = [0]
        used_diffs = set()
        covered = set()
        S_set = {0}

        for _ in range(31):
            if time.time() > deadline:
                break
            if covered == TARGET:
                break

            missing = TARGET - covered
            best_x = None
            best_key = None

            # Generate candidates: for each existing s and missing d, try s+d and s-d
            candidates = set()
            for s in S:
                for d in missing:
                    c1 = s + d
                    if 0 < c1 <= max_val and c1 not in S_set:
                        candidates.add(c1)
                    c2 = s - d
                    if 0 < c2 <= max_val and c2 not in S_set:
                        candidates.add(c2)

            for x in sorted(candidates):
                # Check Sidon validity
                new_diffs = set()
                valid = True
                for s in S:
                    dd = abs(s - x)
                    if dd in used_diffs or dd in new_diffs:
                        valid = False
                        break
                    new_diffs.add(dd)
                if not valid:
                    continue

                gain = len(new_diffs & missing)
                if gain == 0:
                    continue
                # Prefer: more gain, then fewer total new diffs (less "waste"), then smaller x
                key = (-gain, len(new_diffs), x)
                if best_key is None or key < best_key:
                    best_key = key
                    best_x = x

            if best_x is None:
                # Fallback: scan small values
                for x in range(1, min(max_val + 1, 200)):
                    if x in S_set:
                        continue
                    new_diffs = set()
                    valid = True
                    for s in S:
                        dd = abs(s - x)
                        if dd in used_diffs or dd in new_diffs:
                            valid = False
                            break
                        new_diffs.add(dd)
                    if valid and (new_diffs & missing):
                        best_x = x
                        break

            if best_x is None:
                break

            S.append(best_x)
            S_set.add(best_x)
            for s in S[:-1]:
                used_diffs.add(abs(s - best_x))
            covered = used_diffs & TARGET

        if covers_all(S) and is_sidon(S):
            return sorted(S)
        return None

    def backtrack_search(max_val, deadline):
        """
        Depth-first backtracking: build the set incrementally.
        At each step, choose which missing difference to cover next (MRV heuristic),
        then try all valid candidate elements that cover it.
        Uses used_diffs set for O(1) Sidon checks.
        """
        best_solution = None
        best_diameter = max_val + 1

        S = [0]
        S_set = {0}
        used_diffs = set()
        covered = set()

        def dfs(depth):
            nonlocal best_solution, best_diameter

            if time.time() > deadline:
                return

            if covered == TARGET:
                diam = max(S)
                if diam < best_diameter:
                    best_diameter = diam
                    best_solution = S[:]
                return

            if len(S) >= 32:
                return

            missing = TARGET - covered

            # MRV: pick the difference with fewest valid candidate elements
            # For each missing d, candidates are: s+d for s in S, s-d for s in S (if >0)
            # Filter to those that don't conflict with used_diffs
            best_d = None
            best_candidates = None
            min_count = 999

            for d in sorted(missing):
                cands = set()
                for s in S:
                    c1 = s + d
                    if 0 < c1 <= max_val and c1 not in S_set:
                        cands.add(c1)
                    c2 = s - d
                    if 0 < c2 <= max_val and c2 not in S_set:
                        cands.add(c2)

                # Filter by Sidon validity
                valid_cands = []
                for x in cands:
                    new_diffs = set()
                    ok = True
                    for s in S:
                        dd = abs(s - x)
                        if dd in used_diffs or dd in new_diffs:
                            ok = False
                            break
                        new_diffs.add(dd)
                    if ok:
                        valid_cands.append(x)

                if not valid_cands:
                    # Dead end for this d - but other d's might be coverable
                    # This means we need to add elements that cover multiple diffs
                    continue

                if len(valid_cands) < min_count:
                    min_count = len(valid_cands)
                    best_d = d
                    best_candidates = valid_cands

            if best_d is None:
                # No single difference can be covered directly; 
                # try adding any valid element that covers at least one missing diff
                all_cands = set()
                for d in missing:
                    for s in S:
                        c1 = s + d
                        if 0 < c1 <= max_val and c1 not in S_set:
                            all_cands.add(c1)
                        c2 = s - d
                        if 0 < c2 <= max_val and c2 not in S_set:
                            all_cands.add(c2)

                for x in sorted(all_cands):
                    new_diffs = set()
                    ok = True
                    for s in S:
                        dd = abs(s - x)
                        if dd in used_diffs or dd in new_diffs:
                            ok = False
                            break
                        new_diffs.add(dd)
                    if ok and (new_diffs & missing):
                        best_d = None  # mark as "any"
                        best_candidates = [x]
                        break

            if best_candidates is None or not best_candidates:
                return

            # Sort candidates: prefer smaller x (to keep diameter small), then more coverage
            def cand_score(x):
                new_diffs = set()
                for s in S:
                    new_diffs.add(abs(s - x))
                return (x, -len(new_diffs & missing))

            best_candidates.sort(key=cand_score)

            for x in best_candidates:
                # Compute new diffs
                new_diffs = set()
                for s in S:
                    new_diffs.add(abs(s - x))

                # Apply
                S.append(x)
                S_set.add(x)
                old_used = used_diffs.copy()
                old_covered = covered.copy()
                used_diffs |= new_diffs
                covered = used_diffs & TARGET

                dfs(depth + 1)

                # Undo
                S.pop()
                S_set.discard(x)
                used_diffs = old_used
                covered = old_covered

                if best_solution and max(S) >= best_diameter:
                    # Can't do better with this branch
                    continue

        dfs(0)
        return best_solution

    def local_refinement(S, deadline):
        """Try to reduce the maximum element by replacing individual elements."""
        S = sorted(S)
        best = S[:]
        best_max = max(S)

        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(1, len(S)):
                old_val = S[i]
                for new_val in range(1, old_val):
                    if new_val in S:
                        continue
                    trial = S[:]
                    trial[i] = new_val
                    trial.sort()
                    if not is_sidon(trial):
                        continue
                    if not covers_all(trial):
                        continue
                    if max(trial) < best_max:
                        S = trial
                        best = trial
                        best_max = max(S)
                        improved = True
                        break

        return best

    def iterative_deepening(deadline):
        """
        Iterative deepening on diameter bound.
        Start from a lower bound, increase until solution found.
        """
        # Lower bound: need 30 distinct differences, so at least ~sqrt(30) elements,
        # and diameter at least 30 (since diff 30 must exist)
        # But Sidon constraint means diameter >> 30
        # For a Sidon set of size n, diameter >= n^2/4 approximately
        # We need at least ~11 elements (0 + 10 more to get 30 diffs), so diam >= ~30
        # Realistically, start around 50

        # First get an upper bound from greedy
        S_greedy = greedy_construct(200, deadline - 10)
        if S_greedy:
            upper = max(S_greedy)
        else:
            upper = 150

        # Try increasing bounds
        for bound in range(45, upper + 1):
            if time.time() > deadline:
                break
            sol = backtrack_search(bound, deadline - 5)
            if sol:
                return sol

        # Fallback to greedy
        if S_greedy:
            return S_greedy
        return None

    # Main execution
    best_S = None
    best_diam = 10**9

    # Phase 1: Quick greedy for initial solution
    S_greedy = greedy_construct(150, start_time + 5)
    if S_greedy and validate(S_greedy):
        best_S = S_greedy
        best_diam = max(S_greedy)

    # Phase 2: Iterative deepening backtracking
    if time.time() < DEADLINE - 10:
        sol = iterative_deepening(DEADLINE - 10)
        if sol and validate(sol):
            if max(sol) < best_diam:
                best_S = sol
                best_diam = max(sol)

    # Phase 3: Local refinement
    if best_S and time.time() < DEADLINE - 5:
        refined = local_refinement(best_S, DEADLINE - 5)
        if refined and validate(refined):
            if max(refined) < best_diam:
                best_S = refined
                best_diam = max(refined)

    # Phase 4: If still no valid solution, use aggressive fallback
    if best_S is None or not validate(best_S):
        # Build a valid set by adding elements one by one
        S = [0]
        used_diffs = set()
        for d in range(1, 31):
            if d in used_diffs:
                continue
            found = False
            for x in range(1, 500):
                if x in S:
                    continue
                new_diffs = set()
                ok = True
                for s in S:
                    dd = abs(s - x)
                    if dd in used_diffs or dd in new_diffs:
                        ok = False
                        break
                    new_diffs.add(dd)
                if ok and d in new_diffs:
                    S.append(x)
                    used_diffs |= new_diffs
                    found = True
                    break
            if not found:
                # Add two elements to guarantee coverage
                base = 2 * max(S) + 100
                S.append(base)
                S.append(base + d)
                for s in S[:-2]:
                    used_diffs.add(abs(s - base))
                    used_diffs.add(abs(s - (base + d)))
                used_diffs.add(d)
        if validate(S):
            best_S = S

    # Final validation
    if best_S is None:
        # Absolute fallback: known construction
        # Use a simple Sidon set that covers 1..30
        # {0, 1, 4, 10, 12, 17, 21, 25, 28, 30, 34, 36, 39, 41, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78}
        # Actually let's just build one
        S = [0]
        used_diffs = set()
        for x in range(1, 200):
            if x in S:
                continue
            new_diffs = set()
            ok = True
            for s in S:
                dd = abs(s - x)
                if dd in used_diffs or dd in new_diffs:
                    ok = False
                    break
                new_diffs.add(dd)
            if ok:
                S.append(x)
                used_diffs |= new_diffs
            if TARGET.issubset(used_diffs) and len(S) >= 2:
                break
        best_S = S

    return best_S
