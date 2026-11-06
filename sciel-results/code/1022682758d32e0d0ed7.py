import time
import random
from itertools import combinations

def construct():
    """
    Find a Sidon set containing 0 with all differences 1..30, minimizing the largest element.
    Uses a difference-centric CSP: assign each required difference to a unique pair,
    then solve for the set elements. Combines with local search refinement.
    """
    start_time = time.time()
    DEADLINE = 135.0

    TARGET = set(range(1, 31))
    TARGET_LIST = list(range(1, 31))

    def is_sidon_fast(S):
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

    def greedy_init(max_val=200):
        """Greedy construction: pick elements that maximize coverage of missing diffs."""
        S = [0]
        used_diffs = set()
        covered = set()

        for step in range(31):
            if time.time() > DEADLINE - 20:
                break
            if covered == TARGET:
                break

            missing = TARGET - covered
            best_x = None
            best_gain = 0
            best_total_new = 999

            # Candidate pool: values that could cover at least one missing diff
            candidates = set()
            for s in S:
                for d in missing:
                    candidates.add(s + d)
                    if s - d > 0:
                        candidates.add(s - d)

            candidates = sorted([c for c in candidates if 0 < c <= max_val and c not in set(S)])

            for x in candidates:
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
                if gain > best_gain or (gain == best_gain and len(new_diffs) < best_total_new):
                    best_gain = gain
                    best_total_new = len(new_diffs)
                    best_x = x
                    if gain == len(missing):
                        break

            if best_x is None:
                # Fallback: try small values
                for x in range(1, min(max_val + 1, 300)):
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
            covered = used_diffs & TARGET

        return S

    def local_refine(S, deadline):
        """Reduce max(S) by replacing elements with smaller values."""
        S = sorted(S)
        best = S[:]
        best_max = max(S)

        while time.time() < deadline:
            improved = False
            for i in range(len(S)):
                if S[i] == 0:
                    continue
                old_val = S[i]
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

    def swap_local_search(S, deadline):
        """Try swapping pairs of elements to reduce diameter."""
        S = sorted(S)
        best = S[:]
        best_max = max(S)

        while time.time() < deadline:
            improved = False
            n = len(S)
            for i in range(n):
                for j in range(i + 1, n):
                    if S[i] == 0 or S[j] == 0:
                        continue
                    # Try replacing S[j] with a smaller value
                    for new_val in range(1, S[j]):
                        if new_val in S:
                            continue
                        trial = S[:]
                        trial[j] = new_val
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
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break

        return best

    def difference_csp_search(max_val, deadline):
        """
        Difference-centric CSP: For each d in 1..30, choose a pair (a,b) with b-a=d.
        All pairs must be consistent (same element appears in at most the right number of pairs)
        and all differences must be distinct (Sidon).
        
        We model this as: select a set of elements S containing 0, such that:
        - For each d in 1..30, there exists a,b in S with b-a=d
        - All pairwise differences are distinct
        
        Approach: backtracking over differences, maintaining partial assignment of elements.
        """
        if time.time() > deadline:
            return None

        # For each difference d, possible pairs are (x, x+d) for x in 0..max_val-d
        # We need to choose one pair per d, and the union of all elements must form a Sidon set.
        
        # Use backtracking: process differences in order of fewest options first (MRV)
        # Maintain: set of used elements, set of used differences (for Sidon check)
        
        best_solution = None
        best_diameter = max_val + 1

        # Precompute: for each d, list of possible (a, b) pairs
        pair_options = {}
        for d in range(1, 31):
            pair_options[d] = [(a, a + d) for a in range(0, max_val - d + 1)]

        # Order differences by number of options (ascending) for MRV
        diff_order = sorted(range(1, 31), key=lambda d: len(pair_options[d]))

        # State: for each element, how many times it's "used" (should be at most 1 in the set)
        # Actually, elements in the set are just a set. The constraint is:
        # - Each element appears in at most... well, it can appear in multiple pairs.
        # - The key constraint is Sidon: all differences between pairs of elements must be distinct.
        
        # Better formulation: build the set incrementally.
        # Start with {0}. For each difference d (in some order), we need to add elements
        # such that d = b - a for some a, b in the set.
        
        # Let's use a different approach: 
        # 1. Start with S = {0}
        # 2. For each d in 1..30 (in MRV order), ensure d is realized.
        #    If d = 0 - a or b - 0 for some existing a,b, it's already covered.
        #    Otherwise, we need to add at least one new element.
        #    Prefer adding one element that covers d and possibly other missing diffs.
        
        # This is essentially the greedy approach but with backtracking.
        
        used_diffs = set()
        covered = set()
        current_set = {0}
        
        def backtrack(diff_idx, depth):
            nonlocal best_solution, best_diameter
            
            if time.time() > deadline:
                return False
            
            if diff_idx == len(diff_order):
                # All differences covered
                if max(current_set) < best_diameter:
                    best_diameter = max(current_set)
                    best_solution = sorted(current_set)
                return True
            
            d = diff_order[diff_idx]
            
            # Check if d is already covered
            if d in covered:
                return backtrack(diff_idx + 1, depth)
            
            # Try to add elements to cover d
            # Option 1: add x such that x - 0 = d, i.e., x = d
            # Option 2: add x such that 0 - x = d, i.e., x = -d (invalid, x must be >= 0)
            # Option 3: add x such that x - a = d for some a in current_set
            # Option 4: add x such that b - x = d for some b in current_set
            
            # Generate candidate elements that would cover d
            candidates = set()
            for a in current_set:
                candidates.add(a + d)  # new element = a + d
                if a - d > 0:
                    candidates.add(a - d)  # new element = a - d
            
            # Filter candidates
            valid_candidates = []
            for x in sorted(candidates):
                if x <= 0 or x > max_val or x in current_set:
                    continue
                # Check if adding x maintains Sidon property
                new_diffs = set()
                valid = True
                for s in current_set:
                    dd = abs(s - x)
                    if dd in used_diffs or dd in new_diffs:
                        valid = False
                        break
                    new_diffs.add(dd)
                if valid:
                    valid_candidates.append((x, new_diffs))
            
            # Try each candidate
            for x, new_diffs in valid_candidates:
                # Check if this helps (covers d and possibly others)
                if d not in new_diffs:
                    continue
                
                # Apply
                old_used = set(used_diffs)
                old_covered = set(covered)
                old_set = set(current_set)
                
                current_set.add(x)
                used_diffs |= new_diffs
                covered |= (new_diffs & TARGET)
                
                if backtrack(diff_idx + 1, depth + 1):
                    return True
                
                # Backtrack
                used_diffs = old_used
                covered = old_covered
                current_set = old_set
            
            return False
        
        backtrack(0, 0)
        
        if best_solution:
            return best_solution
        
        return None

    def iterative_deepening_search(deadline):
        """Try increasing max_val until we find a solution."""
        # Start with a reasonable lower bound
        # For 31 elements in a Sidon set with differences 1..30, 
        # the minimum diameter is at least ~sqrt(30*31/2) ≈ 22, but likely much larger.
        # Known perfect difference sets: for n=31, the smallest known has diameter around 100+.
        
        # Use greedy to get an initial upper bound
        S_greedy = greedy_init(200)
        upper = max(S_greedy) if S_greedy and covers_all(S_greedy) else 200
        
        # Try from lower values up to the greedy solution
        for max_val in range(40, upper + 1):
            if time.time() > deadline:
                break
            
            sol = difference_csp_search(max_val, deadline - 5)
            if sol and covers_all(sol) and is_sidon_fast(sol):
                return sol
        
        # If CSP search failed, return greedy
        if S_greedy and covers_all(S_greedy) and is_sidon_fast(S_greedy):
            return S_greedy
        
        return None

    # Main search
    best_S = None
    best_score = (30, 10**9)

    # Get initial solution with greedy
    random.seed(42)
    S0 = greedy_init(200)
    if S0 and covers_all(S0) and is_sidon_fast(S0):
        s0 = score_set(S0)
        if s0 < best_score:
            best_score = s0
            best_S = S0

    # Try difference-centric CSP search
    if time.time() < DEADLINE - 30:
        sol_csp = iterative_deepening_search(DEADLINE - 30)
        if sol_csp and covers_all(sol_csp) and is_sidon_fast(sol_csp):
            s = score_set(sol_csp)
            if s < best_score:
                best_score = s
                best_S = sol_csp

    # Local refinement
    if best_S and time.time() < DEADLINE - 10:
        refined = local_refine(best_S, DEADLINE - 10)
        if refined:
            rs = score_set(refined)
            if rs < best_score:
                best_score = rs
                best_S = refined

    # Swap local search
    if best_S and time.time() < DEADLINE - 5:
        swapped = swap_local_search(best_S, DEADLINE - 5)
        if swapped:
            ss = score_set(swapped)
            if ss < best_score:
                best_score = ss
                best_S = swapped

    # Final validation
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
