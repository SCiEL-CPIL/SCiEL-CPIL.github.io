"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We return by 140s.
    deadline = start_time + 135.0 

    # --- Helper: Fast Sidon Check ---
    def is_sidon(arr):
        """
        Checks if arr is a Sidon set.
        Returns True if valid, False otherwise.
        """
        n = len(arr)
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                d = arr[j] - arr[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def covers_required(arr):
        """
        Checks if differences cover 1..30.
        """
        diffs = set()
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(arr[j] - arr[i])
        for d in range(1, 31):
            if d not in diffs:
                return False
        return True

    def is_valid_solution(arr):
        """
        Full validity check.
        """
        if len(arr) > 32 or len(arr) < 2:
            return False
        if 0 not in arr:
            return False
        if min(arr) != 0: # We can always shift to min 0, but task says "including 0" and "differences". 
                          # Actually, if 0 is in set, min is 0.
            return False
        if max(arr) > 10**6:
            return False
        if not is_sidon(arr):
            return False
        if not covers_required(arr):
            return False
        return True

    # --- Core Solver: Backtracking with Forward Checking ---
    # We want to find a set A containing 0, size <= 32, max(A) <= D,
    # such that A is Sidon and {a-b | a,b in A, a>b} covers [1, 30].
    
    # Strategy:
    # 1. Fix 0 in the set.
    # 2. We need to place other elements x_1, ..., x_k in [1, D].
    # 3. The "required differences" are 1..30. 
    # 4. Instead of placing elements blindly, we can view this as:
    #    We have a set of "slots" for differences.
    #    A more effective CSP formulation:
    #    Variables: The elements of the set (excluding 0).
    #    However, the number of elements is variable (2 to 32).
    
    # Alternative CSP View:
    # Let's try to construct the set incrementally, but guided by the *missing differences*.
    # At any step, we have a set S. We have a set of covered differences D_S.
    # Missing differences M = {1..30} \ D_S.
    # We need to add a new element x such that {x - s for s in S} intersects M, 
    # and no new difference conflicts with existing differences.
    
    # This is a "Greedy with Backtracking" approach.
    # To improve efficiency, we use:
    # 1. Variable Ordering: Always pick the smallest missing difference to "force" a structure?
    #    No, that's not quite right. 
    # 2. Heuristic: When adding a new element x, prefer x that covers the *largest* missing difference?
    #    Or the *smallest*? 
    #    Usually, small differences are harder to pack without collision.
    #    Let's try: Pick a missing difference d. We need a pair (a, b) in S_new such that a-b=d.
    #    Since 0 is in S, one pair could be (d, 0). So adding d is a natural move if d is not conflicting.
    #    If adding d conflicts, we might need to add x such that x-s=d for some s in S.
    
    # Let's implement a backtracking search that builds the set.
    # State: current_set, current_diffs, missing_diffs
    # Goal: missing_diffs is empty.
    # Constraint: current_set is Sidon.
    
    def solve_csp(max_diameter):
        """
        Attempts to find a valid Sidon set with max element <= max_diameter.
        Returns the set if found, else None.
        """
        if time.time() > deadline:
            return None
            
        initial_set = [0]
        initial_diffs = set()
        required = set(range(1, 31))
        
        # We will use a recursive backtracking function.
        # To avoid deep recursion limits and overhead, we can use a stack or just recursion with sys.setrecursionlimit.
        # Max depth is 32, so recursion is fine.
        
        # Pre-compute conflicts for potential additions?
        # Given a set S, a candidate x is valid if:
        # 1. x not in S
        # 2. For all s in S, (x-s) not in diffs (if x>s) and (s-x) not in diffs (if s>x, but x is new max? No, x can be anywhere)
        #    Actually, since we build by adding elements, we don't enforce sorted order strictly, but differences are absolute.
        #    Let's maintain S as a sorted list.
        
        # Optimization: 
        # If we know we need to cover specific differences, we can restrict candidates.
        # However, a simple backtracking with good heuristics might be enough.
        
        # Heuristic for candidate generation:
        # Candidates are integers in [1, max_diameter] not in S.
        # To speed up, we can prioritize candidates that cover missing differences.
        
        # Let's use a "Most Constrained Variable" heuristic adapted for construction:
        # At each step, look at the missing differences.
        # If we can add an element x such that x-0 = d (i.e., x=d) for some missing d, and it's valid, do it.
        # Otherwise, look for x that creates multiple missing differences.
        
        # Actually, a stronger approach:
        # The set of differences must be unique.
        # Let's try to fill the set.
        
        def backtrack(S, diffs, missing):
            if time.time() > deadline:
                return None
                
            if not missing:
                return S[:]
                
            if len(S) >= 32:
                return None
                
            # Generate candidates
            # We want to find x in [1, max_diameter] \ S
            # such that adding x doesn't create duplicate differences.
            
            # To avoid checking all 1..max_diameter, we can be smarter.
            # But max_diameter is small (likely < 100). So checking 1..D is O(D * |S|).
            # D ~ 50, |S| ~ 10. 500 ops per node.
            # Depth 30. 500^30 is too big. We need pruning.
            
            # Pruning:
            # If a difference d in missing cannot be formed by any future additions?
            # Hard to know.
            
            # Heuristic Candidate Ordering:
            # 1. Candidates that cover the "hardest" missing differences.
            #    What is hard? Small differences are often hard.
            #    Let's sort missing diffs.
            #    For each candidate x, calculate how many missing diffs it covers.
            #    Prefer candidates covering more.
            
            # However, generating all candidates and scoring them might be slow if D is large.
            # Let's try a simpler heuristic first: 
            # Try candidates in a specific order.
            # Order: x that is in `missing` (covers x-0=x) first?
            
            # Let's collect valid candidates and score them.
            valid_candidates = []
            
            # We can limit the search space by noting that if we have a large gap in S, 
            # we might need to fill it.
            
            # Let's iterate through possible x.
            # To speed up, we can check if x is "promising".
            
            # A key insight: 
            # If we are trying to minimize max element, we should prefer smaller x?
            # Or larger x to jump over conflicts?
            # Usually, greedy "smallest valid" works well for Sidon sets (Golomb rulers).
            
            # Let's try: Iterate x from 1 to max_diameter.
            # Check validity.
            # If valid, check if it covers any missing diffs.
            # If it covers 0 missing diffs, it's a "waste" of a slot, unless necessary for structure.
            # But we MUST cover all 1..30. So every added element should ideally contribute to coverage.
            
            # Let's score candidates by:
            # 1. Number of missing diffs covered.
            # 2. Smaller x preferred (to keep diameter low).
            
            scored_candidates = []
            
            for x in range(1, max_diameter + 1):
                if x in S:
                    continue
                
                # Check Sidon validity for adding x
                new_diffs = set()
                ok = True
                for s in S:
                    d = abs(x - s)
                    if d in diffs or d in new_diffs:
                        ok = False
                        break
                    new_diffs.add(d)
                
                if not ok:
                    continue
                
                # Count how many missing diffs are covered
                covered_count = 0
                for d in new_diffs:
                    if d in missing:
                        covered_count += 1
                
                # If it covers nothing, it's generally bad, but might be needed?
                # Actually, if it covers nothing, it just adds "noise" differences.
                # Since we have limited slots (32), we should avoid noise if possible.
                # But if no candidate covers missing diffs, we might be stuck.
                
                # Score: Higher coverage is better. Smaller x is better.
                # We want to sort by (-covered_count, x)
                scored_candidates.append((covered_count, x, new_diffs))
            
            if not scored_candidates:
                return None
            
            # Sort candidates
            # Primary key: -covered_count (desc)
            # Secondary key: x (asc)
            scored_candidates.sort(key=lambda t: (-t[0], t[1]))
            
            # To avoid symmetry or repetitive failures, we can shuffle candidates with same score?
            # Or just try them in order.
            
            # Optimization: If the best candidate covers 0 missing diffs, 
            # and we have already used many elements, it might be a dead end.
            # But let's try the top few.
            
            # Limit branching factor? 
            # If we have many candidates covering the same number of diffs, trying all is expensive.
            # Let's try at most 10 candidates per level? 
            # No, that might miss the solution. 
            # But with D < 100, the number of valid candidates is usually small.
            
            for count, x, new_diffs in scored_candidates:
                # If this candidate doesn't cover any missing diffs, and we are deep, skip?
                # Let's allow it, but it will be lower priority.
                
                new_S = S + [x]
                new_S.sort()
                new_diffs_set = diffs.union(new_diffs)
                new_missing = missing.difference(new_diffs)
                
                # Recursive call
                result = backtrack(new_S, new_diffs_set, new_missing)
                if result is not None:
                    return result
            
            return None
        
        return backtrack(initial_set, initial_diffs, required)

    # --- Main Search Strategy ---
    # We know that a Sidon set of size k has at least ~k^2 differences.
    # We need 30 differences. 
    # k=10 gives ~45 diffs. k=9 gives ~36 diffs.
    # So size 9 or 10 is likely sufficient.
    # The diameter of a Sidon set of size k is at least ~k^2.
    # For k=10, D ~ 100.
    # Let's binary search or linear search for the minimum D.
    
    best_solution = None
    best_diameter = float('inf')
    
    # Lower bound estimate:
    # For a Sidon set with max element M, the number of differences is M(M-1)/2? No.
    # The number of distinct differences is at most M.
    # We need 30 distinct differences. So M must be at least 30.
    # But Sidon constraint is stronger.
    # Known Golomb rulers:
    # Length 10 (11 marks): 55
    # Length 8 (9 marks): 42
    # We need to cover 1..30.
    # Let's start D from 30 and go up.
    
    # To handle the case where a specific D is hard but D+1 is easy,
    # and to avoid getting stuck in one bad D, we will:
    # 1. Try D in increasing order.
    # 2. For each D, run the CSP solver with a time limit.
    # 3. If found, update best and stop (since we want min D).
    
    # However, CSP solvers can be tricky. 
    # Let's add a "Randomized Restart" mechanism for the CSP.
    # If backtrack fails quickly, try again with different tie-breaking?
    # My current backtrack sorts candidates deterministically.
    # Let's add a bit of randomness to the tie-breaking to escape local minima in the search tree.
    
    def solve_csp_randomized(max_diameter, seed_offset=0):
        if time.time() > deadline:
            return None
        
        initial_set = [0]
        initial_diffs = set()
        required = set(range(1, 31))
        
        def backtrack(S, diffs, missing):
            if time.time() > deadline:
                return None
                
            if not missing:
                return S[:]
                
            if len(S) >= 32:
                return None
                
            valid_candidates = []
            
            for x in range(1, max_diameter + 1):
                if x in S:
                    continue
                
                new_diffs = set()
                ok = True
                for s in S:
                    d = abs(x - s)
                    if d in diffs or d in new_diffs:
                        ok = False
                        break
                    new_diffs.add(d)
                
                if not ok:
                    continue
                
                covered_count = 0
                for d in new_diffs:
                    if d in missing:
                        covered_count += 1
                
                valid_candidates.append((covered_count, x, new_diffs))
            
            if not valid_candidates:
                return None
            
            # Sort by coverage, then use random tie-breaking for equal coverage
            # This helps explore different branches
            valid_candidates.sort(key=lambda t: (-t[0], random.random()))
            
            # Limit branching to top K to prevent explosion?
            # If we limit, we might miss solutions. 
            # But if D is small, branching is naturally limited.
            
            for count, x, new_diffs in valid_candidates:
                new_S = S + [x]
                new_S.sort()
                new_diffs_set = diffs.union(new_diffs)
                new_missing = missing.difference(new_diffs)
                
                result = backtrack(new_S, new_diffs_set, new_missing)
                if result is not None:
                    return result
            
            return None
        
        return backtrack(initial_set, initial_diffs, required)

    # Search Loop
    # We'll try D from 30 up to 200.
    # If we find a solution for D, that's our answer.
    # If time runs out, we return the best found (which might be None, then we need a fallback).
    
    found_solution = None
    
    # To increase robustness, we'll run multiple attempts for each D if it fails quickly.
    # But usually, if D is feasible, the CSP should find it.
    
    for D in range(30, 201):
        if time.time() > deadline - 5:
            break
            
        # Try a few random restarts for this D
        # The first attempt is deterministic-ish (random seed fixed? No, random module is global)
        # Let's just run it.
        
        # If D is small, the search space is tiny.
        # If D is large, it's harder.
        
        # Optimization: If we already have a solution with diameter D_prev, 
        # and we are checking D < D_prev, we only care if we find something better.
        # But we are iterating D upwards, so the first one we find is the minimum.
        
        # Let's add a small time limit per D to avoid getting stuck on infeasible small Ds.
        # But infeasible Ds should fail fast (no candidates).
        # Feasible Ds should succeed.
        # The risk is a "hard" feasible D where the CSP explores a huge bad subtree first.
        
        # Let's try 5 restarts for each D if it doesn't find quickly?
        # No, just run it. The random tie-breaking in sort handles the non-determinism.
        
        sol = solve_csp_randomized(D)
        
        if sol is not None:
            if is_valid_solution(sol):
                found_solution = sol
                best_diameter = D
                break # Found the minimum D
        
    if found_solution is not None:
        return found_solution
    
    # Fallback: If CSP failed (e.g. due to time or search failure), 
    # use a simple greedy to ensure we return *something* valid, even if not optimal.
    # The task requires a valid result. Invalid = 0 score.
    
    def greedy_fallback():
        S = [0]
        diffs = set()
        missing = set(range(1, 31))
        
        # Try to build a valid set with a generous diameter
        x = 1
        while missing:
            if x > 1000: # Safety
                break
            if x not in S:
                new_diffs = set()
                ok = True
                for s in S:
                    d = abs(x - s)
                    if d in diffs or d in new_diffs:
                        ok = False
                        break
                    new_diffs.add(d)
                if ok:
                    S.append(x)
                    S.sort()
                    diffs.update(new_diffs)
                    missing.difference_update(new_diffs)
            x += 1
            
        if not missing:
            return S
        return None

    fb_sol = greedy_fallback()
    if fb_sol is not None:
        return fb_sol
        
    # Absolute last resort: Hardcoded small Sidon set that covers 1..30?
    # Does one exist?
    # A Golomb ruler of length 30+?
    # If we can't find one dynamically, we might return an invalid set and score 0.
    # But greedy_fallback should work for D=1000.
    
    # If even greedy fails, return [0] (invalid, but better than exception?)
    # No, [0] doesn't cover 1..30.
    # Let's assume greedy_fallback works.
    
    return [0] # Should not reach here if greedy works

# EVOLVE-BLOCK-END
