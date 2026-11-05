"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # We want to find a Sidon set A containing 0, |A| <= 32,
    # such that all d in 1..30 are differences in A.
    # Minimize max(A).
    
    # A Sidon set of size n has at most n(n-1)/2 distinct positive differences.
    # For n=32, max diffs = 496. We need 1..30.
    # The diameter D must be large enough to accommodate these.
    # Known small Sidon sets:
    # The "perfect" difference sets are rare. We are looking for a truncated one.
    # Let's use a backtracking search to build the set.
    
    # Strategy:
    # 1. Sort required differences: 1, 2, ..., 30.
    # 2. We build the set A incrementally.
    # 3. We prioritize adding elements that create "new" required differences.
    # 4. We maintain the set of "covered" differences.
    # 5. We use a heuristic ordering for candidate elements to minimize the diameter.
    
    # To speed up, we can fix 0 in the set.
    # We will search for the set A.
    
    # A better approach for "Minimize largest element":
    # Binary search on the maximum element M?
    # Or just iterative deepening / backtracking with pruning based on current max?
    
    # Let's use a randomized backtracking search with restarts, guided by a heuristic.
    # Since the space is small (we just need 1..30 covered), we can be aggressive.
    
    # Pre-computed "good" Sidon sets often come from modular constructions,
    # but for small N, explicit search works.
    
    # Let's try to find a set with small diameter.
    # Lower bound for diameter:
    # If max diff is D, and we have k elements, the differences are distinct.
    # The sum of the smallest k(k-1)/2 differences is bounded by something related to D.
    # Actually, a simple bound: if A = {a_1 < ... < a_k}, then a_k - a_1 >= k-1 is weak.
    # Stronger: The differences a_j - a_i are distinct.
    
    # Let's just run a backtracking search that tries to keep the set compact.
    
    best_solution = None
    best_diameter = 10**6
    
    # Heuristic: 
    # We know 0 is in the set.
    # Let's try to place the next elements.
    # To minimize the max, we should try to fill the "gaps" in differences.
    
    # Algorithm:
    # State: current set A (sorted), set of covered differences C.
    # Goal: C contains 1..30.
    # Constraint: A is Sidon (all pairwise differences distinct).
    
    # To make it fast, we limit the size of A to, say, 15-20 elements.
    # 15 elements -> 105 diffs. Enough to cover 30.
    # If we use 10 elements -> 45 diffs. Might be enough.
    
    # We will perform a depth-first search.
    # At each step, we choose a new element x to add to A.
    # Candidates for x:
    # 1. x such that x - a_i or a_i - x covers a missing difference.
    # 2. To minimize diameter, we prefer small x.
    
    # Optimization:
    # Instead of generic backtracking, we can use a "repair" or "construction" heuristic.
    # But let's stick to CSP.
    
    # To ensure we finish in time, we will use a randomized approach with a time limit.
    # We will run multiple restarts, each trying to find a solution with a tighter bound on the max element.
    
    # Let's define a function to check if a set is valid (Sidon + covers 1..30)
    def is_valid_solution(A):
        if not A or 0 not in A:
            return False
        if len(A) > 32:
            return False
        diffs = set()
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                d = A[j] - A[i]
                if d in diffs:
                    return False # Not Sidon
                diffs.add(d)
        for d in range(1, 31):
            if d not in diffs:
                return False
        return True

    # Search function
    def search(A, covered, max_val, depth):
        nonlocal best_solution, best_diameter
        
        if time.time() > deadline:
            return
        
        # Pruning: if current max is already >= best_diameter, stop?
        # Actually, we want to minimize max. If we find a solution, we update best.
        # If we are at a leaf (covered 1..30), check validity.
        
        if all(d in covered for d in range(1, 31)):
            # We have covered all required differences.
            # Check if it's a valid Sidon set (it should be, by construction, but double check)
            # And check size constraint.
            if len(A) <= 32:
                curr_max = max(A)
                if curr_max < best_diameter:
                    best_diameter = curr_max
                    best_solution = A[:]
                    # We can prune deeper branches if we find a very good solution?
                    # For now, just record and return.
            return
        
        if len(A) >= 32:
            return
            
        # If time is running out, stop
        if time.time() > deadline:
            return

        # Generate candidates for the next element
        # We want to add an element that helps cover missing differences.
        # Missing differences: M = {d in 1..30 | d not in covered}
        
        # Candidate generation strategy:
        # For each missing difference d, and each a in A:
        #   x = a + d  (if x > max(A))
        #   x = a - d  (if x > 0 and x < min(A)? No, A is sorted, so x would be smaller. 
        #                But we add elements to the set, we don't require sorted insertion order 
        #                for the search tree, but A is kept sorted for diff calculation efficiency?
        #                Actually, we can just keep A as a set and sort when checking.
        #                But for "max_val" pruning, we need to know the new max.
        
        # Let's generate a list of potential new elements.
        candidates = set()
        
        # 1. Extend from existing elements to cover missing diffs
        for d in range(1, 31):
            if d in covered:
                continue
            for a in A:
                # Try adding a + d
                x1 = a + d
                # Try adding a - d (if positive)
                if a - d > 0:
                    x2 = a - d
                    candidates.add(x2)
                candidates.add(x1)
                
                # Also, maybe we can bridge two existing elements?
                # If we have a, b in A, and we want to cover d = b-a? No, that's already covered if b-a is in covered.
                # We are looking for NEW differences.
                
        # 2. Heuristic: Also consider small integers that might create many diffs?
        #    Or just stick to the "forced" moves from missing diffs.
        
        # Filter candidates:
        # - Must be > 0 (unless 0 is not in A, but 0 is in A)
        # - Must not be in A
        # - Must result in a valid Sidon set when added
        
        valid_candidates = []
        current_max = max(A) if A else 0
        
        for x in candidates:
            if x in A:
                continue
            if x < 0:
                continue
            
            # Check if adding x violates Sidon property
            # We need to check if x - a_i or a_i - x creates a duplicate difference.
            # Let's just try to add it and check.
            
            new_A = A + [x]
            # Check Sidon
            diffs = set()
            is_sidon = True
            for i in range(len(new_A)):
                for j in range(i+1, len(new_A)):
                    d = new_A[j] - new_A[i]
                    if d in diffs:
                        is_sidon = False
                        break
                    diffs.add(d)
                if not is_sidon:
                    break
            
            if not is_sidon:
                continue
                
            # Check new covered diffs
            new_covered = set(covered)
            for d in diffs:
                if 1 <= d <= 30:
                    new_covered.add(d)
            
            # Pruning: if new_max >= best_diameter, we can skip?
            # Not necessarily, because we might have fewer elements and thus a smaller max later?
            # No, max is monotonic in the set inclusion if we only add positive numbers?
            # Yes, max(A U {x}) >= max(A).
            # So if max(new_A) >= best_diameter, we can prune this branch IF we are only interested in
            # improving the diameter.
            # However, if best_diameter is infinity initially, this doesn't prune.
            # Once we have a solution, this is a strong prune.
            
            if max(new_A) >= best_diameter:
                continue
                
            valid_candidates.append((x, new_A, new_covered))
            
        # Sort candidates to explore promising ones first
        # Heuristic: prefer x that covers more missing differences
        # Or prefer smaller x to keep diameter small?
        # Smaller x usually leads to smaller diameter.
        
        def score_candidate(item):
            x, new_A, new_covered = item
            # Count how many NEW missing diffs are covered
            new_missing_covered = 0
            for d in range(1, 31):
                if d in new_covered and d not in covered:
                    new_missing_covered += 1
            # Prefer smaller x, then more coverage
            return (max(new_A), -new_missing_covered)
            
        valid_candidates.sort(key=score_candidate)
        
        # Randomize a bit to escape local minima?
        # For now, deterministic.
        
        for x, new_A, new_covered in valid_candidates:
            search(new_A, new_covered, max(new_A), depth + 1)
            if time.time() > deadline:
                return
            # If we found a solution with diameter 1? Impossible.
            # If we found a very good solution, we might want to stop early?
            # No, let's let it run until deadline to find the absolute best.
            
    # Initial state
    initial_A = [0]
    initial_covered = set()
    
    # To avoid getting stuck in a bad local minimum or taking too long on one path,
    # we can run multiple restarts with slight variations or different initial seeds?
    # The current search is deterministic.
    # Let's add a simple restart mechanism if the first search doesn't find anything quickly.
    
    # Actually, the search above might be slow if the candidate list is large.
    # Let's optimize the candidate generation.
    
    # Alternative: Use a known good set as a starting point and try to improve it?
    # Or just use the backtracking.
    
    # Let's run the search.
    search(initial_A, initial_covered, 0, 0)
    
    if best_solution is None:
        # Fallback: Return a known valid set if search fails or times out without finding one.
        # We need a valid Sidon set covering 1..30.
        # Let's construct one manually or use the greedy method from the seed as a fallback.
        a = [0]
        def diffs(s):
            return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
        for d in range(1, 31):
            if d in diffs(sorted(a)):
                continue
            found = False
            for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
                trial = sorted(a + [x])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    found = True
                    break
            if not found:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if len(set(diffs(trial))) == len(diffs(trial)):
                        a = trial
                        break
                    y += 1
        return a
        
    return best_solution

# EVOLVE-BLOCK-END
