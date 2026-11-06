import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def check_solution(A):
        """Verify A is a valid Sidon set covering 1..30."""
        diffs = set()
        for i in range(len(A)):
            for j in range(i):
                d = A[i] - A[j]
                if d in diffs:
                    return False
                diffs.add(d)
        # Check if all 1..30 are covered
        for d in range(1, 31):
            if d not in diffs:
                return False
        return True

    def solve_dpll(M, time_limit):
        """
        DPLL-style search for a Sidon set in [0, M] that covers all differences 1..30.
        Forces 0 and M as the first and last elements (since we want to minimize max element).
        Returns a valid set or None.
        """
        FULL_MASK = (1 << 30) - 1
        
        # State:
        # A: sorted list of chosen elements
        # all_diffs: set of all positive differences so far
        # covered_mask: bitmask of differences 1..30 covered
        
        # We start with 0 and M as fixed.
        # Actually, to be flexible, let's just enforce that the final set contains 0 and M.
        # But since we are searching for *a* set with max element M, it's optimal to include M?
        # Not necessarily. The set could be {0, 1, ..., k} with max < M.
        # But if we are searching for the *minimal* M, and we found a solution with max < M,
        # we would have found it in a previous iteration of M.
        # So, when searching for a specific M, we can assume M is in the set.
        # Similarly, 0 must be in the set.
        
        # So, initial A = [0, M]
        # Initial diffs = {M}
        # Initial mask = bit for M if M <= 30, else 0
        
        A = [0, M]
        all_diffs = {M}
        
        covered_mask = 0
        if 1 <= M <= 30:
            covered_mask |= (1 << (M - 1))
            
        # If M > 30, we need to cover 1..30 using intermediate elements.
        # If M <= 30, we already cover M.
        
        # Check if solution is already found (unlikely for small M)
        if covered_mask == FULL_MASK:
            return A[:]

        # Backtracking search
        # We need to add elements in (0, M) such that:
        # 1. Sidon property holds
        # 2. All diffs 1..30 are covered
        
        # Heuristic for choosing next element:
        # We should prioritize elements that cover the most "hard" differences.
        # Or, we can iterate through candidates x in (0, M) that are valid.
        
        # To speed up, we can precompute valid candidates.
        # A candidate x is valid if:
        # - x not in A
        # - For all a in A, (x-a) not in all_diffs and (a-x) not in all_diffs (if x<a)
        #   Since A is sorted, x > a for a < x, and x < a for a > x.
        #   We maintain A sorted.
        
        # Let's implement a recursive solver.
        # To avoid deep recursion overhead, we might use iterative deepening or just standard recursion.
        # Given M is small (<= 100), depth is at most 32.
        
        def backtrack(A, all_diffs, covered_mask):
            if time.time() > time_limit:
                return None
            
            if covered_mask == FULL_MASK:
                return A[:]
                
            # Pruning:
            # 1. If len(A) > 32, return None
            if len(A) > 32:
                return None
                
            # 2. Check if remaining differences can be covered.
            missing = FULL_MASK ^ covered_mask
            missing_count = bin(missing).count('1')
            
            # Max new diffs we can add:
            # If we add k elements, we add at most k * len(A) + k*(k-1)/2 diffs.
            # But we have 32 - len(A) slots.
            slots = 32 - len(A)
            if slots <= 0:
                return None
            
            # Rough bound: each new element adds at most len(A) diffs (with existing) + 1 (with previous new)
            # Actually, if we add elements one by one, the i-th new element adds at most len(A) + (i-1) diffs.
            # Total new diffs <= sum_{i=1}^{slots} (len(A) + i - 1) = slots * len(A) + slots*(slots-1)//2
            max_new_diffs = slots * len(A) + slots * (slots - 1) // 2
            if missing_count > max_new_diffs:
                return None

            # Generate candidates
            # Candidates are x in [1, M-1] such that x not in A.
            # We can iterate x from 1 to M-1.
            # Optimization: Only consider x that cover at least one missing difference.
            
            # To find x that cover missing diffs, we can check:
            # For each missing diff d, x must be a + d or a - d for some a in A.
            # So candidates are union of {a+d, a-d for a in A, d in missing_diffs} intersect [1, M-1].
            
            missing_diffs = [d+1 for d in range(30) if not (covered_mask & (1 << d))]
            
            candidates = set()
            for a in A:
                for d in missing_diffs:
                    x1 = a + d
                    if 1 <= x1 <= M-1 and x1 not in A:
                        candidates.add(x1)
                    x2 = a - d
                    if 1 <= x2 <= M-1 and x2 not in A:
                        candidates.add(x2)
            
            # If no candidates, we might still need to add elements that don't cover missing diffs immediately?
            # No, if we can't cover a missing diff now, and we are constrained to [0, M],
            # adding an element that doesn't cover a missing diff doesn't help cover it later,
            # UNLESS it enables a combination? No, differences are pairwise.
            # If x doesn't create a diff d with any current a, it won't create d with future a' either
            # unless a' - x = d, which means a' = x + d.
            # But if x is added, and later a' is added, the diff is a' - x.
            # So we should consider candidates that *can* lead to covering missing diffs.
            # However, the simple heuristic of "must cover a missing diff immediately" is strong.
            # If it fails, we might miss a solution where we add "bridge" elements.
            # But for minimal M, this is unlikely. Let's stick to this for speed.
            
            # Sort candidates to try smaller ones first? Or larger?
            # Smaller ones might be better for packing.
            for x in sorted(candidates):
                # Check validity of x
                valid = True
                new_diffs = []
                for a in A:
                    d = abs(x - a)
                    if d in all_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                    
                # Update mask
                new_mask = covered_mask
                for d in new_diffs:
                    if 1 <= d <= 30:
                        new_mask |= (1 << (d - 1))
                        
                # If no new mask bits, skip?
                # We already filtered candidates to cover at least one missing diff.
                # But it's possible that the diff is already covered?
                # No, we generated candidates based on missing diffs.
                # So new_mask should be different from covered_mask.
                
                # Recurse
                new_A = A + [x]
                new_A.sort()
                new_all_diffs = all_diffs | set(new_diffs)
                
                res = backtrack(new_A, new_all_diffs, new_mask)
                if res is not None:
                    return res
                    
            return None

        return backtrack(A, all_diffs, covered_mask)

    # Main search loop
    # Lower bound for M:
    # A Sidon set of size n has max element at least ~ n^2/2?
    # For n=32, M ~ 500? No, that's for packing.
    # Here we just need to cover 1..30.
    # With 0 and M, we have 1 diff.
    # We need 29 more.
    # Each new element adds at most len(A) diffs.
    # So we need at least ~ sqrt(30) elements?
    # Actually, known minimal diameter for covering 1..30 with Sidon set:
    # Let's start from a reasonable lower bound.
    # M must be at least 30? No, if M=30, we cover 30.
    # Can we cover 1..29 with elements in [1, 29]?
    # Let's try M from 30 upwards.
    
    # However, the problem states "Minimize the largest element".
    # The seed program found a solution. What was its max element?
    # Score 17.65 implies a specific value.
    # Let's assume the optimal M is around 50-100.
    
    # To be safe and fast, we can try M in increasing order.
    # If we find a solution for M, it's the minimal M.
    
    # But wait, is it possible that a non-minimal M allows a *smaller* max element in the set?
    # No, the max element of the set is M by definition in our search.
    # So we just want the smallest M such that a solution exists.
    
    # Start M from 30.
    # If no solution, try 31, 32, ...
    
    # Optimization: Parallel search or random restarts?
    # DPLL is deterministic.
    
    # Let's add a time check inside the loop.
    
    best_solution = None
    
    # We can also try a greedy initialization to get a good upper bound quickly,
    # then use it to prune?
    # For now, just iterate M.
    
    # To speed up, we can skip M values that are impossible.
    # But let's just try them.
    
    # Start from a slightly higher bound to find *a* solution quickly, then binary search?
    # No, we want the *minimum*.
    
    # Let's try M from 30 to 200.
    # If time runs out, return the best found (which might be None).
    # But we must return a valid set.
    # So we need a fallback.
    
    # Fallback: Greedy construction
    def greedy_construct():
        A = [0]
        all_diffs = set()
        covered = set()
        
        for d in range(1, 31):
            if d in covered:
                continue
            found = False
            # Try to add an element x such that x - a = d or a - x = d for some a in A
            # And Sidon property holds.
            candidates = set()
            for a in A:
                candidates.add(a + d)
                if a - d >= 0:
                    candidates.add(a - d)
            
            for x in sorted(candidates):
                if x in A:
                    continue
                # Check Sidon
                valid = True
                new_diffs = []
                for a in A:
                    diff = abs(x - a)
                    if diff in all_diffs:
                        valid = False
                        break
                    new_diffs.append(diff)
                
                if valid:
                    A.append(x)
                    A.sort()
                    all_diffs.update(new_diffs)
                    covered.update(d for d in new_diffs if 1 <= d <= 30)
                    found = True
                    break
            
            if not found:
                # If we can't add an element to cover d, we might be stuck.
                # Try adding a large element?
                # This greedy is not guaranteed to work.
                pass
        
        # If greedy didn't cover all, we need a more robust fallback.
        # Let's just return a known valid set if we have one?
        # Or expand the set until valid?
        
        # A simple way to ensure validity:
        # Keep adding elements 1, 2, 3... until Sidon property is broken? No.
        # Just add elements far apart.
        while True:
            # Check if all 1..30 covered
            diffs = set()
            for i in range(len(A)):
                for j in range(i):
                    diffs.add(A[i] - A[j])
            if all(d in diffs for d in range(1, 31)):
                return A
            
            # Add next integer that maintains Sidon
            added = False
            for x in range(1, 10000):
                if x in A:
                    continue
                valid = True
                for a in A:
                    d = abs(x - a)
                    if d in diffs:
                        valid = False
                        break
                if valid:
                    A.append(x)
                    A.sort()
                    for a in A:
                        diffs.add(abs(x - a))
                    added = True
                    break
            if not added:
                # Should not happen
                break
        
        return A

    # Try DPLL for M from 30 to 200
    for M in range(30, 201):
        if time.time() > deadline - 5:
            break
        
        res = solve_dpll(M, deadline - 2)
        if res is not None:
            if check_solution(res):
                return res
        
    # If DPLL failed to find a minimal M quickly, or found none,
    # try a slightly larger M with more time?
    # Or just use the greedy fallback.
    # But the greedy might return a very large set.
    
    # Let's try a few more M values with the remaining time.
    # Actually, if DPLL failed for M=30..200, it might be slow.
    # Let's just return the greedy result to ensure validity.
    
    # To improve the greedy, we can try to minimize the max element.
    # But the task is to minimize the max element.
    # If DPLL didn't find the optimal, the greedy might be suboptimal but valid.
    
    # Let's try to run DPLL for a specific M that is likely to work, e.g., M=100.
    # If that fails, use greedy.
    
    # Actually, the seed program's score was 17.65.
    # This corresponds to a specific max element.
    # Let's assume the optimal is found by DPLL.
    
    # If we are running out of time, return the best we have.
    # Since we didn't save intermediate results, we return greedy.
    
    return greedy_construct()

if __name__ == "__main__":
    res = construct()
    print(res)
    print(len(res))
    print(max(res))
