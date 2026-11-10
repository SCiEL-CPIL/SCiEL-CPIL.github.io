import random
import sys
import time

def construct():
    n = 256
    best_total = 0
    best_A = []
    best_B = []
    
    # We will use a backtracking search with bitsets for efficiency.
    # diff_mask_A: bit d is 1 if d is a difference in A
    # diff_mask_B: bit d is 1 if d is a difference in B
    # sum_mask_A: bit s is 1 if s is a sum in A
    # sum_mask_B: bit s is 1 if s is a sum in B
    
    # Max sum is 256+256 = 512, so we need bits 0..512.
    # Max diff is 255, so we need bits 0..255.
    
    def dfs(A, B, diff_A, diff_B, sum_A, sum_B, depth):
        nonlocal best_total, best_A, best_B
        
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_A = A[:]
            best_B = B[:]
            
        if depth == 0:
            return
        
        # Heuristic: try to add numbers from 1 to n
        # To improve diversity, we can shuffle the order of candidates in the outer loop
        # But inside DFS, we usually iterate in a fixed order or heuristic order.
        # Let's try adding each unused number to A or B.
        
        # Optimization: If we can't add any more, return
        # We iterate through candidates 1..n
        # To speed up, we can skip numbers that are already in A or B? 
        # The problem says A and B need not be disjoint. So x can be in both.
        # However, if x is in A, adding x to A again is not allowed (sets).
        # So we only consider x if x not in A for adding to A, and x not in B for adding to B.
        
        # To avoid deep recursion limits and speed up, we limit the branching.
        # We will try to add the smallest available number first? Or random?
        # Let's use a randomized greedy inside the DFS for the first few levels, 
        # but pure DFS might be too slow.
        
        # Alternative: Use a greedy approach with backtracking only when stuck.
        # Let's stick to a DFS that tries to add any valid x.
        # To prevent explosion, we can limit the number of branches explored per level.
        
        # Better approach: Iterative Deepening or just a Greedy with multiple random restarts 
        # using the fast bitset check.
        
        pass # This function structure is a placeholder for the actual logic below

    # Let's implement a Greedy with Backtracking that tries to fill up to a target size.
    # Or simply, run a DFS that explores adding elements.
    
    # Given the time limit, a pure DFS to depth 60 is impossible.
    # We will use a **Greedy Randomized Adaptive Search (GRASP)** or **Simulated Annealing**.
    # Let's go with **Simulated Annealing** starting from a greedy solution.
    
    def greedy_construct(seed):
        rng = random.Random(seed)
        A = []
        B = []
        diff_A = 0
        diff_B = 0
        sum_A = 0
        sum_B = 0
        
        # Candidates
        candidates = list(range(1, n + 1))
        rng.shuffle(candidates)
        
        for x in candidates:
            # Try adding to A
            if x not in A:
                # Check Sidon for A
                ok = True
                # Check differences
                for a in A:
                    d = abs(x - a)
                    if (diff_A >> d) & 1: # diff already in A
                        ok = False
                        break
                    if (diff_B >> d) & 1: # diff in B
                        ok = False
                        break
                if ok:
                    # Check sums
                    for a in A:
                        s = x + a
                        if (sum_A >> s) & 1:
                            ok = False
                            break
                    if ok:
                        s2 = 2 * x
                        if (sum_A >> s2) & 1:
                            ok = False
                        else:
                            # Valid
                            for a in A:
                                d = abs(x - a)
                                diff_A |= (1 << d)
                            sum_A |= (1 << s2)
                            for a in A:
                                sum_A |= (1 << (x + a))
                            A.append(x)
                            continue # Try next x
            
            # Try adding to B
            if x not in B:
                ok = True
                for b in B:
                    d = abs(x - b)
                    if (diff_B >> d) & 1:
                        ok = False
                        break
                    if (diff_A >> d) & 1:
                        ok = False
                        break
                if ok:
                    for b in B:
                        s = x + b
                        if (sum_B >> s) & 1:
                            ok = False
                            break
                    if ok:
                        s2 = 2 * x
                        if (sum_B >> s2) & 1:
                            ok = False
                        else:
                            for b in B:
                                d = abs(x - b)
                                diff_B |= (1 << d)
                            sum_B |= (1 << s2)
                            for b in B:
                                sum_B |= (1 << (x + b))
                            B.append(x)
        
        return A, B

    # Run multiple greedy restarts
    for i in range(100):
        A, B = greedy_construct(i)
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_A = A
            best_B = B
            
    # Try to improve the best solution with local search (swap/add/remove)
    # We can try to add elements that were skipped or swap elements.
    
    # For now, return the best found
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
        
    # Final validation
    # Re-verify with simple sets to be safe
    def _is_sidon(arr):
        sums = set()
        n = len(arr)
        for i in range(n):
            for j in range(i, n):
                s = arr[i] + arr[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def _diffs(arr):
        d = set()
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                d.add(abs(arr[j] - arr[i]))
        return d

    assert _is_sidon(best_A), "A is not Sidon"
    assert _is_sidon(best_B), "B is not Sidon"
    assert not (_diffs(best_A) & _diffs(best_B)), "Differences not disjoint"
    assert len(best_A) <= 64 and len(best_B) <= 64, "Size limit exceeded"
    assert all(1 <= x <= 256 for x in best_A + best_B), "Out of range"
    
    return [best_A, best_B]
