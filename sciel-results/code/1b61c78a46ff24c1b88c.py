import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 4096
    max_sum = 3 * n
    best_set = []
    
    used_sums = bytearray(max_sum + 1)
    
    def reset_sums():
        for i in range(len(used_sums)):
            used_sums[i] = 0
            
    def get_new_sums_indices(x, S):
        """Generate indices of new sums involving x and elements of S.
        Returns None if collision, else list of new sum indices."""
        new_sums = []
        s = 3 * x
        if used_sums[s]:
            return None
        new_sums.append(s)
        
        for y in S:
            s = 2 * x + y
            if used_sums[s]:
                return None
            new_sums.append(s)
            
        k = len(S)
        for i in range(k):
            for j in range(i, k):
                s = x + S[i] + S[j]
                if used_sums[s]:
                    return None
                new_sums.append(s)
                
        return new_sums

    def mark_sums(sums):
        for s in sums:
            used_sums[s] = 1

    def unmark_sums(sums):
        for s in sums:
            used_sums[s] = 0

    def backtrack(S, candidates, target_size):
        """
        Recursive backtracking.
        S: current set
        candidates: list of remaining candidates to try
        target_size: desired size
        """
        if len(S) >= target_size:
            return S[:]
        
        if not candidates:
            return None
            
        # Try each candidate
        # Optimization: Try smaller numbers first? Or random?
        # The knowledge says "explores the smallest valid numbers first".
        # Let's sort candidates? That's expensive. 
        # Let's assume candidates are passed in a specific order.
        
        for i in range(len(candidates)):
            x = candidates[i]
            new_sums = get_new_sums_indices(x, S)
            if new_sums is not None:
                S.append(x)
                mark_sums(new_sums)
                
                # Recurse
                # Create a new list of candidates excluding x
                # To save memory/time, we can pass indices or just slice
                # Slicing is O(N). 
                # Better: pass start index? No, we skip x.
                # Let's just pass the rest of the list.
                result = backtrack(S, candidates[i+1:], target_size)
                
                if result:
                    return result
                    
                # Undo
                unmark_sums(new_sums)
                S.pop()
                
        return None

    # Strategy 1: Fast Greedy to get a good initial set
    def greedy_build(order):
        reset_sums()
        S = []
        for x in order:
            new_sums = get_new_sums_indices(x, S)
            if new_sums is not None:
                S.append(x)
                mark_sums(new_sums)
        return S

    S1 = greedy_build(range(1, n + 1))
    if len(S1) > len(best_set):
        best_set = S1

    # Strategy 2: Backtracking from a small base
    # Start with small numbers, try to reach size 22, 23, 24
    # Since backtracking can explode, we limit the depth or use a heuristic.
    # Let's try backtracking with a limit on recursion depth or nodes visited.
    
    nodes_visited = 0
    MAX_NODES = 100000 # Adjust based on speed
    
    def backtrack_limited(S, candidates, target_size):
        nonlocal nodes_visited
        if nodes_visited > MAX_NODES:
            return None
        nodes_visited += 1
        
        if len(S) >= target_size:
            return S[:]
        
        if not candidates:
            return None
            
        # Heuristic: Try candidates in ascending order?
        # If we want to maximize cardinality, picking small numbers first
        # leaves room for larger numbers? Or does it block them?
        # Usually small numbers create many small sums.
        # Let's try the given order.
        
        for i in range(len(candidates)):
            x = candidates[i]
            new_sums = get_new_sums_indices(x, S)
            if new_sums is not None:
                S.append(x)
                mark_sums(new_sums)
                
                result = backtrack_limited(S, candidates[i+1:], target_size)
                
                if result:
                    return result
                    
                unmark_sums(new_sums)
                S.pop()
                
        return None

    # Try to find a set of size 22, then 23, etc.
    # Use a subset of candidates to keep backtracking tractable?
    # Or just use all 1..4096? That's a huge branching factor.
    # Greedy gives ~21. Backtracking from empty with 4096 candidates is too big.
    
    # Better: Use the greedy result as a guide.
    # Or use a "beam search" or limited backtracking.
    
    # Let's try randomized greedy with the fast bytearray, many times.
    # This is often effective for B_h sets.
    
    for seed in range(2000):
        if time.time() > deadline:
            break
        random.seed(seed)
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        S = greedy_build(candidates)
        
        if len(S) > len(best_set):
            best_set = S
            if len(best_set) >= 24:
                break

    return best_set
