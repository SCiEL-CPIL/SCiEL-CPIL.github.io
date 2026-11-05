import sys
import time
import numpy as np
from itertools import combinations_with_replacement

def construct():
    n = 4096
    max_sum = 3 * n
    
    # Precompute all pairs (i, j) with i <= j and their sum
    # This is useful for generating new sums when adding a new element x
    # Actually, let's just generate them on the fly but efficiently.
    
    # We will use a bytearray for used_sums for O(1) lookup and update
    used_sums = bytearray(max_sum + 1)
    
    # The set of chosen elements
    chosen = []
    
    # Global best
    best_set = []
    best_size = 0
    
    # Time limit
    start_time = time.time()
    timeout = 55.0
    
    # We will implement a backtracking search.
    # To make it fast, we need to be smart about which candidates to try.
    # A key insight for B_h sequences: the greedy algorithm (picking the smallest possible next element)
    # is often optimal or near-optimal for maximizing size in [1, N].
    # However, sometimes a slightly larger choice allows more future elements.
    # Backtracking allows us to explore alternatives if the greedy path leads to a dead end earlier.
    
    # But full backtracking over 4096 candidates at each step is too slow.
    # We need a strong heuristic.
    
    # Heuristic: Try candidates in increasing order, but limit the branching factor.
    # Or, use a "lookahead" heuristic: pick the candidate that minimizes the number of future conflicts.
    # This is expensive.
    
    # Alternative: Use a randomized restart approach with a greedy-like backtracking.
    
    # Let's try a depth-first search with a limited width.
    # At each depth, we only try the first K valid candidates (e.g., K=5 or 10).
    # This keeps the search tree manageable.
    
    # Also, we can use a "jump" heuristic: if we are stuck, try a larger number.
    
    # Let's implement a recursive function.
    # State: current list of chosen elements, current used_sums bytearray.
    # We need to efficiently check if a new element x can be added.
    
    def can_add(x, chosen_list):
        """
        Check if x can be added to chosen_list without creating sum collisions.
        Returns a list of new sums if valid, else None.
        """
        new_sums = []
        
        # 1. (x, x, x)
        s = 3 * x
        if used_sums[s]:
            return None
        new_sums.append(s)
        
        # 2. (a, x, x) for a in chosen_list
        for a in chosen_list:
            s = a + 2 * x
            if used_sums[s]:
                return None
            new_sums.append(s)
        
        # 3. (a, b, x) for a <= b in chosen_list
        n_chosen = len(chosen_list)
        for i in range(n_chosen):
            a = chosen_list[i]
            for j in range(i, n_chosen):
                b = chosen_list[j]
                s = a + b + x
                if used_sums[s]:
                    return None
                new_sums.append(s)
        
        # Optimization: If x is sufficiently large, internal collisions are impossible.
        # Collision condition: a + x = b + c (from a+2x = b+c+x)
        # or 2x = a + b (from 3x = a+b+x)
        # If x > max(chosen_list), 2x > 2*max(chosen_list) >= a+b, so 2x != a+b.
        # If x > 2*max(chosen_list), then x > b+c-a for all a,b,c in chosen_list.
        # Proof: max(b+c-a) <= 2*max(chosen_list) - min(chosen_list) < 2*max(chosen_list).
        # So if x > 2 * max(chosen_list), no internal collisions.
        if n_chosen == 0 or x > 2 * chosen_list[-1]:
            return new_sums
        
        # Check for internal collisions among new_sums
        new_sums_set = set()
        for s in new_sums:
            if s in new_sums_set:
                return None
            new_sums_set.add(s)
        
        return new_sums
    
    def backtrack(start_val, depth):
        nonlocal best_set, best_size, start_time
        
        if time.time() - start_time > timeout:
            return
        
        # Update best
        if depth > best_size:
            best_size = depth
            best_set = chosen[:]
        
        # Pruning: If we can't possibly beat the best size, stop.
        # Max possible size is roughly (n/depth)^(1/3) ? Not a simple bound.
        # We can estimate the maximum number of elements we can still add.
        # Each element x adds about O(depth^2) new sums.
        # Total sums available is ~3n.
        # So we can add at most ~3n / (depth^2) more elements?
        # This is a rough upper bound.
        
        # Let's just try to find a better solution.
        
        # Try candidates from start_val to n
        # To limit branching, we only try a few candidates.
        # Let's try the first 10 valid candidates.
        
        tried = 0
        # Heuristic: Try more candidates at shallow depths to explore the space better.
        # At deeper depths, the search tree is narrower, so fewer tries needed.
        if depth < 5:
            max_tries = 50
        elif depth < 10:
            max_tries = 20
        else:
            max_tries = 10
        
        for x in range(start_val, n + 1):
            if time.time() - start_time > timeout:
                return
            
            if tried >= max_tries:
                break
            
            new_sums = can_add(x, chosen)
            if new_sums is not None:
                # Add x
                chosen.append(x)
                for s in new_sums:
                    used_sums[s] = 1
                
                # Recurse
                backtrack(x + 1, depth + 1)
                
                # Undo
                chosen.pop()
                for s in new_sums:
                    used_sums[s] = 0
                
                tried += 1
                # If we found a solution that extends the best, we might want to continue
                # to see if we can get even more, but backtracking handles this.
        
    # Initialize
    backtrack(1, 0)
    
    return best_set
