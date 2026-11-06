import math
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    n = 24
    # Compute LCM of 1..24
    L = 1
    for i in range(1, n + 1):
        L = math.lcm(L, i)
    
    # Values V[i] = L // i
    V = [0] * (n + 1)
    for i in range(1, n + 1):
        V[i] = L // i

    # Precompute subset sums for small sets? No, dynamic.
    
    def subset_sum_exists(target, values):
        """Check if any subset of values sums to target.
        values is a list of integers.
        Returns True/False.
        """
        if target < 0:
            return False
        if not values:
            return target == 0
        if sum(values) < target:
            return False
        
        # Meet in the middle
        k = len(values)
        if k <= 20:
            half = k // 2
            left = values[:half]
            right = values[half:]
            
            left_sums = set()
            for mask in range(1 << half):
                s = 0
                for i in range(half):
                    if mask & (1 << i):
                        s += left[i]
                left_sums.add(s)
            
            # Check if any right sum completes a left sum to target
            # We can iterate right sums and check if target - s in left_sums
            for mask in range(1 << (k - half)):
                s = 0
                for i in range(k - half):
                    if mask & (1 << i):
                        s += right[i]
                if target - s in left_sums:
                    return True
            return False
        else:
            # Fallback to simple DP or DFS if needed, but k<=24 so 2^12 is fine.
            # Actually 2^20 is ~1M, which might be slow if called many times.
            # But in B&B, we hope to prune early.
            pass
            
    def is_valid_incremental(S_mask, v):
        """
        Check if adding v to S_mask (which is valid) keeps it valid.
        S_mask is an integer bitmask.
        v is an element (1-24).
        """
        # 1. Check if 1/v is sum of subset of S
        # S elements are those bits set in S_mask
        s_elements = [i for i in range(1, n+1) if S_mask & (1 << (i-1))]
        s_values = [V[i] for i in s_elements]
        
        if subset_sum_exists(V[v], s_values):
            return False
        
        # 2. Check for each x in S, if 1/x - 1/v is sum of subset of S \ {x, v}
        # i.e., V[x] - V[v] is sum of subset of S \ {x}
        for x in s_elements:
            target = V[x] - V[v]
            if target <= 0:
                continue
            # Values available: S \ {x}
            avail_values = [V[i] for i in s_elements if i != x]
            if subset_sum_exists(target, avail_values):
                return False
                
        return True

    def is_valid_full(S_mask):
        """Check if S_mask is valid from scratch."""
        s_elements = [i for i in range(1, n+1) if S_mask & (1 << (i-1))]
        if not s_elements:
            return True
        # Just check all pairs? No, need full check.
        # Use incremental logic by building up?
        # Or just check all x in S
        for x in s_elements:
            other_elements = [i for i in s_elements if i != x]
            other_values = [V[i] for i in other_elements]
            if subset_sum_exists(V[x], other_values):
                return False
        return True

    # Phase 1: Greedy to get initial solution
    S_mask = 0
    # Order elements by some heuristic, e.g., descending value (larger i -> smaller V[i]? No, V[i]=L/i, so larger i -> smaller V[i])
    # Smaller V[i] are "smaller" reciprocals. 
    # Let's just iterate 1..24
    for i in range(1, n+1):
        if is_valid_incremental(S_mask, i):
            S_mask |= (1 << (i-1))
            
    best_mask = S_mask
    best_size = bin(S_mask).count('1')
    
    # Phase 2: Branch and Bound
    # We search for a set of size > best_size
    # Candidates: all elements
    # We can use a recursive DFS
    
    def dfs(current_mask, candidates_mask, current_size):
        nonlocal best_mask, best_size, deadline, start_time
        
        if time.time() > deadline:
            return
        
        # Pruning
        if current_size + bin(candidates_mask).count('1') <= best_size:
            return
            
        if candidates_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return
        
        # Pick a candidate
        # Heuristic: pick the one with smallest V[i]? Or largest?
        # Let's pick the lowest set bit for determinism, or random?
        # Lowest set bit
        # v = (candidates_mask & -candidates_mask).bit_length()
        # Actually, let's iterate through all candidates to find the "best" one to branch on?
        # Standard B&B picks one. Let's pick the first available one.
        
        # Find first set bit
        lsb = candidates_mask & -candidates_mask
        v_idx = lsb.bit_length() # 1-based index
        v = v_idx
        
        # Branch 1: Include v
        if is_valid_incremental(current_mask, v):
            new_mask = current_mask | (1 << (v-1))
            new_candidates = candidates_mask & ~(1 << (v-1))
            dfs(new_mask, new_candidates, current_size + 1)
            
        # Branch 2: Exclude v
        new_candidates = candidates_mask & ~(1 << (v-1))
        dfs(current_mask, new_candidates, current_size)
        
    # Start B&B
    # To make it faster, we can order candidates.
    # But the simple "first bit" might be slow if it doesn't prune well.
    # Let's try to improve the initial greedy to be better.
    # Greedy: Add element that minimizes conflicts?
    
    # Let's run a few randomized restarts of the greedy + local search if B&B is too slow?
    # Or just run B&B.
    
    # Optimization: The is_valid_incremental calls subset_sum_exists which is O(2^(k/2)).
    # In the worst case, this is called many times.
    # We can memoize subset_sum_exists(target, tuple(values)).
    # But values depend on the mask.
    
    # Let's add a time check inside the loop more frequently.
    
    # Run B&B
    all_candidates = (1 << n) - 1
    dfs(0, all_candidates, 0)
    
    # If B&B didn't find anything better, return greedy
    # Convert best_mask to list
    result = [i for i in range(1, n+1) if best_mask & (1 << (i-1))]
    
    return sorted(result)
