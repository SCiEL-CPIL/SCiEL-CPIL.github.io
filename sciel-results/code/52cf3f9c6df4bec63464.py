import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s to be safe
    
    required = [1, 2, 5, 11]
    required_set = set(required)
    best_set = set(required)
    best_size = len(best_set)
    
    def get_sums(S):
        sums = set()
        S_list = sorted(S)
        for i in range(len(S_list)):
            for j in range(i, len(S_list)):
                sums.add(S_list[i] + S_list[j])
        return sums

    def is_valid_addition(x, S, sums):
        for y in S:
            if (x + y) in sums:
                return False
        if (2 * x) in sums:
            return False
        return True

    def add_to_sums(x, S, sums):
        for y in S:
            sums.add(x + y)
        sums.add(2 * x)

    def backtrack(S, sums, depth, deadline):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
            
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
        # If we have reached a reasonable size, we can try to stop early?
        # No, we want max size.
        
        # Generate candidates
        candidates = []
        for x in range(1, 285):
            if x not in S:
                if is_valid_addition(x, S, sums):
                    candidates.append(x)
                    
        if not candidates:
            return
            
        # Heuristic: Least Constraining Value (LCV)
        # For each candidate x, count how many other candidates y it blocks.
        # We want to pick x that blocks the fewest y's.
        
        # Optimization: If candidates list is small, just try them.
        # If large, compute LCV score.
        
        # To speed up, we can sample if the list is huge, but 284 is small.
        
        # Calculate LCV score
        # score[x] = number of candidates y (y != x) such that is_valid_addition(y, S + {x}, sums + sums(x)) is False
        # This is expensive: O(N^2 * |S|). 284^2 * 20 ~ 1.6M ops. Might be too slow for every node.
        
        # Alternative Heuristic: Randomize order but bias towards "spread out" numbers?
        # Or just random order with backtracking.
        
        # Let's try a simpler heuristic first: Random order with backtracking.
        # If that's too slow, we can add LCV.
        
        random.shuffle(candidates)
        
        # Optional: Try to limit branching factor if depth is shallow?
        # No, backtracking needs to explore.
        
        for x in candidates:
            if time.time() > deadline:
                return
                
            S.add(x)
            add_to_sums(x, S, sums)
            
            # Recurse
            backtrack(S, sums, depth + 1, deadline)
            
            # Backtrack
            S.remove(x)
            # Remove sums added by x
            for y in S:
                sums.discard(x + y)
            sums.discard(2 * x)
            
            # If we found a solution of size 20, we can stop?
            # The max is 20. If we find 20, we can return immediately.
            if best_size == 20:
                return

    # Run multiple restarts with different random seeds/orders
    # Since backtracking is deterministic if order is fixed, we randomize the order at each step.
    
    # We might need to be smarter than pure random backtracking.
    # Let's add a "greedy" initialization to get a good starting point, 
    # then use backtracking to improve?
    # Or just run backtracking from the required set.
    
    # To make it more effective, we can use a "beam" of backtracking searches?
    # Or just run it once with randomization.
    
    # Let's try a hybrid:
    # 1. Greedy fill to get a set of size ~15-18.
    # 2. Use local search (1-swap, 2-swap) to improve.
    # 3. If stuck, use backtracking on the "gaps" or restart.
    
    # Actually, pure backtracking with good pruning is often best for exact optimization.
    # The "Least Constraining Value" heuristic is key.
    # Let's implement LCV but optimize it.
    
    def backtrack_lcv(S, sums, deadline):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
            
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
        if best_size == 20:
            return
            
        candidates = []
        for x in range(1, 285):
            if x not in S:
                if is_valid_addition(x, S, sums):
                    candidates.append(x)
                    
        if not candidates:
            return
            
        # If few candidates, just try them
        if len(candidates) < 50:
            random.shuffle(candidates)
            for x in candidates:
                if time.time() > deadline:
                    return
                S.add(x)
                add_to_sums(x, S, sums)
                backtrack_lcv(S, sums, deadline)
                S.remove(x)
                for y in S:
                    sums.discard(x + y)
                sums.discard(2 * x)
                if best_size == 20:
                    return
            return
            
        # LCV Heuristic
        # To avoid O(N^2) cost, we can sample a subset of candidates to estimate LCV?
        # Or just use a simpler heuristic: Sort by value?
        # Let's try sorting by value (ascending) first, as smaller numbers might constrain more?
        # Or descending?
        
        # Let's try a mix: Randomize, but keep the top 10% by a quick score.
        # Quick score: How many sums does x create that are "small" (likely to collide)?
        
        # Actually, let's just use Random Backtracking with a limit on the number of branches explored per node.
        # If we explore too many, we might timeout.
        
        # Strategy:
        # 1. Generate candidates.
        # 2. Shuffle.
        # 3. Try the first K candidates (K=10 or 20).
        # 4. If none lead to improvement, try the next K.
        # This is "Branch and Bound" with limited branching.
        
        K = 10
        random.shuffle(candidates)
        
        for i in range(0, len(candidates), K):
            if time.time() > deadline:
                return
            batch = candidates[i:i+K]
            for x in batch:
                if time.time() > deadline:
                    return
                S.add(x)
                add_to_sums(x, S, sums)
                backtrack_lcv(S, sums, deadline)
                S.remove(x)
                for y in S:
                    sums.discard(x + y)
                sums.discard(2 * x)
                if best_size == 20:
                    return

    # Let's run this backtracking
    S = set(required)
    sums = get_sums(S)
    
    # To increase chances, we can run this multiple times with different random states
    # But the backtracking itself is randomized.
    
    # We also need to handle the case where backtracking is too slow.
    # Let's add a timeout check inside the loop.
    
    backtrack_lcv(S, sums, deadline)
    
    # If we didn't find 20, try local search on the best found set?
    # The backtracking should find a good set.
    
    return list(best_set)

# Let's refine the backtracking to be more efficient.
# The LCV calculation is expensive.
# A better heuristic for Golomb rulers:
# Try to place numbers that are "far apart" from existing numbers?
# Or numbers that create large sums?

# Let's stick to the randomized backtracking with limited branching.
# It is robust.
