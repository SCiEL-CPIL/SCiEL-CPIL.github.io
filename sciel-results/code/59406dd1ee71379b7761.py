import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s margin
    
    # Candidates are 2..40
    candidates = list(range(2, 41))
    
    # We want to maximize the size of the set S
    # S is a list of chosen numbers
    # P is the set of all subset products of S
    # We start with S=[], P={1}
    
    best_set = []
    
    # To speed up, we can precompute for each x, its prime factors
    # But we just need to check if x is in P or if x*P intersects P
    
    def is_valid(S, P, x):
        # Check if adding x to S creates a collision
        # New products are {p * x for p in P}
        # Collision if any p * x is in P
        for p in P:
            if p * x in P:
                return False
        return True

    def dfs(S, P, candidates_remaining):
        nonlocal best_set
        if time.time() > deadline:
            return
        
        if len(S) > len(best_set):
            best_set = list(S)
            
        if not candidates_remaining:
            return
            
        # Try to add the next candidate
        # We can iterate through candidates and try adding them
        # To find a large set, we should try adding numbers that are "likely" to be compatible
        # Or just iterate all.
        
        for i in range(len(candidates_remaining)):
            x = candidates_remaining[i]
            
            if is_valid(S, P, x):
                new_P = set(P)
                for p in P:
                    new_P.add(p * x)
                
                dfs(S + [x], new_P, candidates_remaining[i+1:])
                
                # Backtrack
                # No need to explicitly backtrack P if we pass it by value (set copy)
            else:
                # If x is not valid, we can still try other candidates
                # But we must decide if we skip x or not.
                # In a maximal independent set search, we branch: include x or exclude x.
                # However, simple greedy (include if possible, else skip) might not be optimal.
                # Let's try branching:
                # 1. Include x (if valid)
                # 2. Exclude x
                
                # We already tried Include above.
                # Now try Exclude:
                pass # The loop continues to the next candidate, effectively skipping x in the "include" branch.
                # But we need to explicitly branch to "skip x and try others".
                # The current loop structure "for i in ..." tries to include x at position i.
                # If we include x, we recurse on remaining[i+1:].
                # If we don't include x, we should recurse on remaining[i+1:] with SAME S, P.
                
                # Let's restructure to explicit branching.

    # Better structure:
    def dfs2(index, S, P):
        nonlocal best_set
        if time.time() > deadline:
            return
            
        if len(S) > len(best_set):
            best_set = list(S)
            
        if index == len(candidates):
            return
            
        x = candidates[index]
        
        # Option 1: Include x
        if is_valid(S, P, x):
            new_P = set(P)
            for p in P:
                new_P.add(p * x)
            dfs2(index + 1, S + [x], new_P)
            
        # Option 2: Exclude x
        dfs2(index + 1, S, P)

    dfs2(0, [], {1})
    
    # Ensure we return a list of ints
    return [int(x) for x in best_set]
