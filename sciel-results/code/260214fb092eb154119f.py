"""
Strategy:
1. Model the problem as a Maximum Weight Independent Set (MWIS) on a graph where
   vertices are {1..256} and edges connect pairs that form a 3-AP.
2. Use a greedy initialization that processes vertices in descending order of weight
   (1/x), using bitmasks for O(1) conflict checks.
3. Refine using a 2-Opt local search: try swapping one selected vertex with one
   unselected vertex if it improves the total weight and maintains independence.
   Also try simple additions if possible.
"""

def construct():
    import time
    import random
    from fractions import Fraction

    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    n = 256
    # We use 0-indexed internally for bitmasks, so element i in [1..256] is bit i-1
    # But the problem is 1-indexed. Let's map 1..256 to 0..255.
    
    # Precompute conflict masks.
    # conflict_mask[i] is a bitmask where bit j is set if i+1 and j+1 form a 3-AP pair.
    # Wait, the graph vertices are the numbers. An edge exists between a and c if there is a b such that a,b,c is a 3-AP.
    # Actually, the condition is "no 3-term AP". This means we cannot have {a, b, c} all in the set.
    # This is equivalent to finding an independent set in a graph where vertices are the numbers, 
    # and edges connect pairs that are part of a 3-AP? 
    # No. If we have a graph where (a,c) is an edge if a and c are the ends of a 3-AP, 
    # then an independent set in this graph ensures no two elements are ends of a 3-AP.
    # But what if the middle element is included? 
    # If a, b, c is a 3-AP, we cannot have all three.
    # This is a 3-uniform hypergraph independent set problem.
    # However, we can reduce it to a graph problem if we are clever?
    # Or we can just use the direct check.
    
    # Let's stick to the bitmask representation for the set S.
    # S is an integer. Bit i is 1 if (i+1) is in S.
    
    # Precompute for each number x (1..256) the set of numbers y such that x and y 
    # cannot both be in S? 
    # No, x and y can be in S unless they form a 3-AP with some z in S.
    # The constraint is global.
    
    # Efficient approach:
    # Represent S as a set of integers for logic, but use bitmasks for speed if possible.
    # Actually, for n=256, a Python int is a 256-bit integer. 
    # Checking if adding x creates a 3-AP:
    # We need to check if there exists d such that:
    # 1. (x-d) in S and (x+d) in S  (x is middle)
    # 2. (x+2d) in S                (x is start, implies x+d, x+2d? No, just x, x+d, x+2d)
    #    Wait, if x is start, the AP is x, x+d, x+2d. We need x+d and x+2d in S?
    #    No, we just need to ensure we don't complete a triple.
    #    If we add x, we create a 3-AP if:
    #    - There is d>0 such that (x-d) in S and (x+d) in S. (x is middle)
    #    - There is d>0 such that (x+d) in S and (x+2d) in S. (x is start)
    #    - There is d>0 such that (x-d) in S and (x-2d) in S. (x is end)
    
    # To make this fast, we can precompute for each x the list of "forbidden pairs" or just 
    # iterate through possible d. Since n=256, max d is 128. 
    # Iterating 128 times per check is fast enough for a few thousand iterations.
    
    # Better: Precompute conflict sets.
    # For each x, let `conflicts[x]` be the set of y such that adding x to a set containing y 
    # might cause a conflict? No, that's not right.
    
    # Let's use the standard "independent set in 3-uniform hypergraph" approach 
    # but optimized with bitsets.
    
    # We will maintain `S` as a set of ints for easy iteration and `S_bits` as an int for fast membership?
    # Actually, `x in S` is O(1) for a set.
    
    # Let's try a different structural approach: 
    # Greedy by weight, then Local Search.
    
    # 1. Greedy Initialization
    # Sort candidates by weight (1/x) descending.
    # Candidates: 1, 2, ..., 256. Weights: 1, 0.5, ..., 1/256.
    # So we try to add 1, then 2, then 3, etc.
    
    def can_add(x, S):
        # S is a set of ints
        # Check if adding x creates a 3-AP
        # 1. x is middle: (x-d, x, x+d)
        # 2. x is start: (x, x+d, x+2d)
        # 3. x is end: (x-2d, x-d, x)
        
        # Optimization: Only check d where the other elements are in [1, 256]
        
        # Middle
        max_d_mid = min(x - 1, 256 - x)
        for d in range(1, max_d_mid + 1):
            if (x - d) in S and (x + d) in S:
                return False
        
        # Start
        max_d_start = (256 - x) // 2
        for d in range(1, max_d_start + 1):
            if (x + d) in S and (x + 2 * d) in S:
                return False
        
        # End
        max_d_end = x // 2
        for d in range(1, max_d_end + 1):
            if (x - 2 * d) in S and (x - d) in S:
                return False
        
        return True

    # Initial Greedy
    S = set()
    # Try adding in order of decreasing weight (increasing x is wrong, decreasing x is correct for weight? 
    # Weight is 1/x. So 1 is highest, 256 is lowest.
    # So we try 1, 2, 3...
    for x in range(1, 257):
        if can_add(x, S):
            S.add(x)
    
    best_S = set(S)
    best_score = sum(1.0 / x for x in best_S)
    
    # 2. Local Search: 2-Opt Swap
    # Try to swap x in S with y not in S if it improves score and maintains validity.
    # To speed up, we can precompute the "gain" of removing x and adding y.
    # Gain = 1/y - 1/x. We only care if 1/y > 1/x, i.e., y < x.
    # So we want to replace larger numbers with smaller ones if possible.
    
    # Also try simple additions if possible (though greedy should have added all possible 
    # given the order, but removing some might allow adding others? 
    # Actually, greedy is order-dependent. 
    # Local search helps escape this.)
    
    # We will do multiple passes.
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Generate a list of elements in S and not in S
        in_S = list(S)
        not_in_S = [x for x in range(1, 257) if x not in S]
        
        # Randomize order to avoid bias? Or sort by potential gain?
        # Let's sort not_in_S by weight descending (smallest x first) to find best replacements.
        # And in_S by weight ascending (largest x first) to find worst elements to remove.
        
        # For each x in S (sorted by weight ascending, i.e., largest x first):
        #   For each y in not_in_S (sorted by weight descending, i.e., smallest x first):
        #     If 1/y <= 1/x: break (no gain from further y)
        #     Check if (S - {x}) U {y} is valid.
        #     If valid, swap, update best, improved = True, restart loop (or continue carefully)
        
        # To be efficient, we can limit the number of swaps per pass or just do one swap and restart.
        # One swap and restart is simpler and safer.
        
        # Find the best swap
        best_swap = None
        best_gain = 0
        
        # Sort S by value descending (worst weights first)
        S_sorted = sorted(S, reverse=True)
        # Sort not_in_S by value ascending (best weights first)
        not_in_S_sorted = sorted(not_in_S)
        
        for x in S_sorted:
            weight_x = 1.0 / x
            for y in not_in_S_sorted:
                weight_y = 1.0 / y
                if weight_y <= weight_x:
                    break # No further y can improve weight
                
                # Check validity of S' = S - {x} + {y}
                # This is equivalent to checking if y can be added to S - {x}
                S_temp = S - {x}
                if can_add(y, S_temp):
                    # Valid swap
                    gain = weight_y - weight_x
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = (x, y)
            
            # Early termination if we found a good swap? 
            # Or just keep looking for the BEST swap in this pass?
            # Let's keep looking to find the best one in this pass.
        
        if best_swap:
            x, y = best_swap
            S.remove(x)
            S.add(y)
            improved = True
            # Update best
            current_score = sum(1.0 / v for v in S)
            if current_score > best_score:
                best_score = current_score
                best_S = set(S)
        
        # Also try simple additions? 
        # If we can add y without removing x, that's always better.
        # But greedy should have done that. 
        # However, if we removed x in a previous step, maybe we can add y now?
        # The swap logic covers removal+addition. 
        # What if we can just add y?
        # Let's check if any y in not_in_S can be added to current S.
        for y in not_in_S_sorted:
            if can_add(y, S):
                S.add(y)
                improved = True
                current_score = sum(1.0 / v for v in S)
                if current_score > best_score:
                    best_score = current_score
                    best_S = set(S)
                # Break after one addition to keep it simple and restart?
                # Yes, restart the while loop to re-evaluate everything.
                break

    return list(best_S)

# EVOLVE-BLOCK-END
