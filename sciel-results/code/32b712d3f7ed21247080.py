import time
import random

def construct():
    """
    Construct a set of distinct integers in 1..256 such that whenever two subsets
    have different cardinalities, their sums must differ.
    """
    start_time = time.time()
    deadline = 55.0  # Safety margin
    
    best_set = []
    
    # Helper to check if x can be added to current state (A, S, AllS)
    def can_add(S, AllS, x):
        # S is list of sets for cardinalities 0..N
        # AllS is union of all S[k]
        n = len(S) - 1
        for k in range(n + 1):
            if not S[k]:
                continue
            for s in S[k]:
                ns = s + x
                if ns in AllS:
                    # The new sum ns must belong to cardinality k+1
                    # If it's already in AllS, it must be in S[k+1]
                    # If S[k+1] doesn't exist or ns is not in it, conflict
                    if k + 1 >= len(S) or ns not in S[k+1]:
                        return False
        return True

    # Helper to add x to the state
    def do_add(A, S, AllS, x):
        n = len(A)
        # Extend S if necessary
        while len(S) <= n + 1:
            S.append(set())
        
        # Update S in reverse order
        for k in range(n + 1, 0, -1):
            new_sums = {s + x for s in S[k-1]}
            S[k].update(new_sums)
            AllS.update(new_sums)
        
        A.append(x)

    # Generate orderings
    base_order = list(range(1, 257))
    orderings = [
        base_order,
        list(reversed(base_order))
    ]
    
    # Add some random permutations
    # We will generate them on the fly to save memory, but we need to ensure 
    # we try a variety.
    
    attempts = 0
    max_attempts = 100 # We will break based on time anyway
    
    for i in range(max_attempts):
        if time.time() - start_time > deadline:
            break
            
        if i == 0:
            order = base_order
        elif i == 1:
            order = list(reversed(base_order))
        else:
            order = base_order.copy()
            random.shuffle(order)
            
        A = []
        S = [set() for _ in range(20)] # Pre-allocate some, will extend
        S[0] = {0}
        AllS = {0}
        
        for x in order:
            if can_add(S, AllS, x):
                do_add(A, S, AllS, x)
                
        if len(A) > len(best_set):
            best_set = list(A)
            
        # If we found a very large set, we might stop early? 
        # Theoretical max is not known, but 20-30 is likely.
        # If we hit 30, we can relax.
        
    return [int(x) for x in best_set]
