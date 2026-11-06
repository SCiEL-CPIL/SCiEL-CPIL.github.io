"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 256
    
    # Helper to check if x can be added to set S without forming a 3-AP
    def can_add(S, x):
        # S is a set
        for a in S:
            # Case 1: x is the largest term. a, b, x => 2b = a+x => b = (a+x)/2
            if (a + x) % 2 == 0:
                b = (a + x) // 2
                if b in S and b != a and b != x:
                    return False
            # Case 2: x is the middle term. a, x, b => 2x = a+b => b = 2x-a
            b = 2 * x - a
            if b in S and b != a and b != x:
                return False
            # Case 3: x is the smallest term. x, a, b => 2a = x+b => b = 2a-x
            b = 2 * a - x
            if b in S and b != a and b != x:
                return False
        return True

    # 1. Greedy Construction
    # We iterate from 1 to n. Adding small numbers is prioritized because they have higher weight.
    S = set()
    for x in range(1, n + 1):
        if can_add(S, x):
            S.add(x)
    
    # 2. Local Search: Swap-based Hill Climbing
    # We try to replace a larger element u with a smaller element v not in S.
    # This increases the sum of reciprocals (1/v > 1/u).
    # We only check if the new set is valid.
    
    # Precompute list of all candidates 1..n
    all_candidates = list(range(1, n + 1))
    
    improved = True
    while improved:
        improved = False
        # Iterate over elements in S to remove
        # We convert to list to avoid modification during iteration, but we break immediately
        S_list = list(S)
        
        for u in S_list:
            # Try to replace u with v where v < u and v not in S
            # Sorting v candidates by value (ascending) ensures we try the best (smallest) v first?
            # Actually, any valid v < u improves the score.
            # To be efficient, we can just iterate v from 1 to u-1.
            for v in range(1, u):
                if v in S:
                    continue
                
                # Check if v can be added to S \ {u}
                # We temporarily remove u from S for the check
                S.remove(u)
                valid = can_add(S, v)
                if valid:
                    # Perform swap
                    S.add(v)
                    improved = True
                    # Break out of inner loop, restart outer loop to re-evaluate
                    break
                else:
                    # Restore u
                    S.add(u)
            
            if improved:
                break
    
    return list(S)
# EVOLVE-BLOCK-END
