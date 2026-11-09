"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict

def construct():
    n = 512
    
    # Greedy construction with look-ahead to maximize 4-AP-free set size
    # Use a more sophisticated greedy that considers future constraints
    
    # First, let's try a base-3 based construction which is known to be good
    # For 4-AP-free sets, the Salem-Spencer construction (base 3 digits 0,1,2)
    # gives sets of size ~n^(log_3(2)) which is about n^0.63
    
    # Actually, for 4-term AP-free sets, we can use a variant.
    # The classical result is that the largest 4-AP-free subset of [1,n] 
    # has size approximately n / (log n)^(1/3) * (log log n)^(1/3) or similar.
    
    # Let's use a greedy algorithm with backtracking for better results.
    
    # Strategy: Greedy with conflict checking
    # For each number from 1 to n, decide whether to include it
    # A number x can be included if it doesn't form a 4-AP with any 3 already selected numbers
    
    # A 4-AP: a, a+d, a+2d, a+3d
    # When adding x, check if there exist a, d such that:
    #   x = a, a+d, a+2d, or a+3d where the other three are in the set
    
    # More precisely, x participates in a 4-AP if there exist distinct a, b, c, d 
    # forming a 4-AP with x being one of them.
    
    # Let's use a bitset for fast membership checks
    selected = np.zeros(n + 1, dtype=bool)
    result = []
    
    for x in range(1, n + 1):
        # Check if adding x creates a 4-AP
        can_add = True
        
        # Case 1: x is the first term: x, x+d, x+2d, x+3d
        # Need x+d, x+2d, x+3d all in selected
        for d in range(1, (n - x) // 3 + 1):
            if selected[x + d] and selected[x + 2*d] and selected[x + 3*d]:
                can_add = False
                break
        
        if not can_add:
            continue
        
        # Case 2: x is the second term: x-d, x, x+d, x+2d
        # Need x-d, x+d, x+2d all in selected
        for d in range(1, min(x - 1, (n - x) // 2) + 1):
            if selected[x - d] and selected[x + d] and selected[x + 2*d]:
                can_add = False
                break
        
        if not can_add:
            continue
        
        # Case 3: x is the third term: x-2d, x-d, x, x+d
        # Need x-2d, x-d, x+d all in selected
        for d in range(1, min(x // 2, n - x) + 1):
            if selected[x - 2*d] and selected[x - d] and selected[x + d]:
                can_add = False
                break
        
        if not can_add:
            continue
        
        # Case 4: x is the fourth term: x-3d, x-2d, x-d, x
        # Need x-3d, x-2d, x-d all in selected
        for d in range(1, x // 3 + 1):
            if selected[x - 3*d] and selected[x - 2*d] and selected[x - d]:
                can_add = False
                break
        
        if can_add:
            selected[x] = True
            result.append(x)
    
    # Now try to improve by local search: try to swap elements
    # This is a simple improvement step
    
    # Actually, let's try a different approach: use the known optimal constructions
    # For 4-AP-free sets, we can use a greedy with a different ordering
    
    # Let's try: sort candidates by how many conflicts they create, pick least conflicting
    # But for simplicity, the greedy above should give a decent result.
    
    # Let's also try a base-3 inspired construction
    # Numbers whose base-3 representation uses only digits 0 and 1 (no 2s)
    # This gives a 3-AP-free set, which is also 4-AP-free
    # Size is about n^(log_3(2)) ≈ 512^0.6309 ≈ 512^0.63
    
    # 512 = 2^9, so 512^0.63 = 2^(9*0.63) = 2^5.67 ≈ 50.7
    # That's way too small. The greedy should do much better.
    
    # Let's verify our greedy result and return it
    return result

# EVOLVE-BLOCK-END
