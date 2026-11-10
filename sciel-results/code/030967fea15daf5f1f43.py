"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    deadline = time.time() + 85
    N = 256
    
    # Precompute conflict matrix
    conflict = [[False] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) % (i + j) == 0:
                conflict[i][j] = True
                conflict[j][i] = True
    
    def can_add(x, S):
        for y in S:
            if conflict[x][y]:
                return False
        return True
    
    # Start with all odd numbers (mutually compatible)
    odds = list(range(1, N + 1, 2))
    S = set(odds)
    
    # Try to add even numbers greedily
    evens = list(range(2, N + 1, 2))
    for e in evens:
        if can_add(e, S):
            S.add(e)
    
    S_list = list(S)
    
    # Local improvement: try to swap elements to increase size
    # Strategy: try removing one element and adding multiple others
    improved = True
    while improved and time.time() < deadline:
        improved = False
        remaining = set(range(1, N + 1)) - S
        
        # Try adding any remaining element that fits
        for x in list(remaining):
            if can_add(x, S):
                S.add(x)
                remaining.remove(x)
                improved = True
        
        # Try 1-for-2 swaps: remove one, add two
        if not improved and time.time() < deadline:
            for x in list(S):
                S.remove(x)
                added = []
                for y in list(remaining):
                    if can_add(y, S):
                        S.add(y)
                        added.append(y)
                        remaining.remove(y)
                if len(added) >= 2:
                    improved = True
                else:
                    # Revert
                    for y in added:
                        S.remove(y)
                        remaining.add(y)
                    S.add(x)
    
    return list(S)
# EVOLVE-BLOCK-END
