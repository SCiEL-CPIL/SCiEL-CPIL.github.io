"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import itertools
from math import gcd

P = {'n': 256, 'maximum_size': 30}

def construct():
    # Deterministic greedy construction with divisibility-aware candidate selection.
    # We build the set S incrementally.
    # To ensure distinct subset sums, we enforce a_i > sum(S_prev).
    # We pick the smallest valid a_i to maximize the number of elements.
    
    S = []
    sums = {0}
    
    for i in range(30):
        found = False
        # Next element must be > sum(S) to ensure distinct subset sums
        # If S is empty, min_val is 1
        min_val = sum(S) + 1 if S else 1
        
        # Try candidates from min_val to 256
        for cand in range(min_val, 257):
            # Generate new subset sums
            # New sums are {cand + s for s in sums}
            new_sums = set()
            for s in sums:
                new_sums.add(cand + s)
            
            valid = True
            
            # Check 1: No old sum divides a new sum
            # Note: 0 is in sums, but 0 divides nothing (or rather, we skip 0 as divisor)
            for s_old in sums:
                if s_old == 0:
                    continue
                for s_new in new_sums:
                    if s_new % s_old == 0:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Check 2: No new sum divides an old sum
                for s_new in new_sums:
                    for s_old in sums:
                        if s_old == 0:
                            continue
                        if s_old % s_new == 0:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                # Check 3: No two new sums divide each other
                nl = list(new_sums)
                nl.sort()
                for i_n in range(len(nl)):
                    for j_n in range(i_n + 1, len(nl)):
                        if nl[j_n] % nl[i_n] == 0:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                S.append(cand)
                sums = sums.union(new_sums)
                found = True
                break
        
        if not found:
            break
    
    # If the greedy approach yields an empty set (shouldn't happen), fallback
    if not S:
        S = [31, 32, 34]
        
    return list(S)

# EVOLVE-BLOCK-END
