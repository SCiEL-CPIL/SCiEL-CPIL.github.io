"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    deadline = time.time() + 85.0
    n = 512
    # Precompute 4-AP conflict masks: for each element i (0-indexed), 
    # mask[i] has bit j set if j forms a 4-AP with i as one of the terms
    # Actually, we need: for a set S, adding x creates a 4-AP if there exist
    # a, b, c in S such that {a,b,c,x} form a 4-AP.
    # A 4-AP is {a, a+d, a+2d, a+3d} for d>0.
    # If we add x, it could be any of the 4 positions.
    # Precompute for each x, the set of "partner triples" - but that's complex.
    # Better: precompute for each x, a bitmask of all y such that x and y 
    # are in some 4-AP together (with 2 other elements that must also be present).
    
    # Simpler approach: precompute for each x, the list of (d, position) pairs
    # such that if x is at position p in a 4-AP with difference d, the other
    # 3 elements are determined. We check if those 3 are all in the set.
    
    # For bit-parallel: represent set as a 512-bit integer.
    # For each x, precompute: for each possible 4-AP containing x, store the
    # bitmask of the other 3 elements. Then x conflicts with current set S if
    # any of those bitmasks is a subset of S (i.e., mask & S == mask).
    
    # Precompute conflict data
    # For each x in 1..512, find all 4-APs containing x
    # A 4-AP: {a, a+d, a+2d, a+3d}, d>=1, all in [1,512]
    # x can be at position 0,1,2,3
    
    x_conflicts = [[] for _ in range(n+1)]  # x_conflicts[x] = list of (bitmask of other 3)
    
    for d in range(1, 171):  # max d: a+3d <= 512, a>=1, so 3d <= 511, d<=170
        for a in range(1, 512 - 3*d + 1):
            terms = [a, a+d, a+2*d, a+3*d]
            for pos in range(4):
                x = terms[pos]
                others = [terms[j] for j in range(4) if j != pos]
                mask = 0
                for y in others:
                    mask |= (1 << (y-1))
                x_conflicts[x].append(mask)
    
    # Greedy initial solution
    S = 0  # bitmask
    for x in range(1, n+1):
        bit = 1 << (x-1)
        conflict = False
        for mask in x_conflicts[x]:
            if (S & mask) == mask:
                conflict = True
                break
        if not conflict:
            S |= bit
    
    best_S = S
    best_count = bin(S).count('1')
    
    # Simulated annealing
    if time.time() < deadline - 5.0:
        T = 10.0
        T_min = 0.01
        cooling = 0.9995
        rng = random.Random(42)
        
        while time.time() < deadline - 3.0 and T > T_min:
            # Pick a random move: add or remove a random element
            x = rng.randint(1, n)
            bit = 1 << (x-1)
            
            if S & bit:
                # Try removing x
                new_S = S & ~bit
                delta = -1
                # Always accept removals that don't break validity (they can't)
                # But we might want to add back later
                S = new_S
                if bin(S).count('1') > best_count:
                    best_count = bin(S).count('1')
                    best_S = S
            else:
                # Try adding x
                conflict = False
                for mask in x_conflicts[x]:
                    if (S & mask) == mask:
                        conflict = True
                        break
                if not conflict:
                    S |= bit
                    if bin(S).count('1') > best_count:
                        best_count = bin(S).count('1')
                        best_S = S
                else:
                    # Can't add directly; try a swap: remove a conflicting element and add x
                    # Find which elements conflict with x
                    # For each 4-AP containing x, find elements in S that are in the other 3
                    # We need to remove at least one from each conflicting 4-AP
                    # Simple heuristic: find one element to remove that resolves most conflicts
                    # For now, just skip (or do a random swap)
                    pass
            
            T *= cooling
        
        # Use best_S for final result
        S = best_S
    
    # Convert bitmask to sorted list
    result = [x for x in range(1, n+1) if S & (1 << (x-1))]
    return result

# EVOLVE-BLOCK-END
