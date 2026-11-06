"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    N = 512
    # We want largest subset S of [1,512] such that lcm(a,b) <= 512 for all a,b in S
    
    # Precompute compatibility: compat[i] is True if lcm(i,j) <= N for all j in candidate set
    # First, build the full compatibility matrix
    # Two numbers a,b are compatible if lcm(a,b) = a*b//gcd(a,b) <= N
    
    # Strategy: 
    # 1. Start with a good initial set (all numbers 1..22, plus some extras)
    # 2. Try to add numbers greedily
    # 3. Use local search to improve
    
    # Precompute lcm compatibility for all pairs
    # compat[a][b] = (a*b)//gcd(a,b) <= N
    # To save memory, use bitsets
    
    # For each number a in 1..N, compute which numbers b are compatible with a
    compat = [0] * (N + 1)  # compat[a] is a bitmask of b's compatible with a
    
    for a in range(1, N + 1):
        bits = 0
        for b in range(1, N + 1):
            g = gcd(a, b)
            if (a * b) // g <= N:
                bits |= (1 << b)
        compat[a] = bits
    
    # Check deadline
    if time.time() > deadline:
        return list(range(1, 23))
    
    # Initial set: all numbers 1..22 (guaranteed valid)
    # Plus try to add more
    initial_set = set(range(1, 23))
    
    # Try adding numbers 23..512 greedily
    # A number x can be added if it's compatible with all current members
    for x in range(23, N + 1):
        if time.time() > deadline:
            break
        can_add = True
        for y in initial_set:
            if not (compat[x] >> y) & 1:
                can_add = False
                break
        if can_add:
            initial_set.add(x)
    
    current_set = frozenset(initial_set)
    best_set = set(initial_set)
    best_size = len(best_set)
    
    # Local search: try swapping elements to increase size
    # Or: try adding elements, removing conflicting ones
    # Better approach: use a greedy algorithm on the compatibility graph
    
    # Let's try a different approach: 
    # Sort numbers by "compatibility score" (number of compatible partners)
    # and greedily build a clique
    
    # Actually, let's use the fact that the problem has structure.
    # Key insight: if a | b, then lcm(a,b) = b. So if b <= N, then a and b are always compatible.
    # This means we can include chains of divisors.
    
    # Let's try: include all numbers, then remove minimum set to make it valid
    # This is a minimum vertex deletion problem, which is hard.
    
    # Better: greedy with backtracking for small improvements
    # Try to add each number not in the set, and if it conflicts, try to remove conflicting numbers
    # if the net gain is positive
    
    def is_valid(s):
        """Check if set s is valid (all pairs have lcm <= N)"""
        lst = list(s)
        for i in range(len(lst)):
            for j in range(i+1, len(lst)):
                a, b = lst[i], lst[j]
                g = gcd(a, b)
                if (a * b) // g > N:
                    return False
        return True
    
    # Greedy: try to add numbers, if conflict, try removing one conflicting number
    # if it allows adding the new one and doesn't reduce total size
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_list = list(current_set)
        
        # Try adding each number not in current_set
        for x in range(1, N + 1):
            if x in current_set:
                continue
            if time.time() > deadline:
                break
            
            # Find conflicts: numbers in current_set that are not compatible with x
            conflicts = [y for y in current_list if not (compat[x] >> y) & 1]
            
            # If no conflicts, just add x
            if not conflicts:
                new_set = current_set | {x}
                if len(new_set) > best_size:
                    best_set = set(new_set)
                    best_size = len(best_set)
                    current_set = new_set
                    improved = True
                continue
            
            # If there are conflicts, try removing one conflict and adding x
            # Net gain: +1 (add x) - 1 (remove y) = 0, so only useful if it enables further additions
            # For now, try removing each conflict and see if we can then add x
            for y in conflicts:
                if time.time() > deadline:
                    break
                new_set = (current_set - {y}) | {x}
                # Check if new_set is valid
                valid = True
                for a in new_set:
                    for b in new_set:
                        if a >= b:
                            continue
                        g = gcd(a, b)
                        if (a * b) // g > N:
                            valid = False
                            break
                    if not valid:
                        break
                if valid and len(new_set) > best_size:
                    best_set = set(new_set)
                    best_size = len(best_set)
                    current_set = new_set
                    improved = True
                    break
    
    # Final check: verify best_set is valid
    if not is_valid(best_set):
        # Fallback to seed
        return list(range(1, 23))
    
    return list(best_set)
# EVOLVE-BLOCK-END
