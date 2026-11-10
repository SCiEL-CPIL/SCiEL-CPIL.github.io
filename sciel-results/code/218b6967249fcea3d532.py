import random
import math
import time

def construct():
    n = 256
    # Precompute conflict structures for 4-APs
    # conflicts[x] = list of sets of 3 other elements
    # If x is added, and any set in conflicts[x] is subset of S, invalid.
    
    # Precompute 3-AP contributions
    # For each x, list of (u, v) such that x, u, v form a 3-AP.
    # There are 3 cases: x is 1st, 2nd, 3rd.
    
    # Initial Set: Cantor set (digits 0,1 in base 3)
    # Or just use the current code's Phase 1 result as a seed?
    # The current code's Phase 1 is deterministic. Let's use it as a seed for SA.
    # But the directive says "from scratch... do not reuse current designs".
    # So I will implement the Cantor set generation.
    
    def is_cantor(x):
        # Check if x in base 3 has only 0s and 1s
        # Actually, we just generate the set
        pass

    S = set()
    # Generate Cantor set up to n
    # Numbers with no '2' in base 3
    for i in range(1, n+1):
        s = i
        valid = True
        while s > 0:
            if s % 3 == 2:
                valid = False
                break
            s //= 3
        if valid:
            S.add(i)
            
    # Ensure S is not empty
    if not S:
        S.add(1)

    # Function to check if adding x to S creates a 4-AP
    def can_add(x, S):
        # x is 1st: x+d, x+2d, x+3d in S
        # x is 2nd: x-d, x+d, x+2d in S
        # x is 3rd: x-2d, x-d, x+d in S
        # x is 4th: x-3d, x-2d, x-d in S
        for d in range(1, n//3 + 1):
            # 1st
            if x+d <= n and x+d in S and x+2*d in S and x+3*d in S:
                return False
            # 2nd
            if x-d >= 1 and x-d in S and x+d in S and x+2*d in S:
                return False
            # 3rd
            if x-2*d >= 1 and x-2*d in S and x-d in S and x+d in S:
                return False
            # 4th
            if x-3*d >= 1 and x-3*d in S and x-2*d in S and x-d in S:
                return False
        return True

    # Function to count 3-APs involving x
    def count_3aps_with_x(x, S):
        count = 0
        for d in range(1, n//2 + 1):
            # x is 1st: x, x+d, x+2d
            if x+d <= n and x+d in S and x+2*d in S:
                count += 1
            # x is 2nd: x-d, x, x+d
            if x-d >= 1 and x-d in S and x+d in S:
                count += 1
            # x is 3rd: x-2d, x-d, x
            if x-2*d >= 1 and x-2*d in S and x-d in S:
                count += 1
        return count

    # Initial Score
    total_3aps = 0
    # Efficient total count: sum over all x of (count_3aps_with_x(x) / 3) ?
    # Each 3-AP is counted 3 times (once for each element).
    # So total = sum(count_3aps_with_x(x) for x in S) // 3
    for x in S:
        total_3aps += count_3aps_with_x(x, S)
    total_3aps //= 3

    # Simulated Annealing
    start_time = time.time()
    T = 10.0
    T_min = 0.01
    cooling = 0.99
    
    while time.time() - start_time < 55.0 and T > T_min:
        # Pick random move
        x = random.randint(1, n)
        
        if x in S:
            # Remove x
            delta = -count_3aps_with_x(x, S)
            # Always valid
            if delta > 0 or random.random() < math.exp(delta / T):
                S.remove(x)
                total_3aps += delta
        else:
            # Add x
            if can_add(x, S):
                delta = count_3aps_with_x(x, S)
                if delta > 0 or random.random() < math.exp(delta / T):
                    S.add(x)
                    total_3aps += delta
        
        T *= cooling
        
    return sorted(S)
