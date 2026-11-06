"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 512
    # Precompute compatibility matrix
    # compat[i][j] is True if lcm(i, j) <= 512
    # We only need i, j in 1..512
    # To save memory/time, we can just compute lcm on the fly in the inner loops
    # or precompute a list of compatible neighbors for each number.
    
    # Precompute neighbors: neighbors[i] = list of j such that lcm(i,j) <= 512
    neighbors = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            if i == j:
                continue
            # lcm(i, j) = i * j // gcd(i, j)
            g = gcd(i, j)
            if (i * j) // g <= N:
                neighbors[i].append(j)
    
    # Initial seed: Divisors of 360
    # 360 = 2^3 * 3^2 * 5
    seed_num = 360
    S = set()
    for i in range(1, N + 1):
        if seed_num % i == 0:
            S.add(i)
    
    best_S = set(S)
    
    def is_valid_addition(x, current_S):
        # Check if x is compatible with all elements in current_S
        for y in current_S:
            if y not in neighbors[x]:
                return False
        return True
    
    def try_add(S):
        # Try to add any single element
        for x in range(1, N + 1):
            if x in S:
                continue
            if is_valid_addition(x, S):
                S.add(x)
                return True
        return False
    
    def try_swap(S):
        # Try to remove 1 and add 2
        # This increases size by 1
        # We iterate over a in S
        # Then we look for b, c not in S such that (S - {a}) U {b, c} is valid
        # To make it efficient, we can precompute which elements are "blocked" only by a
        # But simple iteration might be fast enough for small sets
        
        # First, identify candidates for b and c
        # They must be compatible with S - {a}
        
        # To speed up, we can just try all pairs (b, c) from the complement
        # Complement size is ~488. Pairs ~ 118k.
        # For each a (24), 118k * 24 ~ 2.8M checks.
        # Each check involves verifying b and c against S-{a} and b-c.
        # This might be slow in Python if done naively.
        
        # Optimization:
        # For a given a, let S' = S - {a}.
        # Valid b must be compatible with all y in S'.
        # Let valid_b = [b for b in range(1, N+1) if b not in S and is_valid_addition(b, S')]
        # Then for each pair b, c in valid_b, check if b and c are compatible.
        
        for a in list(S):
            S_prime = S - {a}
            # Find all b that can be added to S_prime
            valid_b = []
            for b in range(1, N + 1):
                if b in S:
                    continue
                if is_valid_addition(b, S_prime):
                    valid_b.append(b)
            
            # Now find two distinct b, c in valid_b that are compatible with each other
            # We want to maximize the chance of finding a pair, so we can just check all pairs
            # To be faster, we can break early if we find one
            found = False
            for i in range(len(valid_b)):
                b = valid_b[i]
                # Check compatibility with other valid candidates
                for j in range(i + 1, len(valid_b)):
                    c = valid_b[j]
                    if b in neighbors[c]:
                        # Valid swap found
                        S.remove(a)
                        S.add(b)
                        S.add(c)
                        found = True
                        break
                if found:
                    break
            if found:
                return True
        
        return False

    # Main loop
    while time.time() < deadline:
        improved = False
        
        # Try to add
        if try_add(S):
            improved = True
            if len(S) > len(best_S):
                best_S = set(S)
            continue # Restart to allow more additions
        
        # Try to swap
        if try_swap(S):
            improved = True
            if len(S) > len(best_S):
                best_S = set(S)
            continue
        
        # If no improvement, break
        if not improved:
            break
    
    # Return the best set found
    # Ensure it's a list of ints
    return list(best_S)

# EVOLVE-BLOCK-END
