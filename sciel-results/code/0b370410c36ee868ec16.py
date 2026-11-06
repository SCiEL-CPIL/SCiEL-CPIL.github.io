import random
from math import gcd
from itertools import combinations

def lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    # Precompute LCMs for all pairs in [1, 128]
    N = 128
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            val = lcm(i, j)
            lcm_table[i][j] = val
            lcm_table[j][i] = val

    # Helper to check if adding x to S creates a forbidden triple
    def is_valid_addition(S, x):
        for a, b in combinations(S, 2):
            if lcm_table[a][b] == lcm_table[a][x] == lcm_table[b][x]:
                return False
        return True

    # Generate prime powers <= 128
    prime_powers = set()
    for p in range(2, N + 1):
        # Check if p is prime
        is_prime = True
        for d in range(2, int(p**0.5) + 1):
            if p % d == 0:
                is_prime = False
                break
        if is_prime:
            val = p
            while val <= N:
                prime_powers.add(val)
                val *= p
    prime_powers.add(1)
    
    base_set = list(prime_powers)
    all_numbers = set(range(1, N + 1))
    
    best_set = set(base_set)
    best_size = len(best_set)
    
    # Try multiple random orders for adding the remaining numbers
    remaining = list(all_numbers - set(base_set))
    
    # Run several random restarts
    for _ in range(50):
        random.shuffle(remaining)
        current_set = set(base_set)
        
        for x in remaining:
            if is_valid_addition(current_set, x):
                current_set.add(x)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
    return list(best_set)
