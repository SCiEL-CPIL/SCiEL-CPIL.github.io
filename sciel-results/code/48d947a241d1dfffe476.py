import random
from math import lcm
from itertools import combinations

def construct():
    max_n = 128
    # Precompute LCM table
    lcm_table = [[0]*(max_n+1) for _ in range(max_n+1)]
    for i in range(1, max_n+1):
        for j in range(i, max_n+1):
            lcm_table[i][j] = lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
            
    def can_add(current_set, x):
        # Check if adding x to current_set creates a forbidden triple
        # A forbidden triple is {a, b, x} such that lcm(a,b) == lcm(a,x) == lcm(b,x)
        # We only need to check pairs (a, b) in current_set
        for i, a in enumerate(current_set):
            for b in current_set[i+1:]:
                if lcm_table[a][b] == lcm_table[a][x] == lcm_table[b][x]:
                    return False
        return True

    def local_search(initial_set, max_iterations=5000):
        current_set = set(initial_set)
        # First, ensure the initial set is valid (remove conflicts if any)
        # Since we start with valid or empty, this might be skipped, but good for safety
        valid = True
        while valid:
            valid = False
            for a, b, c in combinations(list(current_set), 3):
                if lcm_table[a][b] == lcm_table[b][c] == lcm_table[a][c]:
                    current_set.remove(c)
                    valid = True
        
        best_local = list(current_set)
        
        for _ in range(max_iterations):
            # Try to add any element from 1..128 not in current_set
            added = False
            candidates = [x for x in range(1, max_n+1) if x not in current_set]
            random.shuffle(candidates)
            for x in candidates:
                if can_add(current_set, x):
                    current_set.add(x)
                    added = True
                    break
            
            if not added:
                # Local maximum reached. Perform a kick.
                # Randomly remove 1 or 2 elements to allow new additions
                if len(current_set) > 0:
                    # Remove 1 random element
                    to_remove = random.choice(list(current_set))
                    current_set.remove(to_remove)
                else:
                    # If empty, add a random one
                    current_set.add(random.randint(1, max_n))
            
            if len(current_set) > len(best_local):
                best_local = list(current_set)
                
        return best_local

    best_set = []
    
    # Run multiple restarts with different initializations
    # Strategy 1: Start with empty
    for i in range(20):
        s = local_search([], 2000)
        if len(s) > len(best_set):
            best_set = s
            
    # Strategy 2: Start with "safe" numbers (primes, prime powers)
    # Primes are pairwise coprime, so any subset of primes is valid.
    # Let's start with all primes and try to add composites.
    primes = [n for n in range(2, max_n+1) if all(n % i != 0 for i in range(2, int(n**0.5)+1))]
    for i in range(20):
        # Random subset of primes + some random composites
        init = random.sample(primes, len(primes)//2)
        # Add a few random composites
        composites = [n for n in range(1, max_n+1) if n not in primes and n > 1]
        init += random.sample(composites, min(10, len(composites)))
        s = local_search(init, 2000)
        if len(s) > len(best_set):
            best_set = s
            
    return best_set
