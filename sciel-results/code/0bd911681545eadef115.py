import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = 192
    max_size = 48

    best_solution = []
    best_size = 0

    def is_valid_addition(S, R, x):
        # Check 1: x divides sum of any triple in S
        if len(S) >= 3:
            # O(|S|^3) check
            for i in range(len(S)):
                si = S[i]
                for j in range(i + 1, len(S)):
                    sij = si + S[j]
                    for l in range(j + 1, len(S)):
                        if (sij + S[l]) % x == 0:
                            return False
        
        # Check 2: a divides (b+c+x) for any a in S, b,c in S
        # R[a] contains (b+c)%a for b,c in S, b!=a, c!=a
        for a in S:
            if (-x) % a in R[a]:
                return False
        
        return True

    # Greedy with randomization
    rng = random.Random(42)
    
    # Run multiple trials
    # We can also try different sorting strategies
    strategies = [
        lambda x: (rng.random(), -x), # Random bias to large
        lambda x: (-x, rng.random()), # Large bias to random
        lambda x: (rng.random(), x)   # Random bias to small
    ]
    
    while time.time() < deadline:
        # Pick a strategy
        strat = rng.choice(strategies)
        candidates = list(range(1, n + 1))
        candidates.sort(key=strat)
        
        S = []
        R = {}
        
        for x in candidates:
            if time.time() > deadline:
                break
            
            if is_valid_addition(S, R, x):
                # Add x to S
                # Update R
                # For each a in S (old), add (x+b)%a to R[a] for b in S (old), b!=a
                old_S = S[:]
                for a in old_S:
                    # R[a] already exists
                    for b in old_S:
                        if b != a:
                            R[a].add((x + b) % a)
                
                # Initialize R[x]
                R[x] = set()
                
                S.append(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_solution = S[:]
            if best_size == max_size:
                break

    return best_solution
