import time
import random
from math import gcd
from itertools import combinations

def construct():
    # Strict deadline to avoid timeout
    deadline = time.time() + 50
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute gcd table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def is_bad_triple(a, b, c):
        return gcd_table[a][b] == gcd_table[a][c] == gcd_table[b][c]

    def is_compatible(x, selected):
        """Check if x can be added to selected without creating a bad triple."""
        # selected is a set
        for a in selected:
            for b in selected:
                if a >= b:
                    continue
                if is_bad_triple(a, b, x):
                    return False
        return True

    # 1. Greedy Construction: Prioritize numbers with fewer conflicts
    # Calculate degree: number of bad triples each number is part of
    degrees = [0] * (N + 1)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                if is_bad_triple(a, b, c):
                    degrees[a] += 1
                    degrees[b] += 1
                    degrees[c] += 1
    
    # Sort numbers by degree (ascending) to pick "least problematic" first
    sorted_numbers = sorted(numbers, key=lambda x: degrees[x])
    
    selected = set()
    for x in sorted_numbers:
        if time.time() > deadline - 10:
            break
        if is_compatible(x, selected):
            selected.add(x)
            
    # 2. Local Search: Try to improve by adding or swapping
    # We will run this for a few seconds or until no improvement
    last_improvement = time.time()
    improved = True
    
    while time.time() < deadline - 5 and improved:
        improved = False
        
        # Phase A: Try to add any compatible number
        for x in numbers:
            if x not in selected:
                if is_compatible(x, selected):
                    selected.add(x)
                    improved = True
                    last_improvement = time.time()
                    break # Restart loop after addition
        
        if improved:
            continue
            
        # Phase B: Try swaps (remove one, add one)
        # To speed up, we can limit the number of swap attempts or break early
        current_size = len(selected)
        
        # Shuffle selected to randomize which removals we try first?
        # Or just iterate.
        # Optimization: Only try swaps if we are close to deadline? No, swaps are key to escape local optima.
        
        # Efficient swap check:
        # For each y in selected, for each x not in selected:
        #   if is_compatible(x, selected - {y}):
        #       selected = (selected - {y}) | {x}
        #       improved = True
        #       break
        
        # This is expensive. Let's sample a few y's if the set is large.
        # But N=128 is small. Let's try a few random removals.
        removed_any = False
        for _ in range(20): # Limit random removal attempts per iteration
            if time.time() > deadline - 5:
                break
            if len(selected) == 0:
                break
            y = random.choice(list(selected))
            temp_selected = selected - {y}
            
            for x in numbers:
                if x == y or x in selected:
                    continue
                if is_compatible(x, temp_selected):
                    # Accept swap
                    selected = temp_selected | {x}
                    improved = True
                    last_improvement = time.time()
                    removed_any = True
                    break
            if removed_any:
                break
        
        if not improved and time.time() - last_improvement > 2:
            # Stagnation: try a random restart with a smaller set?
            # Or just break if we've been stuck for a while
            # Given the small N, greedy+swap usually finds a good set quickly.
            # If stuck, we might be at local optimum.
            # Let's try one more aggressive move: remove 2, add 3? Too complex.
            # Just break to save time if no progress for 2s.
            # But we have 50s limit. Let's allow a bit more time.
            if time.time() - last_improvement > 10:
                break

    return list(selected)
