import time
import random

def _score(A):
    # A is a list of distinct ints
    # Calculate sumset
    sums = set()
    prods = set()
    n = len(A)
    for i in range(n):
        for j in range(i, n):
            sums.add(A[i] + A[j])
            prods.add(A[i] * A[j])
    return max(len(sums), len(prods))

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # Initial set: 1 to 20
    A = list(range(1, 21))
    best_score = _score(A)
    best_A = A.copy()
    
    # Local Search
    # To make it deterministic but effective, we can use a fixed seed or just random.
    # Random is fine.
    while time.time() < deadline:
        # Pick a random element to replace
        i = random.randint(0, 19)
        old_val = A[i]
        
        # Pick a candidate y
        # Strategy: try small numbers first? Or random?
        # Let's try a mix.
        # 50% chance: random from 1..10000
        # 50% chance: random from 1..50 (small composites)
        if random.random() < 0.5:
            y = random.randint(1, 50)
        else:
            y = random.randint(1, 10000)
            
        if y == old_val:
            continue
        if y in A:
            continue
            
        # Create new set
        # To avoid creating a new list every time, we can swap, score, swap back
        A[i] = y
        new_score = _score(A)
        
        if new_score < best_score:
            best_score = new_score
            best_A = A.copy()
        else:
            A[i] = old_val
            
    return best_A
