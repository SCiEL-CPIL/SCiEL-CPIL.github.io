import time
import random

SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

def get_small_factor(x):
    result = x
    for p in SMALL_PRIMES:
        while result % p == 0:
            result //= p
    return result

def evaluate(n):
    factors = set()
    for i in range(1, 25):
        factors.add(get_small_factor(n + i))
    return len(factors)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_n = 1000
    best_score = evaluate(best_n)
    
    # Random search with local refinement
    # We will iterate until time runs out
    # Each iteration: pick random n, evaluate, if better, do local search
    
    # To avoid getting stuck in local minima, we use a "reheating" strategy
    # or just rely on the large search space.
    
    # Initial broad scan to find a decent starting point
    # Sample 1000 points
    for _ in range(1000):
        if time.time() > deadline: break
        n = random.randint(1000, 1000000)
        score = evaluate(n)
        if score < best_score:
            best_score = score
            best_n = n
            # Quick local search
            for delta in range(-5, 6):
                c = best_n + delta
                if 1000 <= c <= 1000000:
                    s = evaluate(c)
                    if s < best_score:
                        best_score = s
                        best_n = c

    # Main loop: Random restarts + Local Search
    while time.time() < deadline:
        # Pick a random center
        center = random.randint(1000, 1000000)
        # Evaluate a small neighborhood around center
        # This is more efficient than pure random if good spots are clustered
        # But pure random is safer for global search.
        # Let's do a mix: 50% random, 50% near best
        
        if random.random() < 0.5:
            n = random.randint(1000, 1000000)
        else:
            # Perturb best_n
            n = best_n + random.randint(-100, 100)
            if n < 1000: n = 1000 + random.randint(0, 100)
            if n > 1000000: n = 1000000 - random.randint(0, 100)
            
        score = evaluate(n)
        if score < best_score:
            best_score = score
            best_n = n
            
            # Intensive local search
            # Check all neighbors in a window of size 2*radius
            radius = 20
            for delta in range(-radius, radius + 1):
                c = best_n + delta
                if 1000 <= c <= 1000000:
                    s = evaluate(c)
                    if s < best_score:
                        best_score = s
                        best_n = c
                        # If we found a very good score (e.g., 2 or 3), expand search
                        if s <= 3:
                            radius = 50
                            for delta2 in range(-radius, radius + 1):
                                c2 = best_n + delta2
                                if 1000 <= c2 <= 1000000:
                                    s2 = evaluate(c2)
                                    if s2 < best_score:
                                        best_score = s2
                                        best_n = c2

    return best_n
