import time
from math import gcd

N = 512
DEADLINE = 85.0 # Safety margin

def is_compatible(x, S):
    for y in S:
        if (x // gcd(x, y)) * y > N:
            return False
    return True

def construct():
    start_time = time.time()
    
    # Initial candidate: divisors of 360
    # 360 = 2^3 * 3^2 * 5
    divisors_360 = []
    for i in range(1, 361):
        if 360 % i == 0:
            divisors_360.append(i)
    
    best_S = set(divisors_360)
    best_size = len(best_S)
    
    # Greedy add
    for x in range(1, N + 1):
        if x not in best_S and is_compatible(x, best_S):
            best_S.add(x)
            best_size += 1
    
    # Local search: try to improve
    # We try to swap elements or add elements
    # Since adding is blocked, we try to remove one and add two?
    # Or just remove one and add one if it enables future adds?
    # Simple approach: try all pairs (remove y, add x) if it doesn't decrease size?
    # Actually, we want to increase size.
    # Let's try a limited hill climbing.
    
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        # Try to add any x
        for x in range(1, N + 1):
            if x not in best_S and is_compatible(x, best_S):
                best_S.add(x)
                best_size += 1
                improved = True
                break
        
        if not improved:
            # Try to swap y for x to potentially allow more adds later?
            # Or just find a larger set via random restarts?
            # Given time, a simple restart with random greedy might help.
            # But 24 is likely the max for this structure.
            # Let's try one restart with random order.
            import random
            random.seed(42)
            S_rand = set()
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
            for x in candidates:
                if is_compatible(x, S_rand):
                    S_rand.add(x)
            if len(S_rand) > best_size:
                best_S = S_rand
                best_size = len(S_rand)
                improved = True # Continue loop to try adding more to this new set
    
    return list(best_S)
