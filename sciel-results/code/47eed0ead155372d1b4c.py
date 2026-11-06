"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    # Allow 100 seconds for the main computation to leave margin for overhead and return
    deadline = start_time + 100.0
    
    # Set limit for powerful numbers. 
    # Higher L increases the number of powerful numbers and potential pairs.
    # L = 5e8 gives ~30k powerful numbers, ~4.5e8 pairs.
    # This should take ~20-40s in Python.
    L = 500_000_000
    
    def generate_powerful(limit):
        pows = set()
        # y^3 <= limit
        max_y = int(limit ** (1/3)) + 2
        for y in range(1, max_y + 1):
            y3 = y ** 3
            if y3 > limit:
                break
            # x^2 <= limit / y3
            max_x = int((limit // y3) ** 0.5) + 2
            for x in range(1, max_x + 1):
                val = x * x * y3
                if val > limit:
                    break
                pows.add(val)
        return sorted(list(pows))

    pows = generate_powerful(L)
    
    # Count sums of pairs (a, b) with a <= b
    counts = {}
    n = len(pows)
    iter_count = 0
    aborted = False
    
    for i in range(n):
        a = pows[i]
        for j in range(i, n):
            s = a + pows[j]
            counts[s] = counts.get(s, 0) + 1
            iter_count += 1
            if iter_count % 1000000 == 0:
                if time.time() > deadline:
                    aborted = True
                    break
        if aborted:
            break

    if not counts:
        # Fallback if something went wrong or L was too small to generate anything (unlikely)
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # Find the N with the maximum number of representations
    best_n, best_count = max(counts.items(), key=lambda kv: kv[1])
    
    # Reconstruct the pairs for the best N
    # This is O(N) and very fast
    p_set = set(pows)
    pairs = []
    for a in pows:
        if a > best_n:
            break
        b = best_n - a
        if a <= b and b in p_set:
            pairs.append([a, b])
            
    return {'N': best_n, 'pairs': pairs}
# EVOLVE-BLOCK-END
