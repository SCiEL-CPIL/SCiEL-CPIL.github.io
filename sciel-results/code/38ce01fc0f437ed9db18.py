import random
import time

def construct():
    n = 180
    S_best = set()
    
    # Precompute for each k, all pairs (a,b) with a<b, a,b in [1,n], a!=b
    # such that (a-k)(b-k) = k^2, i.e., ab/(a+b) = k
    conflict_pairs = {}
    for k in range(1, n + 1):
        k2 = k * k
        pairs = set()
        for d in range(1, k2 + 1):
            if k2 % d == 0:
                e = k2 // d
                a = k + d
                b = k + e
                if a == b:
                    continue
                if 1 <= a <= n and 1 <= b <= n:
                    pairs.add((min(a, b), max(a, b)))
        conflict_pairs[k] = pairs
        
    def run_greedy(order):
        S = set()
        for k in order:
            conflict = False
            for (a, b) in conflict_pairs[k]:
                if a in S and b in S:
                    conflict = True
                    break
            if not conflict:
                S.add(k)
        return S

    def is_valid(S):
        S_list = sorted(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                a, b = S_list[i], S_list[j]
                num = a * b
                den = a + b
                if num % den == 0:
                    c = num // den
                    if c in S and c != a and c != b:
                        return False
        return True

    # 1. Deterministic descending order
    order_desc = list(range(n, 0, -1))
    S_current = run_greedy(order_desc)
    if is_valid(S_current):
        S_best = S_current

    # 2. Randomized restarts
    # Use a fixed seed for reproducibility in evaluation if needed, but random is fine
    # We will try 50 random permutations
    start_time = time.time()
    iterations = 0
    while time.time() - start_time < 50: # Keep 10s buffer
        iterations += 1
        order_rand = list(range(1, n + 1))
        random.shuffle(order_rand)
        
        # Optional: Bias towards larger numbers? 
        # The descending order worked well. 
        # Maybe a "mostly descending with noise" approach?
        # Let's stick to pure random first to see if it finds better local optima.
        
        S_rand = run_greedy(order_rand)
        if len(S_rand) > len(S_best):
            if is_valid(S_rand):
                S_best = S_rand

    # 3. Local search / Hill climbing on the best set found
    # Try to add any number from 1..n not in S_best
    # If it causes a conflict, try to remove one of the conflicting numbers?
    # This is complex. Simple addition attempt:
    S_try = set(S_best)
    changed = True
    while changed:
        changed = False
        for k in range(1, n + 1):
            if k not in S_try:
                # Check if adding k is valid
                valid = True
                for (a, b) in conflict_pairs[k]:
                    if a in S_try and b in S_try:
                        valid = False
                        break
                if valid:
                    S_try.add(k)
                    changed = True
                    break # Restart loop as S changed
    
    if len(S_try) > len(S_best) and is_valid(S_try):
        S_best = S_try

    return sorted(S_best)
