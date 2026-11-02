import numpy as np

N = 50

def construct():
    # Find a Sidon set of size K in range(N)
    # A Sidon set has distinct pairwise sums.
    # We can use a greedy approach or a known set.
    # Known Sidon sets for N=50:
    # Let's try to construct one.
    
    def is_sidon(S):
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    # Greedy construction
    S = []
    for x in range(N):
        candidate = S + [x]
        if is_sidon(candidate):
            S.append(x)
        if len(S) >= 8: # Max size approx sqrt(N)
            break
            
    # If greedy fails to get enough, try random restarts
    if len(S) < 8:
        import random
        random.seed(42)
        best_S = S
        for _ in range(1000):
            rand_set = random.sample(range(N), 8)
            if is_sidon(rand_set):
                best_S = rand_set
                break
                
    # Create the height vector
    h = [0.0] * N
    for idx in best_S:
        h[idx] = 1.0
        
    return h
