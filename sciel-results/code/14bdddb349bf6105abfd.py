import random
import time
from itertools import combinations

def is_sidon(subset):
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(S):
    # S is a list of 12 distinct ints
    # Check k from 6 down to 3
    # If any subset of size k is Sidon, return k
    # If none, return 3 (since size 2 is always Sidon)
    for k in range(6, 2, -1):
        for comb in combinations(S, k):
            if is_sidon(comb):
                return k
    return 3

def construct():
    start_time = time.time()
    deadline = 90
    best_set = list(range(12))
    best_score = max_sidon_size(best_set)
    
    # Local search / Random search
    # We want to minimize best_score
    # If best_score becomes 3, we can try to verify if 2 is possible (it's not)
    # So we stop if we find 3? No, we can just return it.
    # But maybe there's a set with max Sidon size 3 that is better? No, 3 is the min possible > 2.
    # So if we find 3, we can return immediately?
    # Yes, 3 is the theoretical minimum for N=12 (since 2 is impossible).
    # Wait, is 3 possible?
    # If 3 is possible, we should find it.
    # If not, we return 4.
    
    while time.time() < deadline - 5:
        # Generate a random candidate
        # Bias towards small range to increase collisions
        # Range [0, 20]
        candidate = set()
        while len(candidate) < 12:
            candidate.add(random.randint(0, 20))
        candidate = list(candidate)
        
        score = max_sidon_size(candidate)
        if score < best_score:
            best_score = score
            best_set = candidate
            if best_score == 3:
                # Can't get better than 3
                break
                
    return best_set
