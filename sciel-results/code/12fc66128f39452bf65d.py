import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 45.0

    n = 16
    k = 4

    all_sets = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)
    
    # Precompute degrees?
    # Degree of S: Number of pairs (T, U) such that {S, T, U} is a sunflower?
    # Or simpler: Number of T such that there exists U forming sunflower with S, T.
    # This is expensive to compute for all S.
    # Let's just use random shuffles.

    def is_sunflower_partner(A, B, X):
        K = A & B
        if (X & K) != K:
            return False
        A_diff = A & ~K
        if (X & A_diff) != 0:
            return False
        B_diff = B & ~K
        if (X & B_diff) != 0:
            return False
        return True

    def can_add(X, family):
        L = len(family)
        for i in range(L):
            A = family[i]
            for j in range(i + 1, L):
                B = family[j]
                if is_sunflower_partner(A, B, X):
                    return False
        return True

    best_family = []
    best_size = 0

    while time.time() < deadline:
        random.shuffle(all_sets)
        current_family = []
        
        for X in all_sets:
            if can_add(X, current_family):
                current_family.append(X)
                
        if len(current_family) > best_size:
            best_size = len(current_family)
            best_family = current_family[:]
            
        if best_size >= 250:
            break

    result = []
    for mask in best_family:
        elems = []
        for i in range(n):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
        
    return result
