import time
import random
from math import gcd
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80  # safety margin under 90s wall-clock
    
    def distinct_sums(A, B):
        return {a + b for a in A for b in B}
    
    def is_pairwise_coprime(vals):
        lst = list(vals)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if gcd(lst[i], lst[j]) != 1:
                    return False
        return True
    
    def score_of(A, B):
        s = distinct_sums(A, B)
        if len(s) < 2:
            return -1
        if not is_pairwise_coprime(s):
            return -1
        return len(s)
    
    # Start with known good solution
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = score_of(best_A, best_B)  # should be 5
    
    # Try to improve via local search and random restarts
    rng = random.Random(42)
    
    def random_set():
        return sorted(rng.sample(range(1, 129), 3))
    
    # Local improvement: try changing one element at a time
    def local_improve(A, B, max_iter=1000):
        A = list(A)
        B = list(B)
        cur_score = score_of(A, B)
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            # Try changing elements in A
            for i in range(3):
                old = A[i]
                for new_val in rng.sample([x for x in range(1, 129) if x != old], min(50, 128)):
                    A[i] = new_val
                    if len(set(A)) == 3:
                        s = score_of(A, B)
                        if s > cur_score:
                            cur_score = s
                            improved = True
                            break
                    A[i] = old
                if improved:
                    break
            # Try changing elements in B
            if not improved:
                for i in range(3):
                    old = B[i]
                    for new_val in rng.sample([x for x in range(1, 129) if x != old], min(50, 128)):
                        B[i] = new_val
                        if len(set(B)) == 3:
                            s = score_of(A, B)
                            if s > cur_score:
                                cur_score = s
                                improved = True
                                break
                        B[i] = old
                    if improved:
                        break
            if not improved:
                break
        return A, B, cur_score
    
    # Phase 1: Local improvement from initial
    A, B, sc = local_improve(best_A, best_B)
    if sc > best_score:
        best_A, best_B, best_score = A, B, sc
    
    # Phase 2: Random restarts
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        A = random_set()
        B = random_set()
        sc = score_of(A, B)
        if sc > best_score:
            best_A, best_B, best_score = A, B, sc
            # Local improve from this point
            A2, B2, sc2 = local_improve(A, B, max_iter=200)
            if sc2 > best_score:
                best_A, best_B, best_score = A2, B2, sc2
        elif sc >= 4:
            # Promising, do local improvement
            A2, B2, sc2 = local_improve(A, B, max_iter=200)
            if sc2 > best_score:
                best_A, best_B, best_score = A2, B2, sc2
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
