"""Rewritten program for pairwise-coprime sumset maximization."""
# EVOLVE-BLOCK-START
import itertools
import math
import time
import random

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    random.seed(42)
    
    # Sums must be pairwise coprime. To avoid factor 2, make all sums odd:
    # A all odd + B all even, or A all even + B all odd.
    
    odds = list(range(1, 129, 2))   # 1,3,5,...,127
    evens = list(range(2, 129, 2))  # 2,4,6,...,128
    
    def evaluate(A, B):
        """Return number of distinct sums if pairwise coprime, else -1."""
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        slist = list(sums)
        for i in range(len(slist)):
            for j in range(i + 1, len(slist)):
                if math.gcd(slist[i], slist[j]) > 1:
                    return -1
        return len(slist)
    
    best_score = 0
    best_A = None
    best_B = None
    
    def update_best(A, B):
        nonlocal best_score, best_A, best_B
        s = evaluate(A, B)
        if s > best_score:
            best_score = s
            best_A = A[:]
            best_B = B[:]
    
    # Try some hand-crafted candidates
    candidates = [
        ([1, 3, 7], [2, 4, 8]),
        ([1, 3, 7], [2, 4, 10]),
        ([1, 5, 7], [2, 4, 8]),
        ([1, 3, 11], [2, 4, 8]),
        ([1, 5, 11], [2, 4, 8]),
        ([3, 5, 7], [2, 4, 8]),
        ([1, 3, 7], [2, 6, 8]),
        ([1, 3, 7], [2, 4, 12]),
        ([1, 5, 11], [2, 6, 8]),
        ([1, 3, 13], [2, 4, 8]),
        ([1, 7, 13], [2, 4, 6]),
        ([1, 7, 13], [4, 10, 16]),  # original baseline
        ([1, 3, 5], [2, 4, 6]),
        ([1, 3, 5], [2, 4, 8]),
        ([1, 3, 5], [2, 6, 8]),
        ([1, 3, 9], [2, 4, 8]),
        ([1, 5, 9], [2, 4, 8]),
        ([3, 5, 9], [2, 4, 8]),
        ([1, 7, 9], [2, 4, 8]),
        ([1, 7, 11], [2, 4, 8]),
    ]
    
    for A, B in candidates:
        update_best(A, B)
    
    # Random search with deadline
    while time.time() < deadline - 5:
        # Type 1: A=odds, B=evens
        A = random.sample(odds, 3)
        B = random.sample(evens, 3)
        update_best(A, B)
        
        # Type 2: A=evens, B=odds
        A2 = random.sample(evens, 3)
        B2 = random.sample(odds, 3)
        update_best(A2, B2)
        
        if time.time() > deadline - 10:
            break
    
    # Local improvement
    if best_A is not None and best_B is not None:
        # Determine type: is best_A odd-dominant or even-dominant?
        # Check if all elements of best_A are odd
        A_is_odd = all(x % 2 == 1 for x in best_A)
        A_pool = odds if A_is_odd else evens
        B_is_odd = all(x % 2 == 1 for x in best_B)
        B_pool = odds if B_is_odd else evens
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            current_score = evaluate(best_A, best_B)
            if current_score < 0:
                break
            
            # Try replacing each element of A
            for i in range(3):
                for _ in range(50):
                    new_A = best_A[:]
                    new_A[i] = random.choice(A_pool)
                    if len(set(new_A)) != 3:
                        continue
                    score = evaluate(new_A, best_B)
                    if score > current_score:
                        best_A = new_A
                        current_score = score
                        improved = True
                        break
                if improved:
                    break
            
            # Try replacing each element of B
            if not improved:
                for i in range(3):
                    for _ in range(50):
                        new_B = best_B[:]
                        new_B[i] = random.choice(B_pool)
                        if len(set(new_B)) != 3:
                            continue
                        score = evaluate(best_A, new_B)
                        if score > current_score:
                            best_B = new_B
                            current_score = score
                            improved = True
                            break
                    if improved:
                        break
    
    if best_A is None:
        return [[1, 7, 13], [4, 10, 16]]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
