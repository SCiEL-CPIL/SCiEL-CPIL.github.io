import itertools
import math
import random
import time

def is_coprime(a, b):
    return math.gcd(a, b) == 1

def is_pairwise_coprime(lst):
    n = len(lst)
    for i in range(n):
        for j in range(i+1, n):
            if not is_coprime(lst[i], lst[j]):
                return False
    return True

def evaluate(A, B):
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    distinct_sums = list(sums)
    if is_pairwise_coprime(distinct_sums):
        return len(distinct_sums)
    return 0

def construct():
    best_score = 0
    best_A = None
    best_B = None
    
    # Seed
    A_seed = [1, 7, 13]
    B_seed = [4, 10, 16]
    score_seed = evaluate(A_seed, B_seed)
    if score_seed > best_score:
        best_score = score_seed
        best_A = A_seed
        best_B = B_seed

    start_time = time.time()
    timeout = 55
    
    current_A = list(best_A)
    current_B = list(best_B)
    current_score = best_score
    
    # Use a fixed seed for reproducibility in testing? No, random is fine.
    # But to be safe, maybe try a few deterministic structures first?
    # No, random search is more robust for this unknown landscape.
    
    while time.time() - start_time < timeout:
        # Random restarts
        if random.random() < 0.05 or current_score == 0:
            current_A = random.sample(range(1, 129), 3)
            current_B = random.sample(range(1, 129), 3)
            current_score = evaluate(current_A, current_B)
            if current_score > best_score:
                best_score = current_score
                best_A = current_A
                best_B = current_B
            continue

        improved = False
        # Try mutating A
        for i in range(3):
            for _ in range(10):
                new_a = random.randint(1, 128)
                if new_a in current_A:
                    continue
                test_A = current_A.copy()
                test_A[i] = new_a
                test_score = evaluate(test_A, current_B)
                if test_score > current_score:
                    current_A = test_A
                    current_score = test_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_A = current_A
                        best_B = current_B
                        if best_score == 9:
                            return [best_A, best_B]
                    break
        
        # Try mutating B
        if not improved:
            for i in range(3):
                for _ in range(10):
                    new_b = random.randint(1, 128)
                    if new_b in current_B:
                        continue
                    test_B = current_B.copy()
                    test_B[i] = new_b
                    test_score = evaluate(current_A, test_B)
                    if test_score > current_score:
                        current_B = test_B
                        current_score = test_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_A = current_A
                            best_B = current_B
                            if best_score == 9:
                                return [best_A, best_B]
                        break

    return [best_A, best_B]
