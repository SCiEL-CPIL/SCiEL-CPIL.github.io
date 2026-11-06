"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from itertools import combinations, product

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _pairwise_coprime(vals):
    """Check if all distinct values are pairwise coprime."""
    distinct = list(set(vals))
    for i in range(len(distinct)):
        for j in range(i + 1, len(distinct)):
            if _gcd(distinct[i], distinct[j]) != 1:
                return False
    return True

def _count_distinct_coprime(A, B):
    """Count distinct values in A+B that are pairwise coprime."""
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    if not sums:
        return 0
    if _pairwise_coprime(list(sums)):
        return len(sums)
    return 0

def _evaluate(A, B):
    """Return the number of distinct pairwise-coprime sums, or 0 if not all coprime."""
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    distinct = list(sums)
    if not distinct:
        return 0
    # Check pairwise coprime
    for i in range(len(distinct)):
        for j in range(i + 1, len(distinct)):
            if _gcd(distinct[i], distinct[j]) != 1:
                return 0
    return len(distinct)

def _try_improve(A, B, deadline):
    """Local search: try swapping elements to improve."""
    best_score = _evaluate(A, B)
    A = list(A)
    B = list(B)
    
    # Precompute all valid candidates in [1,128]
    candidates = list(range(1, 129))
    
    # Try single-element swaps
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(3):
            for new_a in candidates:
                if new_a in A:
                    continue
                old_a = A[i]
                A[i] = new_a
                score = _evaluate(A, B)
                if score > best_score:
                    best_score = score
                    improved = True
                else:
                    A[i] = old_a
            A[i] = old_a if 'old_a' in dir() else A[i]
        
        for i in range(3):
            for new_b in candidates:
                if new_b in B:
                    continue
                old_b = B[i]
                B[i] = new_b
                score = _evaluate(A, B)
                if score > best_score:
                    best_score = score
                    improved = True
                else:
                    B[i] = old_b
        
    return A, B, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Start with known good solution (5 distinct sums)
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = _evaluate(best_A, best_B)
    
    # Precompute: for each possible sum value, check if it can be part of a coprime set
    # Strategy: try to find 9 distinct pairwise-coprime sums
    
    # Key insight: pairwise coprime sums means no two sums share a prime factor
    # Small primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, ...
    # Sum range: 2 to 256
    
    # Let's try a more systematic search with pruning
    # For each candidate A, try to find B that maximizes distinct coprime sums
    
    # First, let's do a targeted search for 9 distinct sums
    # We need 9 values in [2, 256] that are pairwise coprime
    # And each must be expressible as a+b where a in A, b in B
    
    # Let's try random search with smart initialization
    import random
    random.seed(42)
    
    # Try many random configurations
    for trial in range(200000):
        if time.time() > deadline:
            break
        
        # Random A
        A = sorted(random.sample(range(1, 129), 3))
        
        # For this A, try to find good B
        # Greedy: pick b values that create new coprime sums
        used_sums = set()
        B = []
        
        for b_trial in range(50):
            if time.time() > deadline:
                break
            
            # Try random b
            b = random.randint(1, 128)
            if b in B:
                continue
            
            # Check if adding b creates valid sums
            new_sums = set()
            for a in A:
                s = a + b
                if s not in used_sums:
                    new_sums.add(s)
            
            # Check if all new sums are coprime with existing
            all_sums = used_sums | new_sums
            all_sums_list = list(all_sums)
            
            valid = True
            for i in range(len(all_sums_list)):
                for j in range(i + 1, len(all_sums_list)):
                    if _gcd(all_sums_list[i], all_sums_list[j]) != 1:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                B.append(b)
                used_sums = all_sums
                if len(B) == 3:
                    break
        
        if len(B) == 3:
            score = _evaluate(A, B)
            if score > best_score:
                best_score = score
                best_A = A[:]
                best_B = B[:]
                if best_score == 9:
                    break
    
    # Local search from best found
    if time.time() < deadline:
        best_A, best_B, best_score = _try_improve(best_A, best_B, deadline)
    
    # Final check
    final_score = _evaluate(best_A, best_B)
    if final_score > best_score:
        best_score = final_score
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
