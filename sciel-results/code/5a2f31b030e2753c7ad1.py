"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations, permutations

def _check_pairwise_coprime(values):
    """Check if all distinct values are pairwise coprime."""
    vals = list(set(values))
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            if gcd(vals[i], vals[j]) != 1:
                return False
    return True

def _evaluate(A, B):
    """Return (cardinality, is_valid) for the sumset A+B."""
    sums = [a + b for a in A for b in B]
    distinct = set(sums)
    if not _check_pairwise_coprime(distinct):
        return (0, False)
    return (len(distinct), True)

def construct():
    start_time = time.time()
    deadline = start_time + 45  # Safety margin before 60s kill
    
    # Initialize with a known valid solution
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score, valid = _evaluate(best_A, best_B)
    if not valid:
        best_score = 0
        best_A = [1, 2, 3]
        best_B = [4, 5, 6]
    
    if best_score >= 9:
        return [best_A, best_B]
    
    rng = random.Random(12345)
    range_ = range(1, 129)
    
    # Key insight: parity mismatch eliminates even sums automatically
    # All-odd A with all-even B (or vice versa) gives all odd sums
    odd_numbers = list(range(1, 129, 2))   # 1,3,5,...,127
    even_numbers = list(range(2, 129, 2))  # 2,4,6,...,128
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        if best_score >= 9:
            break
        
        # Strategy 1: Random search with parity constraint (odd A, even B or even A, odd B)
        if rng.random() < 0.5:
            A = list(rng.sample(odd_numbers, 3))
            B = list(rng.sample(even_numbers, 3))
        else:
            A = list(rng.sample(even_numbers, 3))
            B = list(rng.sample(odd_numbers, 3))
        
        score, is_valid = _evaluate(A, B)
        if is_valid and score > best_score:
            best_score = score
            best_A = A[:]
            best_B = B[:]
        
        if best_score >= 9:
            break
        
        # Strategy 2: Local perturbation of best solution
        if iteration % 3 == 0:
            # Perturb A
            A2 = best_A[:]
            idx = rng.randint(0, 2)
            if best_A[idx] % 2 == 1:
                A2[idx] = rng.choice(odd_numbers)
            else:
                A2[idx] = rng.choice(even_numbers)
            if len(set(A2)) == 3:
                s2, v2 = _evaluate(A2, best_B)
                if v2 and s2 > best_score:
                    best_score = s2
                    best_A = A2[:]
                    
            # Perturb B
            B2 = best_B[:]
            idx = rng.randint(0, 2)
            if best_B[idx] % 2 == 1:
                B2[idx] = rng.choice(odd_numbers)
            else:
                B2[idx] = rng.choice(even_numbers)
            if len(set(B2)) == 3:
                s3, v3 = _evaluate(best_A, B2)
                if v3 and s3 > best_score:
                    best_score = s3
                    best_B = B2[:]
        
        # Strategy 3: Targeted search - pick sums first, then find A,B
        if iteration % 10 == 0 and best_score < 9:
            # Try to construct 9 distinct pairwise-coprime odd sums
            # Then check if they can be decomposed as a+b
            candidate_sums = list(rng.sample(odd_numbers, min(20, len(odd_numbers))))
            # Filter for pairwise coprime
            valid_sums = []
            for s in candidate_sums:
                ok = True
                for vs in valid_sums:
                    if gcd(s, vs) != 1:
                        ok = False
                        break
                if ok:
                    valid_sums.append(s)
                if len(valid_sums) >= 9:
                    break
            
            if len(valid_sums) >= 9:
                # Check if these 9 sums can be covered by 3x3 grid
                target_sums = set(valid_sums[:9])
                # Try random A from odd, see if B can complete
                for _attempt in range(50):
                    A_try = list(rng.sample(odd_numbers, 3))
                    # For each a in A, we need B such that a+b covers target sums
                    # b = s - a for s in target_sums
                    possible_B = set()
                    for a in A_try:
                        for s in target_sums:
                            b = s - a
                            if 1 <= b <= 128:
                                possible_B.add(b)
                    # We need exactly 3 B values that cover all 9 sums
                    # Try small subsets of possible_B
                    if len(possible_B) >= 3:
                        B_candidates = list(possible_B)
                        if len(B_candidates) > 50:
                            B_candidates = rng.sample(B_candidates, 50)
                        for B_try in combinations(B_candidates, 3):
                            B_set = list(B_try)
                            if len(set(B_set)) == 3:
                                sums_try = set(a + b for a in A_try for b in B_set)
                                if sums_try == target_sums:
                                    s_final, v_final = _evaluate(A_try, B_set)
                                    if v_final and s_final > best_score:
                                        best_score = s_final
                                        best_A = A_try[:]
                                        best_B = B_set[:]
                                        if best_score >= 9:
                                            break
                        if best_score >= 9:
                            break
        
        if time.time() > deadline - 5:
            break
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
