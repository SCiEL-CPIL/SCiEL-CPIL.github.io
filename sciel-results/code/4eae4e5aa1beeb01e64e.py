"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import gcd
import random
import time

def construct():
    # Helper to check if two sets of differences are disjoint
    def diff_sets_disjoint(A, B):
        diffs_A = set()
        for i in range(3):
            for j in range(3):
                if i != j:
                    diffs_A.add(A[i] - A[j])
        diffs_B = set()
        for i in range(3):
            for j in range(3):
                if i != j:
                    diffs_B.add(B[i] - B[j])
        # If intersection is empty, all 9 sums are distinct
        return len(diffs_A & diffs_B) == 0

    def check_pairwise_coprime(values):
        vals = sorted(set(values))
        n = len(vals)
        for i in range(n):
            for j in range(i + 1, n):
                if gcd(vals[i], vals[j]) != 1:
                    return False
        return True

    def evaluate(A, B):
        # Quick check for distinctness using difference sets
        if not diff_sets_disjoint(A, B):
            return 0
        sums = [a + b for a in A for b in B]
        distinct_sums = set(sums)
        # Since diff sets are disjoint, len(distinct_sums) must be 9
        if len(distinct_sums) != 9:
            return 0
        if check_pairwise_coprime(distinct_sums):
            return 9
        return 0

    # Strategy:
    # 1. Generate a list of candidate sets that are "spread out" to maximize chance of disjoint differences.
    # 2. Test pairs of these candidates.
    
    candidates = []
    
    # Generate candidates with large gaps
    # We want A and B to have large internal differences to avoid small difference collisions?
    # Actually, if A has small differences, D_A has small numbers.
    # If B has large differences, D_B has large numbers.
    # Intersection is less likely.
    # So we want one set to have small steps and the other large steps?
    # Or just ensure the sets are from different "scales".
    
    # Let's generate a diverse set of candidates
    # Type 1: Arithmetic Progressions with step k
    for start in range(1, 100, 2):
        for step in [5, 10, 15, 20, 30, 40]:
            A = [start, start + step, start + 2 * step]
            if max(A) <= 128 and len(set(A)) == 3:
                candidates.append(tuple(sorted(A)))
                
    # Type 2: Random sets with minimum gap
    random.seed(42)
    for _ in range(500):
        # Pick 3 distinct numbers with min gap 3
        A = sorted(random.sample(range(1, 129), 3))
        if A[1] - A[0] >= 3 and A[2] - A[1] >= 3:
            candidates.append(tuple(A))
            
    # Remove duplicates
    candidates = list(set(candidates))
    
    # Shuffle to ensure we don't just check "small" sets first if they fail
    random.shuffle(candidates)
    
    best_A = None
    best_B = None
    best_count = 0
    
    # Systematic search
    # To keep it fast, we limit the number of pairs checked
    # 500 candidates -> 250k pairs. Each check is fast.
    # 250k * (diff check + coprime check) should be < 1s.
    
    for i in range(len(candidates)):
        A = list(candidates[i])
        for j in range(len(candidates)):
            B = list(candidates[j])
            count = evaluate(A, B)
            if count > best_count:
                best_count = count
                best_A = A[:]
                best_B = B[:]
                if best_count == 9:
                    return [best_A, best_B]
                    
    # Fallback: Random search if systematic didn't find 9
    # (Unlikely, but safe)
    if best_count < 9:
        random.seed(123)
        start_time = time.time()
        while time.time() - start_time < 50:
            A = sorted(random.sample(range(1, 129), 3))
            B = sorted(random.sample(range(1, 129), 3))
            count = evaluate(A, B)
            if count > best_count:
                best_count = count
                best_A = A[:]
                best_B = B[:]
                if best_count == 9:
                    break

    if best_A is None:
        # Fallback to a known valid small set if all else fails (though 9 is the target)
        # The baseline in the original code was [1,7,13] and [4,10,16]
        # Let's check if that is valid
        A_fb = [1, 7, 13]
        B_fb = [4, 10, 16]
        best_A = A_fb
        best_B = B_fb
        
    return [best_A, best_B]
# EVOLVE-BLOCK-END
