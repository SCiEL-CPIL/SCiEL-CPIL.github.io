import time
import random

def construct():
    start_time = time.time()
    deadline = 88.0  # Safety margin for 90s limit
    n = 256
    
    def elapsed():
        return time.time() - start_time

    def is_valid_solution(A, B):
        if not A or not B:
            return False
        if len(A) > 64 or len(B) > 64:
            return False
        
        # Check Sidon for A
        sums_A = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                s = A[i] + A[j]
                if s in sums_A:
                    return False
                sums_A.add(s)
        
        # Check Sidon for B
        sums_B = set()
        for i in range(len(B)):
            for j in range(i, len(B)):
                s = B[i] + B[j]
                if s in sums_B:
                    return False
                sums_B.add(s)
        
        # Check disjoint differences
        diffs_A = set()
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                diffs_A.add(abs(A[j] - A[i]))
        diffs_B = set()
        for i in range(len(B)):
            for j in range(i+1, len(B)):
                diffs_B.add(abs(B[j] - B[i]))
                
        if diffs_A & diffs_B:
            return False
            
        return True

    def build_sidon_set(candidates, allowed_diffs):
        """
        Greedily build a Sidon set from candidates such that 
        all pairwise differences are in allowed_diffs.
        """
        S = []
        sums = set()
        diffs_used = set()
        
        # Shuffle candidates
        local_candidates = candidates[:]
        random.shuffle(local_candidates)
        
        for x in local_candidates:
            # Check Sidon condition
            valid_sidon = True
            for s in S:
                if (x + s) in sums:
                    valid_sidon = False
                    break
            if valid_sidon and (2*x) in sums:
                valid_sidon = False
            
            if not valid_sidon:
                continue
            
            # Check difference constraint
            valid_diff = True
            new_diffs = []
            for s in S:
                d = abs(x - s)
                if d not in allowed_diffs:
                    valid_diff = False
                    break
                new_diffs.append(d)
            
            if not valid_diff:
                continue
            
            # Add x
            S.append(x)
            for s in S[:-1]:
                sums.add(x + s)
            sums.add(2*x)
            
            # Update diffs used (not strictly needed for constraint if allowed_diffs is static, 
            # but needed if we were doing dynamic allocation)
            # Here allowed_diffs is fixed for this run.
            
            if len(S) >= 20: # Theoretical max approx 16-18, 20 is safe upper bound
                break
                
        return S

    best_score = 0
    best_sets = ([1], [2])
    
    candidates = list(range(1, n + 1))
    all_diffs = set(range(1, n))
    
    # Strategy: Random partition of differences
    # We try many random partitions.
    # To make it faster, we can bias the partition.
    
    for iteration in range(2000):
        if elapsed() > deadline:
            break
            
        # Randomly assign each difference to A, B, or None
        # We want to maximize the chance of finding large sets.
        # A and B should get roughly half the differences.
        p_a = 0.45
        p_b = 0.45
        # p_none = 0.10
        
        P_A = set()
        P_B = set()
        
        for d in all_diffs:
            r = random.random()
            if r < p_a:
                P_A.add(d)
            elif r < p_a + p_b:
                P_B.add(d)
            # else: unused
            
        A = build_sidon_set(candidates, P_A)
        B = build_sidon_set(candidates, P_B)
        
        if A and B:
            # Verify (just in case, though construction guarantees it)
            if is_valid_solution(A, B):
                score = len(A) + len(B)
                if score > best_score:
                    best_score = score
                    best_sets = (A[:], B[:])
                    
        # Occasionally try a deterministic split to ensure we don't miss simple solutions
        if iteration % 10 == 0:
            # Try a few fixed splits if time permits
            for K in [100, 110, 120, 130, 140]:
                if elapsed() > deadline:
                    break
                P_A_fixed = set(range(1, K+1))
                P_B_fixed = set(range(K+1, n))
                A = build_sidon_set(candidates, P_A_fixed)
                B = build_sidon_set(candidates, P_B_fixed)
                if A and B and is_valid_solution(A, B):
                    score = len(A) + len(B)
                    if score > best_score:
                        best_score = score
                        best_sets = (A[:], B[:])

    # Fallback if nothing found
    if best_score == 0:
        A = [1]
        B = [2]
        best_sets = (A, B)

    return list(best_sets)
