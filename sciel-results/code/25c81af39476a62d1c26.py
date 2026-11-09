import time
import numpy as np
import random

N = 256
MAX_EACH = 64

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    best_A = []
    best_B = []
    best_score = 0
    
    def is_valid(A, B):
        """Verify the sets are valid Sidon sets with disjoint differences."""
        if not A or not B:
            return False
        if len(A) > MAX_EACH or len(B) > MAX_EACH:
            return False
        
        sums_A = set()
        diffs_A = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                s = A[i] + A[j]
                if s in sums_A:
                    return False
                sums_A.add(s)
                if i != j:
                    d = abs(A[i] - A[j])
                    diffs_A.add(d)
        
        sums_B = set()
        diffs_B = set()
        for i in range(len(B)):
            for j in range(i, len(B)):
                s = B[i] + B[j]
                if s in sums_B:
                    return False
                sums_B.add(s)
                if i != j:
                    d = abs(B[i] - B[j])
                    diffs_B.add(d)
        
        if diffs_A & diffs_B:
            return False
            
        return True

    def greedy_build(seed):
        """Run a single randomized greedy construction."""
        rng = random.Random(seed)
        A = []
        B = []
        
        # Precompute constraints for speed
        # sums_A: boolean array of size 2*N+1
        # diffs_A: boolean array of size N
        sums_A = [False] * (2 * N + 1)
        sums_B = [False] * (2 * N + 1)
        diffs_A = [False] * (N + 1)
        diffs_B = [False] * (N + 1)
        
        # Candidate pool
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            
            # Decide which set to try to add to
            # Balance strategy: try to add to the smaller set
            if len(A) < len(B):
                order = [0, 1]
            elif len(B) < len(A):
                order = [1, 0]
            else:
                # Randomly pick if equal
                if rng.random() < 0.5:
                    order = [0, 1]
                else:
                    order = [1, 0]
            
            added = False
            for side in order:
                if side == 0:
                    if len(A) >= MAX_EACH:
                        continue
                    # Check Sidon property for A
                    valid_sum = True
                    for a in A:
                        s = x + a
                        if sums_A[s]:
                            valid_sum = False
                            break
                    if valid_sum and sums_A[2 * x]:
                        valid_sum = False
                    
                    if not valid_sum:
                        continue
                        
                    # Check Difference disjointness
                    valid_diff = True
                    for a in A:
                        d = abs(x - a)
                        if d > 0 and (diffs_A[d] or diffs_B[d]):
                            valid_diff = False
                            break
                    
                    if not valid_diff:
                        continue
                        
                    # Add x to A
                    A.append(x)
                    for a in A:
                        sums_A[x + a] = True
                    sums_A[2 * x] = True
                    for a in A:
                        d = abs(x - a)
                        if d > 0:
                            diffs_A[d] = True
                    added = True
                    break
                else:
                    if len(B) >= MAX_EACH:
                        continue
                    # Check Sidon property for B
                    valid_sum = True
                    for b in B:
                        s = x + b
                        if sums_B[s]:
                            valid_sum = False
                            break
                    if valid_sum and sums_B[2 * x]:
                        valid_sum = False
                        
                    if not valid_sum:
                        continue
                        
                    # Check Difference disjointness
                    valid_diff = True
                    for b in B:
                        d = abs(x - b)
                        if d > 0 and (diffs_A[d] or diffs_B[d]):
                            valid_diff = False
                            break
                            
                    if not valid_diff:
                        continue
                        
                    # Add x to B
                    B.append(x)
                    for b in B:
                        sums_B[x + b] = True
                    sums_B[2 * x] = True
                    for b in B:
                        d = abs(x - b)
                        if d > 0:
                            diffs_B[d] = True
                    added = True
                    break
                    
            # If neither worked, we just skip x (greedy doesn't backtrack)
            
        return A, B

    # Run multiple restarts
    num_restarts = 100
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        A, B = greedy_build(i)
        
        # Verify
        if is_valid(A, B):
            score = len(A) + len(B)
            if score > best_score:
                best_score = score
                best_A = A
                best_B = B
                
    # Fallback
    if not best_A:
        best_A = [1]
        best_B = [3]
        
    return [best_A, best_B]
