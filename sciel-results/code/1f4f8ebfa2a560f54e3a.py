import random
import sys
from itertools import combinations

def construct():
    n = 256
    best_sum = 0
    best_A = []
    best_B = []
    
    # We know max size is around 16 for each.
    # Let's try to find a configuration with |A|=16, |B|=16.
    
    # Helper to check if a set is Sidon
    def is_sidon(S):
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def get_diffs(S):
        diffs = set()
        for i in range(len(S)):
            for j in range(len(S)):
                if i != j:
                    d = S[i] - S[j]
                    if d != 0:
                        diffs.add(d)
        return diffs

    # Strategy: Randomized Greedy with Backtracking for A, then Greedy for B
    # We will run multiple iterations to find a good A that allows a large B.
    
    for _ in range(500): # 500 restarts
        A = []
        sums_A = set()
        diffs_A = set()
        
        # Try to build A of size 16
        # Use a heuristic: pick random candidates, check validity
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if len(A) >= 16:
                break
            
            # Check if x can be added to A
            valid = True
            for a in A:
                if a + x in sums_A:
                    valid = False
                    break
            if valid and 2*x in sums_A:
                valid = False
            
            if valid:
                A.append(x)
                for a in A[:-1]:
                    sums_A.add(a + x)
                    diffs_A.add(x - a)
                    diffs_A.add(a - x)
                sums_A.add(2 * x)
        
        if len(A) < 16:
            continue
            
        # Now try to build B
        B = []
        sums_B = set()
        diffs_B = set()
        
        # B must be Sidon and Diff(B) disjoint from Diff(A)
        # Sort candidates by how many conflicts they have with Diff(A)?
        # Or just random shuffle?
        candidates_B = list(range(1, n + 1))
        random.shuffle(candidates_B)
        
        for x in candidates_B:
            # Check Sidon for B
            valid_sidon = True
            for b in B:
                if b + x in sums_B:
                    valid_sidon = False
                    break
            if valid_sidon and 2*x in sums_B:
                valid_sidon = False
                
            if not valid_sidon:
                continue
                
            # Check Diff disjointness
            valid_diff = True
            for b in B:
                d1 = x - b
                d2 = b - x
                if d1 in diffs_A or d2 in diffs_A:
                    valid_diff = False
                    break
            # Also check against existing diffs in B? 
            # Since B is Sidon, diffs within B are unique, but we need to ensure 
            # new diffs don't clash with existing diffs in B? 
            # No, Diff(B) is the set of all diffs. 
            # If we add x, we add new diffs. 
            # We must ensure new diffs are not in Diff(A) AND not in existing Diff(B)?
            # Wait, Diff(B) is defined as the set of differences. 
            # If B is Sidon, all non-zero diffs in B are distinct. 
            # So we just need to ensure the new diffs are not in Diff(A). 
            # The existing diffs in B are already disjoint from Diff(A) by construction.
            # And they are distinct among themselves.
            
            if valid_diff:
                B.append(x)
                for b in B[:-1]:
                    sums_B.add(b + x)
                    diffs_B.add(x - b)
                    diffs_B.add(b - x)
                sums_B.add(2 * x)
                
                # Early stop if we hit 16
                if len(B) >= 16:
                    break
        
        current_sum = len(A) + len(B)
        if current_sum > best_sum:
            best_sum = current_sum
            best_A = A[:]
            best_B = B[:]
            
        if best_sum >= 32: # Theoretical max
            break

    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
        
    return [best_A, best_B]
