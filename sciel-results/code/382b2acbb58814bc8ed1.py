import random
from itertools import combinations, combinations_with_replacement

def construct():
    best_A = []
    best_B = []
    best_size = 0
    
    # Run multiple random restarts
    for _ in range(2000):
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        for x in candidates:
            # Try to add x to A
            can_add_A = True
            # Check Sidon for A
            for a in A:
                if a + x in sums_A:
                    can_add_A = False
                    break
            if can_add_A:
                if x + x in sums_A:
                    can_add_A = False
            
            # Check difference disjointness for A
            if can_add_A:
                for a in A:
                    if x - a in diffs_B or a - x in diffs_B:
                        can_add_A = False
                        break
            
            # Try to add x to B
            can_add_B = True
            for b in B:
                if b + x in sums_B:
                    can_add_B = False
                    break
            if can_add_B:
                if x + x in sums_B:
                    can_add_B = False
            
            if can_add_B:
                for b in B:
                    if x - b in diffs_A or b - x in diffs_A:
                        can_add_B = False
                        break
            
            # Decision logic
            # Prefer adding to the smaller set to balance sizes
            if can_add_A and can_add_B:
                if len(A) <= len(B):
                    # Add to A
                    A.append(x)
                    for a in A[:-1]: # A[:-1] is old A
                        sums_A.add(a + x)
                        diffs_A.add(x - a)
                        diffs_A.add(a - x)
                    sums_A.add(2 * x)
                else:
                    # Add to B
                    B.append(x)
                    for b in B[:-1]:
                        sums_B.add(b + x)
                        diffs_B.add(x - b)
                        diffs_B.add(b - x)
                    sums_B.add(2 * x)
            elif can_add_A:
                A.append(x)
                for a in A[:-1]:
                    sums_A.add(a + x)
                    diffs_A.add(x - a)
                    diffs_A.add(a - x)
                sums_A.add(2 * x)
            elif can_add_B:
                B.append(x)
                for b in B[:-1]:
                    sums_B.add(b + x)
                    diffs_B.add(x - b)
                    diffs_B.add(b - x)
                sums_B.add(2 * x)
        
        current_size = len(A) + len(B)
        if current_size > best_size:
            best_size = current_size
            best_A = A[:]
            best_B = B[:]
            
        # Early termination if we hit the theoretical max or a high bound
        if best_size >= 40: 
            break

    # Ensure non-empty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
        
    return [best_A, best_B]
