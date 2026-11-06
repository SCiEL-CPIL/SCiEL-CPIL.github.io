# EVOLVE-BLOCK-START
import time
import random

def _diffs(lst):
    """Compute nonzero difference set of lst."""
    d = set()
    for i in range(len(lst)):
        for j in range(len(lst)):
            if i != j:
                d.add(lst[j] - lst[i])
    return d

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit
    
    n = 256
    
    def run_greedy(order):
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        for x in order:
            if time.time() - start_time > time_limit:
                break
            
            # Try adding to A
            can_add_A = True
            for a in A:
                if x + a in sums_A:
                    can_add_A = False
                    break
            if can_add_A and x + x in sums_A:
                can_add_A = False
            if can_add_A:
                new_diffs = set()
                for a in A:
                    new_diffs.add(x - a)
                    new_diffs.add(a - x)
                if new_diffs & diffs_B:
                    can_add_A = False
            
            # Try adding to B
            can_add_B = True
            for b in B:
                if x + b in sums_B:
                    can_add_B = False
                    break
            if can_add_B and x + x in sums_B:
                can_add_B = False
            if can_add_B:
                new_diffs = set()
                for b in B:
                    new_diffs.add(x - b)
                    new_diffs.add(b - x)
                if new_diffs & diffs_A:
                    can_add_B = False
            
            # Choose the set with fewer elements (balance), or the one that allows addition
            if can_add_A and can_add_B:
                if len(A) <= len(B):
                    A.append(x)
                    sums_A.add(x + x)
                    for a in A[:-1]:
                        sums_A.add(x + a)
                    for a in A[:-1]:
                        diffs_A.add(x - a)
                        diffs_A.add(a - x)
                else:
                    B.append(x)
                    sums_B.add(x + x)
                    for b in B[:-1]:
                        sums_B.add(x + b)
                    for b in B[:-1]:
                        diffs_B.add(x - b)
                        diffs_B.add(b - x)
            elif can_add_A:
                A.append(x)
                sums_A.add(x + x)
                for a in A[:-1]:
                    sums_A.add(x + a)
                for a in A[:-1]:
                    diffs_A.add(x - a)
                    diffs_A.add(a - x)
            elif can_add_B:
                B.append(x)
                sums_B.add(x + x)
                for b in B[:-1]:
                    sums_B.add(x + b)
                for b in B[:-1]:
                    diffs_B.add(x - b)
                    diffs_B.add(b - x)
        
        return A, B
    
    # Generate different orderings and take the best
    best_size = 0
    best_A = []
    best_B = []
    
    # First: natural order
    A, B = run_greedy(range(1, n + 1))
    size = len(A) + len(B)
    if size > best_size:
        best_size = size
        best_A = A
        best_B = B
    
    if time.time() - start_time > time_limit - 5:
        return [best_A, best_B]
    
    # Try reversed order
    A, B = run_greedy(range(n, 0, -1))
    size = len(A) + len(B)
    if size > best_size:
        best_size = size
        best_A = A
        best_B = B
    
    if time.time() - start_time > time_limit - 5:
        return [best_A, best_B]
    
    # Try several random orderings
    random.seed(42)
    for trial in range(100):
        if time.time() - start_time > time_limit - 5:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        A, B = run_greedy(order)
        size = len(A) + len(B)
        if size > best_size:
            best_size = size
            best_A = A
            best_B = B
    
    # Local improvement: try to add elements that were skipped
    A, B = best_A, best_B
    if time.time() - start_time > time_limit - 3:
        return [A, B]
    
    sums_A = set()
    for i in range(len(A)):
        for j in range(i, len(A)):
            sums_A.add(A[i] + A[j])
    sums_B = set()
    for i in range(len(B)):
        for j in range(i, len(B)):
            sums_B.add(B[i] + B[j])
    diffs_A = _diffs(A)
    diffs_B = _diffs(B)
    
    used = set(A) | set(B)
    
    # Try adding unused elements
    for x in range(1, n + 1):
        if time.time() - start_time > time_limit - 1:
            break
        if x in used:
            continue
        
        # Try adding to A
        ok_A = True
        for a in A:
            if x + a in sums_A:
                ok_A = False
                break
        if ok_A and x + x in sums_A:
            ok_A = False
        if ok_A:
            new_d = set()
            for a in A:
                new_d.add(x - a)
                new_d.add(a - x)
            if new_d & diffs_B:
                ok_A = False
        
        # Try adding to B
        ok_B = True
        for b in B:
            if x + b in sums_B:
                ok_B = False
                break
        if ok_B and x + x in sums_B:
            ok_B = False
        if ok_B:
            new_d = set()
            for b in B:
                new_d.add(x - b)
                new_d.add(b - x)
            if new_d & diffs_A:
                ok_B = False
        
        if ok_A and not ok_B:
            A.append(x)
            sums_A.add(x + x)
            for a in A[:-1]:
                sums_A.add(x + a)
            for a in A[:-1]:
                diffs_A.add(x - a)
                diffs_A.add(a - x)
            used.add(x)
        elif ok_B and not ok_A:
            B.append(x)
            sums_B.add(x + x)
            for b in B[:-1]:
                sums_B.add(x + b)
            for b in B[:-1]:
                diffs_B.add(x - b)
                diffs_B.add(b - x)
            used.add(x)
        elif ok_A and ok_B:
            # Add to smaller set
            if len(A) <= len(B):
                A.append(x)
                sums_A.add(x + x)
                for a in A[:-1]:
                    sums_A.add(x + a)
                for a in A[:-1]:
                    diffs_A.add(x - a)
                    diffs_A.add(a - x)
            else:
                B.append(x)
                sums_B.add(x + x)
                for b in B[:-1]:
                    sums_B.add(x + b)
                for b in B[:-1]:
                    diffs_B.add(x - b)
                    diffs_B.add(b - x)
            used.add(x)
    
    A.sort()
    B.sort()
    return [A, B]

# EVOLVE-BLOCK-END
