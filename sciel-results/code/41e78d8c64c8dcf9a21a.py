# EVOLVE-BLOCK-START
import time
import random

def construct():
    """
    Construct two nonempty Sidon subsets A, B of [1, 256] with at most 64 elements each.
    Pair sums inside each set (including a+a) must be unique.
    Nonzero difference sets of A and B must be disjoint.
    Maximize |A| + |B|.
    """
    start_time = time.time()
    deadline = start_time + 60  # 60 seconds safety margin within 90s limit
    
    n = 256
    candidates = list(range(1, 257))
    
    def fast_greedy(order, prefer_A_first, max_time):
        """
        Fast greedy using incremental sum/diff tracking.
        prefer_A_first: if True, try A before B; else try B before A.
        Returns (A, B, sums_A, diffs_A, sums_B, diffs_B).
        """
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        for x in order:
            if time.time() > max_time:
                break
            
            success = False
            if prefer_A_first:
                # Try A first
                new_diffs = set()
                valid = True
                for a in A:
                    d = abs(a - x)
                    if d > 0:
                        if d in diffs_B:
                            valid = False
                            break
                        new_diffs.add(d)
                if valid:
                    for a in A:
                        s = a + x
                        if s in sums_A:
                            valid = False
                            break
                    if valid and (x + x) in sums_A:
                        valid = False
                if valid:
                    A.append(x)
                    sums_A.add(x + x)
                    for a in A[:-1]:
                        sums_A.add(a + x)
                    diffs_A |= new_diffs
                    success = True
            
            if not success:
                # Try B
                new_diffs = set()
                valid = True
                for b in B:
                    d = abs(b - x)
                    if d > 0:
                        if d in diffs_A:
                            valid = False
                            break
                        new_diffs.add(d)
                if valid:
                    for b in B:
                        s = b + x
                        if s in sums_B:
                            valid = False
                            break
                    if valid and (x + x) in sums_B:
                        valid = False
                if valid:
                    B.append(x)
                    sums_B.add(x + x)
                    for b in B[:-1]:
                        sums_B.add(b + x)
                    diffs_B |= new_diffs
        
        return A, B, sums_A, diffs_A, sums_B, diffs_B
    
    def local_improve(A, B, sums_A, diffs_A, sums_B, diffs_B, max_time):
        """
        Try to add more elements by scanning all candidates multiple times.
        """
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for x in range(1, 257):
                if time.time() > max_time:
                    break
                if x in A or x in B:
                    continue
                
                # Try adding to A
                valid = True
                new_diffs = set()
                for a in A:
                    d = abs(a - x)
                    if d > 0:
                        if d in diffs_B:
                            valid = False
                            break
                        new_diffs.add(d)
                if valid:
                    for a in A:
                        s = a + x
                        if s in sums_A:
                            valid = False
                            break
                    if valid and (x + x) in sums_A:
                        valid = False
                if valid:
                    A.append(x)
                    sums_A.add(x + x)
                    for a in A[:-1]:
                        sums_A.add(a + x)
                    diffs_A |= new_diffs
                    improved = True
                    continue
                
                # Try adding to B
                valid = True
                new_diffs = set()
                for b in B:
                    d = abs(b - x)
                    if d > 0:
                        if d in diffs_A:
                            valid = False
                            break
                        new_diffs.add(d)
                if valid:
                    for b in B:
                        s = b + x
                        if s in sums_B:
                            valid = False
                            break
                    if valid and (x + x) in sums_B:
                        valid = False
                if valid:
                    B.append(x)
                    sums_B.add(x + x)
                    for b in B[:-1]:
                        sums_B.add(b + x)
                    diffs_B |= new_diffs
                    improved = True
        
        return A, B, sums_A, diffs_A, sums_B, diffs_B
    
    best_total = 0
    best_A = []
    best_B = []
    
    # Phase 1: Deterministic greedy with natural order
    A, B, sA, dA, sB, dB = fast_greedy(candidates, True, deadline)
    A, B, sA, dA, sB, dB = local_improve(A, B, sA, dA, sB, dB, deadline)
    total = len(A) + len(B)
    if total > best_total:
        best_total = total
        best_A = A[:]
        best_B = B[:]
    
    # Phase 2: Randomized greedy with various strategies
    random.seed(42)
    
    # Try different number of trials
    num_tries = 200
    
    for trial in range(num_tries):
        if time.time() > deadline:
            break
        
        order = list(range(1, 257))
        random.shuffle(order)
        
        # Alternate which set to prefer
        prefer_A = (trial % 2 == 0)
        
        A, B, sA, dA, sB, dB = fast_greedy(order, prefer_A, deadline)
        
        # Local improvement
        A, B, sA, dA, sB, dB = local_improve(A, B, sA, dA, sB, dB, deadline)
        
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_A = A[:]
            best_B = B[:]
    
    # Phase 3: Try sorted variants (ascending, descending, interleaved)
    for variant in [list(range(1, 257)), list(range(256, 0, -1))]:
        if time.time() > deadline:
            break
        for prefer_A in [True, False]:
            if time.time() > deadline:
                break
            A, B, sA, dA, sB, dB = fast_greedy(variant, prefer_A, deadline)
            A, B, sA, dA, sB, dB = local_improve(A, B, sA, dA, sB, dB, deadline)
            total = len(A) + len(B)
            if total > best_total:
                best_total = total
                best_A = A[:]
                best_B = B[:]
    
    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
