# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    timeout = 50.0  # Safety margin for 60s limit
    
    best_size = 0
    best_A = []
    best_B = []
    
    candidates = list(range(1, 257))
    
    def run_greedy(order, prefer_smaller=False, prefer_A_first=True):
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        for x in order:
            # Check if x can be added to A
            valid_A = True
            # Check sums: x+a for a in A, and 2x
            for a in A:
                if (x + a) in sums_A:
                    valid_A = False
                    break
            if valid_A:
                if (2 * x) in sums_A:
                    valid_A = False
            
            # Check diffs: |x-a| for a in A must not be in diffs_B
            if valid_A:
                for a in A:
                    d = abs(x - a)
                    if d in diffs_B:
                        valid_A = False
                        break
            
            # Check if x can be added to B
            valid_B = True
            for b in B:
                if (x + b) in sums_B:
                    valid_B = False
                    break
            if valid_B:
                if (2 * x) in sums_B:
                    valid_B = False
            
            if valid_B:
                for b in B:
                    d = abs(x - b)
                    if d in diffs_A:
                        valid_B = False
                        break
            
            # Decision logic
            if valid_A and valid_B:
                if prefer_smaller:
                    if len(A) < len(B):
                        add_to_A = True
                    elif len(B) < len(A):
                        add_to_A = False
                    else:
                        # Tie-breaker: alternate or random? Let's say prefer A if prefer_A_first else random
                        add_to_A = prefer_A_first if prefer_A_first is not None else random.choice([True, False])
                else:
                    # Original logic: prefer A
                    add_to_A = True
            elif valid_A:
                add_to_A = True
            elif valid_B:
                add_to_A = False
            else:
                continue
            
            if add_to_A:
                A.append(x)
                for a in A[:-1]:
                    sums_A.add(x + a)
                    diffs_A.add(abs(x - a))
                sums_A.add(2 * x)
            else:
                B.append(x)
                for b in B[:-1]:
                    sums_B.add(x + b)
                    diffs_B.add(abs(x - b))
                sums_B.add(2 * x)
        
        return A, B

    # Strategy 1: Deterministic natural order
    A, B = run_greedy(candidates, prefer_smaller=False, prefer_A_first=True)
    size = len(A) + len(B)
    if size > best_size:
        best_size = size
        best_A = A
        best_B = B
        
    # Strategy 2: Deterministic reverse order
    A, B = run_greedy(reversed(candidates), prefer_smaller=False, prefer_A_first=True)
    size = len(A) + len(B)
    if size > best_size:
        best_size = size
        best_A = A
        best_B = B

    # Strategy 3: Random shuffles with different heuristics
    random.seed(42)
    
    # Run multiple random trials
    while time.time() - start_time < timeout:
        order = list(candidates)
        random.shuffle(order)
        
        # Alternate between heuristics
        if random.random() < 0.5:
            A, B = run_greedy(order, prefer_smaller=False, prefer_A_first=True)
        else:
            A, B = run_greedy(order, prefer_smaller=True, prefer_A_first=random.choice([True, False, None]))
            
        size = len(A) + len(B)
        if size > best_size:
            best_size = size
            best_A = A
            best_B = B

    # Ensure non-empty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
        
    return [best_A, best_B]
# EVOLVE-BLOCK-END
