# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit (evaluator kills at 90s wall-clock)
    
    best_size = 0
    best_A = []
    best_B = []
    
    candidates = list(range(1, 257))
    
    def can_add_to_A(x, A, sums_A, diffs_A, diffs_B):
        for a in A:
            if (x + a) in sums_A:
                return False
        if (2 * x) in sums_A:
            return False
        for a in A:
            d = abs(x - a)
            if d in diffs_B:
                return False
        return True
    
    def can_add_to_B(x, B, sums_B, diffs_A, diffs_B):
        for b in B:
            if (x + b) in sums_B:
                return False
        if (2 * x) in sums_B:
            return False
        for b in B:
            d = abs(x - b)
            if d in diffs_A:
                return False
        return True
    
    def run_greedy(order, prefer_smaller=False, prefer_A_first=True):
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        for x in order:
            valid_A = can_add_to_A(x, A, sums_A, diffs_A, diffs_B)
            valid_B = can_add_to_B(x, B, sums_B, diffs_A, diffs_B)
            
            if valid_A and valid_B:
                if prefer_smaller:
                    if len(A) < len(B):
                        add_to_A = True
                    elif len(B) < len(A):
                        add_to_A = False
                    else:
                        add_to_A = prefer_A_first if prefer_A_first is not None else random.choice([True, False])
                else:
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
    
    def local_search(A, B, max_rounds=3):
        """Try to improve by swapping or moving elements"""
        A = list(A)
        B = list(B)
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        for a in A:
            for a2 in A:
                if a2 >= a:
                    sums_A.add(a + a2)
        for a in A:
            for a2 in A:
                if a2 != a:
                    diffs_A.add(abs(a - a2))
        for b in B:
            for b2 in B:
                if b2 >= b:
                    sums_B.add(b + b2)
        for b in B:
            for b2 in B:
                if b2 != b:
                    diffs_B.add(abs(b - b2))
        
        for _ in range(max_rounds):
            improved = False
            
            # Try adding unassigned elements
            assigned = set(A) | set(B)
            for x in candidates:
                if x in assigned:
                    continue
                if can_add_to_A(x, A, sums_A, diffs_A, diffs_B):
                    A.append(x)
                    for a in A[:-1]:
                        sums_A.add(x + a)
                        diffs_A.add(abs(x - a))
                    sums_A.add(2 * x)
                    improved = True
                    break
                elif can_add_to_B(x, B, sums_B, diffs_A, diffs_B):
                    B.append(x)
                    for b in B[:-1]:
                        sums_B.add(x + b)
                        diffs_B.add(abs(x - b))
                    sums_B.add(2 * x)
                    improved = True
                    break
            
            if not improved:
                break
        
        return A, B
    
    # Strategy 1: Deterministic natural order with smaller-set preference
    A, B = run_greedy(candidates, prefer_smaller=True, prefer_A_first=True)
    A, B = local_search(A, B)
    size = len(A) + len(B)
    if size > best_size:
        best_size = size
        best_A = A
        best_B = B
        
    # Strategy 2: Deterministic reverse order with smaller-set preference
    A, B = run_greedy(list(reversed(candidates)), prefer_smaller=True, prefer_A_first=False)
    A, B = local_search(A, B)
    size = len(A) + len(B)
    if size > best_size:
        best_size = size
        best_A = A
        best_B = B
        
    # Strategy 3: Random shuffles with different heuristics
    random.seed(42)
    
    trial_count = 0
    while time.time() < deadline:
        order = list(candidates)
        random.shuffle(order)
        
        trial_count += 1
        # Vary heuristics based on trial number
        if trial_count % 3 == 0:
            A, B = run_greedy(order, prefer_smaller=True, prefer_A_first=True)
        elif trial_count % 3 == 1:
            A, B = run_greedy(order, prefer_smaller=True, prefer_A_first=False)
        else:
            A, B = run_greedy(order, prefer_smaller=False, prefer_A_first=True)
        
        # Apply local search
        A, B = local_search(A, B)
        
        size = len(A) + len(B)
        if size > best_size:
            best_size = size
            best_A = A
            best_B = B
        
        # Check time more frequently
        if time.time() > deadline - 5:
            break

    # Ensure non-empty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
        
    return [best_A, best_B]
# EVOLVE-BLOCK-END
