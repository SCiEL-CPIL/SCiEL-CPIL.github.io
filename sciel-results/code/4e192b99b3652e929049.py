import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin for 90s limit
    
    n = 256
    best_size = 0
    best_A = []
    best_B = []
    
    def build_greedy(order):
        A = []
        B = []
        sums_A = set()
        diffs_A = set()
        sums_B = set()
        diffs_B = set()
        
        for x in order:
            # Try adding to A
            valid_A = True
            # Check sums
            if 2*x in sums_A:
                valid_A = False
            else:
                for a in A:
                    if (x + a) in sums_A:
                        valid_A = False
                        break
            
            # Check diffs
            if valid_A:
                for a in A:
                    d = abs(x - a)
                    if d in diffs_A or d in diffs_B:
                        valid_A = False
                        break
            
            # Try adding to B
            valid_B = True
            if 2*x in sums_B:
                valid_B = False
            else:
                for b in B:
                    if (x + b) in sums_B:
                        valid_B = False
                        break
            
            if valid_B:
                for b in B:
                    d = abs(x - b)
                    if d in diffs_B or d in diffs_A:
                        valid_B = False
                        break
            
            if valid_A and valid_B:
                # Heuristic: add to the set with fewer elements to balance
                if len(A) <= len(B):
                    # Add to A
                    A.append(x)
                    for a in A[:-1]:
                        sums_A.add(x + a)
                        diffs_A.add(abs(x - a))
                    sums_A.add(2*x)
                else:
                    # Add to B
                    B.append(x)
                    for b in B[:-1]:
                        sums_B.add(x + b)
                        diffs_B.add(abs(x - b))
                    sums_B.add(2*x)
            elif valid_A:
                A.append(x)
                for a in A[:-1]:
                    sums_A.add(x + a)
                    diffs_A.add(abs(x - a))
                sums_A.add(2*x)
            elif valid_B:
                B.append(x)
                for b in B[:-1]:
                    sums_B.add(x + b)
                    diffs_B.add(abs(x - b))
                sums_B.add(2*x)
            # else: skip x
            
        return A, B

    # Main loop
    # Run a few deterministic orders first
    orders = [
        list(range(1, n + 1)),
        list(range(n, 0, -1)),
    ]
    
    for order in orders:
        A, B = build_greedy(order)
        size = len(A) + len(B)
        if size > best_size:
            best_size = size
            best_A = A
            best_B = B

    # Then random orders
    while time.time() < deadline:
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        A, B = build_greedy(order)
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
