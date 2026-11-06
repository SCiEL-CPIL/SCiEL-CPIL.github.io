import time
import random

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin for 90s limit
    
    n = 256
    best_A = [1]
    best_B = [2]
    best_size = 2
    
    def greedy(order):
        A = []
        B = []
        D_A = set()
        D_B = set()
        S_A = set() # Set of all pair sums a_i + a_j (i<=j)
        S_B = set()
        
        for x in order:
            canA = False
            canB = False
            
            # Check if x can be added to A
            # 1. Sidon Check: New sums must not intersect existing sums
            new_sums_A = {x + x}
            for s in A:
                new_sums_A.add(x + s)
            
            if not new_sums_A & S_A:
                # 2. Difference Disjointness Check: New diffs must not intersect D_B
                new_diffs_A = set()
                for s in A:
                    new_diffs_A.add(x - s)
                    new_diffs_A.add(s - x)
                
                if not new_diffs_A & D_B:
                    canA = True
            
            # Check if x can be added to B
            new_sums_B = {x + x}
            for s in B:
                new_sums_B.add(x + s)
            
            if not new_sums_B & S_B:
                new_diffs_B = set()
                for s in B:
                    new_diffs_B.add(x - s)
                    new_diffs_B.add(s - x)
                
                if not new_diffs_B & D_A:
                    canB = True
            
            # Decision
            if canA and canB:
                # Balance the sets
                if len(A) <= len(B):
                    A.append(x)
                    S_A.update(new_sums_A)
                    D_A.update(new_diffs_A)
                else:
                    B.append(x)
                    S_B.update(new_sums_B)
                    D_B.update(new_diffs_B)
            elif canA:
                A.append(x)
                S_A.update(new_sums_A)
                D_A.update(new_diffs_A)
            elif canB:
                B.append(x)
                S_B.update(new_sums_B)
                D_B.update(new_diffs_B)
                
        # Ensure non-empty
        if not A:
            A = [1]
            S_A = {2}
        if not B:
            B = [2]
            S_B = {4}
            
        return A, B

    # 1. Try natural order
    A, B = greedy(list(range(1, n + 1)))
    best_A, best_B = A, B
    best_size = len(A) + len(B)
    
    # 2. Try random permutations
    # We can afford many iterations
    while time.time() - start_time < time_limit:
        order = list(range(1, n + 1))
        random.shuffle(order)
        A, B = greedy(order)
        size = len(A) + len(B)
        if size > best_size:
            best_A, best_B = A, B
            best_size = size

    return [best_A, best_B]
