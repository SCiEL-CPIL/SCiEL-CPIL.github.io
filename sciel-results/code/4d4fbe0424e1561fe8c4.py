import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    n = 256
    max_size = 64
    
    def can_add_to_A(x, A, sums_A, diffs_B):
        if x + x in sums_A:
            return False
        for a in A:
            if x + a in sums_A:
                return False
            if x - a in diffs_B:
                return False
            if a - x in diffs_B:
                return False
        return True

    def can_add_to_B(x, B, sums_B, diffs_A):
        if x + x in sums_B:
            return False
        for b in B:
            if x + b in sums_B:
                return False
            if x - b in diffs_A:
                return False
            if b - x in diffs_A:
                return False
        return True

    def get_state(A, B):
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums_A.add(A[i] + A[j])
        for i in range(len(A)):
            for j in range(len(A)):
                if i != j:
                    diffs_A.add(A[j] - A[i])
                    
        for i in range(len(B)):
            for j in range(i, len(B)):
                sums_B.add(B[i] + B[j])
        for i in range(len(B)):
            for j in range(len(B)):
                if i != j:
                    diffs_B.add(B[j] - B[i])
                    
        return sums_A, sums_B, diffs_A, diffs_B

    def greedy_construct(order):
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        for x in order:
            if time.time() > deadline:
                break
            if len(A) >= max_size and len(B) >= max_size:
                break
                
            # Try A
            if len(A) < max_size and can_add_to_A(x, A, sums_A, diffs_B):
                A.append(x)
                for a in A[:-1]:
                    sums_A.add(x + a)
                    diffs_A.add(x - a)
                    diffs_A.add(a - x)
                sums_A.add(x + x)
            # Try B
            elif len(B) < max_size and can_add_to_B(x, B, sums_B, diffs_A):
                B.append(x)
                for b in B[:-1]:
                    sums_B.add(x + b)
                    diffs_B.add(x - b)
                    diffs_B.add(b - x)
                sums_B.add(x + x)
                
        return A, B, sums_A, sums_B, diffs_A, diffs_B

    def local_search(A, B, sums_A, sums_B, diffs_A, diffs_B):
        in_A = set(A)
        in_B = set(B)
        
        # We will try to improve the solution.
        # Priority: 1. Add new elements. 2. Swap/Move to allow adding.
        
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            
            # 1. Try to add any unused element
            unused = [x for x in range(1, n + 1) if x not in in_A and x not in in_B]
            random.shuffle(unused)
            
            for x in unused:
                if time.time() > deadline - 1:
                    break
                    
                # Try add to A
                if len(A) < max_size and can_add_to_A(x, A, sums_A, diffs_B):
                    A.append(x)
                    for a in A[:-1]:
                        sums_A.add(x + a)
                        diffs_A.add(x - a)
                        diffs_A.add(a - x)
                    sums_A.add(x + x)
                    in_A.add(x)
                    improved = True
                    break # Restart loop with new state
                    
                # Try add to B
                if len(B) < max_size and can_add_to_B(x, B, sums_B, diffs_A):
                    B.append(x)
                    for b in B[:-1]:
                        sums_B.add(x + b)
                        diffs_B.add(x - b)
                        diffs_B.add(b - x)
                    sums_B.add(x + x)
                    in_B.add(x)
                    improved = True
                    break # Restart loop with new state
            
            if improved:
                continue
            
            # 2. If no direct addition, try swaps or moves to free up space
            # Try a limited number of random swaps/moves
            attempts = 0
            while attempts < 20 and time.time() < deadline - 1:
                attempts += 1
                
                # Random Swap: remove a from A and b from B, then try to add them back in different sets
                if random.random() < 0.5 and len(A) > 0 and len(B) > 0:
                    a_idx = random.randint(0, len(A) - 1)
                    b_idx = random.randint(0, len(B) - 1)
                    a_val = A[a_idx]
                    b_val = B[b_idx]
                    
                    # Save original state
                    A_orig = A[:]
                    B_orig = B[:]
                    sums_A_orig = sums_A.copy()
                    sums_B_orig = sums_B.copy()
                    diffs_A_orig = diffs_A.copy()
                    diffs_B_orig = diffs_B.copy()
                    
                    # Remove a from A
                    A_temp = A[:a_idx] + A[a_idx+1:]
                    # Remove b from B
                    B_temp = B[:b_idx] + B[b_idx+1:]
                    
                    # Recompute state for temp sets (expensive but only for 20 attempts)
                    sums_A_temp, sums_B_temp, diffs_A_temp, diffs_B_temp = get_state(A_temp, B_temp)
                    
                    # Try to add a_val to B_temp
                    if can_add_to_B(a_val, B_temp, sums_B_temp, diffs_A_temp):
                        B_temp.append(a_val)
                        for b in B_temp[:-1]:
                            sums_B_temp.add(a_val + b)
                            diffs_B_temp.add(a_val - b)
                            diffs_B_temp.add(b - a_val)
                        sums_B_temp.add(a_val + a_val)
                        
                        # Try to add b_val to A_temp
                        if can_add_to_A(b_val, A_temp, sums_A_temp, diffs_B_temp):
                            A_temp.append(b_val)
                            for a in A_temp[:-1]:
                                sums_A_temp.add(b_val + a)
                                diffs_A_temp.add(b_val - a)
                                diffs_A_temp.add(a - b_val)
                            sums_A_temp.add(b_val + b_val)
                            
                            # Accept the swap
                            A = A_temp
                            B = B_temp
                            sums_A = sums_A_temp
                            sums_B = sums_B_temp
                            diffs_A = diffs_A_temp
                            diffs_B = diffs_B_temp
                            in_A = set(A)
                            in_B = set(B)
                            improved = True
                            break
                    else:
                        # Restore original state
                        A = A_orig
                        B = B_orig
                        sums_A = sums_A_orig
                        sums_B = sums_B_orig
                        diffs_A = diffs_A_orig
                        diffs_B = diffs_B_orig
                        in_A = set(A)
                        in_B = set(B)
                else:
                    # Random Move: move an element from A to B or B to A
                    if len(A) > 0 and len(B) < max_size:
                        a_idx = random.randint(0, len(A) - 1)
                        a_val = A[a_idx]
                        
                        # Save original state
                        A_orig = A[:]
                        B_orig = B[:]
                        sums_A_orig = sums_A.copy()
                        sums_B_orig = sums_B.copy()
                        diffs_A_orig = diffs_A.copy()
                        diffs_B_orig = diffs_B.copy()
                        
                        # Remove a_val from A
                        A_temp = A[:a_idx] + A[a_idx+1:]
                        B_temp = B[:]
                        
                        # Recompute state
                        sums_A_temp, sums_B_temp, diffs_A_temp, diffs_B_temp = get_state(A_temp, B_temp)
                        
                        # Try to add a_val to B_temp
                        if can_add_to_B(a_val, B_temp, sums_B_temp, diffs_A_temp):
                            B_temp.append(a_val)
                            for b in B_temp[:-1]:
                                sums_B_temp.add(a_val + b)
                                diffs_B_temp.add(a_val - b)
                                diffs_B_temp.add(b - a_val)
                            sums_B_temp.add(a_val + a_val)
                            
                            # Accept the move
                            A = A_temp
                            B = B_temp
                            sums_A = sums_A_temp
                            sums_B = sums_B_temp
                            diffs_A = diffs_A_temp
                            diffs_B = diffs_B_temp
                            in_A = set(A)
                            in_B = set(B)
                            improved = True
                            break
                    elif len(B) > 0 and len(A) < max_size:
                        b_idx = random.randint(0, len(B) - 1)
                        b_val = B[b_idx]
                        
                        # Save original state
                        A_orig = A[:]
                        B_orig = B[:]
                        sums_A_orig = sums_A.copy()
                        sums_B_orig = sums_B.copy()
                        diffs_A_orig = diffs_A.copy()
                        diffs_B_orig = diffs_B.copy()
                        
                        # Remove b_val from B
                        B_temp = B[:b_idx] + B[b_idx+1:]
                        A_temp = A[:]
                        
                        # Recompute state
                        sums_A_temp, sums_B_temp, diffs_A_temp, diffs_B_temp = get_state(A_temp, B_temp)
                        
                        # Try to add b_val to A_temp
                        if can_add_to_A(b_val, A_temp, sums_A_temp, diffs_B_temp):
                            A_temp.append(b_val)
                            for a in A_temp[:-1]:
                                sums_A_temp.add(b_val + a)
                                diffs_A_temp.add(b_val - a)
                                diffs_A_temp.add(a - b_val)
                            sums_A_temp.add(b_val + b_val)
                            
                            # Accept the move
                            A = A_temp
                            B = B_temp
                            sums_A = sums_A_temp
                            sums_B = sums_B_temp
                            diffs_A = diffs_A_temp
                            diffs_B = diffs_B_temp
                            in_A = set(A)
                            in_B = set(B)
                            improved = True
                            break

        return A, B, sums_A, sums_B, diffs_A, diffs_B

    best_A, best_B = [], []
    best_score = 0
    
    # Use multiple ordering strategies
    strategies = []
    
    # Strategy 1: Random
    strategies.append(lambda: list(range(1, n + 1)))
    
    # Strategy 2: Ascending
    strategies.append(lambda: list(range(1, n + 1)))
    
    # Strategy 3: Descending
    strategies.append(lambda: list(range(n, 0, -1)))
    
    # Strategy 4: Interleaved (odd then even)
    strategies.append(lambda: [x for x in range(1, n + 1, 2)] + [x for x in range(2, n + 1, 2)])
    
    # Strategy 5: Reverse interleaved
    strategies.append(lambda: [x for x in range(2, n + 1, 2)] + [x for x in range(1, n + 1, 2)])
    
    num_trials = 200
    
    for trial in range(num_trials):
        if time.time() > deadline - 2:
            break
        
        # Choose strategy
        strat_idx = trial % len(strategies)
        order = strategies[strat_idx]()
        
        # Shuffle if not deterministic strategy
        if strat_idx == 0:
            random.shuffle(order)
        
        A, B, sums_A, sums_B, diffs_A, diffs_B = greedy_construct(order)
        
        # Polish
        A, B, sums_A, sums_B, diffs_A, diffs_B = local_search(A, B, sums_A, sums_B, diffs_A, diffs_B)
        
        score = len(A) + len(B)
        if score > best_score:
            best_score = score
            best_A, best_B = A[:], B[:]
            
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
        
    return [best_A, best_B]
