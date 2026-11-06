import time
import random

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit
    
    n = 256
    best_size = 0
    best_A = []
    best_B = []
    
    def try_add(x, A, sums_A, diffs_A, diffs_B):
        for a in A:
            s = x + a
            if s in sums_A:
                return False
            d = abs(x - a)
            if d in diffs_A:
                return False
            if d in diffs_B:
                return False
        return True

    def try_add_B(x, B, sums_B, diffs_B, diffs_A):
        for b in B:
            s = x + b
            if s in sums_B:
                return False
            d = abs(x - b)
            if d in diffs_B:
                return False
            if d in diffs_A:
                return False
        return True

    def update_state(x, A, sums_A, diffs_A):
        for a in A:
            sums_A.add(x + a)
            diffs_A.add(abs(x - a))
        A.append(x)

    def rebuild_state(A):
        sums = set()
        diffs = set()
        for i in range(len(A)):
            for j in range(len(A)):
                if i <= j:
                    sums.add(A[i] + A[j])
                if i < j:
                    diffs.add(A[j] - A[i])
        return sums, diffs

    def run_greedy(order, strategy):
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        for x in order:
            if strategy == 'A_first':
                if try_add(x, A, sums_A, diffs_A, diffs_B):
                    update_state(x, A, sums_A, diffs_A)
                elif try_add_B(x, B, sums_B, diffs_B, diffs_A):
                    update_state(x, B, sums_B, diffs_B)
            elif strategy == 'B_first':
                if try_add_B(x, B, sums_B, diffs_B, diffs_A):
                    update_state(x, B, sums_B, diffs_B)
                elif try_add(x, A, sums_A, diffs_A, diffs_B):
                    update_state(x, A, sums_A, diffs_A)
            else: # balanced
                if len(A) <= len(B):
                    if try_add(x, A, sums_A, diffs_A, diffs_B):
                        update_state(x, A, sums_A, diffs_A)
                    elif try_add_B(x, B, sums_B, diffs_B, diffs_A):
                        update_state(x, B, sums_B, diffs_B)
                else:
                    if try_add_B(x, B, sums_B, diffs_B, diffs_A):
                        update_state(x, B, sums_B, diffs_B)
                    elif try_add(x, A, sums_A, diffs_A, diffs_B):
                        update_state(x, A, sums_A, diffs_A)
                        
        return A, B

    def local_search(A, B):
        A = list(A)
        B = list(B)
        sums_A, diffs_A = rebuild_state(A)
        sums_B, diffs_B = rebuild_state(B)
        
        used = set(A) | set(B)
        candidates = [x for x in range(1, n+1) if x not in used]
        
        improved = True
        # Limit iterations to prevent long stalls in a single local search
        max_iters = 1000
        iters = 0
        
        while improved and iters < max_iters:
            iters += 1
            improved = False
            
            # 1. Try to add any candidate
            for i in range(len(candidates)):
                x = candidates[i]
                if try_add(x, A, sums_A, diffs_A, diffs_B):
                    update_state(x, A, sums_A, diffs_A)
                    used.add(x)
                    candidates.pop(i)
                    improved = True
                    break
                elif try_add_B(x, B, sums_B, diffs_B, diffs_A):
                    update_state(x, B, sums_B, diffs_B)
                    used.add(x)
                    candidates.pop(i)
                    improved = True
                    break
            
            # 2. If no addition, try swaps
            if not improved:
                # Try swapping an element from A
                for i in range(len(A)):
                    a = A[i]
                    A.remove(a)
                    used.remove(a)
                    sums_A_new, diffs_A_new = rebuild_state(A)
                    
                    success = False
                    for j in range(len(candidates)):
                        x = candidates[j]
                        if try_add(x, A, sums_A_new, diffs_A_new, diffs_B):
                            update_state(x, A, sums_A_new, diffs_A_new)
                            sums_A = sums_A_new
                            diffs_A = diffs_A_new
                            used.add(x)
                            candidates.pop(j)
                            success = True
                            break
                        elif try_add_B(x, B, sums_B, diffs_B, diffs_A_new):
                            update_state(x, B, sums_B, diffs_B)
                            sums_A = sums_A_new
                            diffs_A = diffs_A_new
                            used.add(x)
                            candidates.pop(j)
                            success = True
                            break
                    
                    if not success:
                        A.append(a)
                        used.add(a)
                        sums_A, diffs_A = rebuild_state(A)
                    else:
                        improved = True
                        break # Restart loop after a swap
                
                if not improved:
                    # Try swapping from B
                    for i in range(len(B)):
                        b = B[i]
                        B.remove(b)
                        used.remove(b)
                        sums_B_new, diffs_B_new = rebuild_state(B)
                        
                        success = False
                        for j in range(len(candidates)):
                            x = candidates[j]
                            if try_add_B(x, B, sums_B_new, diffs_B_new, diffs_A):
                                update_state(x, B, sums_B_new, diffs_B_new)
                                sums_B = sums_B_new
                                diffs_B = diffs_B_new
                                used.add(x)
                                candidates.pop(j)
                                success = True
                                break
                            elif try_add(x, A, sums_A, diffs_A, diffs_B_new):
                                update_state(x, A, sums_A, diffs_A)
                                sums_B = sums_B_new
                                diffs_B = diffs_B_new
                                used.add(x)
                                candidates.pop(j)
                                success = True
                                break
                                
                        if not success:
                            B.append(b)
                            used.add(b)
                            sums_B, diffs_B = rebuild_state(B)
                        else:
                            improved = True
                            break

        return A, B

    strategies = ['A_first', 'B_first', 'balanced']
    while time.time() - start_time < time_limit:
        order = list(range(1, n+1))
        random.shuffle(order)
        strat = random.choice(strategies)
        
        A, B = run_greedy(order, strat)
        
        # Local search
        A, B = local_search(A, B)
        
        size = len(A) + len(B)
        if size > best_size:
            best_size = size
            best_A = A
            best_B = B
            
    return [best_A, best_B]
