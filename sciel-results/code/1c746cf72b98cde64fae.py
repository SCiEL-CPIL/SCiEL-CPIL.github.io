"""Rewritten program for the suite task; incremental state greedy + local search."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    n = 256
    max_size = 64
    rng = random.Random(42)
    
    best_total = 0
    best_A = []
    best_B = []
    
    def rebuild_state(A, B):
        sums_A = set()
        diffs_A = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums_A.add(A[i] + A[j])
        for i in range(len(A)):
            for j in range(len(A)):
                if i != j:
                    diffs_A.add(A[j] - A[i])
        
        sums_B = set()
        diffs_B = set()
        for i in range(len(B)):
            for j in range(i, len(B)):
                sums_B.add(B[i] + B[j])
        for i in range(len(B)):
            for j in range(len(B)):
                if i != j:
                    diffs_B.add(B[j] - B[i])
                    
        return sums_A, diffs_A, sums_B, diffs_B

    def can_add_to_A(x, A, sums_A, diffs_A, diffs_B):
        if x + x in sums_A:
            return False
        for a in A:
            if x + a in sums_A:
                return False
            if (x - a) in diffs_B:
                return False
            if (a - x) in diffs_B:
                return False
        return True

    def can_add_to_B(x, B, sums_B, diffs_A, diffs_B):
        if x + x in sums_B:
            return False
        for b in B:
            if x + b in sums_B:
                return False
            if (x - b) in diffs_A:
                return False
            if (b - x) in diffs_A:
                return False
        return True

    def add_to_A(x, A, sums_A, diffs_A):
        A.append(x)
        sums_A.add(x + x)
        for a in A[:-1]:
            sums_A.add(x + a)
            diffs_A.add(x - a)
            diffs_A.add(a - x)
            
    def add_to_B(x, B, sums_B, diffs_B):
        B.append(x)
        sums_B.add(x + x)
        for b in B[:-1]:
            sums_B.add(x + b)
            diffs_B.add(x - b)
            diffs_B.add(b - x)

    def greedy_construct(order_A, order_B):
        A = []
        B = []
        sums_A = set()
        diffs_A = set()
        sums_B = set()
        diffs_B = set()
        
        # Interleaved greedy to balance constraints
        i_A = 0
        i_B = 0
        while (len(A) < max_size and i_A < len(order_A)) or (len(B) < max_size and i_B < len(order_B)):
            # Try adding to A
            if len(A) < max_size:
                found = False
                while i_A < len(order_A):
                    x = order_A[i_A]
                    i_A += 1
                    if x not in A and can_add_to_A(x, A, sums_A, diffs_A, diffs_B):
                        add_to_A(x, A, sums_A, diffs_A)
                        found = True
                        break
                if found:
                    continue
            
            # Try adding to B
            if len(B) < max_size:
                found = False
                while i_B < len(order_B):
                    x = order_B[i_B]
                    i_B += 1
                    if x not in B and can_add_to_B(x, B, sums_B, diffs_A, diffs_B):
                        add_to_B(x, B, sums_B, diffs_B)
                        found = True
                        break
                if found:
                    continue
                    
            # If neither could be added in this pass, break to avoid infinite loop if orders are exhausted
            if not found:
                break

        return A, B

    def local_search(A, B, max_iters=200):
        A = list(A)
        B = list(B)
        
        for _ in range(max_iters):
            improved = False
            sums_A, diffs_A, sums_B, diffs_B = rebuild_state(A, B)
            
            # Phase 1: Additions
            if len(A) < max_size:
                for x in range(1, n + 1):
                    if x not in A and can_add_to_A(x, A, sums_A, diffs_A, diffs_B):
                        add_to_A(x, A, sums_A, diffs_A)
                        improved = True
                        break
            if len(B) < max_size:
                if improved:
                    sums_A, diffs_A, sums_B, diffs_B = rebuild_state(A, B)
                for x in range(1, n + 1):
                    if x not in B and can_add_to_B(x, B, sums_B, diffs_A, diffs_B):
                        add_to_B(x, B, sums_B, diffs_B)
                        improved = True
                        break

            if improved:
                continue

            # Phase 2: Swaps
            # Try swapping an element in A with a new one
            swapped = False
            if len(A) > 0:
                sums_A, diffs_A, sums_B, diffs_B = rebuild_state(A, B)
                for a in A[:]:
                    # Try removing a
                    A_new = [x for x in A if x != a]
                    # Rebuild state for A_new
                    sums_A_new = set()
                    diffs_A_new = set()
                    for i in range(len(A_new)):
                        for j in range(i, len(A_new)):
                            sums_A_new.add(A_new[i] + A_new[j])
                    for i in range(len(A_new)):
                        for j in range(len(A_new)):
                            if i != j:
                                diffs_A_new.add(A_new[j] - A_new[i])
                    
                    # Check if we can add a new element x to A_new
                    for x in range(1, n + 1):
                        if x in A: continue
                        if x + x in sums_A_new: continue
                        valid = True
                        for a2 in A_new:
                            if x + a2 in sums_A_new:
                                valid = False
                                break
                            if (x - a2) in diffs_B:
                                valid = False
                                break
                            if (a2 - x) in diffs_B:
                                valid = False
                                break
                        if valid:
                            # Perform swap
                            A = A_new + [x]
                            # Rebuild full state for next iteration
                            improved = True
                            swapped = True
                            break
                    if swapped:
                        break
            
            if swapped:
                continue

            # Try swapping an element in B with a new one
            if len(B) > 0:
                sums_A, diffs_A, sums_B, diffs_B = rebuild_state(A, B)
                for b in B[:]:
                    B_new = [x for x in B if x != b]
                    sums_B_new = set()
                    diffs_B_new = set()
                    for i in range(len(B_new)):
                        for j in range(i, len(B_new)):
                            sums_B_new.add(B_new[i] + B_new[j])
                    for i in range(len(B_new)):
                        for j in range(len(B_new)):
                            if i != j:
                                diffs_B_new.add(B_new[j] - B_new[i])
                    
                    for x in range(1, n + 1):
                        if x in B: continue
                        if x + x in sums_B_new: continue
                        valid = True
                        for b2 in B_new:
                            if x + b2 in sums_B_new:
                                valid = False
                                break
                            if (x - b2) in diffs_A:
                                valid = False
                                break
                            if (b2 - x) in diffs_A:
                                valid = False
                                break
                        if valid:
                            B = B_new + [x]
                            improved = True
                            swapped = True
                            break
                    if swapped:
                        break
            
            if not improved:
                break
                
        return A, B

    def valid_solution(A, B):
        if not A or not B:
            return False
        if len(A) > max_size or len(B) > max_size:
            return False
        sums_A, diffs_A, sums_B, diffs_B = rebuild_state(A, B)
        if diffs_A & diffs_B:
            return False
        return True

    for seed in range(30):
        rng.seed(seed)
        all_elements = list(range(1, n + 1))
        order_A = all_elements[:]
        order_B = all_elements[:]
        rng.shuffle(order_A)
        rng.shuffle(order_B)
        
        A, B = greedy_construct(order_A, order_B)
        A, B = local_search(A, B, max_iters=100)
        
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_A = A[:]
            best_B = B[:]
            
        if best_total >= 120:
            break

    if not valid_solution(best_A, best_B):
        A = list(range(1, 17))
        B = list(range(17, 33))
        if not valid_solution(A, B):
            A = [1, 2, 4, 8, 16, 32, 64, 128]
            B = [3, 5, 9, 17, 33, 65, 129]
            B = [x for x in B if x <= n]
            best_A = A
            best_B = B

    return [best_A, best_B]
# EVOLVE-BLOCK-END
