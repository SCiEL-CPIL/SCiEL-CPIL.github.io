import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    n = 256
    best_A = []
    best_B = []
    best_score = 0
    rng = random.Random(42)

    def can_add(x, S, S_sums, other_diffs):
        if x + x in S_sums:
            return False
        for a in S:
            if x + a in S_sums:
                return False
        for a in S:
            if (x - a) in other_diffs or (a - x) in other_diffs:
                return False
        return True

    def get_diffs(S):
        diffs = set()
        for i in range(len(S)):
            for j in range(len(S)):
                if i != j:
                    diffs.add(S[i] - S[j])
        return diffs

    def get_sums(S):
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.add(S[i] + S[j])
        return sums

    def is_valid_sidon(S):
        if len(S) <= 1:
            return True
        s = get_sums(S)
        return len(s) == len(S) * (len(S) + 1) // 2

    def count_addable(unassigned, A, B, sums_A, sums_B, diffs_A, diffs_B):
        """Count how many unassigned elements can be added to either set."""
        count = 0
        for x in unassigned:
            if can_add(x, A, sums_A, diffs_B) or can_add(x, B, sums_B, diffs_A):
                count += 1
        return count

    def greedy_build(elements):
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        for x in elements:
            if time.time() > deadline:
                break
            can_A = can_add(x, A, sums_A, diffs_B)
            can_B = can_add(x, B, sums_B, diffs_A)
            
            if can_A and can_B:
                # Choose the set with fewer elements to balance
                if len(A) <= len(B):
                    A.append(x)
                    for a in A[:-1]:
                        sums_A.add(x + a)
                    sums_A.add(x + x)
                    for a in A[:-1]:
                        diffs_A.add(x - a)
                        diffs_A.add(a - x)
                else:
                    B.append(x)
                    for b in B[:-1]:
                        sums_B.add(x + b)
                    sums_B.add(x + x)
                    for b in B[:-1]:
                        diffs_B.add(x - b)
                        diffs_B.add(b - x)
            elif can_A:
                A.append(x)
                for a in A[:-1]:
                    sums_A.add(x + a)
                sums_A.add(x + x)
                for a in A[:-1]:
                    diffs_A.add(x - a)
                    diffs_A.add(a - x)
            elif can_B:
                B.append(x)
                for b in B[:-1]:
                    sums_B.add(x + b)
                sums_B.add(x + x)
                for b in B[:-1]:
                    diffs_B.add(x - b)
                    diffs_B.add(b - x)
        
        return A, B

    def local_search(A, B):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            sums_A = get_sums(A)
            sums_B = get_sums(B)
            diffs_A = get_diffs(A)
            diffs_B = get_diffs(B)
            
            # Try to add unassigned elements (try all, add all possible)
            assigned = set(A) | set(B)
            unassigned = [x for x in range(1, n + 1) if x not in assigned]
            
            added_any = False
            for x in unassigned[:]:
                if time.time() > deadline:
                    break
                if can_add(x, A, sums_A, diffs_B):
                    A.append(x)
                    for a in A[:-1]:
                        sums_A.add(x + a)
                    sums_A.add(x + x)
                    for a in A[:-1]:
                        diffs_A.add(x - a)
                        diffs_A.add(a - x)
                    added_any = True
                elif can_add(x, B, sums_B, diffs_A):
                    B.append(x)
                    for b in B[:-1]:
                        sums_B.add(x + b)
                    sums_B.add(x + x)
                    for b in B[:-1]:
                        diffs_B.add(x - b)
                        diffs_B.add(b - x)
                    added_any = True
            
            if added_any:
                improved = True
                continue

            # Recompute after additions
            sums_A = get_sums(A)
            sums_B = get_sums(B)
            diffs_A = get_diffs(A)
            diffs_B = get_diffs(B)
            assigned = set(A) | set(B)
            unassigned = [x for x in range(1, n + 1) if x not in assigned]
            
            # Try moving elements between sets
            # Move from A to B
            for i in range(len(A)):
                if time.time() > deadline:
                    break
                x = A[i]
                A_new = A[:i] + A[i+1:]
                B_new = B + [x]
                
                if not is_valid_sidon(B_new):
                    continue
                if not is_valid_sidon(A_new):
                    continue
                d_a_new = get_diffs(A_new)
                d_b_new = get_diffs(B_new)
                if d_a_new & d_b_new:
                    continue
                
                s_a_new = get_sums(A_new)
                s_b_new = get_sums(B_new)
                # Count how many more we could add
                old_count = count_addable(unassigned, A, B, sums_A, sums_B, diffs_A, diffs_B)
                new_count = count_addable(unassigned, A_new, B_new, s_a_new, s_b_new, d_a_new, d_b_new)
                
                if new_count > old_count:
                    A = A_new
                    B = B_new
                    improved = True
                    break
            
            if improved:
                continue

            # Move from B to A
            for i in range(len(B)):
                if time.time() > deadline:
                    break
                x = B[i]
                B_new = B[:i] + B[i+1:]
                A_new = A + [x]
                
                if not is_valid_sidon(A_new):
                    continue
                if not is_valid_sidon(B_new):
                    continue
                d_a_new = get_diffs(A_new)
                d_b_new = get_diffs(B_new)
                if d_a_new & d_b_new:
                    continue
                
                s_a_new = get_sums(A_new)
                s_b_new = get_sums(B_new)
                old_count = count_addable(unassigned, A, B, sums_A, sums_B, diffs_A, diffs_B)
                new_count = count_addable(unassigned, A_new, B_new, s_a_new, s_b_new, d_a_new, d_b_new)
                
                if new_count > old_count:
                    A = A_new
                    B = B_new
                    improved = True
                    break
            
            if improved:
                continue

            # Try swapping an element in A with an unassigned element
            for i in range(len(A)):
                if time.time() > deadline:
                    break
                x = A[i]
                for z in unassigned:
                    if time.time() > deadline:
                        break
                    A_new = A[:i] + A[i+1:] + [z]
                    if not is_valid_sidon(A_new):
                        continue
                    d_a_new = get_diffs(A_new)
                    if d_a_new & diffs_B:
                        continue
                    
                    s_a_new = get_sums(A_new)
                    old_count = count_addable(unassigned, A, B, sums_A, sums_B, diffs_A, diffs_B)
                    new_count = count_addable([w for w in unassigned if w != z], A_new, B, s_a_new, sums_B, d_a_new, diffs_B)
                    
                    if new_count > old_count or rng.random() < 0.03:
                        A = A_new
                        improved = True
                        break
                if improved:
                    break
            
            if improved:
                continue

            # Try swapping an element in B with an unassigned element
            for j in range(len(B)):
                if time.time() > deadline:
                    break
                y = B[j]
                for z in unassigned:
                    if time.time() > deadline:
                        break
                    B_new = B[:j] + B[j+1:] + [z]
                    if not is_valid_sidon(B_new):
                        continue
                    d_b_new = get_diffs(B_new)
                    if d_b_new & diffs_A:
                        continue
                    
                    s_b_new = get_sums(B_new)
                    old_count = count_addable(unassigned, A, B, sums_A, sums_B, diffs_A, diffs_B)
                    new_count = count_addable([w for w in unassigned if w != z], A, B_new, sums_A, s_b_new, diffs_A, d_b_new)
                    
                    if new_count > old_count or rng.random() < 0.03:
                        B = B_new
                        improved = True
                        break
                if improved:
                    break
            
            # Try 2-swap: swap one element from A with one from B
            if not improved:
                for i in range(len(A)):
                    if time.time() > deadline:
                        break
                    for j in range(len(B)):
                        if time.time() > deadline:
                            break
                        a = A[i]
                        b = B[j]
                        A_new = A[:i] + A[i+1:] + [b]
                        B_new = B[:j] + B[j+1:] + [a]
                        
                        if not is_valid_sidon(A_new) or not is_valid_sidon(B_new):
                            continue
                        d_a_new = get_diffs(A_new)
                        d_b_new = get_diffs(B_new)
                        if d_a_new & d_b_new:
                            continue
                        
                        s_a_new = get_sums(A_new)
                        s_b_new = get_sums(B_new)
                        old_count = count_addable(unassigned, A, B, sums_A, sums_B, diffs_A, diffs_B)
                        new_count = count_addable(unassigned, A_new, B_new, s_a_new, s_b_new, d_a_new, d_b_new)
                        
                        if new_count > old_count:
                            A = A_new
                            B = B_new
                            improved = True
                            break
                    if improved:
                        break
        
        return A, B

    # Main loop: greedy + local search with random restarts
    for trial in range(3000):
        if time.time() > deadline:
            break
        
        elements = list(range(1, n + 1))
        rng.shuffle(elements)
        
        A, B = greedy_build(elements)
        A, B = local_search(A, B)
        
        score = len(A) + len(B)
        if score > best_score:
            best_A = A[:]
            best_B = B[:]
            best_score = score

    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    return [best_A, best_B]
