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
            
            # Try to add unassigned elements
            assigned = set(A) | set(B)
            unassigned = [x for x in range(1, n + 1) if x not in assigned]
            
            for x in unassigned:
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
                    improved = True
                    break
                if can_add(x, B, sums_B, diffs_A):
                    B.append(x)
                    for b in B[:-1]:
                        sums_B.add(x + b)
                    sums_B.add(x + x)
                    for b in B[:-1]:
                        diffs_B.add(x - b)
                        diffs_B.add(b - x)
                    improved = True
                    break
            
            if improved:
                continue

            # Try moving elements between sets to free up difference space
            # Move from A to B
            for i, x in enumerate(A):
                if time.time() > deadline:
                    break
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
                
                # Check if this move enables adding more elements
                s_a_new = get_sums(A_new)
                s_b_new = get_sums(B_new)
                added = 0
                for z in unassigned:
                    if can_add(z, A_new, s_a_new, d_b_new):
                        added += 1
                        break
                    if can_add(z, B_new, s_b_new, d_a_new):
                        added += 1
                        break
                
                if added > 0:
                    A = A_new
                    B = B_new
                    improved = True
                    break
            
            if improved:
                continue

            # Move from B to A
            for i, x in enumerate(B):
                if time.time() > deadline:
                    break
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
                added = 0
                for z in unassigned:
                    if can_add(z, A_new, s_a_new, d_b_new):
                        added += 1
                        break
                    if can_add(z, B_new, s_b_new, d_a_new):
                        added += 1
                        break
                
                if added > 0:
                    A = A_new
                    B = B_new
                    improved = True
                    break
            
            if improved:
                continue

            # Try swapping an element in A with an unassigned element
            for i, x in enumerate(A):
                if time.time() > deadline:
                    break
                for z in unassigned:
                    A_new = A[:i] + A[i+1:] + [z]
                    if not is_valid_sidon(A_new):
                        continue
                    d_a_new = get_diffs(A_new)
                    if d_a_new & diffs_B:
                        continue
                    
                    # Check if this enables adding more to B
                    s_a_new = get_sums(A_new)
                    added = 0
                    for w in unassigned:
                        if w != z:
                            if can_add(w, B, sums_B, d_a_new):
                                added += 1
                                break
                    
                    if added > 0 or rng.random() < 0.05:
                        A = A_new
                        improved = True
                        break
                if improved:
                    break
            
            if improved:
                continue

            # Try swapping an element in B with an unassigned element
            for j, y in enumerate(B):
                if time.time() > deadline:
                    break
                for z in unassigned:
                    B_new = B[:j] + B[j+1:] + [z]
                    if not is_valid_sidon(B_new):
                        continue
                    d_b_new = get_diffs(B_new)
                    if d_b_new & diffs_A:
                        continue
                    
                    s_b_new = get_sums(B_new)
                    added = 0
                    for w in unassigned:
                        if w != z:
                            if can_add(w, A, sums_A, d_b_new):
                                added += 1
                                break
                    
                    if added > 0 or rng.random() < 0.05:
                        B = B_new
                        improved = True
                        break
                if improved:
                    break
        
        return A, B

    # Main loop: greedy + local search with random restarts
    for trial in range(5000):
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
