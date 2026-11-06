import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    N = 256
    MAX_SIZE_A = 16
    MAX_SIZE_B = 16

    best_size = 0
    best_A = []
    best_B = []

    def compute_sums_mask(S):
        mask = 0
        for i in range(len(S)):
            for j in range(i, len(S)):
                mask |= (1 << (S[i] + S[j]))
        return mask

    def compute_diffs_mask(S):
        mask = 0
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                mask |= (1 << abs(S[i] - S[j]))
        return mask

    def is_valid_add(x, S, sums_mask, forbidden_diffs_mask):
        # Check Sidon sums
        if (sums_mask >> (2 * x)) & 1:
            return False
        for a in S:
            if (sums_mask >> (x + a)) & 1:
                return False
        
        # Check difference disjointness
        for a in S:
            d = abs(x - a)
            if d == 0:
                continue
            if (forbidden_diffs_mask >> d) & 1:
                return False
        return True

    def build_greedy(order, forbidden_diffs_mask, max_size):
        S = []
        sums_mask = 0
        diffs_mask = 0
        
        for x in order:
            if len(S) >= max_size:
                break
            if x in S:
                continue
            
            # Check validity
            valid = True
            if (sums_mask >> (2 * x)) & 1:
                valid = False
            else:
                for a in S:
                    if (sums_mask >> (x + a)) & 1:
                        valid = False
                        break
            
            if valid:
                for a in S:
                    d = abs(x - a)
                    if d > 0 and (forbidden_diffs_mask >> d) & 1:
                        valid = False
                        break

            if valid:
                S.append(x)
                # Update masks
                for a in S[:-1]:
                    sums_mask |= (1 << (x + a))
                sums_mask |= (1 << (2 * x))
                for a in S[:-1]:
                    d = abs(x - a)
                    if d > 0:
                        diffs_mask |= (1 << d)
        
        return S, sums_mask, diffs_mask

    def polish(A, B, max_time):
        A = A[:]
        B = B[:]
        
        sums_A = compute_sums_mask(A)
        diffs_A = compute_diffs_mask(A)
        sums_B = compute_sums_mask(B)
        diffs_B = compute_diffs_mask(B)
        
        forbidden_for_A = diffs_B
        forbidden_for_B = diffs_A
        
        improved = True
        while improved and time.time() < max_time:
            improved = False
            
            # Try to add to A
            candidates = [x for x in range(1, N + 1) if x not in A]
            random.shuffle(candidates)
            for x in candidates:
                if len(A) >= MAX_SIZE_A:
                    break
                if is_valid_add(x, A, sums_A, forbidden_for_A):
                    A.append(x)
                    # Update sums_A and diffs_A
                    for a in A[:-1]:
                        sums_A |= (1 << (x + a))
                    sums_A |= (1 << (2 * x))
                    for a in A[:-1]:
                        d = abs(x - a)
                        if d > 0:
                            diffs_A |= (1 << d)
                    forbidden_for_B |= (1 << 0) # placeholder, recompute below
                    # Recompute forbidden masks since diffs changed
                    # Actually, we can just update the specific masks
                    # forbidden_for_A is diffs_B (unchanged)
                    # forbidden_for_B is diffs_A (changed)
                    # We need to update forbidden_for_B
                    # But we pass masks to is_valid_add, so we just use the updated diffs_A
                    improved = True
                    break
            
            if improved:
                forbidden_for_B = diffs_A
                continue

            # Try to add to B
            candidates = [x for x in range(1, N + 1) if x not in B]
            random.shuffle(candidates)
            for x in candidates:
                if len(B) >= MAX_SIZE_B:
                    break
                if is_valid_add(x, B, sums_B, forbidden_for_B):
                    B.append(x)
                    for b in B[:-1]:
                        sums_B |= (1 << (x + b))
                    sums_B |= (1 << (2 * x))
                    for b in B[:-1]:
                        d = abs(x - b)
                        if d > 0:
                            diffs_B |= (1 << d)
                    improved = True
                    break
            
            if improved:
                forbidden_for_A = diffs_B
                continue

            # Try Swap: Remove one from A, add one to A
            if len(A) > 0:
                idx_to_remove = random.randint(0, len(A) - 1)
                x_removed = A[idx_to_remove]
                A_temp = A[:]
                A_temp.pop(idx_to_remove)
                
                sums_temp = compute_sums_mask(A_temp)
                diffs_temp = compute_diffs_mask(A_temp)
                
                forbidden_temp_B = diffs_B
                candidates = [x for x in range(1, N + 1) if x not in A_temp and x != x_removed]
                random.shuffle(candidates)
                
                found_swap = False
                for x in candidates:
                    if len(A_temp) >= MAX_SIZE_A:
                        break
                    if is_valid_add(x, A_temp, sums_temp, forbidden_temp_B):
                        # Accept swap
                        A = A_temp + [x]
                        sums_A = compute_sums_mask(A)
                        diffs_A = compute_diffs_mask(A)
                        forbidden_for_B = diffs_A
                        improved = True
                        found_swap = True
                        break
                
                if found_swap:
                    continue

            # Try Swap: Remove one from B, add one to B
            if len(B) > 0:
                idx_to_remove = random.randint(0, len(B) - 1)
                x_removed = B[idx_to_remove]
                B_temp = B[:]
                B_temp.pop(idx_to_remove)
                
                sums_temp = compute_sums_mask(B_temp)
                diffs_temp = compute_diffs_mask(B_temp)
                
                forbidden_temp_A = diffs_A
                candidates = [x for x in range(1, N + 1) if x not in B_temp and x != x_removed]
                random.shuffle(candidates)
                
                found_swap = False
                for x in candidates:
                    if len(B_temp) >= MAX_SIZE_B:
                        break
                    if is_valid_add(x, B_temp, sums_temp, forbidden_temp_A):
                        B = B_temp + [x]
                        sums_B = compute_sums_mask(B)
                        diffs_B = compute_diffs_mask(B)
                        forbidden_for_A = diffs_B
                        improved = True
                        found_swap = True
                        break
                
                if found_swap:
                    continue

        return A, B

    while time.time() < deadline:
        # Random restart
        order_A = list(range(1, N + 1))
        random.shuffle(order_A)
        A, sums_A, diffs_A = build_greedy(order_A, 0, MAX_SIZE_A)
        
        if not A:
            continue

        order_B = list(range(1, N + 1))
        random.shuffle(order_B)
        B, sums_B, diffs_B = build_greedy(order_B, diffs_A, MAX_SIZE_B)
        
        current_size = len(A) + len(B)
        
        # Polish
        if current_size > best_size or (current_size >= best_size - 2 and random.random() < 0.7):
            A_pol, B_pol = polish(A, B, deadline - 1.0)
            current_size = len(A_pol) + len(B_pol)
            
            if current_size > best_size:
                best_size = current_size
                best_A = A_pol
                best_B = B_pol
                
        if best_size >= 32:
            break

    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]

    return [best_A, best_B]
