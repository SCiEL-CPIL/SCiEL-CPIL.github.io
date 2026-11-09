import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    n = 256
    max_size = 64
    
    # Helper to check if x can be added to S, given sums_S and diffs_T (T is the other set)
    def can_add(x, S, sums_S, diffs_T):
        if x + x in sums_S:
            return False
        for s in S:
            if x + s in sums_S:
                return False
            # Check difference constraints with the OTHER set T
            # The new differences introduced by adding x to S are x-s and s-x.
            # These must NOT be in diffs_T.
            if x - s in diffs_T:
                return False
            if s - x in diffs_T:
                return False
        return True

    def add_to(S, x, sums_S, diffs_S, diffs_T):
        """
        Adds x to S. Updates sums_S, diffs_S.
        Note: diffs_T is not modified, but we must ensure disjointness was checked.
        Returns updated sums_S, diffs_S.
        """
        S.append(x)
        sums_S.add(x + x)
        for s in S[:-1]:
            sums_S.add(x + s)
            diffs_S.add(x - s)
            diffs_S.add(s - x)
        return sums_S, diffs_S

    def remove_from(S, x, sums_S, diffs_S):
        """
        Removes x from S. Updates sums_S, diffs_S.
        """
        S.remove(x)
        sums_S.remove(x + x)
        for s in S:
            sums_S.remove(x + s)
            diffs_S.remove(x - s)
            diffs_S.remove(s - x)
        return sums_S, diffs_S

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
            
            # Try adding to A
            if x not in A and len(A) < max_size and can_add(x, A, sums_A, diffs_B):
                sums_A, diffs_A = add_to(A, x, sums_A, diffs_A, diffs_B)
            # Try adding to B
            elif x not in B and len(B) < max_size and can_add(x, B, sums_B, diffs_A):
                sums_B, diffs_B = add_to(B, x, sums_B, diffs_B, diffs_A)
                
        return A, B, sums_A, sums_B, diffs_A, diffs_B

    def local_search(A, B, sums_A, sums_B, diffs_A, diffs_B):
        in_A = set(A)
        in_B = set(B)
        
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            
            # 1. Try to add any unused element to A or B
            # We iterate through all numbers 1..256
            # Optimization: only check if not already in the target set
            for x in range(1, n + 1):
                if time.time() > deadline - 1:
                    break
                
                # Try add to A
                if x not in in_A and len(A) < max_size and can_add(x, A, sums_A, diffs_B):
                    sums_A, diffs_A = add_to(A, x, sums_A, diffs_A, diffs_B)
                    in_A.add(x)
                    improved = True
                    break # Restart loop with new state
                    
                # Try add to B
                if x not in in_B and len(B) < max_size and can_add(x, B, sums_B, diffs_A):
                    sums_B, diffs_B = add_to(B, x, sums_B, diffs_B, diffs_A)
                    in_B.add(x)
                    improved = True
                    break # Restart loop with new state
            
            if improved:
                continue
            
            # 2. Try Swaps: a in A, b in B -> b in A, a in B
            # We need to check if this swap is valid.
            # Remove a from A, b from B.
            # Check if b can be added to A (with new diffs_B)
            # Check if a can be added to B (with new diffs_A)
            
            # To avoid O(|A||B|) full checks with expensive removals every time,
            # we can just iterate. Given small sizes, it's fine.
            
            A_orig = A[:]
            B_orig = B[:]
            sums_A_orig = sums_A.copy()
            sums_B_orig = sums_B.copy()
            diffs_A_orig = diffs_A.copy()
            diffs_B_orig = diffs_B.copy()
            
            swap_found = False
            for a in A_orig:
                if time.time() > deadline - 1:
                    break
                for b in B_orig:
                    if time.time() > deadline - 1:
                        break
                    
                    # Temporarily remove a from A and b from B
                    A_temp = A_orig.copy()
                    B_temp = B_orig.copy()
                    A_temp.remove(a)
                    B_temp.remove(b)
                    
                    sums_A_temp = sums_A_orig.copy()
                    sums_B_temp = sums_B_orig.copy()
                    diffs_A_temp = diffs_A_orig.copy()
                    diffs_B_temp = diffs_B_orig.copy()
                    
                    # Remove a from A's state
                    for s in A_temp:
                        sums_A_temp.discard(a + s)
                        diffs_A_temp.discard(a - s)
                        diffs_A_temp.discard(s - a)
                    sums_A_temp.discard(a + a)
                    
                    # Remove b from B's state
                    for s in B_temp:
                        sums_B_temp.discard(b + s)
                        diffs_B_temp.discard(b - s)
                        diffs_B_temp.discard(s - b)
                    sums_B_temp.discard(b + b)
                    
                    # Now try to add b to A_temp
                    if can_add(b, A_temp, sums_A_temp, diffs_B_temp):
                        # Add b to A_temp
                        sums_A_temp.add(b + b)
                        for s in A_temp:
                            sums_A_temp.add(b + s)
                            diffs_A_temp.add(b - s)
                            diffs_A_temp.add(s - b)
                        A_temp.append(b)
                        
                        # Now try to add a to B_temp
                        if can_add(a, B_temp, sums_B_temp, diffs_A_temp):
                            # Add a to B_temp
                            sums_B_temp.add(a + a)
                            for s in B_temp:
                                sums_B_temp.add(a + s)
                                diffs_B_temp.add(a - s)
                                diffs_B_temp.add(s - a)
                            B_temp.append(a)
                            
                            # Accept swap
                            A = A_temp
                            B = B_temp
                            sums_A = sums_A_temp
                            sums_B = sums_B_temp
                            diffs_A = diffs_A_temp
                            diffs_B = diffs_B_temp
                            in_A = set(A)
                            in_B = set(B)
                            improved = True
                            swap_found = True
                            break
                if swap_found:
                    break
            
            if swap_found:
                continue

            # 3. Try Moves: a in A -> B, or b in B -> A
            # Move a from A to B:
            # Remove a from A. Check if a can be added to B (with updated diffs_A).
            
            move_found = False
            for a in A:
                if time.time() > deadline - 1:
                    break
                if len(B) >= max_size:
                    continue
                    
                A_temp = A.copy()
                B_temp = B.copy()
                A_temp.remove(a)
                
                sums_A_temp = sums_A.copy()
                sums_B_temp = sums_B.copy()
                diffs_A_temp = diffs_A.copy()
                diffs_B_temp = diffs_B.copy()
                
                # Remove a from A's state
                for s in A_temp:
                    sums_A_temp.discard(a + s)
                    diffs_A_temp.discard(a - s)
                    diffs_A_temp.discard(s - a)
                sums_A_temp.discard(a + a)
                
                # Try to add a to B_temp
                if can_add(a, B_temp, sums_B_temp, diffs_A_temp):
                    # Add a to B_temp
                    sums_B_temp.add(a + a)
                    for s in B_temp:
                        sums_B_temp.add(a + s)
                        diffs_B_temp.add(a - s)
                        diffs_B_temp.add(s - a)
                    B_temp.append(a)
                    
                    # Accept move
                    A = A_temp
                    B = B_temp
                    sums_A = sums_A_temp
                    sums_B = sums_B_temp
                    diffs_A = diffs_A_temp
                    diffs_B = diffs_B_temp
                    in_A = set(A)
                    in_B = set(B)
                    improved = True
                    move_found = True
                    break
            
            if move_found:
                continue

            # Move b from B to A
            for b in B:
                if time.time() > deadline - 1:
                    break
                if len(A) >= max_size:
                    continue
                    
                A_temp = A.copy()
                B_temp = B.copy()
                B_temp.remove(b)
                
                sums_A_temp = sums_A.copy()
                sums_B_temp = sums_B.copy()
                diffs_A_temp = diffs_A.copy()
                diffs_B_temp = diffs_B.copy()
                
                # Remove b from B's state
                for s in B_temp:
                    sums_B_temp.discard(b + s)
                    diffs_B_temp.discard(b - s)
                    diffs_B_temp.discard(s - b)
                sums_B_temp.discard(b + b)
                
                # Try to add b to A_temp
                if can_add(b, A_temp, sums_A_temp, diffs_B_temp):
                    # Add b to A_temp
                    sums_A_temp.add(b + b)
                    for s in A_temp:
                        sums_A_temp.add(b + s)
                        diffs_A_temp.add(b - s)
                        diffs_A_temp.add(s - b)
                    A_temp.append(b)
                    
                    # Accept move
                    A = A_temp
                    B = B_temp
                    sums_A = sums_A_temp
                    sums_B = sums_B_temp
                    diffs_A = diffs_A_temp
                    diffs_B = diffs_B_temp
                    in_A = set(A)
                    in_B = set(B)
                    improved = True
                    move_found = True
                    break
            
            if move_found:
                continue

    best_A, best_B = [], []
    best_score = 0
    
    # Try multiple strategies
    for trial in range(20):
        if time.time() > deadline:
            break
        
        order = list(range(1, n+1))
        random.shuffle(order)
        
        A, B, sums_A, sums_B, diffs_A, diffs_B = greedy_construct(order)
        
        # Local search
        local_search(A, B, sums_A, sums_B, diffs_A, diffs_B)
        
        score = len(A) + len(B)
        if score > best_score:
            best_score = score
            best_A, best_B = A[:], B[:]
            
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
        
    return [best_A, best_B]
