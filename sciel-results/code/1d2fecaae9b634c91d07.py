import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    N = 256
    best_size = 0
    best_A = []
    best_B = []
    
    # Pre-compute all possible differences and sums? 
    # We will compute on the fly for simplicity, but optimize the check.
    
    def is_valid_sidon(S):
        """Check if S is a Sidon set. S is a list of unique integers."""
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def get_diffs(S):
        """Get the set of non-zero differences for S."""
        diffs = set()
        for i in range(len(S)):
            for j in range(len(S)):
                if i != j:
                    diffs.add(S[i] - S[j])
        return diffs

    def check_add(x, S, sums_set, diffs_set, other_diffs):
        """
        Check if x can be added to S.
        sums_set: set of all pair sums in S (including a+a)
        diffs_set: set of non-zero diffs in S
        other_diffs: set of non-zero diffs in the other set
        Returns (valid, new_sums, new_diffs)
        """
        # Check sum conflicts
        if x + x in sums_set:
            return False, None, None
        for a in S:
            if x + a in sums_set:
                return False, None, None
        
        # Check diff conflicts with other set
        for a in S:
            d1 = x - a
            d2 = a - x
            if d1 in other_diffs or d2 in other_diffs:
                return False, None, None
        
        # Check diff conflicts within S (Sidon condition implies diffs are unique? No, Sidon is about sums.
        # Wait, Sidon set definition: all pair sums a+b (a<=b) are unique.
        # This implies that differences are not necessarily unique, but the problem statement says 
        # "Pair sums inside each set, including a+a, must be unique."
        # And "Their nonzero difference sets must be disjoint."
        # It does NOT say differences within a set must be unique.
        # So we only need to check sum uniqueness and diff disjointness.
        
        # However, adding x to S might create a duplicate sum with an existing pair.
        # We checked x+x and x+a against sums_set.
        # Is that sufficient?
        # sums_set contains all sums a+b for a,b in S.
        # If x+x is in sums_set, invalid.
        # If x+a is in sums_set for any a in S, invalid.
        # Yes, this covers all new sums involving x.
        
        # Now construct new sums and diffs
        new_sums = set(sums_set)
        new_sums.add(x + x)
        for a in S:
            new_sums.add(x + a)
            
        new_diffs = set(diffs_set)
        for a in S:
            new_diffs.add(x - a)
            new_diffs.add(a - x)
            
        return True, new_sums, new_diffs

    def local_improve(A, B):
        """
        Attempt to improve A and B by adding, moving, or swapping elements.
        Returns improved A, B.
        """
        A = A[:]
        B = B[:]
        
        # Initialize state
        sums_A = set()
        diffs_A = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums_A.add(A[i] + A[j])
            for j in range(len(A)):
                if i != j:
                    diffs_A.add(A[i] - A[j])
                    
        sums_B = set()
        diffs_B = set()
        for i in range(len(B)):
            for j in range(i, len(B)):
                sums_B.add(B[i] + B[j])
            for j in range(len(B)):
                if i != j:
                    diffs_B.add(B[i] - B[j])
                    
        # Check initial validity (should be valid by construction)
        # If not, return as is
        
        changed = True
        while changed:
            changed = False
            
            # 1. Try to add any unused element
            used = set(A) | set(B)
            unused = [x for x in range(1, N + 1) if x not in used]
            random.shuffle(unused)
            
            for x in unused:
                # Try adding to A
                valid, new_sums_A, new_diffs_A = check_add(x, A, sums_A, diffs_A, diffs_B)
                if valid:
                    A.append(x)
                    sums_A = new_sums_A
                    diffs_A = new_diffs_A
                    changed = True
                    break
                    
                # Try adding to B
                valid, new_sums_B, new_diffs_B = check_add(x, B, sums_B, diffs_B, diffs_A)
                if valid:
                    B.append(x)
                    sums_B = new_sums_B
                    diffs_B = new_diffs_B
                    changed = True
                    break
                    
            if changed:
                continue

            # 2. Try moving an element from A to B or B to A
            # Move a from A to B:
            # A' = A \ {a}, B' = B \cup {a}
            # Check if A' is valid Sidon (it is, subset of Sidon)
            # Check if B' is valid Sidon
            # Check if diffs(A') and diffs(B') are disjoint
            
            # We need to recompute sums/diffs for the modified sets.
            # This is expensive if done naively. 
            # Optimization: 
            # Removing a from A:
            # sums_A' = sums_A - {sums involving a}
            # diffs_A' = diffs_A - {diffs involving a}
            # Adding a to B:
            # We need to check if a can be added to B.
            
            # Let's implement a helper to remove an element efficiently.
            def remove_element(S, x, sums, diffs):
                new_sums = set()
                new_diffs = set()
                # Rebuilding from scratch is O(|S|^2). |S| <= 64, so 4096 ops. 
                # Doing this for every element in A and B is 128 * 4096 ~ 500k ops.
                # This is acceptable for a few iterations.
                
                S_new = [y for y in S if y != x]
                for i in range(len(S_new)):
                    for j in range(i, len(S_new)):
                        new_sums.add(S_new[i] + S_new[j])
                    for j in range(len(S_new)):
                        if i != j:
                            new_diffs.add(S_new[i] - S_new[j])
                return S_new, new_sums, new_diffs

            # Try moving from A to B
            for idx_a in range(len(A)):
                a = A[idx_a]
                A_new, sums_A_new, diffs_A_new = remove_element(A, a, sums_A, diffs_A)
                
                # Check if a can be added to B
                valid, sums_B_new, diffs_B_new = check_add(a, B, sums_B, diffs_B, diffs_A_new)
                if valid:
                    # Check if diffs_A_new and diffs_B_new are disjoint
                    if not (diffs_A_new & diffs_B_new):
                        A = A_new
                        B = B + [a]
                        sums_A = sums_A_new
                        diffs_A = diffs_A_new
                        sums_B = sums_B_new
                        diffs_B = diffs_B_new
                        changed = True
                        break
                        
            if changed:
                continue

            # Try moving from B to A
            for idx_b in range(len(B)):
                b = B[idx_b]
                B_new, sums_B_new, diffs_B_new = remove_element(B, b, sums_B, diffs_B)
                
                # Check if b can be added to A
                valid, sums_A_new, diffs_A_new = check_add(b, A, sums_A, diffs_A, diffs_B_new)
                if valid:
                    # Check if diffs_A_new and diffs_B_new are disjoint
                    if not (diffs_A_new & diffs_B_new):
                        B = B_new
                        A = A + [b]
                        sums_B = sums_B_new
                        diffs_B = diffs_B_new
                        sums_A = sums_A_new
                        diffs_A = diffs_A_new
                        changed = True
                        break
                        
            if changed:
                continue

            # 3. Try swapping a in A with b in B
            # A' = A \ {a} \cup {b}, B' = B \ {b} \cup {a}
            for idx_a in range(len(A)):
                a = A[idx_a]
                for idx_b in range(len(B)):
                    b = B[idx_b]
                    if a == b: continue
                    
                    A_without_a, sums_A_wo_a, diffs_A_wo_a = remove_element(A, a, sums_A, diffs_A)
                    B_without_b, sums_B_wo_b, diffs_B_wo_b = remove_element(B, b, sums_B, diffs_B)
                    
                    # Check if b can be added to A_without_a
                    valid_A, sums_A_new, diffs_A_new = check_add(b, A_without_a, sums_A_wo_a, diffs_A_wo_a, diffs_B_wo_b)
                    if not valid_A:
                        continue
                        
                    # Check if a can be added to B_without_b
                    valid_B, sums_B_new, diffs_B_new = check_add(a, B_without_b, sums_B_wo_b, diffs_B_wo_b, diffs_A_new)
                    if not valid_B:
                        continue
                        
                    # Check disjointness of new diffs
                    if not (diffs_A_new & diffs_B_new):
                        A = A_without_a + [b]
                        B = B_without_b + [a]
                        sums_A = sums_A_new
                        diffs_A = diffs_A_new
                        sums_B = sums_B_new
                        diffs_B = diffs_B_new
                        changed = True
                        break
                if changed:
                    break
                    
        return A, B

    # Main loop: Multiple restarts with different heuristics
    max_restarts = 1000
    for i in range(max_restarts):
        if time.time() > deadline:
            break
            
        A = []
        B = []
        
        # Heuristic 1: Small to Large
        # Heuristic 2: Large to Small
        # Heuristic 3: Random
        
        if i % 3 == 0:
            candidates = list(range(1, N + 1))
        elif i % 3 == 1:
            candidates = list(range(N, 0, -1))
        else:
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
            
        # Greedy construction
        sums_A = set()
        diffs_A = set()
        sums_B = set()
        diffs_B = set()
        
        for x in candidates:
            # Try adding to A
            valid, new_sums_A, new_diffs_A = check_add(x, A, sums_A, diffs_A, diffs_B)
            if valid:
                A.append(x)
                sums_A = new_sums_A
                diffs_A = new_diffs_A
                continue
                
            # Try adding to B
            valid, new_sums_B, new_diffs_B = check_add(x, B, sums_B, diffs_B, diffs_A)
            if valid:
                B.append(x)
                sums_B = new_sums_B
                diffs_B = new_diffs_B
                continue
                
        # Local improvement
        A, B = local_improve(A, B)
        
        current_size = len(A) + len(B)
        if current_size > best_size:
            best_size = current_size
            best_A = A[:]
            best_B = B[:]
            
    return [best_A, best_B]
