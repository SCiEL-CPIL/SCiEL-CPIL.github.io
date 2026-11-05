import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    best_score = 0
    best_A = [1]
    best_B = [2]
    
    # Pre-compute all possible differences and sums for quick lookup?
    # Actually, sets are fine.
    
    def rebuild_diffs_sums(S):
        diffs = set()
        sums = set()
        for i, a in enumerate(S):
            sums.add(2 * a)
            for j in range(i + 1, len(S)):
                b = S[j]
                d = abs(a - b)
                diffs.add(d)
                sums.add(a + b)
        return diffs, sums

    def can_add(x, S, diffs_S, sums_S, diffs_other):
        # Check Sidon condition for S
        if 2 * x in sums_S:
            return False
        for a in S:
            d = abs(x - a)
            if d in diffs_S:
                return False
            if d in diffs_other:
                return False
            if x + a in sums_S:
                return False
        return True

    def add_to_set(x, S, diffs_S, sums_S):
        S.append(x)
        sums_S.add(2 * x)
        for a in S:
            if a != x:
                d = abs(x - a)
                diffs_S.add(d)
                sums_S.add(x + a)

    def remove_from_set(x, S, diffs_S, sums_S):
        """
        Incremental removal is tricky because diffs/sums might be shared.
        For correctness and simplicity in local search where we often rebuild,
        we will use a helper that rebuilds if the set is small, or 
        carefully updates. 
        Given N=256, rebuilding is O(k^2) which is fast for k~16.
        16^2 = 256 ops. Very fast.
        So we can just rebuild after every modification for simplicity and correctness.
        """
        S.remove(x)
        # Rebuild is safer and fast enough
        new_diffs, new_sums = rebuild_diffs_sums(S)
        # Update the sets in place to avoid re-assignment in caller?
        # Caller will reassign or we return them.
        # Let's return new sets.
        return new_diffs, new_sums

    def do_local_search(A, B, max_iterations=500):
        """
        Local search with Add, Move, Swap, and Neutral Swaps.
        """
        diffs_A, sums_A = rebuild_diffs_sums(A)
        diffs_B, sums_B = rebuild_diffs_sums(B)
        A_set = set(A)
        B_set = set(B)
        
        # Track best in this local search
        best_local_score = len(A) + len(B)
        best_local_A = A.copy()
        best_local_B = B.copy()
        
        iterations = 0
        stall_count = 0
        
        while iterations < max_iterations:
            iterations += 1
            if time.time() > deadline:
                break
            
            made_improvement = False
            made_neutral_move = False
            
            # 1. Try to Add
            # Shuffle candidates for randomness
            candidates = [x for x in range(1, 257) if x not in A_set and x not in B_set]
            random.shuffle(candidates)
            
            added = False
            for x in candidates:
                if can_add(x, A, diffs_A, sums_A, diffs_B):
                    add_to_set(x, A, diffs_A, sums_A)
                    A_set.add(x)
                    made_improvement = True
                    added = True
                    break
                if can_add(x, B, diffs_B, sums_B, diffs_A):
                    add_to_set(x, B, diffs_B, sums_B)
                    B_set.add(x)
                    made_improvement = True
                    added = True
                    break
            
            if made_improvement:
                best_local_score = len(A) + len(B)
                best_local_A = A.copy()
                best_local_B = B.copy()
                stall_count = 0
                continue
            
            # 2. Try to Move (A to B or B to A)
            # This keeps size same, but might enable future adds.
            # Check if moving a from A to B is valid.
            # Condition: A\{a} is Sidon (always true if A was Sidon), 
            # B U {a} is Sidon, and diffs(A\{a}) disjoint from diffs(B U {a}).
            # Note: diffs change.
            
            moved = False
            for a in list(A):
                # Temporarily remove a
                A_temp = [x for x in A if x != a]
                diffs_A_temp, sums_A_temp = rebuild_diffs_sums(A_temp)
                
                # Check if a can be added to B
                if can_add(a, B, diffs_B, sums_B, diffs_A_temp):
                    # It's a valid move!
                    # Update A
                    A.remove(a)
                    A_set.discard(a)
                    diffs_A, sums_A = diffs_A_temp, sums_A_temp
                    
                    # Add to B
                    add_to_set(a, B, diffs_B, sums_B)
                    B_set.add(a)
                    
                    made_neutral_move = True
                    moved = True
                    break
            
            if moved:
                # Neutral move doesn't improve score, but changes state
                stall_count += 1
                if stall_count > 50:
                    # If we've been doing neutral moves for too long without improvement,
                    # maybe try a kick?
                    pass
                continue
            
            for b in list(B):
                B_temp = [x for x in B if x != b]
                diffs_B_temp, sums_B_temp = rebuild_diffs_sums(B_temp)
                
                if can_add(b, A, diffs_A, sums_A, diffs_B_temp):
                    B.remove(b)
                    B_set.discard(b)
                    diffs_B, sums_B = diffs_B_temp, sums_B_temp
                    
                    add_to_set(b, A, diffs_A, sums_A)
                    A_set.add(b)
                    
                    made_neutral_move = True
                    moved = True
                    break
            
            if moved:
                stall_count += 1
                continue
            
            # 3. Try Swap (a in A with b in B)
            # Remove a from A, remove b from B.
            # Add b to A, add a to B.
            # Check validity.
            swapped = False
            for a in list(A):
                A_temp = [x for x in A if x != a]
                diffs_A_temp, sums_A_temp = rebuild_diffs_sums(A_temp)
                for b in list(B):
                    B_temp = [x for x in B if x != b]
                    diffs_B_temp, sums_B_temp = rebuild_diffs_sums(B_temp)
                    
                    # Check if b can be added to A_temp
                    # and a can be added to B_temp
                    # And new diffs are disjoint.
                    
                    # Check b in A_temp
                    if can_add(b, A_temp, diffs_A_temp, sums_A_temp, diffs_B_temp):
                         # Now check if a can be added to B_temp with the NEW diffs of A?
                         # No, can_add checks against diffs_other.
                         # If we add b to A_temp, diffs_A_temp changes.
                         # So we must simulate the addition to get new diffs_A.
                         
                         # Simulate adding b to A_temp
                         A_sim = A_temp + [b]
                         diffs_A_sim, sums_A_sim = rebuild_diffs_sums(A_sim)
                         
                         # Now check if a can be added to B_temp against diffs_A_sim
                         if can_add(a, B_temp, diffs_B_temp, sums_B_temp, diffs_A_sim):
                             # Valid Swap!
                             # Apply changes
                             A.remove(a)
                             A_set.discard(a)
                             B.remove(b)
                             B_set.discard(b)
                             
                             # Add b to A
                             add_to_set(b, A, diffs_A, sums_A) # Wait, diffs_A is old. 
                             # We need to reset A to A_temp first?
                             # Easier:
                             A[:] = A_temp
                             B[:] = B_temp
                             diffs_A, sums_A = rebuild_diffs_sums(A)
                             diffs_B, sums_B = rebuild_diffs_sums(B)
                             
                             A_set = set(A)
                             B_set = set(B)
                             
                             add_to_set(b, A, diffs_A, sums_A)
                             A_set.add(b)
                             add_to_set(a, B, diffs_B, sums_B)
                             B_set.add(a)
                             
                             made_neutral_move = True
                             swapped = True
                             break
                if swapped:
                    break
            
            if swapped:
                stall_count += 1
                continue
            
            # 4. If stuck, try a "Kick": Remove a random element and re-greedy?
            # Or just break and restart.
            if not made_improvement and not made_neutral_move:
                break
                
        return best_local_A, best_local_B, best_local_score

    def polish(A, B):
        """Final greedy add."""
        diffs_A, sums_A = rebuild_diffs_sums(A)
        diffs_B, sums_B = rebuild_diffs_sums(B)
        A_set = set(A)
        B_set = set(B)
        
        while True:
            if time.time() > deadline:
                break
            any_added = False
            candidates = [x for x in range(1, 257) if x not in A_set and x not in B_set]
            random.shuffle(candidates)
            for x in candidates:
                if can_add(x, A, diffs_A, sums_A, diffs_B):
                    add_to_set(x, A, diffs_A, sums_A)
                    A_set.add(x)
                    any_added = True
                    break
                if can_add(x, B, diffs_B, sums_B, diffs_A):
                    add_to_set(x, B, diffs_B, sums_B)
                    B_set.add(x)
                    any_added = True
                    break
            if not any_added:
                break
        
        return A, B

    attempts = 0
    while time.time() < deadline:
        attempts += 1
        random.seed(attempts * 98765 + 111)
        
        A = []
        B = []
        diffs_A = set()
        diffs_B = set()
        sums_A = set()
        sums_B = set()
        
        # Greedy Initialization
        # Alternate between A and B to balance sizes
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        for x in candidates:
            valid_A = can_add(x, A, diffs_A, sums_A, diffs_B)
            valid_B = can_add(x, B, diffs_B, sums_B, diffs_A)
            
            if valid_A and valid_B:
                if len(A) < len(B):
                    add_to_set(x, A, diffs_A, sums_A)
                elif len(B) < len(A):
                    add_to_set(x, B, diffs_B, sums_B)
                else:
                    if random.random() < 0.5:
                        add_to_set(x, A, diffs_A, sums_A)
                    else:
                        add_to_set(x, B, diffs_B, sums_B)
            elif valid_A:
                add_to_set(x, A, diffs_A, sums_A)
            elif valid_B:
                add_to_set(x, B, diffs_B, sums_B)
        
        # Local Search
        A, B, local_score = do_local_search(A, B)
        
        # Polish
        A, B = polish(A, B)
        
        score = len(A) + len(B)
        if score > best_score:
            best_score = score
            best_A = A.copy()
            best_B = B.copy()
            
        # Early stop if we hit theoretical max?
        # Max Sidon size in [1, 256] is 16.
        # If we get 32, we can stop.
        if best_score >= 32:
            break

    return [best_A, best_B]
