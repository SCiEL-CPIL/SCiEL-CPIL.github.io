import random
import time
import math

def construct():
    n = 256
    max_size = 64
    
    # Helper to check if x can be added to a set with given sums/diffs
    # Returns True if valid, False otherwise
    def can_add_to_A(x, A, sums_A, diffs_B):
        if 2 * x in sums_A:
            return False
        for a in A:
            if (a + x) in sums_A:
                return False
            d = x - a
            if d != 0 and d in diffs_B:
                return False
        return True

    def can_add_to_B(x, B, sums_B, diffs_A):
        if 2 * x in sums_B:
            return False
        for b in B:
            if (b + x) in sums_B:
                return False
            d = x - b
            if d != 0 and d in diffs_A:
                return False
        return True

    def add_to_A(x, A, sums_A, diffs_A):
        for a in A:
            sums_A.add(a + x)
            d = x - a
            if d != 0:
                diffs_A.add(d)
                diffs_A.add(-d)
        sums_A.add(2 * x)
        A.append(x)
        
    def remove_from_A(x, A, sums_A, diffs_A):
        # Rebuild sums and diffs is easier and fast enough for size < 64
        A.remove(x)
        sums_A.clear()
        diffs_A.clear()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums_A.add(A[i] + A[j])
        for i in range(len(A)):
            for j in range(len(A)):
                if i != j:
                    d = A[i] - A[j]
                    diffs_A.add(d)
                    diffs_A.add(-d)

    def add_to_B(x, B, sums_B, diffs_B):
        for b in B:
            sums_B.add(b + x)
            d = x - b
            if d != 0:
                diffs_B.add(d)
                diffs_B.add(-d)
        sums_B.add(2 * x)
        B.append(x)

    def remove_from_B(x, B, sums_B, diffs_B):
        B.remove(x)
        sums_B.clear()
        diffs_B.clear()
        for i in range(len(B)):
            for j in range(i, len(B)):
                sums_B.add(B[i] + B[j])
        for i in range(len(B)):
            for j in range(len(B)):
                if i != j:
                    d = B[i] - B[j]
                    diffs_B.add(d)
                    diffs_B.add(-d)

    # Initial construction using greedy
    def initial_greedy():
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        for x in range(1, n + 1):
            if len(A) >= max_size and len(B) >= max_size:
                break
            a_ok = len(A) < max_size and can_add_to_A(x, A, sums_A, diffs_B)
            b_ok = len(B) < max_size and can_add_to_B(x, B, sums_B, diffs_A)
            
            if a_ok and b_ok:
                if len(A) <= len(B):
                    add_to_A(x, A, sums_A, diffs_A)
                else:
                    add_to_B(x, B, sums_B, diffs_B)
            elif a_ok:
                add_to_A(x, A, sums_A, diffs_A)
            elif b_ok:
                add_to_B(x, B, sums_B, diffs_B)
                
        return A, B, sums_A, sums_B, diffs_A, diffs_B

    A, B, sums_A, sums_B, diffs_A, diffs_B = initial_greedy()
    current_score = len(A) + len(B)
    best_score = current_score
    best_A, best_B = A[:], B[:]
    
    start_time = time.time()
    T = 100.0 # Initial temperature
    T_min = 0.1
    cooling = 0.999
    
    universe = list(range(1, n + 1))
    
    while time.time() - start_time < 55:
        T *= cooling
        if T < T_min:
            T = T_min
            
        move_type = random.random()
        
        # Try to apply a move
        new_score = current_score
        new_A = A
        new_B = B
        new_sums_A = sums_A
        new_sums_B = sums_B
        new_diffs_A = diffs_A
        new_diffs_B = diffs_B
        
        # We need to be careful with state mutation. 
        # Since sets are mutable, we should copy or revert.
        # Copying sets of size ~2000 is fast.
        
        if move_type < 0.4: # Add
            x = random.choice(universe)
            if x not in A and x not in B:
                # Try add to A
                if len(A) < max_size and can_add_to_A(x, A, sums_A, diffs_B):
                    new_A = A + [x]
                    new_sums_A = set(sums_A)
                    new_diffs_A = set(diffs_A)
                    add_to_A(x, new_A, new_sums_A, new_diffs_A)
                    # Wait, add_to_A modifies A in place. 
                    # I need a version that doesn't modify original, or I revert.
                    # Let's use a "tentative" approach.
                    
                    # Actually, let's just copy the state for the trial.
                    trial_A = A[:]
                    trial_sums_A = set(sums_A)
                    trial_diffs_A = set(diffs_A)
                    add_to_A(x, trial_A, trial_sums_A, trial_diffs_A)
                    
                    if len(trial_A) > len(A):
                        new_score = current_score + 1
                        new_A = trial_A
                        new_sums_A = trial_sums_A
                        new_diffs_A = trial_diffs_A
                        # B remains same
                        # Check if this move is accepted
                        # Delta = 1
                        if random.random() < math.exp(1 / T):
                            A = new_A
                            sums_A = new_sums_A
                            diffs_A = new_diffs_A
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_A, best_B = A[:], B[:]
                            continue # Restart loop to avoid double moves? Or just proceed?
                        else:
                            # Reject
                            pass
                    # If add to A failed, try B
                if len(B) < max_size and can_add_to_B(x, B, sums_B, diffs_A):
                    trial_B = B[:]
                    trial_sums_B = set(sums_B)
                    trial_diffs_B = set(diffs_B)
                    add_to_B(x, trial_B, trial_sums_B, trial_diffs_B)
                    
                    if len(trial_B) > len(B):
                        new_score = current_score + 1
                        if random.random() < math.exp(1 / T):
                            B = trial_B
                            sums_B = trial_sums_B
                            diffs_B = trial_diffs_B
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_A, best_B = A[:], B[:]
                            continue
                            
        elif move_type < 0.7: # Swap in A
            if A:
                x = random.choice(A)
                y = random.choice(universe)
                if y != x and y not in A:
                    # Remove x, Add y
                    # Check if y can be added to A without x
                    # This is complex. Simpler: Remove x, then try add y.
                    # But we need to ensure the intermediate state is valid? No, just the final state.
                    # However, removing x changes sums/diffs.
                    
                    trial_A = A[:]
                    trial_sums_A = set(sums_A)
                    trial_diffs_A = set(diffs_A)
                    remove_from_A(x, trial_A, trial_sums_A, trial_diffs_A)
                    
                    if can_add_to_A(y, trial_A, trial_sums_A, diffs_B):
                        add_to_A(y, trial_A, trial_sums_A, trial_diffs_A)
                        # Size is same. 
                        # Acceptance: Always accept equal size moves to explore? 
                        # Or use SA with delta=0? exp(0/T)=1. So always accept.
                        # This might wander. Maybe only accept if it allows future adds? 
                        # For now, accept equal size moves with prob 0.5 or so to avoid loops?
                        # Let's accept equal size moves always for now, but limit swaps.
                        if random.random() < 0.5: # Random walk
                            A = trial_A
                            sums_A = trial_sums_A
                            diffs_A = trial_diffs_A
                            # current_score unchanged
                            if current_score > best_score:
                                best_A, best_B = A[:], B[:]
                            
        elif move_type < 0.9: # Swap in B
            if B:
                x = random.choice(B)
                y = random.choice(universe)
                if y != x and y not in B:
                    trial_B = B[:]
                    trial_sums_B = set(sums_B)
                    trial_diffs_B = set(diffs_B)
                    remove_from_B(x, trial_B, trial_sums_B, trial_diffs_B)
                    
                    if can_add_to_B(y, trial_B, trial_sums_B, diffs_A):
                        add_to_B(y, trial_B, trial_sums_B, trial_diffs_B)
                        if random.random() < 0.5:
                            B = trial_B
                            sums_B = trial_sums_B
                            diffs_B = trial_diffs_B
                            if current_score > best_score:
                                best_A, best_B = A[:], B[:]
                            
        else: # Transfer A to B or B to A
            if A and len(B) < max_size:
                x = random.choice(A)
                # Remove from A, Add to B
                trial_A = A[:]
                trial_sums_A = set(sums_A)
                trial_diffs_A = set(diffs_A)
                remove_from_A(x, trial_A, trial_sums_A, trial_diffs_A)
                
                if can_add_to_B(x, B, sums_B, trial_diffs_A): # Note: diffs_A changed
                    # Wait, can_add_to_B checks against diffs_A. 
                    # The new diffs_A is trial_diffs_A.
                    # So we check if x can be added to B given the NEW diffs_A?
                    # No, the constraint is disjointness between final A and final B.
                    # Final A is trial_A. Final B is B + [x].
                    # So we need to check if diffs of (B+[x]) are disjoint from diffs of trial_A.
                    # can_add_to_B(x, B, sums_B, diffs_A_trial) does exactly this?
                    # Let's check can_add_to_B definition.
                    # It checks if new diffs of B collide with diffs_A passed as arg.
                    # So yes, pass trial_diffs_A.
                    
                    trial_B = B[:]
                    trial_sums_B = set(sums_B)
                    trial_diffs_B = set(diffs_B)
                    add_to_B(x, trial_B, trial_sums_B, trial_diffs_B)
                    
                    # Size is same.
                    if random.random() < 0.5:
                        A = trial_A
                        B = trial_B
                        sums_A = trial_sums_A
                        sums_B = trial_sums_B
                        diffs_A = trial_diffs_A
                        diffs_B = trial_diffs_B

    return [best_A, best_B]
