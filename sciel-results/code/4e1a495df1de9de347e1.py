import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    MIN_SUM = 128
    MAX_SUM = 256
    N = 129  # Elements 0..128
    
    # Precompute neighbors for fast updates? 
    # When adding a, we affect sums a+b for b in A.
    
    def get_initial_set():
        # Heuristic initialization: Two intervals at the ends to cover extreme sums efficiently?
        # Or a random set that is repaired.
        # Let's try a set that is a union of two intervals: [0, L] and [128-R, 128]
        # This helps cover low sums (0+0..L+L) and high sums (128+128..128+128-R)
        # But we need to cover [128, 256].
        # 0+128=128, 128+128=256.
        # If we have [0, 10] and [118, 128], sums cover [0, 20] U [236, 256] U [118, 138]... 
        # This leaves a huge gap in the middle.
        # Better: Random set, then repair gaps by adding elements that cover the most gaps.
        
        A = set()
        # Start with a random set of size ~30
        k = 30
        A = set(random.sample(range(N), k))
        
        # Repair phase: While there are gaps, add an element that covers the most gaps
        # Calculate current gaps
        mult = np.zeros(MAX_SUM + 1, dtype=np.int32)
        for a in A:
            for b in A:
                if a <= b: # Unordered, but for count it's easier to just do ordered and divide later? 
                    # No, for gap checking, we just need > 0.
                    # Let's stick to the incremental update logic defined below.
                    pass
        
        # Let's just use the fast incremental logic to repair.
        # Initialize mult array
        mult = np.zeros(MAX_SUM + 1, dtype=np.int32)
        for a in A:
            for b in A:
                if a <= b:
                    mult[a+b] += 1
        
        # Check gaps
        target_range = range(MIN_SUM, MAX_SUM + 1)
        
        max_repair_steps = 50
        for _ in range(max_repair_steps):
            gaps = [s for s in target_range if mult[s] == 0]
            if not gaps:
                break
            
            # Find element x in [0, 128] not in A that covers the most gaps
            # For a candidate x, it covers gap s if s-x is in A and s-x >= x (or just s-x in A for existence)
            # Actually, if we add x, it forms pairs (x, y) for all y in A.
            # New sums: x + y for y in A.
            # We want to maximize the number of new sums that are in `gaps`.
            
            best_x = None
            best_gain = -1
            # Sample candidates to save time? Or iterate all 129? 129 is small.
            for x in range(N):
                if x in A:
                    continue
                gain = 0
                for y in A:
                    s = x + y
                    if MIN_SUM <= s <= MAX_SUM and mult[s] == 0:
                        gain += 1
                # Also check if x+x covers a gap
                if MIN_SUM <= 2*x <= MAX_SUM and mult[2*x] == 0:
                    gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None or best_gain == 0:
                # Can't repair with single additions? Break.
                break
            
            # Add best_x
            A.add(best_x)
            # Update mult
            for y in A:
                if y == best_x:
                    continue
                s = best_x + y
                if s <= MAX_SUM:
                    mult[s] += 1
            s = 2 * best_x
            if s <= MAX_SUM:
                mult[s] += 1
                
        return A, mult

    def calc_score(A, mult):
        # mult is array of counts for unordered pairs
        # Check gaps
        for s in range(MIN_SUM, MAX_SUM + 1):
            if mult[s] == 0:
                return float('inf')
        peak = np.max(mult[MIN_SUM:MAX_SUM+1])
        return peak

    def local_search(A, mult, deadline):
        current_score = calc_score(A, mult)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Identify bottleneck sums
            # Find max multiplicity
            target_range = np.arange(MIN_SUM, MAX_SUM + 1)
            mult_slice = mult[MIN_SUM:MAX_SUM+1]
            max_mult = np.max(mult_slice)
            
            if max_mult <= 1:
                break
                
            # Get indices of sums with max multiplicity
            high_mult_indices = np.where(mult_slice == max_mult)[0] + MIN_SUM
            
            # Pick a random bottleneck sum
            target_sum = random.choice(high_mult_indices)
            
            # Get contributors to this sum
            # Pairs (a, b) in A such that a+b = target_sum and a <= b
            contributors = []
            for a in A:
                b = target_sum - a
                if b in A and a <= b:
                    contributors.append(a)
            
            if not contributors:
                continue
            
            A_list = list(A)
            available = [x for x in range(N) if x not in A]
            
            # Strategy 1: 1-Swap (Remove a, Add b)
            # We want to reduce mult[target_sum].
            # Removing a contributor reduces mult[target_sum] by 1.
            # Adding b might increase mult[target_sum] if target_sum - b is in A.
            
            random.shuffle(contributors)
            for a in contributors:
                if time.time() > deadline:
                    break
                    
                # Try removing a and adding a random b
                # To be efficient, try a few random b's
                num_candidates = min(10, len(available))
                if num_candidates == 0:
                    continue
                candidates = random.sample(available, num_candidates)
                
                for b in candidates:
                    # Check if b helps or hurts
                    # We remove a.
                    # We add b.
                    
                    # Delta for mult[target_sum]:
                    # -1 for pair (a, target_sum-a)
                    # +1 if (b, target_sum-b) is a valid pair in NEW A
                    # Note: if b == target_sum - a, then we are swapping a with b.
                    # If b is such that target_sum - b is in A (and != b, or == b), then mult[target_sum] increases.
                    
                    # Let's just simulate the change on a copy of mult? 
                    # Copying 257 ints is fast.
                    new_mult = mult.copy()
                    
                    # Remove a
                    for y in A:
                        if y == a:
                            continue
                        s = a + y
                        if s <= MAX_SUM:
                            new_mult[s] -= 1
                    s_aa = 2*a
                    if s_aa <= MAX_SUM:
                        new_mult[s_aa] -= 1
                        
                    # Add b
                    for y in A:
                        if y == b:
                            continue
                        s = b + y
                        if s <= MAX_SUM:
                            new_mult[s] += 1
                    s_bb = 2*b
                    if s_bb <= MAX_SUM:
                        new_mult[s_bb] += 1
                        
                    new_score = calc_score(A, new_mult)
                    if new_score < current_score:
                        # Accept move
                        # Update A
                        A.remove(a)
                        A.add(b)
                        # Update mult in place (we already have new_mult)
                        mult[:] = new_mult[:]
                        current_score = new_score
                        improved = True
                        # Recalculate available
                        available = [x for x in range(N) if x not in A]
                        break
                if improved:
                    break
            
            if improved:
                continue
                
            # Strategy 2: 2-Swap (Remove a1, a2, Add b1, b2)
            # This is more complex.
            # Pick two contributors a1, a2.
            # Pick two random available b1, b2.
            
            if len(contributors) >= 2 and len(available) >= 2:
                random.shuffle(contributors)
                a1, a2 = contributors[0], contributors[1]
                # Ensure a1 != a2
                if a1 == a2:
                    continue
                    
                num_candidates = min(5, len(available))
                candidates = random.sample(available, num_candidates)
                # We need to pick 2 distinct b's from candidates? Or just iterate pairs?
                # With 5 candidates, 10 pairs. Manageable.
                
                for i in range(len(candidates)):
                    for j in range(i+1, len(candidates)):
                        b1, b2 = candidates[i], candidates[j]
                        
                        new_mult = mult.copy()
                        
                        # Remove a1
                        for y in A:
                            if y == a1:
                                continue
                            s = a1 + y
                            if s <= MAX_SUM:
                                new_mult[s] -= 1
                        s = 2*a1
                        if s <= MAX_SUM:
                            new_mult[s] -= 1
                            
                        # Remove a2
                        for y in A:
                            if y == a2:
                                continue
                            s = a2 + y
                            if s <= MAX_SUM:
                                new_mult[s] -= 1
                        s = 2*a2
                        if s <= MAX_SUM:
                            new_mult[s] -= 1
                            
                        # Add b1
                        for y in A:
                            if y == b1:
                                continue
                            s = b1 + y
                            if s <= MAX_SUM:
                                new_mult[s] += 1
                        s = 2*b1
                        if s <= MAX_SUM:
                            new_mult[s] += 1
                            
                        # Add b2
                        for y in A:
                            if y == b2:
                                continue
                            s = b2 + y
                            if s <= MAX_SUM:
                                new_mult[s] += 1
                        s = 2*b2
                        if s <= MAX_SUM:
                            new_mult[s] += 1
                            
                        new_score = calc_score(A, new_mult)
                        if new_score < current_score:
                            A.remove(a1)
                            A.remove(a2)
                            A.add(b1)
                            A.add(b2)
                            mult[:] = new_mult[:]
                            current_score = new_score
                            improved = True
                            available = [x for x in range(N) if x not in A]
                            break
                    if improved:
                        break
                        
        return current_score, A

    # Main loop
    best_score = float('inf')
    best_A = None
    
    # Multiple restarts
    for restart in range(20):
        if time.time() > deadline - 5:
            break
            
        A, mult = get_initial_set()
        
        # Run local search
        score, A = local_search(A, mult, deadline)
        
        if score < best_score:
            best_score = score
            best_A = A
            
        if best_score == 1:
            break
            
    # Final polish: Try to remove elements if possible (0-swap)
    if best_A is not None:
        mult = np.zeros(MAX_SUM + 1, dtype=np.int32)
        for a in best_A:
            for b in best_A:
                if a <= b:
                    mult[a+b] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            A_list = list(best_A)
            random.shuffle(A_list)
            for a in A_list:
                if time.time() > deadline:
                    break
                new_mult = mult.copy()
                # Remove a
                for y in best_A:
                    if y == a:
                        continue
                    s = a + y
                    if s <= MAX_SUM:
                        new_mult[s] -= 1
                s = 2*a
                if s <= MAX_SUM:
                    new_mult[s] -= 1
                    
                new_score = calc_score(best_A, new_mult)
                if new_score <= best_score:
                    best_A.remove(a)
                    mult = new_mult
                    best_score = new_score
                    improved = True
                    break
                    
    if best_A is None:
        # Fallback
        return list(range(64, 129))
        
    return sorted(best_A)

if __name__ == "__main__":
    start = time.time()
    A = construct()
    end = time.time()
    print(f"Time: {end-start:.2f}s")
    print(f"Set size: {len(A)}")
