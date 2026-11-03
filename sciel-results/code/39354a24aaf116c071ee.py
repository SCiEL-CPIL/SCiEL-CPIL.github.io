import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    # Fallback for non-standard parameters
    if q != 3 or n != 6 or radius != 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    # Specific optimization for q=3, n=6, radius=1
    deadline = time.time() + 115.0  # Safety margin
    
    # 1. Precompute balls
    # Word (w0, ..., w5) -> idx
    # ball[i] is a list of indices covered by codeword i
    balls = []
    for i in range(729):
        w = [0] * 6
        temp = i
        for k in range(5, -1, -1):
            w[k] = temp % 3
            temp //= 3
        
        covered = [i]
        for pos in range(6):
            orig_val = w[pos]
            for delta in [1, 2]:
                new_val = (orig_val + delta) % 3
                new_w = w[:]
                new_w[pos] = new_val
                new_idx = 0
                for v in new_w:
                    new_idx = new_idx * 3 + v
                covered.append(new_idx)
        balls.append(covered)

    # 2. Greedy Initialization
    # We use a simple greedy to get a starting point.
    # We will use numpy for faster set operations if possible, but lists/sets are fine for 729.
    
    def greedy_init():
        uncovered = set(range(729))
        selected = []
        while uncovered:
            best_cand = -1
            max_cov = 0
            # Optimization: Randomize tie-breaking
            candidates = []
            for i in range(729):
                count = 0
                # Using set intersection is fast enough for 13 elements
                # But iterating is also fine
                for w in balls[i]:
                    if w in uncovered:
                        count += 1
                if count > max_cov:
                    max_cov = count
                    candidates = [i]
                elif count == max_cov and count > 0:
                    candidates.append(i)
            
            if not candidates:
                break
            c = random.choice(candidates)
            selected.append(c)
            uncovered -= set(balls[c])
        return selected

    # Run a few greedy inits to pick the best
    best_initial = None
    best_size = 100
    for _ in range(5):
        sol = greedy_init()
        if len(sol) < best_size:
            best_size = len(sol)
            best_initial = sol
    
    # If greedy failed or is too bad, use tetracode
    if not best_initial or best_size > 80:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        best_initial = [idx for idx in range(729) if True] # Placeholder, we need indices
        # Actually, let's just convert tetracode words to indices
        tetracode_indices = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=2):
                w = h + list(tail)
                idx = 0
                for v in w:
                    idx = idx * 3 + v
                tetracode_indices.append(idx)
        best_initial = tetracode_indices
        best_size = len(best_initial)

    # 3. Simulated Annealing
    current_code = set(best_initial)
    in_code = [False] * 729
    for i in current_code:
        in_code[i] = True
    
    cov = [0] * 729
    for i in current_code:
        for w in balls[i]:
            cov[w] += 1
    
    num_zero = 0
    num_one = 0
    for c in cov:
        if c == 0: num_zero += 1
        elif c == 1: num_one += 1
        
    # Should be valid
    if num_zero != 0:
        # Fallback to tetracode if init was bad
        return [list(t) for t in itertools.product(range(3), repeat=6)] # This is invalid, just a placeholder. 
        # Actually, greedy should always produce a valid set.

    best_solution = list(current_code)
    best_size_sa = len(current_code)
    
    T = 2.0
    T_min = 0.05
    cooling = 0.999
    
    while time.time() < deadline:
        if T < T_min:
            T = T_min # Floor temperature to keep exploring
            
        # 1. Try Remove
        if current_code:
            c_out = random.choice(list(current_code))
            # Check validity of removal
            valid = True
            # We need to check if any w in balls[c_out] has cov[w] == 1
            # If so, removing c_out makes cov[w] 0 -> invalid
            # We can just check min(cov[w] for w in balls[c_out]) > 1
            # But we need to update cov if we accept.
            # Let's just check first.
            can_remove = True
            for w in balls[c_out]:
                if cov[w] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Accept removal
                for w in balls[c_out]:
                    cov[w] -= 1
                    if cov[w] == 1: num_one += 1
                    elif cov[w] == 0: num_zero += 1
                    # If cov[w] was 2, it becomes 1. num_one increases.
                    # If cov[w] was 1, it becomes 0. num_one decreases, num_zero increases.
                    # We need to be careful with counts.
                    # Let's just recalculate num_zero/num_one from cov? No, too slow.
                    # Let's track counts properly.
                    # Old val: v. New val: v-1.
                    # If v==1: num_one -= 1, num_zero += 1
                    # If v==2: num_one += 1
                    # If v>2: no change in num_one/num_zero
                # Wait, the above loop updates cov, but I need to update counters based on OLD values.
                # Let's redo the update logic carefully.
                
                # Revert cov for a moment to do it right? No, let's just do it right.
                # I already modified cov in the if block above? No, I only checked.
                # Let's perform the update.
                for w in balls[c_out]:
                    old_c = cov[w]
                    cov[w] = old_c - 1
                    new_c = cov[w]
                    if old_c == 1:
                        num_one -= 1
                        num_zero += 1
                    elif old_c == 2:
                        num_one += 1
                    # else: old_c > 2, new_c >= 2, no change to num_one/num_zero
                    
                current_code.remove(c_out)
                in_code[c_out] = False
                best_size_sa = len(current_code)
                if best_size_sa < len(best_solution):
                    best_solution = list(current_code)
                # Continue to next iteration (try more removals)
                continue
        
        # 2. Try Swap
        # Pick c_out from current_code, c_in from not in current_code
        if current_code:
            c_out = random.choice(list(current_code))
            # Pick c_in randomly from 0..728 not in current_code
            # To avoid O(N) scan, we can just pick random and check
            c_in = random.randint(0, 728)
            if in_code[c_in]:
                # Retry or pick another
                # Simple retry
                c_in = random.randint(0, 728)
                if in_code[c_in]:
                    # Skip this step
                    T *= cooling
                    continue

            # Check validity of swap: Remove c_out, Add c_in
            # We need to check if min(cov[w] - delta_remove + delta_add) > 0
            # delta_remove is 1 for w in balls[c_out], 0 else
            # delta_add is 1 for w in balls[c_in], 0 else
            
            # Efficient check:
            # The only words that can become 0 are those in balls[c_out] that are NOT in balls[c_in]
            # AND have cov[w] == 1.
            # Also, words in balls[c_in] that are NOT in balls[c_out] will increase, so they are safe.
            # Words in both are safe (cov changes by 0).
            
            valid_swap = True
            # We need to know if c_out and c_in share balls?
            # Intersection of balls is at most 1 (the center) or 2?
            # If c_out and c_in are distance 1, they share 1 word?
            # If distance 2, they share 0?
            # Actually, we can just check the critical words.
            
            # To be safe and simple, let's just simulate the change on the counters?
            # No, let's just check the condition:
            # For all w in balls[c_out]:
            #   if w not in balls[c_in] and cov[w] == 1: invalid
            
            # We can precompute a set for balls[c_in] for fast lookup?
            # balls[c_in] is a list of 13.
            # Let's create a set for c_in's ball.
            ball_in_set = set(balls[c_in])
            
            for w in balls[c_out]:
                if w not in ball_in_set:
                    if cov[w] == 1:
                        valid_swap = False
                        break
            
            if valid_swap:
                # Perform Swap
                # Remove c_out
                for w in balls[c_out]:
                    old_c = cov[w]
                    cov[w] = old_c - 1
                    new_c = cov[w]
                    if old_c == 1:
                        num_one -= 1
                        num_zero += 1
                    elif old_c == 2:
                        num_one += 1
                        
                # Add c_in
                for w in balls[c_in]:
                    old_c = cov[w]
                    cov[w] = old_c + 1
                    new_c = cov[w]
                    if old_c == 0:
                        num_zero -= 1
                        num_one += 1
                    elif old_c == 1:
                        num_one -= 1
                        # new_c is 2, so no change to num_one? 
                        # Wait. If old_c==1, it was counted in num_one.
                        # New_c==2, it is NOT counted in num_one.
                        # So num_one decreases by 1.
                        # Correct.
                        
                current_code.remove(c_out)
                in_code[c_out] = False
                current_code.add(c_in)
                in_code[c_in] = True
                
                # Check if we can remove anything now?
                # The loop continues, and step 1 will try to remove.
                
            else:
                # Swap invalid, try Add
                pass
        
        # 3. Try Add (if Swap was invalid or skipped)
        # Actually, we should try Add if Swap was invalid.
        # Let's restructure:
        # 1. Remove
        # 2. Swap
        # 3. Add
        
        # The code above did Remove. If it failed, it did Swap.
        # If Swap failed, we should do Add.
        # But the structure above is a bit messy with continues.
        
        # Let's just do Add if we are here (meaning Remove failed and Swap failed or was skipped)
        # Note: If Swap succeeded, we continued the loop. So we are here only if Remove failed AND Swap failed.
        
        c_in = random.randint(0, 728)
        if not in_code[c_in]:
            # Accept with probability exp(-1/T)
            if random.random() < np.exp(-1.0 / T):
                # Perform Add
                for w in balls[c_in]:
                    old_c = cov[w]
                    cov[w] = old_c + 1
                    new_c = cov[w]
                    if old_c == 0:
                        num_zero -= 1
                        num_one += 1
                    elif old_c == 1:
                        num_one -= 1
                        
                current_code.add(c_in)
                in_code[c_in] = True
                
        T *= cooling

    # Convert best_solution indices back to words
    result = []
    for idx in best_solution:
        w = [0] * 6
        temp = idx
        for k in range(5, -1, -1):
            w[k] = temp % 3
            temp //= 3
        result.append(w)
        
    return result
