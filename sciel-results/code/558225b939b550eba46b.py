import time
import itertools

def construct():
    start_time = time.time()
    deadline = 55.0  # 55 seconds limit, keep a margin for safety
    
    # 1. Generate smooth numbers up to 128
    smooth_set = set()
    smooth_list = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth_set.add(x)
            smooth_list.append(x)
    
    smooth_list.sort()
    
    # Precompute divisibility relations for smooth numbers
    # divs[i] is a set of indices j such that smooth_list[i] divides smooth_list[j]
    # This helps in antichain check
    S = len(smooth_list)
    val_to_idx = {v: i for i, v in enumerate(smooth_list)}
    
    # Helper to check if a set of terms forms an antichain
    # terms are values
    def is_antichain(terms):
        # Convert to indices for faster lookup? Or just use values.
        # Values are small.
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                a, b = terms[i], terms[j]
                if a > b:
                    a, b = b, a
                if b % a == 0:
                    return False
        return True

    answers = [None] * 129  # Index 0 unused, 1..128

    for n in range(1, 129):
        # Check deadline
        if time.time() > deadline:
            # If we run out of time, we must return something valid.
            # Since we process in order, if we hit deadline, we have a problem.
            # But 128 targets is small. This should not happen.
            # For safety, fill remaining with a greedy valid solution if possible, or raise.
            # Given the speed, this is unlikely.
            pass

        found = False
        
        # Try 1 term
        if n in smooth_set:
            answers[n] = [n]
            found = True
        else:
            # Try 2 terms
            # x + y = n
            # x, y in smooth, x != y, x not divide y
            # Iterate x in smooth_list
            for x in smooth_list:
                if x >= n:
                    break
                y = n - x
                if y in smooth_set and y != x:
                    # Check antichain
                    if x < y:
                        if y % x != 0:
                            answers[n] = [x, y]
                            found = True
                            break
                    else: # y < x
                        if x % y != 0:
                            answers[n] = [x, y]
                            found = True
                            break
            if found:
                continue

            # Try 3 terms
            # x + y + z = n
            # x, y, z distinct, antichain
            for x in smooth_list:
                if x >= n:
                    break
                rem1 = n - x
                for y in smooth_list:
                    if y >= rem1:
                        break
                    z = rem1 - y
                    if z in smooth_set and z != x and z != y and x != y:
                        if is_antichain([x, y, z]):
                            answers[n] = [x, y, z]
                            found = True
                            break
                if found:
                    break
            if found:
                continue
            
            # Fallback for k >= 4
            # This is very unlikely to be needed for n <= 128
            # But we must ensure validity.
            # Use a generic backtracking search for minimal k
            # Since k=1,2,3 failed, try k=4
            candidates = [s for s in smooth_list if s < n]
            for k in range(4, 11):
                if len(candidates) < k:
                    break
                # Use combinations
                # For k=4, C(36,4) ~ 58k. Fast.
                for terms in itertools.combinations(candidates, k):
                    if sum(terms) == n and is_antichain(terms):
                        answers[n] = list(terms)
                        found = True
                        break
                if found:
                    break
        
        if not found:
            # Should not happen. 
            # If it does, we return an invalid state which scores 0.
            # We can try to force a solution by relaxing distinctness? No.
            # We can try to include non-smooth? No.
            # Pass
            pass

    return answers[1:]
