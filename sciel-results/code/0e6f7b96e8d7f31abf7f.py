import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    N = 256
    
    A = set()
    remaining = set(range(1, N + 1))
    
    def is_valid_add(x, A):
        # Check 4-AP
        # x is 1st
        for d in range(1, (N - x) // 3 + 1):
            if x + d in A and x + 2 * d in A and x + 3 * d in A:
                return False
        # x is 2nd
        for d in range(1, min(x - 1, (N - x) // 2) + 1):
            if x - d in A and x + d in A and x + 2 * d in A:
                return False
        # x is 3rd
        for d in range(1, min(x // 2, N - x) + 1):
            if x - 2 * d in A and x - d in A and x + d in A:
                return False
        # x is 4th
        for d in range(1, (x - 1) // 3 + 1):
            if x - 3 * d in A and x - 2 * d in A and x - d in A:
                return False
        return True

    def count_3ap_gain(x, A):
        gain = 0
        # x is 1st
        for d in range(1, (N - x) // 2 + 1):
            if x + d in A and x + 2 * d in A:
                gain += 1
        # x is 2nd
        for d in range(1, min(x - 1, N - x) + 1):
            if x - d in A and x + d in A:
                gain += 1
        # x is 3rd
        for d in range(1, x // 2 + 1):
            if x - 2 * d in A and x - d in A:
                gain += 1
        return gain

    def count_total_3ap(A):
        count = 0
        for b in A:
            for d in range(1, min(b - 1, N - b) + 1):
                if b - d in A and b + d in A:
                    count += 1
        return count

    def compute_loss(x, A):
        loss = 0
        for d in range(1, (N - x) // 2 + 1):
            if x + d in A and x + 2 * d in A:
                loss += 1
        for d in range(1, min(x - 1, N - x) + 1):
            if x - d in A and x + d in A:
                loss += 1
        for d in range(1, x // 2 + 1):
            if x - 2 * d in A and x - d in A:
                loss += 1
        return loss

    # Greedy Phase: pick element with max 3-AP gain that doesn't create 4-AP
    while remaining and time.time() < deadline - 40:
        best_x = -1
        best_score = -1
        for x in remaining:
            if is_valid_add(x, A):
                score = count_3ap_gain(x, A)
                if score > best_score:
                    best_score = score
                    best_x = x
        if best_x == -1:
            break
        A.add(best_x)
        remaining.remove(best_x)
    
    current_score = count_total_3ap(A)
    
    # Local Search Phase: 1-swap moves
    # Try to replace a in A with b in remaining if it improves score and maintains 4-AP-freeness
    improved = True
    max_passes = 50
    pass_count = 0
    
    while improved and pass_count < max_passes and time.time() < deadline:
        improved = False
        pass_count += 1
        
        # Compute loss for each element in A
        A_losses = {}
        for a in A:
            A_losses[a] = compute_loss(a, A)
        
        # Compute gain for each element in remaining (if valid to add to A)
        R_gains = {}
        for b in remaining:
            if is_valid_add(b, A):
                R_gains[b] = count_3ap_gain(b, A)
            else:
                R_gains[b] = -1
        
        # Sort A by loss ascending (remove low-value elements first)
        sorted_A = sorted(A_losses.keys(), key=lambda x: A_losses[x])
        # Sort R by gain descending (add high-value elements first)
        sorted_R = sorted(R_gains.keys(), key=lambda x: R_gains[x], reverse=True)
        
        # Try swaps: for each a in sorted_A (up to 10), for each b in sorted_R (up to 10)
        max_a_tries = 10
        max_b_tries = 10
        found_improvement = False
        
        for a in sorted_A[:max_a_tries]:
            if found_improvement:
                break
            if time.time() > deadline:
                break
                
            for b in sorted_R[:max_b_tries]:
                if time.time() > deadline:
                    break
                    
                # Skip if b is not valid to add to A (already filtered, but double-check)
                if R_gains[b] < 0:
                    continue
                    
                # Check if b is valid to add to A - {a}
                # Temporarily remove a
                A.remove(a)
                
                valid_b = is_valid_add(b, A)
                
                if valid_b:
                    # Compute exact gain of b in A - {a}
                    gain_b = count_3ap_gain(b, A)
                    # Loss of a is A_losses[a] (computed before removing a)
                    # New score = current_score - A_losses[a] + gain_b
                    new_score = current_score - A_losses[a] + gain_b
                    
                    if new_score > current_score:
                        # Accept swap
                        A.add(b)
                        remaining.remove(b)
                        current_score = new_score
                        improved = True
                        found_improvement = True
                        break
                else:
                    # Restore a
                    A.add(a)
            
            # If we didn't find an improvement with this a, restore a if it was removed
            if a not in A:
                A.add(a)
        
        # If we found an improvement, the set has changed, so restart the pass
        # (the while loop will handle this)
    
    return sorted(A)
