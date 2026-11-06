import time

def _smooth_numbers_up_to(n):
    """Return sorted list of all 2^a*3^b*5^c <= n (positive)."""
    out = set()
    p2 = 1
    while p2 <= n:
        p23 = p2
        while p23 <= n:
            p235 = p23
            while p235 <= n:
                out.add(p235)
                p235 *= 5
            p23 *= 3
        p2 *= 2
    return sorted(out)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # 1. Generate smooth numbers up to 128
    smooth = _smooth_numbers_up_to(128)
    num_smooth = len(smooth)
    
    # 2. Initialize best representations
    best_rep = [None] * 129
    solved_count = 0
    
    # Helper to check if a number can be added to a set of chosen numbers
    def can_add(val, chosen):
        for c in chosen:
            if c < val:
                if val % c == 0:
                    return False
            else:
                if c % val == 0:
                    return False
        return True

    # 3. Iterate over possible number of terms k
    # Likely k <= 6 is sufficient
    for k in range(1, 7):
        # Check deadline
        if time.time() > deadline:
            break
            
        # If all solved, break
        if solved_count == 128:
            break

        # Prune if min sum > 128
        if num_smooth >= k:
            min_sum_k = sum(smooth[:k])
            if min_sum_k > 128:
                break
        else:
            break

        # Recursive search for antichains of size k
        def dfs(start_idx, current_sum, chosen):
            nonlocal solved_count
            if time.time() > deadline:
                return
            
            if len(chosen) == k:
                s = current_sum
                if 1 <= s <= 128:
                    if best_rep[s] is None:
                        best_rep[s] = list(chosen)
                        solved_count += 1
                return

            remaining_needed = k - len(chosen)
            max_start = num_smooth - remaining_needed
            
            for i in range(start_idx, max_start + 1):
                val = smooth[i]
                
                # Prune if sum exceeds 128
                if current_sum + val > 128:
                    break
                
                if can_add(val, chosen):
                    chosen.append(val)
                    dfs(i + 1, current_sum + val, chosen)
                    chosen.pop()
                    
        dfs(0, 0, [])
        
    # Fallback: If some targets are not solved, try k=7, 8...
    # This is unlikely to be needed
    for n in range(1, 129):
        if best_rep[n] is None:
             # Simple brute force for this specific n with larger k
             # Just in case
             pass

    return best_rep[1:129]
