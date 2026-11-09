import time
import random
import sys

# EVOLVE-BLOCK-START

def _is_valid_full(values):
    """Check if a full sequence has all distinct consecutive block sums."""
    n = len(values)
    seen = set()
    for i in range(n):
        total = 0
        for j in range(i, n):
            total += values[j]
            if total in seen:
                return False
            seen.add(total)
    return True

def _greedy_forward(values, max_val=256):
    """Greedy: pick smallest valid next element."""
    current = list(values)
    cur_sums = set()
    for i in range(len(current)):
        total = 0
        for j in range(i, len(current)):
            total += current[j]
            cur_sums.add(total)
    
    for x in range((current[-1] if current else 0) + 1, max_val + 1):
        tails = [x]
        total = x
        for a in reversed(current):
            total += a
            tails.append(total)
        if not cur_sums.intersection(tails):
            current.append(x)
            cur_sums.update(tails)
    return current

def _max_extension_by_sum_capacity(k_curr, t_curr, last_val, max_val=256):
    """
    Estimate the maximum number of additional elements 'm' we can add
    such that the total number of distinct block sums does not exceed
    the maximum possible total sum.
    
    Current length: k_curr
    Current total sum: t_curr
    Current last value: last_val
    
    If we add m elements, the new length is k = k_curr + m.
    Total number of block sums = k * (k + 1) / 2.
    
    The maximum possible total sum of the new sequence:
    The new elements must be > last_val and <= max_val.
    To maximize the sum, we pick the largest m available values.
    Available values: last_val + 1, ..., max_val.
    Count of available values: N_avail = max_val - last_val.
    If m > N_avail, return N_avail (value bound).
    
    Max sum of m largest values from (last_val+1 ... max_val):
    Sum = m * max_val - (m * (m - 1)) / 2
    
    Total Max Sum T_max = t_curr + m * max_val - m * (m - 1) / 2
    
    Constraint:
    k * (k + 1) / 2 <= T_max
    
    We solve for the largest integer m >= 0 satisfying this.
    """
    if last_val >= max_val:
        return 0
    
    n_avail = max_val - last_val
    if n_avail == 0:
        return 0
    
    # Binary search for m in [0, n_avail]
    lo, hi = 0, n_avail
    best_m = 0
    
    while lo <= hi:
        mid = (lo + hi) // 2
        k_new = k_curr + mid
        num_sums = k_new * (k_new + 1) // 2
        
        # Max sum with 'mid' additional elements
        # The elements are the 'mid' largest available: 
        # max_val, max_val-1, ..., max_val-mid+1
        # Sum = mid * max_val - (0 + 1 + ... + mid-1) = mid * max_val - mid*(mid-1)/2
        max_add_sum = mid * max_val - mid * (mid - 1) // 2
        t_max = t_curr + max_add_sum
        
        if num_sums <= t_max:
            best_m = mid
            lo = mid + 1
        else:
            hi = mid - 1
            
    return best_m

def _dfs_csp(start_seq, max_val=256, deadline=None, best_len=0):
    """
    DFS with Backtracking and Sum-Capacity Pruning.
    """
    if deadline and time.time() > deadline:
        return start_seq, best_len

    current = list(start_seq)
    if not current:
        return current, best_len
        
    last_val = current[-1]
    k_curr = len(current)
    
    # Calculate current sums and total sum
    cur_sums = set()
    t_curr = 0
    for i in range(k_curr):
        total = 0
        for j in range(i, k_curr):
            total += current[j]
            cur_sums.add(total)
        t_curr += current[i] # This is wrong, t_curr is sum of all elements
    
    # Recalculate t_curr correctly
    t_curr = sum(current)
    
    # Pruning: Value bound
    max_by_value = max_val - last_val
    # Pruning: Sum capacity bound
    max_by_sum = _max_extension_by_sum_capacity(k_curr, t_curr, last_val, max_val)
    
    max_possible_additional = min(max_by_value, max_by_sum)
    
    if k_curr + max_possible_additional <= best_len:
        return current, best_len

    best_found = list(current)
    
    def dfs(seq, sums_set, total_sum, last, k):
        nonlocal best_found, best_len
        if deadline and time.time() > deadline:
            return
        
        if k > best_len:
            best_len = k
            best_found = list(seq)
            
        if last >= max_val:
            return
            
        # Pruning bounds
        max_by_val = max_val - last
        max_by_sum = _max_extension_by_sum_capacity(k, total_sum, last, max_val)
        max_add = min(max_by_val, max_by_sum)
        
        if k + max_add <= best_len:
            return
            
        # Try candidates
        # Optimization: Try candidates that are "promising" first?
        # Smaller values leave more room for sum?
        # Actually, smaller values increase the "capacity" for more elements?
        # No, smaller values reduce the total sum, which reduces the RHS of the capacity bound.
        # So LARGER values might allow more elements by increasing the sum capacity?
        # Let's check: T_max = T_curr + Sum(New).
        # If we pick larger values, T_max is larger.
        # The LHS (num_sums) is fixed by k.
        # So larger values relax the constraint.
        # Therefore, we should try LARGER values first to keep the capacity high?
        # But we also need strictly increasing.
        # Let's try iterating from last+1 to max_val.
        
        for x in range(last + 1, max_val + 1):
            if deadline and time.time() > deadline:
                return
            
            # Check validity of x
            tails = [x]
            total = x
            for a in reversed(seq):
                total += a
                tails.append(total)
            
            # Check intersection
            valid = True
            for t in tails:
                if t in sums_set:
                    valid = False
                    break
            
            if valid:
                new_sums = set(sums_set)
                new_sums.update(tails)
                new_total = total_sum + x
                dfs(seq + [x], new_sums, new_total, x, k + 1)
                
    dfs(current, cur_sums, t_curr, last_val, k_curr)
    
    return best_found, best_len

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    best = []
    best_len = 0
    
    # 1. Get a good initial lower bound using Greedy
    greedy_seq = _greedy_forward([])
    if len(greedy_seq) > best_len:
        best = greedy_seq
        best_len = len(best)
        
    # 2. Run DFS with the new Sum-Capacity pruning
    found_seq, found_len = _dfs_csp([], max_val=256, deadline=deadline, best_len=best_len)
    if found_len > best_len:
        best = found_seq
        best_len = found_len
        
    # 3. If we have time, try randomized restarts with different starting prefixes
    if time.time() < deadline - 5:
        rng = random.Random(42)
        for _ in range(10):
            if time.time() > deadline:
                break
            prefix = []
            cur_sums = set()
            last = 0
            t_curr = 0
            try:
                for _ in range(3):
                    found = False
                    for _ in range(50):
                        x = rng.randint(last + 1, 256)
                        tails = [x]
                        total = x
                        for a in reversed(prefix):
                            total += a
                            tails.append(total)
                        if not cur_sums.intersection(tails):
                            prefix.append(x)
                            cur_sums.update(tails)
                            last = x
                            t_curr += x
                            found = True
                            break
                    if not found:
                        break
                if len(prefix) == 3:
                    found_seq, found_len = _dfs_csp(prefix, max_val=256, deadline=deadline, best_len=best_len)
                    if found_len > best_len:
                        best = found_seq
                        best_len = found_len
            except Exception:
                pass

    # Final validation
    if not _is_valid_full(best):
        best = _greedy_forward([])
        
    return best

# EVOLVE-BLOCK-END
