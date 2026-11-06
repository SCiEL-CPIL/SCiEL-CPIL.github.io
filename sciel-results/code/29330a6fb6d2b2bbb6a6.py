import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s limit, 10s safety margin
    
    # 1. Get a baseline with greedy
    best_seq = greedy()
    best_len = len(best_seq)
    
    # 2. Try to extend using DFS
    # Start from best_len + 1
    for target_len in range(best_len + 1, 40):
        if time.time() > deadline:
            break
            
        # DFS to find a sequence of length target_len
        seq = dfs(target_len, [], 0, [0])
        if seq is not None:
            best_seq = seq
            best_len = target_len
        else:
            # If we failed to find a sequence of this length,
            # it's either impossible or we ran out of time.
            if time.time() > deadline:
                break
            # If we finished search quickly without finding, assume impossible
            # for this length and break, as longer lengths are harder.
            break
            
    return best_seq

def greedy():
    values = []
    sums = set()
    for x in range(1, 257):
        tails = [x]
        total = x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    return values

def dfs(target_len, current_seq, sums_bitset, suffix_sums):
    global deadline
    
    k = len(current_seq)
    if k == target_len:
        return current_seq[:]
        
    if time.time() > deadline:
        return None
        
    last_x = current_seq[-1] if k > 0 else 0
    
    # Pruning:
    # Min sum increase for remaining (target_len - k) elements
    m = target_len - k
    min_increment = m * last_x + m * (m + 1) // 2
    current_total_sum = suffix_sums[-1] # Sum of all elements (last suffix sum)
    potential_max_sum = current_total_sum + min_increment
    
    # We need total sum >= M_final
    M_final = target_len * (target_len + 1) // 2
    if potential_max_sum < M_final:
        return None

    # Try candidates
    for x in range(last_x + 1, 257):
        if time.time() > deadline:
            return None
            
        # Check collision
        # New sums are x + s for s in suffix_sums
        new_sums_bitset = 0
        for s in suffix_sums:
            val = x + s
            new_sums_bitset |= (1 << val)
            
        if (new_sums_bitset & sums_bitset) != 0:
            continue
            
        # Valid move
        new_sums_bitset_total = sums_bitset | new_sums_bitset
        
        # Update suffix sums
        # New suffixes: 0, x, x+s1, ...
        new_suffix_sums = [0] + [x + s for s in suffix_sums]
        
        # Recurse
        result = dfs(target_len, current_seq + [x], new_sums_bitset_total, new_suffix_sums)
        if result is not None:
            return result
            
    return None

# Global variable for deadline access in dfs
deadline = 0
def construct_wrapper():
    global deadline
    start_time = time.time()
    deadline = start_time + 80
    return construct()

# Override construct to use wrapper logic or just define deadline in scope
# The prompt requires construct() to be the entry point.
# I will refactor to pass deadline or use a global.
# Using a global is cleaner for the signature.

def construct():
    global deadline
    start_time = time.time()
    deadline = start_time + 80
    
    # 1. Get a baseline with greedy
    best_seq = greedy()
    best_len = len(best_seq)
    
    # 2. Try to extend using DFS
    for target_len in range(best_len + 1, 40):
        if time.time() > deadline:
            break
            
        seq = dfs(target_len, [], 0, [0])
        if seq is not None:
            best_seq = seq
            best_len = target_len
        else:
            if time.time() > deadline:
                break
            break
            
    return best_seq

