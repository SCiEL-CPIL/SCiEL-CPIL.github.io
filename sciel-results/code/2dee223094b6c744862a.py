import time

def construct():
    start_time = time.time()
    DEADLINE = 85  # 85s guard (90s kill)
    
    # Generate all 2,3,5-smooth numbers up to 128
    smooth = set()
    a = 1
    while a <= 128:
        b = a
        while b <= 128:
            c = b
            while c <= 128:
                smooth.add(c)
                c *= 5
            b *= 3
        a *= 2
    smooth_list = sorted(smooth)
    S = len(smooth_list)
    smooth_set = set(smooth_list)
    val_to_idx = {v: i for i, v in enumerate(smooth_list)}
    
    # Precompute incompatibility masks
    # incompatible[i] is a bitmask of indices j where smooth_list[j] is incompatible with smooth_list[i]
    # Incompatible means one divides the other.
    incompatible = [0] * S
    for i in range(S):
        mask = 0
        vi = smooth_list[i]
        for j in range(S):
            if i == j:
                continue
            vj = smooth_list[j]
            if vi % vj == 0 or vj % vi == 0:
                mask |= (1 << j)
        incompatible[i] = mask

    # best_size[sum] = minimum number of terms
    # best_set[sum] = tuple of terms achieving that size
    best_size = [float('inf')] * 129
    best_set = [None] * 129
    
    # Initialize with single elements (antichains of size 1)
    for i in range(S):
        v = smooth_list[i]
        if v <= 128:
            if 1 < best_size[v]:
                best_size[v] = 1
                best_set[v] = (v,)

    # DFS to find all antichains
    # idx: current index in smooth_list
    # curr_sum: current sum
    # curr_size: current number of elements
    # curr_set: list of elements chosen
    # forbidden: bitmask of indices that cannot be chosen
    
    def dfs(idx, curr_sum, curr_size, curr_set, forbidden):
        if curr_sum > 128:
            return
        if time.time() - start_time > DEADLINE:
            return
            
        if idx == S:
            if curr_size > 0 and curr_size < best_size[curr_sum]:
                best_size[curr_sum] = curr_size
                best_set[curr_sum] = tuple(curr_set)
            return
        
        # Option 1: Skip smooth_list[idx]
        dfs(idx + 1, curr_sum, curr_size, curr_set, forbidden)
        
        # Option 2: Include smooth_list[idx] if not forbidden
        if not ((forbidden >> idx) & 1):
            val = smooth_list[idx]
            new_sum = curr_sum + val
            if new_sum <= 128:
                new_forbidden = forbidden | incompatible[idx] | (1 << idx)
                curr_set.append(val)
                dfs(idx + 1, new_sum, curr_size + 1, curr_set, new_forbidden)
                curr_set.pop()

    dfs(0, 0, 0, [], 0)
    
    answers = []
    for n in range(1, 129):
        if best_set[n] is not None:
            answers.append(list(best_set[n]))
        else:
            # Fallback, should not happen
            answers.append([n])
            
    return answers
