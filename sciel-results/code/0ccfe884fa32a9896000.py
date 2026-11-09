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
    
    # Precompute incompatibility: two numbers are incompatible if one divides the other
    # incompatible[i] is a bitmask of indices j where smooth_list[j] is incompatible with smooth_list[i]
    incompatible = [0] * S
    for i in range(S):
        for j in range(S):
            if i == j:
                continue
            if smooth_list[i] % smooth_list[j] == 0 or smooth_list[j] % smooth_list[i] == 0:
                incompatible[i] |= (1 << j)
    
    # For each n, get candidate smooth numbers <= n, sorted descending
    cand_indices = [[] for _ in range(129)]
    for n in range(1, 129):
        indices = [i for i in range(S) if smooth_list[i] <= n]
        indices.sort(key=lambda i: smooth_list[i], reverse=True)
        cand_indices[n] = indices
    
    def find_antichain(n, k):
        if k == 1:
            if n in smooth_set:
                return [n]
            return None
        
        indices = cand_indices[n]
        if len(indices) < k:
            return None
        
        path = []
        
        def dfs(start_idx, rem_sum, rem_k, forb_mask):
            if rem_k == 0:
                return path[:] if rem_sum == 0 else None
            
            # Pruning: check if enough compatible numbers remain
            avail = []
            for i in range(start_idx, len(indices)):
                idx = indices[i]
                if not ((forb_mask >> idx) & 1):
                    avail.append(idx)
            
            if len(avail) < rem_k:
                return None
            
            # Pruning: sum bounds
            max_sum = sum(smooth_list[avail[i]] for i in range(min(rem_k, len(avail))))
            min_sum = sum(smooth_list[avail[i]] for i in range(len(avail) - min(rem_k, len(avail)), len(avail)))
            
            if rem_sum > max_sum or rem_sum < min_sum:
                return None
            
            for i in range(start_idx, len(indices)):
                idx = indices[i]
                val = smooth_list[idx]
                if val > rem_sum:
                    continue
                if (forb_mask >> idx) & 1:
                    continue
                
                path.append(val)
                new_forb = forb_mask | incompatible[idx] | (1 << idx)
                res = dfs(i + 1, rem_sum - val, rem_k - 1, new_forb)
                if res is not None:
                    return res
                path.pop()
            
            return None
        
        return dfs(0, n, k, 0)
    
    answers = []
    for n in range(1, 129):
        found = None
        for k in range(1, 31):
            if time.time() - start_time > DEADLINE:
                break
            res = find_antichain(n, k)
            if res is not None:
                found = res
                break
        
        if found is None:
            # Fallback: should not happen for 1-128
            found = [n]
        
        answers.append(found)
        
    return answers
