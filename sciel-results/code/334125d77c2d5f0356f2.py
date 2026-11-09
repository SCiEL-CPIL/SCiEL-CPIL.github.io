import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s guard
    
    # 1. Generate smooth numbers up to 128
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
    n_smooth = len(smooth_list)
    
    # Map value to index for quick lookup
    val_to_idx = {v: i for i, v in enumerate(smooth_list)}
    
    # 2. Precompute incompatible masks
    # incompatible_mask[i] has bit j set if smooth_list[i] and smooth_list[j] are comparable
    # i.e., smooth_list[i] % smooth_list[j] == 0 or smooth_list[j] % smooth_list[i] == 0
    incompatible_mask = [0] * n_smooth
    for i in range(n_smooth):
        vi = smooth_list[i]
        for j in range(n_smooth):
            vj = smooth_list[j]
            if i == j:
                continue
            if vi % vj == 0 or vj % vi == 0:
                incompatible_mask[i] |= (1 << j)
        # Also mark self as incompatible to prevent re-picking
        incompatible_mask[i] |= (1 << i)

    def find_min_antichain(n):
        # Check k=1
        if n in val_to_idx:
            return [n]
            
        memo = {}
        
        def dfs(rem, k, forb_mask, end_idx):
            if k == 0:
                return [] if rem == 0 else None
            
            if rem < 0:
                return None
                
            key = (rem, k, forb_mask, end_idx)
            if key in memo:
                return memo[key]
                
            # Pruning: Check if enough numbers available and if sum is possible
            # We need to pick k numbers from indices <= end_idx that are not in forb_mask
            
            avail_indices = []
            for i in range(end_idx + 1):
                if not (forb_mask & (1 << i)):
                    avail_indices.append(i)
                    
            if len(avail_indices) < k:
                memo[key] = None
                return None
                
            # Filter to those that can fit in 'rem' (value <= rem)
            # Note: avail_indices is sorted by index (value).
            valid_avail = [i for i in avail_indices if smooth_list[i] <= rem]
            
            if len(valid_avail) < k:
                memo[key] = None
                return None
                
            # Min sum: k smallest
            min_sum = sum(smooth_list[i] for i in valid_avail[:k])
            # Max sum: k largest
            max_sum = sum(smooth_list[i] for i in valid_avail[-k:])
            
            if rem < min_sum or rem > max_sum:
                memo[key] = None
                return None
                
            # Search: Try larger values first to reduce 'rem' quickly
            # Iterate from end_idx down to 0
            for i in range(end_idx, -1, -1):
                if (forb_mask & (1 << i)):
                    continue
                val = smooth_list[i]
                if val > rem:
                    continue # Smaller values might still fit
                
                new_forb = forb_mask | incompatible_mask[i]
                # Next search picks from indices < i
                res = dfs(rem - val, k - 1, new_forb, i - 1)
                if res is not None:
                    res.append(val)
                    memo[key] = res
                    return res
                    
            memo[key] = None
            return None
            
        for k in range(2, 31):
            if time.time() > deadline:
                pass
            res = dfs(n, k, 0, n_smooth - 1)
            if res is not None:
                return res
                
        return None

    answers = []
    for n in range(1, 129):
        if time.time() > deadline:
            while len(answers) < 128:
                answers.append([1])
            return answers
            
        rep = find_min_antichain(n)
        if rep is None:
            rep = [1]
        answers.append(rep)
        
    return answers
