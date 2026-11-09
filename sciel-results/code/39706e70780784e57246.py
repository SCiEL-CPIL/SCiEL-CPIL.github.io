import math
import time

def construct():
    N = 512
    n = N - 1
    a_of = list(range(1, N))
    w = [1.0 / (N - (i + 1)) for i in range(n)]

    # Precompute SPF (Smallest Prime Factor)
    spf = [0] * N
    for i in range(2, N):
        if spf[i] == 0:
            for j in range(i, N, i):
                if spf[j] == 0:
                    spf[j] = i
    # spf[1] remains 0

    # Precompute neighbors bitmask
    neighbors = [0] * n
    for i in range(n):
        ai = i + 1
        mask = 0
        for j in range(n):
            if i != j and math.gcd(ai, j + 1) > 1:
                mask |= (1 << j)
        neighbors[i] = mask

    all_mask = (1 << n) - 1

    # Greedy initial solution
    order_desc = sorted(range(n), key=lambda i: w[i], reverse=True)
    chosen = []
    best_val = 0.0
    for i in order_desc:
        ai = i + 1
        ok = True
        for c in chosen:
            if math.gcd(ai, c) > 1:
                ok = False
                break
        if ok:
            chosen.append(ai)
            best_val += w[i]
    
    best_set = chosen[:]
    start = time.time()
    TIME_LIMIT = 110.0

    # Bound calculation
    # max_w_by_spf array size N
    max_w_by_spf = [0.0] * N
    
    def calc_bound(mask):
        if mask == 0:
            return 0.0
        total = 0.0
        # Reset max_w_by_spf? 
        # Instead of resetting whole array, we can just use a local dict or a timestamp trick.
        # Given N=511, resetting a list of 512 floats is fast.
        for k in range(N):
            max_w_by_spf[k] = 0.0
        
        if mask & 1:
            total += w[0]
        
        m = mask & ~1
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            p = spf[idx]
            if w[idx] > max_w_by_spf[p]:
                max_w_by_spf[p] = w[idx]
            m ^= lsb
        
        for p in range(2, N):
            if max_w_by_spf[p] > 0:
                total += max_w_by_spf[p]
        
        return total

    # Stack: (mask, value, chosen_list)
    # chosen_list contains values (1..511)
    stack = [(all_mask, 0.0, [])]
    
    count = 0
    while stack:
        if count & 0x1FF == 0:
            if time.time() - start > TIME_LIMIT:
                break
        count += 1
        
        mask, val, c_list = stack.pop()
        
        # Bound
        ub = val + calc_bound(mask)
        if ub <= best_val + 1e-15:
            continue
            
        if mask == 0:
            if val > best_val:
                best_val = val
                best_set = c_list[:]
            continue
        
        # Select vertex: max weight in mask
        v_idx = -1
        max_w_in_mask = -1.0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            if w[idx] > max_w_in_mask:
                max_w_in_mask = w[idx]
                v_idx = idx
            m ^= lsb
            
        if v_idx == -1:
            continue
            
        v_val = v_idx + 1
        w_v = w[v_idx]
        
        # Include branch
        new_mask_inc = mask & ~(1 << v_idx) & ~neighbors[v_idx]
        new_val_inc = val + w_v
        new_list_inc = c_list + [v_val]
        
        # Exclude branch
        new_mask_exc = mask & ~(1 << v_idx)
        new_val_exc = val
        new_list_exc = c_list
        
        # Push Exclude first, then Include
        stack.append((new_mask_exc, new_val_exc, new_list_exc))
        stack.append((new_mask_inc, new_val_inc, new_list_inc))

    return best_set
