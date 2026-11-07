import sys
sys.setrecursionlimit(10000)

def _dfs(i_idx, visited, adj, match_val, match_time, current_time):
    avail = adj[i_idx] & ~visited
    while avail:
        d_bit = avail & -avail
        d = d_bit.bit_length() - 1
        avail ^= d_bit
        visited |= d_bit
        if match_time[d] != current_time or _dfs(match_val[d], visited, adj, match_val, match_time, current_time):
            match_val[d] = i_idx
            match_time[d] = current_time
            return True
    return False

def _precompute():
    n = 12
    period = 27720
    max_L = 144
    
    div_masks = [[0] * i for i in range(1, n + 1)]
    for i in range(1, n + 1):
        for r in range(i):
            mask = 0
            target = (-r) % i
            d = target
            if d == 0:
                d = i
            while d <= max_L:
                mask |= (1 << d)
                d += i
            div_masks[i][r] = mask

    results = [0] * period
    
    match_val = [-1] * (max_L + 1)
    match_time = [0] * (max_L + 1)
    current_time = 0
    
    for m in range(period):
        found = False
        for L in range(n, max_L + 1):
            current_time += 1
            range_mask = (1 << (L + 1)) - 2
            
            adj = [0] * n
            for i in range(1, n + 1):
                adj[i-1] = div_masks[i][m % i] & range_mask
            
            matched_count = 0
            for i_idx in range(n):
                if _dfs(i_idx, 0, adj, match_val, match_time, current_time):
                    matched_count += 1
            
            if matched_count == n:
                results[m] = L
                found = True
                break
        
        if not found:
            results[m] = max_L
    
    return results

_results = _precompute()

def construct():
    period = 27720
    best_r = 0
    best_val = -1
    for r in range(period):
        if _results[r] > best_val:
            best_val = _results[r]
            best_r = r
    return best_r
