import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    t0 = time.time()
    DEADLINE = 85.0
    n = P['n']
    full = (1 << n) - 1
    N = 1 << n

    adj = [0] * n
    for a, b in P['edges']:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)

    is_clique = [False] * N
    is_indep = [False] * N
    is_clique[0] = True
    is_indep[0] = True
    for mask in range(1, N):
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        if is_clique[rest] and (adj[v] & rest) == rest:
            is_clique[mask] = True
        if is_indep[rest] and (adj[v] & rest) == 0:
            is_indep[mask] = True

    valid = [False] * N
    for m in range(N):
        valid[m] = is_clique[m] or is_indep[m]

    # Precompute valid masks containing each vertex, sorted by size desc
    valid_masks_for_v = [[] for _ in range(n)]
    for m in range(1, N):
        if valid[m]:
            temp = m
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                valid_masks_for_v[v].append(m)
                temp ^= lsb
    
    for v in range(n):
        valid_masks_for_v[v].sort(key=lambda x: x.bit_count(), reverse=True)

    best = n + 1
    best_partition = None
    current_partition = []
    # memo[mask] = minimum depth reached for this mask
    memo = [100] * N
    memo[0] = 0

    def solve(mask, depth):
        nonlocal best, best_partition, current_partition
        
        if time.time() - t0 > DEADLINE:
            return False
        
        if mask == 0:
            if depth < best:
                best = depth
                best_partition = [list(p) for p in current_partition]
            return True
            
        if depth >= best:
            return False
            
        if depth >= memo[mask]:
            return False
            
        memo[mask] = depth

        if valid[mask]:
            if depth + 1 < best:
                best = depth + 1
                verts = []
                m = mask
                while m:
                    lsb = m & (-m)
                    verts.append(lsb.bit_length() - 1)
                    m ^= lsb
                current_partition.append(verts)
                best_partition = [list(p) for p in current_partition]
                current_partition.pop()
            return True

        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        
        for S in valid_masks_for_v[v]:
            if (S & mask) == S:
                current_partition.append(S)
                if solve(mask ^ S, depth + 1):
                    current_partition.pop()
                    return True
                current_partition.pop()
                
        return False

    solve(full, 0)
    
    if best_partition is None:
        return [[i] for i in range(n)]
        
    return best_partition
