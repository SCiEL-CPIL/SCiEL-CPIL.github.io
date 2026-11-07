import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

N = P['n']
DEADLINE = 75.0

def _greedy_partition():
    """Greedy fallback."""
    es = set()
    for u, v in P['edges']:
        es.add((u, v))
        es.add((v, u))

    def is_clique(lst):
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if (lst[i], lst[j]) not in es:
                    return False
        return True

    def is_indep(lst):
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if (lst[i], lst[j]) in es:
                    return False
        return True

    parts = []
    for v in range(N):
        placed = False
        for part in parts:
            trial = part + [v]
            if is_clique(trial) or is_indep(trial):
                part.append(v)
                placed = True
                break
        if not placed:
            parts.append([v])
    return parts

def construct():
    start = time.time()
    
    # Precompute adjacency matrix
    adj = [[False] * N for _ in range(N)]
    for u, v in P['edges']:
        adj[u][v] = True
        adj[v][u] = True
        
    # Precompute is_clique and is_indep for all masks
    size = 1 << N
    is_clique = [False] * size
    is_indep = [False] * size
    is_clique[0] = True
    is_indep[0] = True
    
    for mask in range(1, size):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        if rest == 0:
            is_clique[mask] = True
            is_indep[mask] = True
        else:
            # Check clique
            ok_clique = is_clique[rest]
            if ok_clique:
                tmp = rest
                while tmp:
                    lsb_r = tmp & -tmp
                    u = lsb_r.bit_length() - 1
                    if not adj[v][u]:
                        ok_clique = False
                        break
                    tmp ^= lsb_r
            is_clique[mask] = ok_clique
            
            # Check independent set
            ok_indep = is_indep[rest]
            if ok_indep:
                tmp = rest
                while tmp:
                    lsb_r = tmp & -tmp
                    u = lsb_r.bit_length() - 1
                    if adj[v][u]:
                        ok_indep = False
                        break
                    tmp ^= lsb_r
            is_indep[mask] = ok_indep

    is_valid = [is_clique[m] or is_indep[m] for m in range(size)]

    def solve_k(k):
        bins = [0] * k
        
        def dfs(v, non_empty_count):
            if time.time() - start > DEADLINE:
                return False, None, True
            
            if non_empty_count + (N - v) < k:
                return False, None, False
                
            if v == N:
                if non_empty_count == k:
                    parts = []
                    for b in range(k):
                        if bins[b] != 0:
                            part_verts = []
                            m = bins[b]
                            while m:
                                lsb = m & -m
                                u = lsb.bit_length() - 1
                                part_verts.append(u)
                                m ^= lsb
                            parts.append(part_verts)
                    return True, parts, False
                else:
                    return False, None, False
            
            for b in range(k):
                if bins[b] == 0:
                    if b == non_empty_count:
                        new_mask = 1 << v
                        bins[b] = new_mask
                        res, parts, timed_out = dfs(v + 1, non_empty_count + 1)
                        if timed_out:
                            return False, None, True
                        if res:
                            return True, parts, False
                        bins[b] = 0
                else:
                    old_mask = bins[b]
                    new_mask = old_mask | (1 << v)
                    if is_valid[new_mask]:
                        bins[b] = new_mask
                        res, parts, timed_out = dfs(v + 1, non_empty_count)
                        if timed_out:
                            bins[b] = old_mask
                            return False, None, True
                        if res:
                            return True, parts, False
                        bins[b] = old_mask
            
            return False, None, False

        if k == 0:
            return N == 0, []
            
        bins[0] = 1 << 0
        res, parts, timed_out = dfs(1, 1)
        if timed_out:
            return False, None
        return res, parts

    for k in range(1, N + 1):
        found, parts = solve_k(k)
        if found:
            final = []
            for part in parts:
                final.append([int(v) for v in part])
            return final
            
    # Fallback
    return [[int(v) for v in p] for p in _greedy_partition()]

