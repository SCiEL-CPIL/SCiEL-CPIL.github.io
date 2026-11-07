"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _build_graph():
    n = P['n']
    adj = [[False] * n for _ in range(n)]
    for u, v in P['edges']:
        adj[u][v] = True
        adj[v][u] = True
    adj_mask = [0] * n
    for u in range(n):
        for v in range(n):
            if adj[u][v]:
                adj_mask[u] |= (1 << v)
    return adj, adj_mask

def _is_clique_mask(mask, adj_mask):
    verts = []
    m = mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        verts.append(v)
        m ^= lsb
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            if not (adj_mask[verts[i]] & (1 << verts[j])):
                return False
    return True

def _is_independent_mask(mask, adj_mask):
    verts = []
    m = mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        verts.append(v)
        m ^= lsb
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            if (adj_mask[verts[i]] & (1 << verts[j])):
                return False
    return True

def _precompute_valid_parts(adj_mask, n):
    valid = [False] * (1 << n)
    valid[0] = False
    for mask in range(1, 1 << n):
        if _is_clique_mask(mask, adj_mask) or _is_independent_mask(mask, adj_mask):
            valid[mask] = True
    return valid

def _greedy_upper_bound(adj_mask, n):
    remaining = (1 << n) - 1
    parts = 0
    while remaining:
        parts += 1
        lsb = remaining & (-remaining)
        v = lsb.bit_length() - 1
        candidate = lsb
        for u in range(v + 1, n):
            if not (remaining & (1 << u)):
                continue
            new_candidate = candidate | (1 << u)
            if _is_clique_mask(new_candidate, adj_mask) or _is_independent_mask(new_candidate, adj_mask):
                candidate = new_candidate
        remaining ^= candidate
    return parts

def _backtrack_min_parts(adj_mask, valid, n, deadline):
    full_mask = (1 << n) - 1
    masks_by_vertex = [[] for _ in range(n)]
    for mask in range(1, 1 << n):
        if valid[mask]:
            for v in range(n):
                if mask & (1 << v):
                    masks_by_vertex[v].append(mask)
    for v in range(n):
        masks_by_vertex[v].sort(key=lambda m: m.bit_count(), reverse=True)
    
    ub = _greedy_upper_bound(adj_mask, n)
    best_k = [ub]
    best_partition = [None]
    nodes = [0]
    
    def dfs(remaining, k_used, path):
        nodes[0] += 1
        if time.time() > deadline or nodes[0] > 2000000:
            return False
        if remaining == 0:
            if k_used < best_k[0]:
                best_k[0] = k_used
                best_partition[0] = list(path)
            return True
        if k_used >= best_k[0]:
            return False
        best_v = -1
        best_count = float('inf')
        m = remaining
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            count = 0
            for pm in masks_by_vertex[v]:
                if (pm & remaining) == pm:
                    count += 1
                    if count >= best_count:
                        break
            if count < best_count:
                best_count = count
                best_v = v
                if count <= 1:
                    break
            m ^= lsb
        if best_v == -1:
            return False
        for part_mask in masks_by_vertex[best_v]:
            if (part_mask & remaining) != part_mask:
                continue
            path.append(part_mask)
            dfs(remaining ^ part_mask, k_used + 1, path)
            path.pop()
        return False
    
    for target_k in range(ub, 0, -1):
        best_k[0] = target_k
        best_partition[0] = None
        nodes[0] = 0
        dfs(full_mask, 0, [])
        if best_partition[0] is not None:
            parts = []
            for m in best_partition[0]:
                parts.append([i for i in range(n) if m & (1 << i)])
            return parts
    return None

def construct():
    adj, adj_mask = _build_graph()
    n = P['n']
    valid = _precompute_valid_parts(adj_mask, n)
    deadline = time.time() + 55
    result = _backtrack_min_parts(adj_mask, valid, n, deadline)
    if result is not None:
        parts = []
        for part in result:
            parts.append([int(v) for v in part])
        return parts
    es = {tuple(e) for e in P['edges']}
    parts = []
    for v in range(n):
        for part in parts:
            trial = part + [v]
            hits = [(u, w) in es for i, u in enumerate(trial) for w in trial[i + 1:]]
            if all(hits) or not any(hits):
                part.append(v)
                break
        else:
            parts.append([v])
    return parts

# EVOLVE-BLOCK-END
