import time
from collections import deque

def construct():
    deadline = time.time() + 85

    # Precompute modular inverses
    inv = [0] * 11
    for x in range(1, 11):
        inv[x] = pow(x, -1, 101)

    # Precompute all 1023 non-empty subsets: residue and size
    n_subsets = 1 << 10
    subset_residue = [0] * n_subsets
    subset_size = [0] * n_subsets
    subset_terms = [[] for _ in range(n_subsets)]
    
    for mask in range(1, n_subsets):
        lsb = mask & (-mask)
        bit = lsb.bit_length() - 1
        prev = mask ^ lsb
        subset_residue[mask] = (subset_residue[prev] + inv[bit + 1]) % 101
        subset_size[mask] = subset_size[prev] + 1
        subset_terms[mask] = subset_terms[prev] + [bit + 1]

    # For each residue 0..100, list subset indices (mask-1) that produce it, sorted by size
    residue_to_subsets = [[] for _ in range(101)]
    for mask in range(1, n_subsets):
        idx = mask - 1
        residue_to_subsets[subset_residue[mask]].append(idx)

    for r in range(101):
        residue_to_subsets[r].sort(key=lambda i: (subset_size[i + 1], i))

    # Precompute cumulative subset counts to skip impossible k
    # count_subsets[k] = number of subsets with size <= k
    count_subsets = [0] * 11
    for k in range(1, 11):
        count_subsets[k] = count_subsets[k-1] + sum(1 for m in range(1, n_subsets) if subset_size[m] == k)

    def try_greedy(max_k):
        if time.time() > deadline - 2:
            return None
        pairU = [-1] * 101
        pairV = [-1] * (n_subsets - 1)
        # Heuristic: process residues with fewer options first
        items = []
        for r in range(101):
            opts = []
            for idx in residue_to_subsets[r]:
                if subset_size[idx + 1] <= max_k:
                    opts.append(idx)
                else:
                    break # Since sorted by size
            if not opts:
                return None
            items.append((len(opts), r, opts))
        
        items.sort(key=lambda x: x[0])
        
        for _, r, opts in items:
            found = False
            for idx in opts:
                if pairV[idx] == -1:
                    pairU[r] = idx
                    pairV[idx] = r
                    found = True
                    break
            if not found:
                return None
        return pairU

    def hopcroft_karp(max_k):
        if time.time() > deadline - 2:
            return 0, None
            
        # Build adjacency
        adj = [[] for _ in range(101)]
        for r in range(101):
            for idx in residue_to_subsets[r]:
                if subset_size[idx + 1] <= max_k:
                    adj[r].append(idx)
                else:
                    break

        pairU = [-1] * 101
        pairV = [-1] * (n_subsets - 1)
        dist = [0] * 101

        def bfs():
            q = deque()
            found = False
            for u in range(101):
                if pairU[u] == -1:
                    dist[u] = 0
                    q.append(u)
                else:
                    dist[u] = -1
            while q:
                u = q.popleft()
                for v in adj[u]:
                    pu = pairV[v]
                    if pu == -1:
                        found = True
                    elif dist[pu] == -1:
                        dist[pu] = dist[u] + 1
                        q.append(pu)
            return found

        def dfs(u):
            for v in adj[u]:
                pu = pairV[v]
                if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                    pairU[u] = v
                    pairV[v] = u
                    return True
            dist[u] = -1
            return False

        matching = 0
        while time.time() < deadline - 3:
            if not bfs():
                break
            for u in range(101):
                if pairU[u] == -1:
                    if dfs(u):
                        matching += 1
        return matching, pairU

    best_pairU = None
    for k in range(1, 11):
        if count_subsets[k] < 101:
            continue
            
        # Try greedy first
        pairU = try_greedy(k)
        if pairU is not None:
            best_pairU = pairU
            break
            
        # Try HK
        matching, pairU_hk = hopcroft_karp(k)
        if matching == 101:
            best_pairU = pairU_hk
            break

    if best_pairU is not None:
        result = []
        for r in range(101):
            v = best_pairU[r]
            if v >= 0:
                result.append(subset_terms[v + 1])
            else:
                result.append([1])
        return result

    # Fallback
    used = [False] * (n_subsets - 1)
    result = []
    for r in range(101):
        assigned = False
        for idx in residue_to_subsets[r]:
            if not used[idx]:
                used[idx] = True
                result.append(subset_terms[idx + 1])
                assigned = True
                break
        if not assigned:
            result.append([1])
    return result
