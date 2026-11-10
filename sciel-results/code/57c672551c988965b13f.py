import time
from collections import deque

def construct():
    deadline = time.time() + 85

    # Precompute modular inverses
    inv = [0] * 11
    for x in range(1, 11):
        inv[x] = pow(x, -1, 101)

    # Precompute all 1023 non-empty subsets: residue and size
    # Use iterative DP for speed
    n_subsets = 1 << 10
    subset_residue = [0] * n_subsets
    subset_size = [0] * n_subsets
    subset_terms = [[] for _ in range(n_subsets)]
    
    # For each mask, compute residue, size, and terms
    for mask in range(1, n_subsets):
        # Find lowest set bit
        lsb = mask & (-mask)
        bit = lsb.bit_length() - 1  # 0-indexed, corresponds to denominator bit+1
        prev = mask ^ lsb
        subset_residue[mask] = (subset_residue[prev] + inv[bit + 1]) % 101
        subset_size[mask] = subset_size[prev] + 1
        subset_terms[mask] = subset_terms[prev] + [bit + 1]

    # For each residue 0..100, list subset indices (mask-1) that produce it, sorted by size
    residue_to_subsets = [[] for _ in range(101)]
    for mask in range(1, n_subsets):
        idx = mask - 1  # 0-based index into arrays
        residue_to_subsets[subset_residue[mask]].append(idx)

    # Sort by size for each residue
    for r in range(101):
        residue_to_subsets[r].sort(key=lambda i: (subset_size[i + 1], i))

    # Hopcroft-Karp to find minimum k with perfect matching
    def hopcroft_karp(max_k):
        # Build adjacency: for each residue r, which subsets (of size <= max_k) map to it
        adj = [[] for _ in range(101)]
        for r in range(101):
            for idx in residue_to_subsets[r]:
                if subset_size[idx + 1] <= max_k:
                    adj[r].append(idx)

        pairU = [-1] * 101
        pairV = [-1] * (n_subsets - 1)  # 1023 entries
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

    # Find minimum k
    best_pairU = None
    for k in range(1, 11):
        matching, pairU = hopcroft_karp(k)
        if matching == 101:
            best_pairU = pairU
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

    # Fallback: greedy by size
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
