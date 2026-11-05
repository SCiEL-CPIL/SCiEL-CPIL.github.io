"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

_START = time.time()
_DEADLINE = 85.0

def _elapsed():
    return time.time() - _START

def construct():
    MOD = 101
    N = 10
    NUM_SUBSETS = 1 << N  # 1024
    
    # Precompute inverses
    inv = [0] * (N + 1)
    for x in range(1, N + 1):
        inv[x] = pow(x, -1, MOD)
    
    # Precompute subset sums and sizes using DP
    subset_sum = [0] * NUM_SUBSETS
    subset_size = [0] * NUM_SUBSETS
    for mask in range(1, NUM_SUBSETS):
        lsb = mask & (-mask)
        x = lsb.bit_length()  # 1-indexed
        prev = mask ^ lsb
        subset_sum[mask] = (subset_sum[prev] + inv[x]) % MOD
        subset_size[mask] = subset_size[prev] + 1
    
    # Map residue -> list of masks, sorted by size
    res_to_masks = defaultdict(list)
    for mask in range(1, NUM_SUBSETS):
        r = subset_sum[mask]
        res_to_masks[r].append(mask)
    
    for r in res_to_masks:
        res_to_masks[r].sort(key=lambda m: subset_size[m])
    
    # Hopcroft-Karp bipartite matching
    # Left: residues 0..100, Right: masks 1..1023
    # adj[r] = list of masks available for residue r
    def hopcroft_karp(adj, max_size):
        """Returns matching: match_r[mask] = residue, match_l[residue] = mask"""
        match_r = [-1] * NUM_SUBSETS  # mask -> residue
        match_l = [-1] * MOD          # residue -> mask
        dist = [0] * MOD
        
        def bfs():
            queue = deque()
            found = False
            for r in range(MOD):
                if match_l[r] == -1:
                    dist[r] = 0
                    queue.append(r)
                else:
                    dist[r] = -1
            while queue:
                r = queue.popleft()
                for mask in adj[r]:
                    if subset_size[mask] > max_size:
                        continue
                    mr = match_r[mask]
                    if mr == -1:
                        found = True
                    elif dist[mr] == -1:
                        dist[mr] = dist[r] + 1
                        queue.append(mr)
            return found
        
        def dfs(r, visited):
            for mask in adj[r]:
                if subset_size[mask] > max_size:
                    continue
                if visited[mask]:
                    continue
                visited[mask] = True
                mr = match_r[mask]
                if mr == -1 or (dist[mr] == dist[r] + 1 and dfs(mr, visited)):
                    match_r[mask] = r
                    match_l[r] = mask
                    return True
            dist[r] = -1
            return False
        
        matching = 0
        while bfs():
            for r in range(MOD):
                if match_l[r] == -1:
                    if dfs(r, [False] * NUM_SUBSETS):
                        matching += 1
        return matching, match_l, match_r
    
    # Binary search on max_size K
    lo, hi = 1, N
    best_K = N
    while lo <= hi:
        if _elapsed() > _DEADLINE - 5:
            break
        mid = (lo + hi) // 2
        adj = [None] * MOD
        feasible = True
        for r in range(MOD):
            if r not in res_to_masks:
                feasible = False
                break
            adj[r] = [m for m in res_to_masks[r] if subset_size[m] <= mid]
            if not adj[r]:
                feasible = False
                break
        if not feasible:
            lo = mid + 1
            continue
        matching, _, _ = hopcroft_karp(adj, mid)
        if matching == MOD:
            best_K = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    # Final matching with best_K
    adj = [None] * MOD
    for r in range(MOD):
        if r in res_to_masks:
            adj[r] = [m for m in res_to_masks[r] if subset_size[m] <= best_K]
        else:
            adj[r] = []
    
    matching, match_l, match_r = hopcroft_karp(adj, best_K)
    
    # Build rows
    rows = [None] * MOD
    for r in range(MOD):
        mask = match_l[r]
        if mask == -1:
            # Fallback
            if res_to_masks[r]:
                mask = res_to_masks[r][0]
            else:
                mask = NUM_SUBSETS - 1
        terms = [x for x in range(1, N + 1) if (mask >> (x - 1)) & 1]
        rows[r] = terms
    
    return rows

# EVOLVE-BLOCK-END
