"""Rewritten program: longest path via maximum weight matching on a transformed graph."""
import random
from collections import defaultdict

# Build the pinned graph once at import time (deterministic, matches the seed graph)
def _build_graph():
    n = 100
    rng = random.Random(748)
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 1.8 / 100:
                edges.add((i, j))
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return n, adj

N, ADJ = _build_graph()

# --- Maximum Weight Matching via Hungarian algorithm on bipartite representation ---
# A simple path in G can be seen as a matching in a bipartite graph where each vertex
# appears on both sides, with edges (u_L, v_R) for each undirected edge (u,v) in G,
# plus self-loops (u_L, u_R) to allow a vertex to be a path endpoint (degree 1) or
# internal (degree 2). We want to select edges so that each vertex has degree <= 2,
# and the resulting subgraph is a single path (connected, acyclic, max degree 2).
#
# A simpler and more robust approach for n=100: use a branch-and-bound / DP over
# subsets is too slow. Instead, we use a well-known trick:
#
# Formulate as: find a maximum-size set of edges forming a simple path.
# Equivalent to: maximum weight matching in a bipartite graph where we assign
# weights to encourage long paths. But matching alone doesn't enforce connectivity.
#
# Given n=100 is small, we can use a more direct approach:
# 1. Try all pairs of endpoints (s, t) — at most 100*99/2 ~ 5000 pairs.
# 2. For each pair, check if there's a simple path from s to t, and find the longest one.
#    Finding the longest simple path is NP-hard, but for n=100 with average degree ~1.8,
#    the graph is sparse. We can use DFS with pruning.
#
# Even better: since the graph is very sparse (avg degree 1.8, so ~90 edges),
# most connected components are small. We can decompose into components and solve
# each component exactly, then combine.
#
# Let's first find connected components.

def _connected_components():
    visited = [False] * N
    comps = []
    for i in range(N):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in ADJ[u]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            comps.append(comp)
    return comps

COMPS = _connected_components()

# For each component, find the longest simple path.
# For small components (size <= 20), use DP over subsets.
# For larger components, use heuristic + local search.

def _longest_path_in_component(comp):
    n = len(comp)
    if n == 0:
        return []
    if n == 1:
        return [comp[0]]
    if n == 2:
        if comp[0] in ADJ[comp[1]]:
            return [comp[0], comp[1]]
        else:
            return [comp[0]]
    
    # Map local indices
    idx = {v: i for i, v in enumerate(comp)}
    local_adj = [set() for _ in range(n)]
    for u in comp:
        for v in ADJ[u]:
            if v in idx:
                local_adj[idx[u]].add(idx[v])
    
    # For n <= 18, use DP over subsets
    if n <= 18:
        # dp[mask][last] = True if there's a path visiting exactly mask, ending at last
        size = 1 << n
        dp = [bytearray(n) for _ in range(size)]
        # Initialize
        for i in range(n):
            dp[1 << i][i] = 1
        
        best_len = 1
        best_mask = 1
        best_last = 0
        
        for mask in range(size):
            for last in range(n):
                if not dp[mask][last]:
                    continue
                # Try to extend
                for nxt in local_adj[last]:
                    if mask & (1 << nxt):
                        continue
                    new_mask = mask | (1 << nxt)
                    dp[new_mask][nxt] = 1
                    if new_mask.bit_count() > best_len:
                        best_len = new_mask.bit_count()
                        best_mask = new_mask
                        best_last = nxt
        
        # Reconstruct
        path = [best_last]
        mask = best_mask
        last = best_last
        while mask.bit_count() > 1:
            prev_mask = mask ^ (1 << last)
            for prev in range(n):
                if prev_mask & (1 << prev) and dp[prev_mask][prev] and prev in local_adj[last]:
                    path.append(prev)
                    mask = prev_mask
                    last = prev
                    break
            else:
                break
        path.reverse()
        return [comp[i] for i in path]
    
    # For larger components, use a heuristic: greedy with backtracking
    # Try all starting vertices, use DFS with pruning
    best_path = [comp[0]]
    
    for start in comp:
        start_idx = idx[start]
        # DFS
        path = [start_idx]
        used = {start_idx}
        
        def dfs(current, path, used):
            nonlocal best_path
            if len(path) > len(best_path):
                best_path = path[:]
            # Prune: if remaining possible nodes can't beat best
            remaining = n - len(used)
            if len(path) + remaining <= len(best_path):
                return
            for nxt in local_adj[current]:
                if nxt not in used:
                    used.add(nxt)
                    path.append(nxt)
                    dfs(nxt, path, used)
                    path.pop()
                    used.discard(nxt)
        
        dfs(start_idx, path, used)
    
    return [comp[i] for i in best_path]

# Compute longest path in each component
component_paths = []
for comp in COMPS:
    lp = _longest_path_in_component(comp)
    if lp:
        component_paths.append(lp)

# The overall longest path is the longest among all components
# (since paths can't span components)
if component_paths:
    BEST_PATH = max(component_paths, key=len)
else:
    BEST_PATH = [0]

def construct():
    return list(BEST_PATH)
