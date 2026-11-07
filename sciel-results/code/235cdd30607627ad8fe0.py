"""Improved program: embed trees T1..T20 edge-disjointly in K20 via backtracking."""
import time
from random import Random

_START = time.time()
_DEADLINE = 120.0  # seconds; return best found before this

def _instance_trees(n=20, seed=611):
    rng = Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

_TREES = _instance_trees()
_N = 20
_ALL_EDGES = set()
for i in range(_N):
    for j in range(i + 1, _N):
        _ALL_EDGES.add((i, j))

# Precompute tree edges as frozensets for fast lookup
_TREE_EDGES = [set((min(a,b), max(a,b)) for a, b in t) for t in _TREES]

def _build_adj(tree_edges):
    """Build adjacency list for a tree."""
    k = len(tree_edges) + 1 if tree_edges else 1
    if k == 1:
        return [[]]
    adj = [[] for _ in range(k)]
    for u, v in tree_edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _embed_tree(tree_edges, used_edges, rng):
    """
    Try to embed tree_edges (list of (u,v) pairs on vertices 0..k-1) 
    into K20 using distinct host vertices, edge-disjoint from used_edges.
    Returns list of k ints (host vertex for tree vertex i) or None.
    """
    k = len(tree_edges) + 1 if tree_edges else 1
    if k == 1:
        # Single vertex: pick any unused vertex
        for v in range(_N):
            return [v]
    
    # Build tree adjacency
    adj = _build_adj(tree_edges)
    
    # Choose root: a leaf to minimize branching
    leaves = [v for v in range(k) if len(adj[v]) == 1]
    root = leaves[0] if leaves else 0
    
    # BFS order from root for embedding
    from collections import deque
    order = [root]
    parent = [-1] * k
    visited = {root}
    q = deque([root])
    while q:
        u = q.popleft()
        for w in adj[u]:
            if w not in visited:
                visited.add(w)
                parent[w] = u
                order.append(w)
                q.append(w)
    
    # Backtracking search
    assignment = [-1] * k
    used_hosts = set()
    
    def dfs(idx):
        if time.time() > _DEADLINE:
            return False
        if idx == k:
            return True
        node = order[idx]
        # Candidate host vertices: not used, and edge to parent's host not in used_edges
        candidates = []
        for h in range(_N):
            if h in used_hosts:
                continue
            p = parent[node]
            if p != -1:
                ph = assignment[p]
                e = (min(ph, h), max(ph, h))
                if e in used_edges:
                    continue
            candidates.append(h)
        
        # Shuffle for diversity
        rng.shuffle(candidates)
        
        for h in candidates:
            assignment[node] = h
            used_hosts.add(h)
            if dfs(idx + 1):
                return True
            used_hosts.remove(h)
            assignment[node] = -1
        return False
    
    if dfs(0):
        return assignment
    return None

def _greedy_embed_all(seed=0):
    """
    Greedy: embed trees in order of decreasing complexity (more edges first),
    then smaller ones. Use randomized backtracking.
    Returns (result_list, total_edges) or (None, 0).
    """
    rng = Random(seed)
    used_edges = set()
    result = [None] * _N
    total_edges = 0
    
    # Order: embed larger trees first (harder to fit)
    order = list(range(_N))
    # Sort by number of edges descending, with random tie-break
    order.sort(key=lambda i: (-len(_TREE_EDGES[i]), rng.random()))
    
    for ti in order:
        tree_edges = _TREES[ti]
        if not tree_edges:
            # T1: single vertex
            # Find any host vertex
            for h in range(_N):
                result[ti] = [h]
                break
            continue
        
        # Try multiple random restarts for this tree
        max_restarts = 50
        success = False
        for _ in range(max_restarts):
            if time.time() > _DEADLINE - 5:
                break
            emb = _embed_tree(tree_edges, used_edges, rng)
            if emb is not None:
                # Verify: all edges are valid and not in used_edges
                valid = True
                for u, v in tree_edges:
                    hu, hv = emb[u], emb[v]
                    e = (min(hu, hv), max(hu, hv))
                    if e in used_edges:
                        valid = False
                        break
                if valid and len(set(emb)) == len(emb):
                    result[ti] = emb
                    for u, v in tree_edges:
                        hu, hv = emb[u], emb[v]
                        used_edges.add((min(hu, hv), max(hu, hv)))
                    total_edges += len(tree_edges)
                    success = True
                    break
        
        if not success:
            return None, 0
    
    # Verify all entries are valid
    for i in range(_N):
        if result[i] is None:
            return None, 0
        if not isinstance(result[i], list):
            return None, 0
        for v in result[i]:
            if not isinstance(v, int) or isinstance(v, bool):
                return None, 0
            if v < 0 or v >= _N:
                return None, 0
    
    return result, total_edges

def construct():
    t0 = time.time()
    
    # Try multiple seeds to find a full embedding
    best_result = None
    best_score = 0
    
    # First try: identity-based or simple heuristic for large trees
    # Then randomized backtracking with various seeds
    
    for seed in range(200):
        if time.time() > _DEADLINE - 10:
            break
        res, score = _greedy_embed_all(seed)
        if res is not None and score > best_score:
            best_score = score
            best_result = res
            if best_score == 190:  # Maximum possible: sum of (k-1) for k=1..20 = 190
                break
    
    if best_result is not None:
        # Ensure all values are Python ints
        final = []
        for entry in best_result:
            if entry is None:
                final.append(None)
            else:
                final.append([int(v) for v in entry])
        return final
    
    # Fallback: only embed T20 (as in seed program)
    return [None] * 19 + [list(range(_N))]

# Pre-compute to avoid repeated work
if __name__ == "__main__":
    res = construct()
    total = sum(len(r) - 1 for r in res if r is not None)
    print(f"Total edges: {total}")
