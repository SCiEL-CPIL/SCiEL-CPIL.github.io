"""Rewritten program: structural change to ILP-style set cover with exact verification."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

_START_TIME = time.time()
_TIME_LIMIT = 75.0

def _time_ok():
    return (time.time() - _START_TIME) < _TIME_LIMIT

def _bfs(adj, n, start):
    dist = [-1] * n
    dist[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

def _verify(n, base_edges, added_edges):
    """Check: no duplicates with base, triangle-free, connected, diameter <= 4."""
    base_set = set()
    for u, v in base_edges:
        base_set.add((min(u, v), max(u, v)))
    
    added_set = set()
    for u, v in added_edges:
        key = (min(u, v), max(u, v))
        if key in base_set or key in added_set:
            return False
        added_set.add(key)
    
    all_edges = base_set | added_set
    adj = [set() for _ in range(n)]
    for u, v in all_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Triangle-free
    for i in range(n):
        ni = list(adj[i])
        for idx in range(len(ni)):
            j = ni[idx]
            for idx2 in range(idx + 1, len(ni)):
                k = ni[idx2]
                if k in adj[j]:
                    return False
    
    # Connected + diameter
    for start in range(n):
        dist = _bfs(adj, n, start)
        if -1 in dist or max(dist) > 4:
            return False
    
    return True

def _compute_edge_benefits(n, base_edges):
    """For each candidate edge, compute which pairs (u,v) it would help (reduce distance)."""
    base_set = set()
    for u, v in base_edges:
        base_set.add((min(u, v), max(u, v)))
    
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute all-pairs distances on base graph
    dist = [[0] * n for _ in range(n)]
    for s in range(n):
        d = _bfs(adj, n, s)
        for t in range(n):
            dist[s][t] = d[t]
    
    # Candidate edges (not in base)
    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in base_set:
                candidates.append((i, j))
    
    # For each candidate edge, compute which pairs it can shorten
    # Edge (a,b) helps pair (x,y) if min(dist[x][a]+1+dist[b][y], dist[x][b]+1+dist[a][y]) < dist[x][y]
    edge_helps = {}
    for a, b in candidates:
        helps = set()
        for x in range(n):
            for y in range(x + 1, n):
                if dist[x][y] <= 4:
                    continue
                # New path via (a,b)
                nd1 = dist[x][a] + 1 + dist[b][y]
                nd2 = dist[x][b] + 1 + dist[a][y]
                if min(nd1, nd2) < dist[x][y]:
                    helps.add((x, y))
        if helps:
            edge_helps[(a, b)] = helps
    
    return candidates, edge_helps

def _greedy_set_cover(n, base_edges, candidates, edge_helps):
    """Greedy set cover: repeatedly pick edge that covers most uncovered pairs."""
    # All pairs with distance > 4 in base
    base_set = set()
    for u, v in base_edges:
        base_set.add((min(u, v), max(u, v)))
    
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    dist = [[0] * n for _ in range(n)]
    for s in range(n):
        d = _bfs(adj, n, s)
        for t in range(n):
            dist[s][t] = d[t]
    
    uncovered = set()
    for x in range(n):
        for y in range(x + 1, n):
            if dist[x][y] > 4:
                uncovered.add((x, y))
    
    if not uncovered:
        return []
    
    selected = []
    selected_set = set()
    current_uncovered = set(uncovered)
    
    while current_uncovered and _time_ok():
        best_edge = None
        best_count = 0
        best_cover = set()
        
        for edge in candidates:
            if edge in selected_set:
                continue
            if edge not in edge_helps:
                continue
            # Count how many currently uncovered pairs this edge helps
            count = len(edge_helps[edge] & current_uncovered)
            if count > best_count:
                best_count = count
                best_edge = edge
                best_cover = edge_helps[edge] & current_uncovered
        
        if best_edge is None or best_count == 0:
            break
        
        selected.append(best_edge)
        selected_set.add(best_edge)
        current_uncovered -= best_cover
    
    return selected

def _local_search_reduce(n, base_edges, added_edges):
    """Try to remove edges one by one to minimize count."""
    current = list(added_edges)
    
    improved = True
    while improved and _time_ok():
        improved = False
        for i in range(len(current)):
            trial = current[:i] + current[i+1:]
            if _verify(n, base_edges, trial):
                current = trial
                improved = True
                break
    
    return current

def _try_fixed_patterns(n, base_edges):
    """Try several hand-crafted patterns and return the best valid one."""
    base_set = set()
    for u, v in base_edges:
        base_set.add((min(u, v), max(u, v)))
    
    patterns = []
    
    # Pattern 1: stride-6 edges
    p1 = []
    for i in range(18):
        j = i + 6
        if (i, j) not in base_set:
            p1.append((i, j))
    patterns.append(p1)
    
    # Pattern 2: stride-6 + stride-12
    p2 = list(p1)
    for i in range(12):
        j = i + 12
        if (i, j) not in base_set:
            p2.append((i, j))
    patterns.append(p2)
    
    # Pattern 3: stride-5
    p3 = []
    for i in range(19):
        j = i + 5
        if (i, j) not in base_set:
            p3.append((i, j))
    patterns.append(p3)
    
    # Pattern 4: stride-7
    p4 = []
    for i in range(17):
        j = i + 7
        if (i, j) not in base_set:
            p4.append((i, j))
    patterns.append(p4)
    
    # Pattern 5: stride-4
    p5 = []
    for i in range(20):
        j = i + 4
        if (i, j) not in base_set:
            p5.append((i, j))
    patterns.append(p5)
    
    # Pattern 6: symmetric stride-6
    p6 = []
    for i in range(12):
        j = i + 6
        if (i, j) not in base_set:
            p6.append((i, j))
        # Also mirror
        k = 23 - i
        l = 23 - (i + 6)
        if k != l and (min(k,l), max(k,l)) not in base_set and (min(k,l), max(k,l)) not in [(min(a,b),max(a,b)) for a,b in p6]:
            p6.append((min(k,l), max(k,l)))
    patterns.append(p6)
    
    best = None
    for pat in patterns:
        if _verify(n, base_edges, pat):
            if best is None or len(pat) < len(best):
                best = pat
    
    return best

def construct():
    n = P['n']
    base = P['edges']
    
    # Phase 1: Try fixed patterns first (fast)
    pattern_result = _try_fixed_patterns(n, base)
    
    # Phase 2: Greedy set cover
    candidates, edge_helps = _compute_edge_benefits(n, base)
    greedy_result = _greedy_set_cover(n, base, candidates, edge_helps)
    
    # Phase 3: Local search to reduce
    if greedy_result and _verify(n, base, greedy_result):
        greedy_result = _local_search_reduce(n, base, greedy_result)
    
    # Choose best valid result
    best = None
    if pattern_result:
        best = pattern_result
    if greedy_result and _verify(n, base, greedy_result):
        if best is None or len(greedy_result) < len(best):
            best = greedy_result
    
    if best is None:
        # Fallback: return empty (shouldn't happen)
        return []
    
    # Ensure proper types
    result = []
    for u, v in best:
        result.append([int(u), int(v)])
    
    return result
# EVOLVE-BLOCK-END
