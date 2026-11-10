"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
from collections import deque

P = {'n': 12, 'm': 40}

def _build_static_graph(n):
    """
    Build the static DAG part of the flow graph.
    Returns adjacency list structure for the 2^n nodes.
    We will add Source and Sink dynamically.
    """
    num_nodes = 1 << n
    # We need to store edges for residual graph updates?
    # It's easier to just rebuild the full graph for each lambda if it's fast enough.
    # 4096 nodes, 24576 edges.
    # Building it takes ~0.1s?
    # Let's try to optimize by precomputing the immediate subsets.
    
    # For each node v, list of immediate subsets u
    # This allows fast edge generation
    immediate_subsets = [[] for _ in range(num_nodes)]
    for v in range(num_nodes):
        mask = v
        while mask:
            lsb = mask & -mask
            u = v ^ lsb
            immediate_subsets[v].append(u)
            mask ^= lsb
            
    return num_nodes, immediate_subsets

def _run_max_closure(n, immediate_subsets, weights, lambda_val):
    num_nodes = 1 << n
    source = num_nodes
    sink = num_nodes + 1
    total_nodes = num_nodes + 2
    
    graph = [[] for _ in range(total_nodes)]
    
    def add_edge(u, v, cap):
        graph[u].append([v, cap, len(graph[v])])
        graph[v].append([u, 0, len(graph[u]) - 1])
        
    INF = 10**9
    
    # Add Source/Sink edges
    for v in range(num_nodes):
        w = weights[v] - lambda_val
        if w > 0:
            add_edge(source, v, int(w))
        elif w < 0:
            add_edge(v, sink, int(-w))
            
    # Add DAG edges
    for v in range(num_nodes):
        for u in immediate_subsets[v]:
            add_edge(u, v, INF)
            
    # Run Dinic
    level = [-1] * total_nodes
    it = [0] * total_nodes
    
    def bfs():
        for i in range(total_nodes):
            level[i] = -1
        q = deque([source])
        level[source] = 0
        while q:
            u = q.popleft()
            for v, cap, rev in graph[u]:
                if cap > 0 and level[v] < 0:
                    level[v] = level[u] + 1
                    q.append(v)
        return level[sink] >= 0
        
    def dfs(u, f):
        if u == sink:
            return f
        while it[u] < len(graph[u]):
            v, cap, rev = graph[u][it[u]]
            if cap > 0 and level[u] < level[v]:
                d = dfs(v, min(f, cap))
                if d > 0:
                    graph[u][it[u]][1] -= d
                    graph[v][rev][1] += d
                    return d
            it[u] += 1
        return 0
        
    flow = 0
    while bfs():
        it = [0] * total_nodes
        while True:
            f = dfs(source, 10**18)
            if f == 0:
                break
            flow += f
            
    # Find reachable
    reachable = [False] * total_nodes
    q = deque([source])
    reachable[source] = True
    while q:
        u = q.popleft()
        for v, cap, rev in graph[u]:
            if cap > 0 and not reachable[v]:
                reachable[v] = True
                q.append(v)
                
    selected = [i for i in range(num_nodes) if reachable[i]]
    return selected

def _generate_family(n, m, seed=42):
    num_nodes, immediate_subsets = _build_static_graph(n)
    weights = [(1 << (v.bit_count())) - 1 for v in range(num_nodes)]
    
    best_family = None
    best_score = -1
    
    # Search for lambda
    # Weights are up to 4095.
    # Try range
    for lam in range(0, 500, 1):
        selected = _run_max_closure(n, immediate_subsets, weights, lam)
        size = len(selected)
        
        if size == m:
            score = sum(weights[i] for i in selected)
            if score > best_score:
                best_score = score
                best_family = []
                for mask in selected:
                    s = []
                    for i in range(n):
                        if mask & (1 << i):
                            s.append(i)
                    best_family.append(s)
                    
        # If size is much smaller, we can stop?
        # Size is non-increasing with lambda.
        if size < m - 10:
            break
        if size > m + 10 and lam > 100:
            # If it's still too big, maybe increase step?
            pass

    if best_family is None:
        # Fallback
        best_family = [list(range(i)) for i in range(m)]
        
    return best_family, best_score

def construct():
    family, score = _generate_family(P['n'], P['m'])
    assert len(family) == P['m']
    assert len(set(tuple(s) for s in family)) == P['m']
    for s in family:
        for x in s:
            assert isinstance(x, int) and not isinstance(x, bool)
            assert 0 <= x < P['n']
        assert len(s) == len(set(s))
    return family
# EVOLVE-BLOCK-END
