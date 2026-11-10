"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations, permutations
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_hamiltonian_cycles(vertices, adj, k, deadline):
    """
    Find k edge-disjoint Hamiltonian cycles on the given vertex set using backtracking.
    """
    vertex_set = set(vertices)
    n = len(vertices)
    vlist = sorted(vertices)
    v_to_local = {v: i for i, v in enumerate(vlist)}
    
    # Build local adjacency
    local_adj = [set() for _ in range(n)]
    for u in vlist:
        for v in adj[u]:
            if v in vertex_set:
                local_adj[v_to_local[u]].add(v_to_local[v])
    
    # Total edges on this vertex set
    all_edges_local = set()
    for i in range(n):
        for j in local_adj[i]:
            if i < j:
                all_edges_local.add((i, j))
    
    if len(all_edges_local) < k * n:
        return None
    
    used_edges = set()
    found = []
    
    def find_cycle(visited, current, start, path):
        if len(path) == n:
            edge = (min(start, current), max(start, current))
            if start in local_adj[current] and edge not in used_edges:
                return path[:]
            return None
        for nxt in sorted(local_adj[current]):
            if nxt not in visited:
                edge = (min(current, nxt), max(current, nxt))
                if edge not in used_edges:
                    visited.add(nxt)
                    path.append(nxt)
                    res = find_cycle(visited, nxt, start, path)
                    if res is not None:
                        return res
                    path.pop()
                    visited.remove(nxt)
        return None
    
    def backtrack(idx, target):
        if idx == target:
            return True
        if time.time() > deadline:
            return False
        
        start = 0
        visited = {start}
        path = [start]
        
        cyc = find_cycle(visited, start, start, path)
        if cyc is None:
            return False
        
        for i in range(n):
            u, v = cyc[i], cyc[(i+1) % n]
            used_edges.add((min(u, v), max(u, v)))
        found.append([vlist[v] for v in cyc])
        
        if backtrack(idx + 1, target):
            return True
        
        found.pop()
        for i in range(n):
            u, v = cyc[i], cyc[(i+1) % n]
            used_edges.discard((min(u, v), max(u, v)))
        return False
    
    if backtrack(0, k):
        return found
    return None

def construct():
    deadline = time.time() + 80
    
    adj = _build_adj()
    n = P['n']
    full_set = list(range(n))
    
    total_edges = len(P['edges'])
    max_possible = total_edges // n
    
    for k in range(max_possible, 0, -1):
        result = _find_hamiltonian_cycles(full_set, adj, k, deadline)
        if result is not None:
            cycles = []
            for cyc in result:
                cycles.append([int(v) for v in cyc])
            return cycles
    
    # Fallback: find any triangle
    edge_set = set()
    for u, v in P['edges']:
        edge_set.add((min(u, v), max(u, v)))
    for a in range(n):
        for b in range(a + 1, n):
            if (a, b) not in edge_set:
                continue
            for c in range(b + 1, n):
                if (b, c) in edge_set and (a, c) in edge_set:
                    return [[a, b, c]]
    
    return [[0, 1, 2]]
# EVOLVE-BLOCK-END
