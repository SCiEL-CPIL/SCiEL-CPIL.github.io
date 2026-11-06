"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n = 16
    q3_adj = {
        0: [1, 2, 4],
        1: [0, 3, 5],
        2: [0, 3, 6],
        3: [1, 2, 7],
        4: [0, 5, 6],
        5: [1, 4, 7],
        6: [2, 4, 7],
        7: [3, 5, 6]
    }

    # Precompute all 8-subsets
    all_8sets = list(combinations(range(16), 8))
    # Precompute 8-sets containing each pair (for fast incremental checks)
    pair_to_8sets = {}
    for vs in all_8sets:
        for i in range(8):
            for j in range(i+1, 8):
                pair = (vs[i], vs[j])
                if pair not in pair_to_8sets:
                    pair_to_8sets[pair] = []
                pair_to_8sets[pair].append(vs)

    def contains_q3_fast(vs, adj_bits):
        """Check if 8 vertices vs induce a Q3. adj_bits[v] = bitmask of neighbors."""
        # Build local adjacency
        local_adj = [0] * 8
        for i in range(8):
            u = vs[i]
            mask = adj_bits[u]
            for j in range(8):
                if j != i and (mask >> vs[j]) & 1:
                    local_adj[i] |= (1 << j)
        
        # Degree check
        for i in range(8):
            if bin(local_adj[i]).count('1') < 3:
                return False
        
        # Backtrack
        assignment = [-1] * 8
        vs_used = [False] * 8
        
        def backtrack(q3_v):
            if q3_v == 8:
                return True
            for vs_i in range(8):
                if vs_used[vs_i]:
                    continue
                ok = True
                for nb in q3_adj[q3_v]:
                    if nb < q3_v:
                        if not ((local_adj[vs_i] >> assignment[nb]) & 1):
                            ok = False
                            break
                if ok:
                    assignment[q3_v] = vs_i
                    vs_used[vs_i] = True
                    if backtrack(q3_v + 1):
                        return True
                    vs_used[vs_i] = False
                    assignment[q3_v] = -1
            return False
        
        return backtrack(0)

    def check_edge_creates_q3(u, v, adj_bits):
        """Check if adding edge (u,v) creates a Q3. Only check 8-sets containing u and v."""
        if u > v:
            u, v = v, u
        relevant_8sets = pair_to_8sets.get((u, v), [])
        for vs in relevant_8sets:
            if contains_q3_fast(vs, adj_bits):
                return True
        return False

    # Baseline: K3,13
    k313 = set((a, b) for a in range(3) for b in range(3, 16))
    best_edges = set(k313)
    best_count = len(k313)

    def edges_to_adj_bits(edges):
        adj = [0] * 16
        for (u, v) in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    current_adj = edges_to_adj_bits(k313)
    current_edges = set(k313)

    candidate_edges = [(i, j) for i in range(16) for j in range(i + 1, 16)
                       if (i, j) not in current_edges]

    def edge_priority(e):
        u, v = e
        if u < 3 and v < 3:
            return 0
        if u >= 3 and v >= 3:
            return 1
        return 2

    candidate_edges.sort(key=edge_priority)

    for edge in candidate_edges:
        if time.time() > deadline - 35:
            break
        u, v = edge
        # Temporarily add edge
        new_adj = current_adj[:]
        new_adj[u] |= (1 << v)
        new_adj[v] |= (1 << u)
        
        if not check_edge_creates_q3(u, v, new_adj):
            current_adj = new_adj
            current_edges.add(edge)

    if len(current_edges) > best_count:
        best_count = len(current_edges)
        best_edges = set(current_edges)

    # Local search: try adding edges, and if Q3 forms, try swapping
    random.seed(42)
    local_adj = edges_to_adj_bits(best_edges)
    local_edges = set(best_edges)
    
    all_possible = [(i, j) for i in range(16) for j in range(i + 1, 16)]
    
    # Build list of missing edges
    missing_edges = [(i, j) for (i, j) in all_possible if (i, j) not in local_edges]
    random.shuffle(missing_edges)
    
    improvement_count = 0
    for idx, edge in enumerate(missing_edges):
        if time.time() > deadline - 5:
            break
        u, v = edge
        
        new_adj = local_adj[:]
        new_adj[u] |= (1 << v)
        new_adj[v] |= (1 << u)
        
        if not check_edge_creates_q3(u, v, new_adj):
            local_adj = new_adj
            local_edges.add(edge)
            improvement_count += 1
            if len(local_edges) > best_count:
                best_count = len(local_edges)
                best_edges = set(local_edges)
    
    # Second pass: try more missing edges after improvements
    missing_edges2 = [(i, j) for (i, j) in all_possible if (i, j) not in local_edges]
    random.shuffle(missing_edges2)
    
    for edge in missing_edges2:
        if time.time() > deadline - 5:
            break
        u, v = edge
        
        new_adj = local_adj[:]
        new_adj[u] |= (1 << v)
        new_adj[v] |= (1 << u)
        
        if not check_edge_creates_q3(u, v, new_adj):
            local_adj = new_adj
            local_edges.add(edge)
            if len(local_edges) > best_count:
                best_count = len(local_edges)
                best_edges = set(local_edges)

    result = [[u, v] for (u, v) in best_edges]
    return result
# EVOLVE-BLOCK-END
