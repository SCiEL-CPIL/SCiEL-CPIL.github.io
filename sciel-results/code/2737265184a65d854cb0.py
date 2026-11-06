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

    # Precompute all 8-subsets and map pairs to 8-sets
    all_8sets = list(combinations(range(16), 8))
    pair_to_8sets = {}
    for vs in all_8sets:
        for i in range(8):
            for j in range(i+1, 8):
                pair = (vs[i], vs[j])
                if pair not in pair_to_8sets:
                    pair_to_8sets[pair] = []
                pair_to_8sets[pair].append(vs)

    # Precompute Q3 isomorphism check using bitmask comparison
    # For each 8-set, we need to check if the induced subgraph is Q3
    # Q3 has 12 edges and is 3-regular bipartite
    
    def is_q3_subgraph(vs, adj_bits):
        """Fast check: does vs induce exactly Q3?"""
        # Build local adjacency bitmask
        local_adj = [0] * 8
        for i in range(8):
            u = vs[i]
            mask = adj_bits[u]
            for j in range(8):
                if j != i and (mask >> vs[j]) & 1:
                    local_adj[i] |= (1 << j)
        
        # Check degrees: each vertex must have degree >= 3 in induced subgraph
        # For Q3 exactly, each must have degree exactly 3
        # But we're checking if it CONTAINS Q3, so degree >= 3 is necessary
        for i in range(8):
            deg = bin(local_adj[i]).count('1')
            if deg < 3:
                return False
        
        # Now check if the induced subgraph on these 8 vertices contains Q3
        # Use backtracking with pruning
        assignment = [-1] * 8
        vs_used = [False] * 8
        
        def backtrack(q3_v):
            if q3_v == 8:
                return True
            # Try each unassigned vertex
            for vs_i in range(8):
                if vs_used[vs_i]:
                    continue
                ok = True
                for nb in q3_adj[q3_v]:
                    if nb < q3_v:
                        if assignment[nb] == -1:
                            ok = False
                            break
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
        """Check if adding edge (u,v) creates a Q3."""
        if u > v:
            u, v = v, u
        relevant_8sets = pair_to_8sets.get((u, v), [])
        for vs in relevant_8sets:
            if is_q3_subgraph(vs, adj_bits):
                return True
        return False

    def edges_to_adj_bits(edges):
        adj = [0] * 16
        for (u, v) in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    # Multiple strategies to build initial graph
    best_edges = set()
    best_count = 0

    # Strategy 1: Greedy from K3,13
    k313 = set((a, b) for a in range(3) for b in range(3, 16))
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
        if time.time() > deadline - 40:
            break
        u, v = edge
        new_adj = current_adj[:]
        new_adj[u] |= (1 << v)
        new_adj[v] |= (1 << u)
        
        if not check_edge_creates_q3(u, v, new_adj):
            current_adj = new_adj
            current_edges.add(edge)

    if len(current_edges) > best_count:
        best_count = len(current_edges)
        best_edges = set(current_edges)

    # Strategy 2: Start from random sparse graph and add edges
    random.seed(42)
    for attempt in range(3):
        if time.time() > deadline - 30:
            break
        
        # Start with a random subset of edges
        all_possible = [(i, j) for i in range(16) for j in range(i + 1, 16)]
        random.shuffle(all_possible)
        
        # Start with ~20 edges
        initial_edges = set(all_possible[:20])
        # Verify no Q3
        valid = True
        adj_check = edges_to_adj_bits(initial_edges)
        for vs in all_8sets:
            if is_q3_subgraph(vs, adj_check):
                valid = False
                break
        
        if not valid:
            # Remove edges until valid
            for e in list(initial_edges):
                if time.time() > deadline - 30:
                    break
                initial_edges.discard(e)
                adj_check = edges_to_adj_bits(initial_edges)
                has_q3 = False
                for vs in all_8sets:
                    if is_q3_subgraph(vs, adj_check):
                        has_q3 = True
                        break
                if not has_q3:
                    break
        
        if not valid:
            continue

        local_adj = edges_to_adj_bits(initial_edges)
        local_edges = set(initial_edges)
        
        # Greedy add
        missing = [(i, j) for (i, j) in all_possible if (i, j) not in local_edges]
        random.shuffle(missing)
        
        for edge in missing:
            if time.time() > deadline - 25:
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

    # Local search with swaps
    local_adj = edges_to_adj_bits(best_edges)
    local_edges = set(best_edges)
    
    all_possible = [(i, j) for i in range(16) for j in range(i + 1, 16)]
    
    for iteration in range(5):
        if time.time() > deadline - 5:
            break
        
        missing_edges = [(i, j) for (i, j) in all_possible if (i, j) not in local_edges]
        random.shuffle(missing_edges)
        
        improved = False
        for edge in missing_edges:
            if time.time() > deadline - 5:
                break
            u, v = edge
            
            new_adj = local_adj[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            
            if not check_edge_creates_q3(u, v, new_adj):
                local_adj = new_adj
                local_edges.add(edge)
                improved = True
                if len(local_edges) > best_count:
                    best_count = len(local_edges)
                    best_edges = set(local_edges)
        
        if not improved:
            break

    result = [[u, v] for (u, v) in best_edges]
    return result
# EVOLVE-BLOCK-END
