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

    # Precompute all 8-subsets and pair mappings
    all_8sets = list(combinations(range(16), 8))
    pair_to_8sets = {}
    for vs in all_8sets:
        for i in range(8):
            for j in range(i+1, 8):
                pair = (vs[i], vs[j])
                if pair not in pair_to_8sets:
                    pair_to_8sets[pair] = []
                pair_to_8sets[pair].append(vs)

    # Precompute all Q3 embeddings: list of (tuple_of_8_vertices, adj_pattern)
    # For each 8-set, we check if it contains a Q3 by testing all 8! / 8 / 2 = 2520 labelings
    # But that's too many. Instead, use a smarter approach:
    # A set of 8 vertices contains Q3 iff there exists a bijection f: {0..7} -> vs
    # such that for all q3 edges (a,b), f(a)-f(b) is an edge in the induced subgraph.
    # We can use the fact that Q3 has 12 edges and 8 vertices, all degree 3.

    # Key insight: For a given 8-set vs, we can check Q3 by:
    # 1. All 8 vertices must have degree >= 3 within vs
    # 2. The induced subgraph must have at least 12 edges
    # 3. Then try to find a Q3 embedding via backtracking with pruning

    def contains_q3_fast(vs, adj_bits):
        """Check if 8 vertices vs induce a Q3. adj_bits[v] = bitmask of neighbors."""
        # Build local adjacency as bitmasks
        local_adj = [0] * 8
        edge_count = 0
        for i in range(8):
            u = vs[i]
            mask = adj_bits[u]
            for j in range(i+1, 8):
                if (mask >> vs[j]) & 1:
                    local_adj[i] |= (1 << j)
                    local_adj[j] |= (1 << i)
                    edge_count += 1
        
        if edge_count < 12:
            return False
        
        # Degree check: all must have degree >= 3
        for i in range(8):
            if bin(local_adj[i]).count('1') < 3:
                return False
        
        # Backtracking search for Q3 embedding
        # Fix vertex 0 of Q3 to vs[0] to break symmetry
        assignment = [-1] * 8
        assignment[0] = 0  # q3 vertex 0 maps to vs index 0
        used = [False] * 8
        used[0] = True
        
        def backtrack(q3_v):
            if q3_v == 8:
                return True
            for vs_i in range(8):
                if used[vs_i]:
                    continue
                # Check all neighbors of q3_v that are already assigned
                ok = True
                for nb in q3_adj[q3_v]:
                    if nb < q3_v:
                        if not ((local_adj[vs_i] >> assignment[nb]) & 1):
                            ok = False
                            break
                if ok:
                    assignment[q3_v] = vs_i
                    used[vs_i] = True
                    if backtrack(q3_v + 1):
                        return True
                    used[vs_i] = False
                    assignment[q3_v] = -1
            return False
        
        return backtrack(1)

    def check_edge_creates_q3(u, v, adj_bits):
        """Check if adding edge (u,v) creates a Q3. Only check 8-sets containing u and v."""
        if u > v:
            u, v = v, u
        relevant_8sets = pair_to_8sets.get((u, v), [])
        for vs in relevant_8sets:
            if contains_q3_fast(vs, adj_bits):
                return True
        return False

    def edges_to_adj_bits(edges):
        adj = [0] * 16
        for (u, v) in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def count_edges(adj_bits):
        count = 0
        for i in range(16):
            count += bin(adj_bits[i]).count('1')
        return count // 2

    # Greedy construction: start empty, add edges in order of "safety"
    # An edge is safe if it doesn't create a Q3.
    # Order edges by how many Q3-containing 8-sets they're in (fewer = safer)
    
    all_edges = [(i, j) for i in range(16) for j in range(i + 1, 16)]
    
    # Precompute for each edge, the number of 8-sets it appears in
    # (this is constant: C(14,6) = 3003 for all edges, so not useful for ordering)
    # Instead, order by vertex degrees: prefer edges between low-degree vertices early
    
    # Start with a good baseline: complete 3-partite graph K_{4,4,4,4} has 48 edges
    # But that might contain Q3. Let's use a different approach.
    
    # Strategy: greedy add edges, prioritizing those that are "safe"
    # Use a priority queue based on how many potential Q3s the edge participates in
    
    current_adj = [0] * 16
    current_edges = set()
    
    # Order edges: prioritize edges within groups (to build dense clusters)
    # Vertices 0-3, 4-7, 8-11, 12-15
    def edge_order(e):
        u, v = e
        gu = u // 4
        gv = v // 4
        if gu == gv:
            return 0  # Within group
        else:
            return 1  # Between groups
    
    sorted_edges = sorted(all_edges, key=edge_order)
    
    for edge in sorted_edges:
        if time.time() > deadline - 60:
            break
        u, v = edge
        new_adj = current_adj[:]
        new_adj[u] |= (1 << v)
        new_adj[v] |= (1 << u)
        
        if not check_edge_creates_q3(u, v, new_adj):
            current_adj = new_adj
            current_edges.add(edge)
    
    best_count = len(current_edges)
    best_edges = set(current_edges)
    best_adj = current_adj[:]
    
    # Local search with swaps: try to add an edge, and if it creates Q3,
    # try removing a different edge to make it work
    random.seed(42)
    
    for restart in range(3):
        if time.time() > deadline - 15:
            break
        
        local_adj = best_adj[:]
        local_edges = set(best_edges)
        
        # Try adding missing edges
        missing = [(i, j) for i in range(16) for j in range(i+1, 16) if (i, j) not in local_edges]
        random.shuffle(missing)
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for edge in missing:
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
                        best_adj = local_adj[:]
                    break  # Restart missing list after improvement
        
        # Swap-based improvement: for each missing edge, try to find an edge to remove
        if time.time() < deadline - 5:
            missing2 = [(i, j) for i in range(16) for j in range(i+1, 16) if (i, j) not in local_edges]
            random.shuffle(missing2)
            
            for edge in missing2:
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
                        best_adj = local_adj[:]
                else:
                    # Try swapping: remove one existing edge, then add this one
                    # Only try a few swaps to save time
                    for _ in range(5):
                        if time.time() > deadline - 5:
                            break
                        # Pick a random existing edge to try removing
                        existing = list(local_edges)
                        if not existing:
                            break
                        remove_edge = random.choice(existing)
                        ru, rv = remove_edge
                        
                        # Remove remove_edge, add edge
                        trial_adj = local_adj[:]
                        trial_adj[ru] &= ~(1 << rv)
                        trial_adj[rv] &= ~(1 << ru)
                        trial_adj[u] |= (1 << v)
                        trial_adj[v] |= (1 << u)
                        
                        # Check if trial graph is Q3-free
                        # Only need to check 8-sets containing u,v or ru,rv
                        # Simplification: just check if the new edge creates Q3
                        if not check_edge_creates_q3(u, v, trial_adj):
                            local_adj = trial_adj
                            local_edges.discard(remove_edge)
                            local_edges.add(edge)
                            if len(local_edges) > best_count:
                                best_count = len(local_edges)
                                best_edges = set(local_edges)
                                best_adj = local_adj[:]
                            break  # Success, move to next missing edge
    
    result = [[u, v] for (u, v) in best_edges]
    return result
# EVOLVE-BLOCK-END
