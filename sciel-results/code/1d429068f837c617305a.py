import time
import itertools
import random
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    n = 16
    total_possible_edges = n * (n - 1) // 2  # 120
    
    # Q3 structure: 3-cube with 8 vertices, 12 edges
    Q3_EDGES = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]
    
    # Precompute Q3 adjacency as bitmasks for fast checking
    q3_adj_mask = [0] * 8
    for u, v in Q3_EDGES:
        q3_adj_mask[u] |= (1 << v)
        q3_adj_mask[v] |= (1 << u)
    
    # All 8-vertex subsets of 0..15
    all_8_subsets = list(itertools.combinations(range(n), 8))
    
    def find_one_q3_fast(adj_mask):
        """
        Fast Q3 detection using bitmask operations.
        Returns (True, edge_list) if a Q3 is found, (False, None) otherwise.
        adj_mask: list of 16 integers, each a bitmask of neighbors.
        """
        # Iterate over all 8-element subsets
        for S in all_8_subsets:
            S_set = set(S)
            
            # Quick filter: each vertex must have degree >= 3 within S
            ok = True
            for v in S:
                mask = adj_mask[v] & ((1 << 16) - 1)
                deg = 0
                for u in S:
                    if u != v and (mask >> u) & 1:
                        deg += 1
                if deg < 3:
                    ok = False
                    break
            if not ok:
                continue
            
            # Backtracking with bitmask pruning
            mapping = [-1] * 8
            used = 0  # bitmask of used vertices in S
            
            def dfs(idx):
                nonlocal used
                if time.time() > deadline - 0.5:
                    return False  # Abort if too slow
                
                if idx == 8:
                    return True
                
                # Neighbors of idx in Q3 that are already assigned (index < idx)
                assigned_nbs = []
                for nb in range(8):
                    if nb < idx and (q3_adj_mask[idx] >> nb) & 1:
                        assigned_nbs.append(nb)
                
                # Try each unused vertex in S
                for v in S:
                    if (used >> v) & 1:
                        continue
                    
                    # Check if v is adjacent to all already-assigned neighbors of idx
                    valid = True
                    for nb in assigned_nbs:
                        if not (adj_mask[mapping[nb]] >> v) & 1:
                            valid = False
                            break
                    
                    if valid:
                        mapping[idx] = v
                        used |= (1 << v)
                        if dfs(idx + 1):
                            return True
                        used &= ~(1 << v)
                        mapping[idx] = -1
                
                return False
            
            if dfs(0):
                # Extract the 12 edges of the found Q3
                edges = []
                for u, v in Q3_EDGES:
                    edges.append((mapping[u], mapping[v]))
                return (True, edges)
        
        return (False, None)
    
    def get_edge_list(adj_mask):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_mask[u] >> v) & 1:
                    edges.append([u, v])
        return edges
    
    # Strategy: Use structural insight about Q3-free graphs
    # A graph is Q3-free if it doesn't contain an induced subgraph isomorphic to Q3
    # Key insight: If we can partition vertices into sets where certain density constraints are met
    
    # Start with a good seed: try different complete bipartite graphs
    # K_{a,b} is Q3-free if min(a,b) < 8 (since Q3 needs 8 vertices)
    # Actually K_{a,b} contains Q3 iff both a >= 4 and b >= 4 (Q3 is bipartite with parts of size 4)
    
    # So K_{3,13} = 39 edges is Q3-free
    # K_{4,12} = 48 edges - need to check if it contains Q3
    # K_{4,12}: Q3 has bipartition (even, odd) with 4 each. 
    # In K_{4,12}, we can pick 4 from part A and 4 from part B, getting K_{4,4} which contains Q3
    
    # Better approach: Use a random graph with controlled density and iteratively remove Q3s
    
    def _make_random_graph(p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj
    
    def _make_complete_bipartite(a, b):
        adj = [0] * n
        # Part A: vertices 0..a-1, Part B: vertices a..a+b-1
        for u in range(a):
            for v in range(a, a + b):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj
    
    def _count_edges(adj):
        return sum(bin(adj[u]).count('1') for u in range(n)) // 2
    
    # Multiple strategies with different starting points
    strategies = [
        ("K3,13", lambda: _make_complete_bipartite(3, 13)),
        ("K2,14", lambda: _make_complete_bipartite(2, 14)),
        ("Random0.80", lambda: _make_random_graph(0.80, 1)),
        ("Random0.85", lambda: _make_random_graph(0.85, 2)),
        ("Random0.90", lambda: _make_random_graph(0.90, 3)),
        ("Random0.95", lambda: _make_random_graph(0.95, 4)),
    ]
    
    best_m = 0
    best_adj = [0] * n
    best_edges = []
    
    for strat_name, make_fn in strategies:
        if time.time() > deadline - 5.0:
            break
        
        adj = make_fn()
        current_edges = _count_edges(adj)
        
        # Greedy removal loop with randomized edge selection
        max_iterations = 200
        for iteration in range(max_iterations):
            if time.time() > deadline - 3.0:
                break
            
            found, edges = find_one_q3_fast(adj)
            
            if not found:
                # Q3-free!
                if current_edges > best_m:
                    best_m = current_edges
                    best_adj = adj[:]
                    best_edges = get_edge_list(adj)
                break
            
            # Remove a random edge from the found Q3
            random.shuffle(edges)
            u, v = edges[0]
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            current_edges -= 1
        
        # Also update best if we have a Q3-free graph even if not improved
        if current_edges > best_m:
            found, _ = find_one_q3_fast(adj)
            if not found:
                best_m = current_edges
                best_adj = adj[:]
                best_edges = get_edge_list(adj)
    
    # Try to add edges to the best found so far
    if time.time() < deadline - 5.0 and best_m > 0:
        adj = best_adj[:]
        current_edges = best_m
        
        # Generate candidate edges not in current graph
        candidates = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] >> v) & 1:
                    candidates.append((u, v))
        
        random.shuffle(candidates)
        
        for u, v in candidates:
            if time.time() > deadline - 2.0:
                break
            
            # Try adding edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            current_edges += 1
            
            found, _ = find_one_q3_fast(adj)
            if found:
                # Remove the edge we just added
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                current_edges -= 1
            else:
                # Edge addition successful
                best_m = current_edges
                best_adj = adj[:]
                best_edges = get_edge_list(adj)
    
    return best_edges
