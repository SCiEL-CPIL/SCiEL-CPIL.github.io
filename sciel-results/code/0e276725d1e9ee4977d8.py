"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_triangle_free(adj, n):
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            for w in adj[v]:
                if w > v and w in adj[u]:
                    return False
    return True

def _max_independent_set_size(adj, n):
    # Branch and bound for maximum independent set size
    best = [0]
    # Use bitmasks for efficiency
    adj_mask = [0] * n
    for u in range(n):
        m = 0
        for v in adj[u]:
            m |= (1 << v)
        adj_mask[u] = m

    def backtrack(candidates, size):
        if size + len(candidates) <= best[0]:
            return
        if not candidates:
            if size > best[0]:
                best[0] = size
            return
        # Greedy lower bound
        # Pick a vertex with max degree in the candidate subgraph to branch
        # Actually for independent set, we want to pick a vertex and either include or exclude
        # Heuristic: pick vertex with minimum degree in candidate subgraph? 
        # Or just use first vertex.
        v = candidates & (-candidates)  # lowest set bit
        v_idx = v.bit_length() - 1
        
        # Include v: remove v and its neighbors from candidates
        new_cand = candidates & ~(adj_mask[v_idx] | v)
        backtrack(new_cand, size + 1)
        
        # Exclude v
        backtrack(candidates & ~v, size)

    backtrack((1 << n) - 1, 0)
    return best[0]

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    # Start with Andrasfai graph on 26 vertices
    n = 26
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n % 3 == 1:
                edges.add((i, j))
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Verify initial graph is valid
    assert _is_triangle_free(adj, n), "Initial graph has triangle"
    alpha = _max_independent_set_size(adj, n)
    assert alpha <= 9, f"Initial graph has independent set of size {alpha}"
    
    best_result = {'n': n, 'edges': [[u, v] for u, v in sorted(edges)]}
    
    # Try to add vertices one by one
    for target_n in range(n + 1, 41):
        if time.time() > deadline - 5.0:
            break
        
        # Try to add a new vertex
        added = False
        for attempt in range(100):
            if time.time() > deadline - 2.0:
                break
            
            # Randomly decide which edges to add for the new vertex
            # The new vertex can connect to a subset of existing vertices
            # such that no triangle is formed and no independent set of size 10 is formed
            
            # A vertex v is "safe" to connect to new vertex if:
            # 1. No two neighbors of v are adjacent (would form triangle)
            # 2. Adding the edge doesn't create an independent set of size 10
            
            # Actually, let's think differently. We add vertex n.
            # We choose which existing vertices it connects to.
            # Constraint: The set of non-neighbors of new vertex (plus new vertex) 
            # must not contain an independent set of size 9 (which would make size 10 with new vertex)
            # And the neighbors of new vertex must form an independent set in the original graph
            # (otherwise triangle with new vertex)
            
            # Let S be the set of vertices connected to new vertex.
            # S must be an independent set in G.
            # The complement T = V \ S must not contain an independent set of size 9 in G.
            
            # Let's try random subsets
            # First, find all maximal independent sets? Too expensive.
            # Instead, random sampling.
            
            # Random subset of vertices
            if attempt == 0:
                # Try empty set (new vertex isolated)
                S = set()
            else:
                # Random subset
                S = set(random.sample(range(n), random.randint(0, min(n, 15))))
            
            # Check if S is independent
            valid = True
            for u, v in combinations(S, 2):
                if v in adj[u]:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check if T contains independent set of size 9
            T = set(range(n)) - S
            # Need to check if T has independent set of size 9
            # This is expensive, so we only do it for promising candidates
            
            if len(T) < 9:
                continue  # Can't have independent set of size 9
            
            # Quick check: if T itself is not independent, we need deeper check
            # For now, skip the expensive check and just try to add
            # We'll verify after adding
            
            # Create new graph
            new_n = n + 1
            new_adj = [set(s) for s in adj]
            new_adj.append(set(S))
            for v in S:
                new_adj[v].add(new_n)
            
            # Check triangle-free
            if not _is_triangle_free(new_adj, new_n):
                continue
            
            # Check independent set
            if _max_independent_set_size(new_adj, new_n) > 9:
                continue
            
            # Valid!
            adj = new_adj
            n = new_n
            edges = set()
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.add((u, v))
            added = True
            break
        
        if not added:
            break
    
    return {'n': n, 'edges': [[u, v] for u, v in sorted(edges)]}
# EVOLVE-BLOCK-END
