import random
import itertools
import sys

def construct():
    # Try to construct a C5-free graph with alpha < 9 on n=35 vertices
    # Use random greedy edge addition with C5 constraint
    # Then verify alpha < 9 exactly
    
    for n in [35, 34, 33, 32, 31, 30, 29, 28]:
        for trial in range(500):
            adj = [set() for _ in range(n)]
            edges = []
            
            # Shuffle all possible edges
            all_edges = list(itertools.combinations(range(n), 2))
            random.shuffle(all_edges)
            
            for u, v in all_edges:
                if u in adj[v]:
                    continue
                # Check if adding edge (u,v) creates a C5
                # A C5 containing edge (u,v) needs a path of length 4 between u and v
                # i.e., u-a-b-c-v where a in N(u), c in N(v), b in N(a)∩N(c)
                creates_c5 = False
                for a in adj[u]:
                    for c in adj[v]:
                        if a == c:
                            continue
                        if adj[a] & adj[c]:
                            creates_c5 = True
                            break
                    if creates_c5:
                        break
                if not creates_c5:
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append([u, v])
            
            # Check if alpha < 9 (no independent set of size 9)
            if not has_independent_set(n, adj, 9):
                return {'n': n, 'edges': edges}
    
    # Fallback: deterministic construction for n=28
    # Disjoint union of 7 K4's - each K4 has no C5, alpha of K4 is 1, so alpha of union is 7 < 9
    edges = []
    for b in range(7):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    return {'n': 28, 'edges': edges}

def has_independent_set(n, adj, k):
    """Check if there exists an independent set of size k using branch and bound."""
    # Use a recursive backtracking with pruning
    # Order vertices by degree (higher degree first for better pruning)
    order = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
    # Map from original vertex to position in order
    pos = [0] * n
    for i, v in enumerate(order):
        pos[v] = i
    
    # Convert adjacency to bitmasks for faster operations
    adj_mask = [0] * n
    for i in range(n):
        for j in adj[i]:
            adj_mask[i] |= (1 << j)
    
    # Use recursive search with candidates as a set
    def dfs(candidates, count):
        if count == k:
            return True
        if not candidates:
            return False
        
        # Upper bound: if len(candidates) < k - count, prune
        if len(candidates) < k - count:
            return False
        
        # Greedy coloring upper bound on independence number of candidate subgraph
        # This is expensive, so use a simpler bound: len(candidates)
        # Better: try to find a vertex to branch on
        
        # Pick a vertex from candidates (use first in some order)
        # For better pruning, pick vertex with highest degree in induced subgraph
        v = None
        max_deg = -1
        for u in candidates:
            deg_in_candidates = len(adj[u] & candidates)
            if deg_in_candidates > max_deg:
                max_deg = deg_in_candidates
                v = u
        
        if v is None:
            return False
        
        # Option 1: Include v
        # New candidates = candidates - {v} - N(v)
        new_candidates_incl = candidates - {v} - adj[v]
        if dfs(new_candidates_incl, count + 1):
            return True
        
        # Option 2: Exclude v
        new_candidates_excl = candidates - {v}
        if dfs(new_candidates_excl, count):
            return True
        
        return False
    
    all_vertices = frozenset(range(n))
    return dfs(all_vertices, 0)
