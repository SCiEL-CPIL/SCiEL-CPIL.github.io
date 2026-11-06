"""Rewritten program: Q3-free graph maximization on 16 vertices."""
import time
import random
from itertools import combinations

def _creates_q3(adj, u, v):
    """
    Check if adding edge (u, v) to the graph represented by adj creates a Q3 subgraph.
    The new Q3 must contain the edge (u, v).
    
    A Q3 is a 3-regular bipartite graph on 8 vertices. If we add edge (u,v),
    the Q3 must have u and v adjacent. In Q3, each vertex has 3 neighbors.
    
    Structure: If (u,v) is an edge in Q3, then:
    - u has 2 other neighbors a, b in Q3
    - v has 2 other neighbors c, d in Q3
    - a, b are non-adjacent (bipartite)
    - c, d are non-adjacent (bipartite)
    - There are cross edges: either (a~c and b~d) or (a~d and b~c)
    - There are 2 more vertices e, f such that:
      - e adjacent to a, b and one of {c,d}
      - f adjacent to c, d and one of {a,b}
      - Actually, let me reconsider the Q3 structure more carefully.
    """
    # In Q3 with edge (u,v), u and v are in opposite bipartite sets.
    # u's neighbors in Q3: v, a, b (a, b in same bipartite set as u's other neighbors)
    # v's neighbors in Q3: u, c, d (c, d in same bipartite set as v)
    # Wait, u and v are adjacent, so they're in different bipartite sets.
    # Let's say u is in set A, v is in set B.
    # u's other neighbors a, b are in set B.
    # v's other neighbors c, d are in set A.
    # In Q3: a~v, b~v, c~u, d~u
    # Cross edges between {a,b} and {c,d}: In Q3, each of a,b,c,d has degree 3.
    # a is adjacent to u, v, and one more. Since a is in set B, its neighbors are in set A: u and one of {c,d}.
    # Actually in Q3, the bipartite sets each have 4 vertices.
    # Set A: {u, c, d, e} and Set B: {v, a, b, f} (for example)
    # Edges of Q3: u-v, u-c, u-d, u-e, v-a, v-b, v-f, a-c, a-d, a-? ...
    # Let me just enumerate: Q3 vertices are 3-bit strings. Edge between x,y if Hamming distance 1.
    # 
    # Key property: If (u,v) is an edge, then there are exactly 2 common "structures".
    # Let me use a different approach: just check all 8-vertex subsets containing u and v.
    
    # More efficient: iterate over pairs (a,b) of non-adjacent neighbors of u (excluding v)
    # and pairs (c,d) of non-adjacent neighbors of v (excluding u), then check for the remaining structure.
    
    nu = [x for x in adj[u] if x != v]
    nv = [x for x in adj[v] if x != u]
    
    if len(nu) < 2 or len(nv) < 2:
        return False
    
    # Find non-adjacent pairs in nu
    pairs_u = []
    for i in range(len(nu)):
        for j in range(i + 1, len(nu)):
            a, b = nu[i], nu[j]
            if b not in adj[a]:
                pairs_u.append((a, b))
    
    # Find non-adjacent pairs in nv
    pairs_v = []
    for i in range(len(nv)):
        for j in range(i + 1, len(nv)):
            c, d = nv[i], nv[j]
            if d not in adj[c]:
                pairs_v.append((c, d))
    
    if not pairs_u or not pairs_v:
        return False
    
    for a, b in pairs_u:
        for c, d in pairs_v:
            # Check cross edges: either (a~c and b~d) or (a~d and b~c)
            cross1 = (c in adj[a] and d in adj[b])
            cross2 = (d in adj[a] and c in adj[b])
            if not (cross1 or cross2):
                continue
            
            # Now we need to find e and f such that they complete the Q3.
            # e must be adjacent to a and b (and not to u or v in the Q3 context, but could be in graph)
            # f must be adjacent to c and d
            # And e~f must hold
            # Also e != f, e not in {u,v,a,b,c,d}, f not in {u,v,a,b,c,d}
            
            common_ab = adj[a] & adj[b]
            common_cd = adj[c] & adj[d]
            
            # Filter out vertices that are part of the 6 already identified
            excluded = {u, v, a, b, c, d}
            
            e_candidates = [x for x in common_ab if x not in excluded]
            f_candidates = [x for x in common_cd if x not in excluded]
            
            for e in e_candidates:
                for f in f_candidates:
                    if e == f:
                        continue
                    if f in adj[e]:
                        # Verify this is actually a Q3: all 8 vertices form a 3-regular bipartite graph
                        # Vertices: u, v, a, b, c, d, e, f
                        # Edges should be: u-v, u-c, u-d, u-e, v-a, v-b, v-f, 
                        #                    a-c, a-d, a-?, ... 
                        # Actually we already checked the key structural conditions.
                        # Let's verify it's Q3 by checking all degrees and no other edges within the 8 vertices.
                        verts = {u, v, a, b, c, d, e, f}
                        # Check that within these 8 vertices, the induced subgraph is exactly Q3
                        # Q3 has 12 edges.
                        edge_count = 0
                        for x in verts:
                            for y in verts:
                                if x < y and y in adj[x]:
                                    edge_count += 1
                        # In Q3, each vertex has degree 3, so total edges = 8*3/2 = 12
                        if edge_count == 12:
                            # Also verify each vertex has degree exactly 3 within the subgraph
                            deg_ok = True
                            for x in verts:
                                d = sum(1 for y in verts if y != x and y in adj[x])
                                if d != 3:
                                    deg_ok = False
                                    break
                            if deg_ok:
                                return True
    return False


def _greedy_construct(adj, edges_to_try, deadline):
    """Try to add edges greedily, returning the edge list."""
    current_edge_list = []
    for u, v in edges_to_try:
        if time.time() > deadline:
            break
        if v not in adj[u]:  # not already added
            if not _creates_q3(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
                current_edge_list.append((u, v))
    return current_edge_list


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_edges = 0
    best_edge_list = []
    
    # Try multiple random orders with different strategies
    for seed in range(500):
        if time.time() > deadline - 5:
            break
        
        random.seed(seed)
        adj = [set() for _ in range(n)]
        
        # Different strategies for edge ordering
        strategy = seed % 4
        
        if strategy == 0:
            # Pure random
            edges_to_try = all_edges[:]
            random.shuffle(edges_to_try)
        elif strategy == 1:
            # Start with a known good structure (K3,13) then fill in
            # K_{3,13}: vertices 0,1,2 connect to 3..15
            edges_to_try = []
            for i in range(3):
                for j in range(3, 16):
                    edges_to_try.append((i, j))
            # Add edges among {3..15} and among {0,1,2}
            remaining = [(i, j) for i, j in all_edges 
                        if not (i < 3 and j >= 3)]
            random.shuffle(remaining)
            edges_to_try.extend(remaining)
        elif strategy == 2:
            # Random but biased toward high-degree vertices first
            edges_to_try = all_edges[:]
            # Sort by sum of vertex indices (arbitrary bias)
            random.shuffle(edges_to_try)
        else:
            # Two-phase: random then re-shuffle
            edges_to_try = all_edges[:]
            random.shuffle(edges_to_try)
        
        current_edge_list = _greedy_construct(adj, edges_to_try, deadline)
        
        if len(current_edge_list) > best_edges:
            best_edges = len(current_edge_list)
            best_edge_list = current_edge_list[:]
    
    # Local improvement: try to add more edges to the best solution
    if best_edge_list:
        adj = [set() for _ in range(n)]
        for u, v in best_edge_list:
            adj[u].add(v)
            adj[v].add(u)
        
        # Try adding any missing edge
        for u, v in all_edges:
            if time.time() > deadline - 2:
                break
            if v not in adj[u]:
                if not _creates_q3(adj, u, v):
                    adj[u].add(v)
                    adj[v].add(u)
                    best_edge_list.append((u, v))
                    best_edges += 1
        
        # Repeat local improvement a few times
        for _ in range(3):
            if time.time() > deadline - 2:
                break
            improved = False
            for u, v in all_edges:
                if time.time() > deadline - 2:
                    break
                if v not in adj[u]:
                    if not _creates_q3(adj, u, v):
                        adj[u].add(v)
                        adj[v].add(u)
                        best_edge_list.append((u, v))
                        best_edges += 1
                        improved = True
            if not improved:
                break
    
    return [[u, v] for u, v in best_edge_list]
