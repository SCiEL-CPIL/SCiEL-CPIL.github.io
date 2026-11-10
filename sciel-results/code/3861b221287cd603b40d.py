"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import itertools
from functools import lru_cache

def _is_q3_free(adj):
    """
    Check if graph with adjacency matrix adj (16x16, symmetric, 0/1) is Q3-free.
    Q3 has 8 vertices, 12 edges. We check all subsets of 8 vertices.
    For each subset, check if it contains a subgraph isomorphic to Q3.
    """
    n = 16
    # Precompute: for each set of 8 vertices (as a frozenset), check if Q3 is a subgraph
    # Q3 structure: 8 vertices, each of degree 3, bipartite with parts of size 4,
    # and the bipartite graph is the incidence of edges of a cube.
    # Actually, let's use a more direct approach.
    # Q3 is the 3-cube. Its vertices can be labeled 0..7 (binary 000..111).
    # Two vertices are adjacent if they differ in exactly one bit.
    
    # For a set S of 8 vertices to contain Q3 as a subgraph, there must exist
    # an injection f: {0..7} -> S such that for all u,v in {0..7},
    # if popcount(u XOR v) == 1, then f(u) and f(v) are adjacent in G.
    
    # This is expensive to check for all 8-subsets. Let's use a smarter method.
    
    # Alternative: enumerate all ways to pick 8 vertices and check if they form a supergraph of Q3.
    # Number of 8-subsets from 16 is C(16,8) = 12870. For each, we need to check if Q3 is a subgraph.
    # To check if Q3 is a subgraph of an 8-vertex induced subgraph, we can try all 8! = 40320 bijections,
    # but that's too slow. Instead, we can use the fact that Q3 is 3-regular and bipartite.
    
    # Better approach: For each 8-subset S, check if the induced subgraph on S contains Q3.
    # We can do this by checking if there exists a 3-regular bipartite subgraph on S with parts of size 4
    # that is isomorphic to Q3.
    
    # Even better: Precompute the edge sets of Q3. Q3 has 12 edges.
    # For each 8-subset S, the induced subgraph has some set of edges. We need to check if
    # there's a subset of 12 edges forming Q3.
    
    # Let's use a different strategy: enumerate all possible embeddings of Q3 into the 16 vertices.
    # An embedding is determined by choosing 8 distinct vertices and a bijection to {0..7}.
    # But we can fix the first vertex to 0 to break symmetry... no, the vertices are labeled.
    
    # Let's just do: for each combination of 8 vertices, check if Q3 is a subgraph.
    # To check efficiently: 
    #   1. The subgraph must have at least 12 edges (since Q3 has 12 edges).
    #   2. It must be possible to find a 3-regular bipartite subgraph isomorphic to Q3.
    
    # Actually, let's use a canonical form approach. For each 8-subset, compute the adjacency
    # matrix of the induced subgraph, and check if Q3 is a subgraph.
    
    # Q3 adjacency (on vertices 0..7, where vertex i corresponds to binary i):
    q3_edges = set()
    for i in range(8):
        for j in range(i+1, 8):
            xor = i ^ j
            if xor & (xor - 1) == 0:  # power of 2, i.e., differ in one bit
                q3_edges.add((i, j))
    # q3_edges has 12 elements
    
    # For each 8-subset of {0..15}, we need to check if there's a bijection phi: {0..7} -> S
    # such that for all (i,j) in q3_edges, (phi(i), phi(j)) is an edge in G.
    
    # To make this feasible, we can use backtracking with pruning.
    # For each 8-subset S, we try to embed Q3 into it.
    
    # Let's precompute all 8-subsets
    all_8_subsets = list(itertools.combinations(range(16), 8))
    
    for S in all_8_subsets:
        S_set = set(S)
        # Check if the induced subgraph on S has at least 12 edges
        edge_count = 0
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                if adj[S[i]][S[j]]:
                    edge_count += 1
        if edge_count < 12:
            continue
        
        # Try to embed Q3 into S
        # We'll use backtracking: assign each Q3 vertex (0..7) to a distinct vertex in S
        # such that all Q3 edges map to G edges.
        
        # Order Q3 vertices for backtracking
        # Use a simple ordering 0,1,2,...,7
        assignment = {}
        
        def backtrack(q3_v):
            if q3_v == 8:
                return True
            for s_v in S:
                if s_v in assignment.values():
                    continue
                # Check if assigning q3_v -> s_v is consistent with already assigned vertices
                ok = True
                for prev_q3_v in range(q3_v):
                    if (min(prev_q3_v, q3_v), max(prev_q3_v, q3_v)) in q3_edges:
                        if not adj[assignment[prev_q3_v]][s_v]:
                            ok = False
                            break
                if ok:
                    assignment[q3_v] = s_v
                    if backtrack(q3_v + 1):
                        return True
                    del assignment[q3_v]
            return False
        
        if backtrack(0):
            return False
    
    return True

def construct():
    """
    Construct a Q3-free graph on 16 vertices with maximum number of edges.
    
    Strategy: Start with K4,4,4,4 (complete 4-partite with parts of size 4),
    which is Q3-free. Then add intra-part edges via local search, verifying
    Q3-freeness after each addition.
    """
    n = 16
    # Initial: complete 4-partite graph with parts {0,1,2,3}, {4,5,6,7}, {8,9,10,11}, {12,13,14,15}
    # Edges: all pairs (u,v) where u and v are in different parts
    adj = [[0]*n for _ in range(n)]
    edges = []
    
    parts = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]
    ]
    
    for i in range(n):
        for j in range(i+1, n):
            # Check if i and j are in different parts
            part_i = next(p for p in range(4) if i in parts[p])
            part_j = next(p for p in range(4) if j in parts[p])
            if part_i != part_j:
                adj[i][j] = 1
                adj[j][i] = 1
                edges.append([i, j])
    
    # Now try to add intra-part edges
    # Intra-part edge candidates: all pairs within each part
    intra_candidates = []
    for part in parts:
        for i in range(len(part)):
            for j in range(i+1, len(part)):
                intra_candidates.append((part[i], part[j]))
    
    random.seed(42)
    random.shuffle(intra_candidates)
    
    best_edges = edges[:]
    best_adj = [row[:] for row in adj]
    best_count = len(edges)
    
    # Local search: try adding each intra-part edge, check if Q3-free
    for u, v in intra_candidates:
        if adj[u][v]:
            continue
        # Try adding edge (u, v)
        adj[u][v] = 1
        adj[v][u] = 1
        if _is_q3_free(adj):
            edges.append([u, v])
            # Keep it
        else:
            # Remove it
            adj[u][v] = 0
            adj[v][u] = 0
    
    # Additional local search: try swapping edges to gain more
    # For simplicity, do a few rounds of trying to add edges and removing others
    for iteration in range(5):
        progress = False
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    continue
                adj[u][v] = 1
                adj[v][u] = 1
                if _is_q3_free(adj):
                    edges.append([u, v])
                    progress = True
                else:
                    adj[u][v] = 0
                    adj[v][u] = 0
        if not progress:
            break
    
    # Return the edge list
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                result.append([i, j])
    
    return result
# EVOLVE-BLOCK-END
