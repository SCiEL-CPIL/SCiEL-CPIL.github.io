"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

_N = 16
_E = _N * (_N - 1) // 2  # 120 edges

def _edge_idx(u, v):
    if u > v:
        u, v = v, u
    # u < v, u in 0..14, v in u+1..15
    # number of edges with first vertex < u: sum_{i=0}^{u-1} (15-i) = 15u - u(u-1)/2
    # edges with first vertex == u and second vertex < v: v - u - 1
    return 15 * u - u * (u - 1) // 2 + (v - u - 1)

def _idx_to_edge(idx):
    # Inverse of _edge_idx
    u = 0
    while True:
        start = 15 * u - u * (u - 1) // 2
        count = 15 - u
        if idx < start + count:
            v = u + 1 + (idx - start)
            return u, v
        u += 1

def _build_q3_conflicts():
    """
    Build conflict sets: conflict[i] is a bitmask of edges that conflict with edge i.
    Two edges conflict if they can both be in a Q3 subgraph together.
    A Q3 has 8 vertices and 12 edges. Two edges of K16 conflict if there exists
    a set of 8 vertices containing both edge endpoints, and a Q3 subgraph on those
    8 vertices containing both edges.
    
    For efficiency: two edges (a,b) and (c,d) conflict if:
    - They share at most... actually in Q3, any two edges can coexist as long as
      the 4+ endpoints fit in 8 vertices and the Q3 structure can accommodate them.
    - More precisely: the union of endpoints has size <= 8, AND there exists a
      labeling of those vertices into {0,1}^3 such that both edges correspond to
      Hamming distance 1 pairs in the cube.
    
    Key insight: In Q3, two edges conflict (i.e., cannot both be absent in a Q3-free
    graph... wait, no. In the MIS formulation:
    - Nodes = edges of K16
    - We want to select a maximum set of edges with NO Q3 subgraph
    - Two edges "conflict" if they MUST both be present to form a Q3? No...
    
    Actually, the conflict graph for MIS: two edges conflict if they CANNOT both be
    in a Q3-free graph. That's not pairwise - it's a higher-order constraint.
    
    Let me reconsider. The correct formulation:
    - A set of edges is Q3-free iff it does NOT contain any Q3 subgraph.
    - A Q3 subgraph is a specific set of 12 edges on 8 vertices.
    - So the constraint is: for every possible Q3 embedding (8 vertices + 12 edges),
      not all 12 edges are selected.
    
    This is NOT a pairwise conflict problem - it's a hypergraph independent set problem.
    However, we can still use branch and bound with the constraint that no Q3 is complete.
    
    Alternative approach: Use the fact that Q3 is bipartite (4+4). 
    A Q3-free graph means no induced... no, no subgraph isomorphic to Q3.
    
    Let me think about this differently. The seed solution K3,13 works because Q3 is
    bipartite with parts of size 4. If we have a bipartition (A,B) with |A|<4, no Q3
    can exist. K3,13 has parts of size 3 and 13.
    
    For a better approach, let's use a different structural insight:
    Q3 has 8 vertices, 12 edges, each vertex degree 3.
    
    Strategy: Try to find a dense Q3-free graph by:
    1. Start with a good initial solution (e.g., K3,13 plus some intra-part edges)
    2. Use local search / simulated annealing to add edges while checking Q3-freeness
    3. Use branch and bound for final optimization
    
    For Q3 detection: check if the current graph contains a Q3 subgraph.
    A Q3 subgraph requires 8 vertices where the induced subgraph contains Q3 as a subgraph.
    
    Efficient Q3 check: For each set of 8 vertices, check if Q3 is a subgraph.
    But C(16,8) = 12870, and checking each is expensive.
    
    Better: Enumerate all Q3 embeddings. A Q3 embedding is determined by:
    - Choosing 8 vertices from 16: C(16,8) = 12870
    - For each, checking if the 12 Q3 edges are present in our graph
    
    Actually, let's precompute all possible Q3 edge sets. Two vertices in Q3 are
    connected iff their labels differ in exactly 1 bit. So for a fixed set of 8 vertices,
    we need to check if there's a bijection to {0,1}^3 such that the 12 edges of Q3
    are all present.
    
    For the branch and bound approach with the pairwise conflict approximation:
    Actually, let me use a different structural approach.
    
    Key observation from the inspiration: K3,13 + 13-cycle on the 13-vertex side gives
    39 + 13 = 52 edges. This works because adding edges within the 13-set creates
    degree-2 vertices in any potential Q3's larger part, preventing the 3-regular
    structure needed.
    
    Let me try a different construction: Use a Turan-like or random dense graph and
    verify Q3-freeness, then optimize.
    
    For now, let me implement a robust approach:
    1. Precompute all Q3 subgraphs (as sets of edge indices)
    2. Use branch and bound: at each step, decide which edges to include
    3. Prune if any Q3 is fully included
    
    But 120 edges is too many for naive B&B. Let me use a smarter approach.
    
    Alternative: Since Q3 is 3-regular on 8 vertices, a necessary condition for Q3
    to exist is that there are 8 vertices each with degree >= 3 in the subgraph induced
    by those 8 vertices.
    
    Let me implement a practical approach:
    1. Start with a good construction
    2. Greedily add edges, checking Q3-freeness after each addition
    3. Use local search to improve
    """
    pass

def _has_q3(adj_matrix):
    """Check if the graph with adjacency matrix contains a Q3 subgraph."""
    # Q3 vertices labeled 0-7, edges where labels differ in 1 bit
    q3_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            diff = i ^ j
            if diff in (1, 2, 4):  # Hamming distance 1
                q3_edges.append((i, j))
    
    # Try all mappings of Q3 vertices to graph vertices
    # Optimization: use backtracking
    from itertools import permutations
    
    # Precompute: for each pair of graph vertices, whether they're adjacent
    # adj_matrix[u][v] is True if edge exists
    
    # Backtracking: assign Q3 vertices 0..7 to distinct graph vertices
    # such that all Q3 edges map to graph edges
    
    assigned = [-1] * 8  # q3_vertex -> graph_vertex
    used = set()
    
    def backtrack(qv):
        if qv == 8:
            return True
        for gv in range(_N):
            if gv in used:
                continue
            # Check all edges from qv to already-assigned Q3 vertices
            ok = True
            for pv in range(qv):
                if pv ^ qv in (1, 2, 4):
                    if not adj_matrix[gv][assigned[pv]]:
                        ok = False
                        break
            if ok:
                assigned[qv] = gv
                used.add(gv)
                if backtrack(qv + 1):
                    return True
                used.remove(gv)
                assigned[qv] = -1
        return False
    
    return backtrack(0)

def _count_edges(edge_list):
    return len(edge_list)

def _is_q3_free(edge_list):
    """Check if the given edge list forms a Q3-free graph."""
    adj = [[False] * _N for _ in range(_N)]
    for u, v in edge_list:
        adj[u][v] = True
        adj[v][u] = True
    return not _has_q3(adj)

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Start with K3,13 construction (39 edges)
    # Part A = {0,1,2}, Part B = {3,...,15}
    edges = [[a, b] for a in range(3) for b in range(3, 16)]
    best_edges = list(edges)
    best_count = len(edges)
    
    # Try to add edges within Part B (the 13-vertex set)
    # Adding edges within B is safe if it doesn't create a Q3
    # A Q3 would need 4 vertices in A and 4 in B, but A only has 3 vertices,
    # so no Q3 can exist! Wait, that means we can add ALL edges within B.
    # K3,13 + K13 = 39 + 78 = 117 edges, and it's Q3-free!
    
    # Actually wait - Q3 is bipartite with parts of size 4 each.
    # In K3,13, the bipartition is (3, 13). Any Q3 subgraph would need
    # 4 vertices from one part and 4 from the other. But we only have 3
    # in part A, so no Q3 can exist regardless of edges within B.
    # 
    # Can we also add edges within A? K3,13 + K13 + K3 = 39 + 78 + 3 = 120 = K16!
    # But K16 definitely contains Q3. So adding edges within A might create Q3.
    #
    # Wait, I need to reconsider. The bipartition argument:
    # Q3 is bipartite with parts of size 4. If our graph has a bipartition
    # (A,B) with |A| < 4, then no Q3 can exist as a subgraph, because any Q3
    # subgraph would need 4 vertices in one part of its own bipartition, which
    # must map to one side of our graph's bipartition... 
    #
    # NO! That's wrong. A subgraph doesn't need to respect the host graph's
    # bipartition. The Q3 subgraph has its own bipartition. The issue is:
    # can we find 8 vertices in our graph that induce (contain) a Q3?
    #
    # In K3,13 + K13: Consider 4 vertices from B and 4 vertices from... 
    # we need 8 vertices total. From A we can take at most 3. So we take
    # 3 from A and 5 from B, or 2 from A and 6 from B, etc.
    # For a Q3, we need the 8 vertices to contain Q3 as a subgraph.
    # Q3 is bipartite (4,4). The 8 vertices in our graph don't need to be
    # split 4-4 between A and B.
    #
    # Example: Take vertices a1,a2,a3 from A and b1,b2,b3,b4 from B.
    # Edges: a1-b1, a1-b2, a2-b1, a2-b2, a3-b1, a3-b2 (from K3,13 part)
    # Plus b1-b2, b1-b3, b2-b3, b2-b4, b3-b4, b1-b4 (from K13 part)
    # This is a 7-vertex subgraph. Need 8 vertices for Q3.
    #
    # Take a1,a2,a3 from A and b1,b2,b3,b4 from B (7 vertices). 
    # We need 8 vertices, so add another b5 from B.
    # 
    # For Q3: need 8 vertices, 12 edges, each vertex degree 3 in the subgraph.
    # In our subgraph on {a1,a2,a3,b1,b2,b3,b4,b5}:
    # - a1 connected to b1,b2,b3,b4,b5 (degree 5 in subgraph)
    # - b1 connected to a1,a2,a3,b2,b3,b4,b5 (degree 7 in subgraph)
    # - etc.
    # 
    # For Q3 to be a subgraph, we need to find 12 edges forming Q3 within these.
    # Each vertex in Q3 has degree exactly 3. So we need to select a subset of
    # edges where each of the 8 vertices has degree exactly 3.
    #
    # This is getting complex. Let me just verify computationally.
    
    # Add all edges within B
    for i in range(3, 16):
        for j in range(i+1, 16):
            if time.time() > deadline:
                break
            edges.append([i, j])
    
    # Verify Q3-freeness
    if _is_q3_free(edges):
        best_edges = list(edges)
        best_count = len(edges)
    else:
        # Need to remove some edges. Fall back to K3,13
        best_edges = [[a, b] for a in range(3) for b in range(3, 16)]
        best_count = len(best_edges)
    
    # Try to add edges within A
    for i in range(3):
        for j in range(i+1, 3):
            if time.time() > deadline:
                break
            candidate = best_edges + [[i, j]]
            if _is_q3_free(candidate):
                best_edges = candidate
                best_count = len(best_edges)
    
    # Try to add cross edges that might have been missed
    # (K3,13 already has all cross edges, so nothing to add)
    
    # Local search: try to add edges between A and B that create cycles
    # Actually K3,13 already has ALL cross edges. So the only edges we can add
    # are within A (3 edges) and within B (78 edges).
    # We already tried all within B and all within A.
    
    # If K3,13+K13 is Q3-free, we have 39+78=117 edges.
    # If not, we need to be more careful.
    
    # Let's do a more careful search if we have time
    if time.time() < deadline - 10 and best_count < 117:
        # Try subsets of B-edges
        # Start with K3,13 and greedily add B-edges
        current = [[a, b] for a in range(3) for b in range(3, 16)]
        adj = [[False] * _N for _ in range(_N)]
        for u, v in current:
            adj[u][v] = True
            adj[v][u] = True
        
        b_edges = [(i, j) for i in range(3, 16) for j in range(i+1, 16)]
        
        for i, j in b_edges:
            if time.time() > deadline - 5:
                break
            if not adj[i][j]:
                adj[i][j] = True
                adj[j][i] = True
                current.append([i, j])
                # Check Q3-freeness
                if _has_q3(adj):
                    # Remove this edge
                    current.pop()
                    adj[i][j] = False
                    adj[j][i] = False
                else:
                    pass  # Keep the edge
        
        if len(current) > best_count:
            best_edges = current
            best_count = len(current)
    
    return best_edges
# EVOLVE-BLOCK-END
