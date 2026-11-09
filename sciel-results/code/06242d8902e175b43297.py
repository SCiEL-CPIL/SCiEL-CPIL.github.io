import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n = 16
    
    # Adjacency matrix (symmetric, 0-indexed)
    adj = np.zeros((n, n), dtype=np.int8)
    
    # Start with K_{3,13}: U={0,1,2}, V={3,...,15}
    for u in range(3):
        for v in range(3, n):
            adj[u, v] = 1
            adj[v, u] = 1
    
    edge_count = 3 * 13  # 39 edges
    
    def check_q3(vertices):
        """Check if the induced subgraph on 8 vertices contains Q3 as a subgraph.
        Q3 is a 3-regular bipartite graph on 8 vertices (4+4).
        We need to find a bipartition (A,B) of the 8 vertices into two sets of 4
        such that there is a perfect matching of MISSING edges between A and B.
        Equivalently, the present edges between A and B must be 3-regular on each side.
        """
        # vertices: tuple of 8 ints
        # Build local adjacency: 8x8 matrix
        local_adj = np.zeros((8, 8), dtype=np.int8)
        for i in range(8):
            for j in range(i+1, 8):
                if adj[vertices[i], vertices[j]]:
                    local_adj[i, j] = 1
                    local_adj[j, i] = 1
        
        # Try all ways to split 8 vertices into two groups of 4
        # There are C(8,4)/2 = 35 distinct bipartitions (since A,B and B,A are same)
        # But we need to check both orientations? No, Q3 is bipartite so the bipartition is determined.
        # Actually we just need: does there exist a 4-vertex subset A such that the bipartite
        # graph between A and B=V\A is 3-regular (each vertex has degree 3 in the bipartite subgraph)?
        
        from itertools import combinations
        
        for A in combinations(range(8), 4):
            A_set = set(A)
            B_set = set(range(8)) - A_set
            B_list = list(B_set)
            
            # Check if each vertex in A has exactly 3 neighbors in B
            ok = True
            for i in A:
                deg = 0
                for j in B_list:
                    if local_adj[i, j]:
                        deg += 1
                if deg != 3:
                    ok = False
                    break
            if not ok:
                continue
            
            # Check if each vertex in B has exactly 3 neighbors in A
            for j in B_list:
                deg = 0
                for i in A:
                    if local_adj[i, j]:
                        deg += 1
                if deg != 3:
                    ok = False
                    break
            if ok:
                return True
        return False
    
    # Greedy edge addition with local Q3 check
    # For each candidate edge, only check 8-vertex sets that include both endpoints
    # There are C(14,6) = 3003 such sets per edge candidate
    
    from itertools import combinations
    
    all_pairs = list(combinations(range(n), 2))
    
    # Order candidates: try edges that create denser local structures first
    # Prioritize edges within V (high-degree side) and within U
    def edge_priority(e):
        u, v = e
        # Prefer edges where both endpoints have high degree
        du = adj[u].sum()
        dv = adj[v].sum()
        return -(du + dv)
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Get current non-edges
        non_edges = [e for e in all_pairs if adj[e[0], e[1]] == 0]
        non_edges.sort(key=edge_priority)
        
        for u, v in non_edges:
            if time.time() > deadline:
                break
            
            # Temporarily add edge
            adj[u, v] = 1
            adj[v, u] = 1
            
            # Check all 8-vertex subsets containing u and v
            remaining = [x for x in range(n) if x != u and x != v]
            safe = True
            for combo in combinations(remaining, 6):
                verts = tuple([u, v] + list(combo))
                if check_q3(verts):
                    safe = False
                    break
            
            if safe:
                edge_count += 1
                changed = True
            else:
                # Remove edge
                adj[u, v] = 0
                adj[v, u] = 0
    
    # Build result
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j]:
                edges.append([i, j])
    
    return edges
