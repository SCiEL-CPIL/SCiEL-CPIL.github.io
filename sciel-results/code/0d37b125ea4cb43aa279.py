import random
import time
import itertools

def _build_line_graph_edges(H_edges, n_left, n_right):
    """Given edges of a bipartite graph (as list of (l, r) pairs), return edges of the line graph."""
    # Each edge in H is a vertex in L(H). Vertex i in L(H) corresponds to H_edges[i].
    n = len(H_edges)
    # Map vertex in H to list of edge indices
    left_adj = {}
    right_adj = {}
    for idx, (l, r) in enumerate(H_edges):
        left_adj.setdefault(l, []).append(idx)
        right_adj.setdefault(r, []).append(idx)
    
    L_edges = set()
    # For each left vertex, all pairs of edges at it form a clique in L(H)
    for l, edge_indices in left_adj.items():
        for i in range(len(edge_indices)):
            for j in range(i+1, len(edge_indices)):
                a, b = edge_indices[i], edge_indices[j]
                L_edges.add((min(a,b), max(a,b)))
    # For each right vertex
    for r, edge_indices in right_adj.items():
        for i in range(len(edge_indices)):
            for j in range(i+1, len(edge_indices)):
                a, b = edge_indices[i], edge_indices[j]
                L_edges.add((min(a,b), max(a,b)))
    return list(L_edges)

def _max_triangle_free_induced(G_edges, n=18):
    """Find the largest subset of vertices inducing a triangle-free subgraph."""
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in G_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find all triangles
    triangles = []
    for i in range(n):
        for j in adj[i]:
            if j <= i:
                continue
            for k in adj[j]:
                if k > j and k in adj[i]:
                    triangles.append((i, j, k))
    
    # For each triangle, at least one vertex must be excluded.
    # Use branch and bound: try to find largest triangle-free induced subgraph.
    # Equivalent to: minimum vertex cover of the hypergraph of triangles.
    # Score = 18 - min vertex cover size.
    
    if not triangles:
        return n
    
    # Branch and bound on triangles
    best_cover = [n]  # worst case
    
    def solve(triangles_remaining, covered, depth):
        if not triangles_remaining:
            if len(covered) < best_cover[0]:
                best_cover[0] = len(covered)
            return
        # Lower bound: need at least 1 more
        if len(covered) + 1 >= best_cover[0]:
            return
        # Pick triangle with fewest uncovered vertices... actually pick first
        tri = triangles_remaining[0]
        rest = triangles_remaining[1:]
        # Try removing each vertex of the triangle
        for v in tri:
            new_covered = list(covered)
            new_covered.append(v)
            # Filter triangles that still need covering
            new_tris = [t for t in rest if v not in t]
            solve(new_tris, new_covered, depth+1)
    
    solve(triangles, [], 0)
    return n - best_cover[0]

def _is_k4_free(G_edges, n=18):
    """Check if graph is K4-free."""
    adj = [set() for _ in range(n)]
    for u, v in G_edges:
        adj[u].add(v)
        adj[v].add(u)
    for i in range(n):
        for j in adj[i]:
            if j <= i:
                continue
            for k in adj[j]:
                if k > j and k in adj[i]:
                    for m in adj[k]:
                        if m > k and m in adj[i] and m in adj[j]:
                            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    random.seed(42)
    
    # Start with two disjoint K_{3,3}
    # H has 12 vertices: left 0-5, right 6-11
    # Edges of two K_{3,3}
    H_edges = []
    for copy in range(2):
        base_l = copy * 3
        base_r = 6 + copy * 3
        for l in range(3):
            for r in range(3):
                H_edges.append((base_l + l, base_r + r))
    # 18 edges total
    
    # Compute score
    G_edges = _build_line_graph_edges(H_edges, 6, 6)
    current_score = _max_triangle_free_induced(G_edges)
    
    best_score = current_score
    best_H = list(H_edges)
    
    # Simulated annealing: swap edges in H
    # H must have max degree <= 3 to keep L(H) K4-free
    # H is bipartite with 12 vertices (6 left, 6 right), 18 edges
    # 3-regular bipartite on 6+6 = K_{3,3} ∪ K_{3,3} is one option
    # But we can have other 3-regular bipartite graphs on 6+6
    
    # Actually, let's generalize: allow H to have up to 12 left vertices and 12 right vertices
    # but keep total edges = 18, max degree <= 3
    
    # Let's try a different approach: use a 3-regular bipartite graph on 6+6
    # There are several such graphs. Let's enumerate or randomize.
    
    # For a 3-regular bipartite graph on 6+6, the adjacency matrix is 6x6 with row/col sums = 3
    # Let's randomize by permuting edges
    
    def get_score_from_adj(adj_matrix):
        """adj_matrix is 6x6 with 0/1 entries, row/col sums = 3"""
        H_edges_local = []
        for i in range(6):
            for j in range(6):
                if adj_matrix[i][j]:
                    H_edges_local.append((i, 6+j))
        if len(H_edges_local) != 18:
            return 999
        G_edges_local = _build_line_graph_edges(H_edges_local, 6, 6)
        if not _is_k4_free(G_edges_local):
            return 999
        return _max_triangle_free_induced(G_edges_local)
    
    # Initial adj: two blocks of 3x3 all-ones
    adj = [[0]*6 for _ in range(6)]
    for i in range(3):
        for j in range(3):
            adj[i][j] = 1
            adj[i+3][j+3] = 1
    
    current_adj = [row[:] for row in adj]
    current_score = get_score_from_adj(current_adj)
    best_adj = [row[:] for row in current_adj]
    best_score_adj = current_score
    
    # Try all permutations of columns within the two groups, and also mixed structures
    # Actually, let's do a local search: swap edges
    
    def valid_adj(adj_m):
        for i in range(6):
            if sum(adj_m[i]) != 3:
                return False
            if sum(adj_m[i][j] for i in range(6)) != 3:
                return False
        return True
    
    # Random search: generate random 3-regular bipartite graphs on 6+6
    best_found = best_score_adj
    best_adj_found = [row[:] for row in best_adj]
    
    num_trials = 0
    while time.time() < deadline:
        num_trials += 1
        # Generate random 3-regular bipartite graph
        # Use configuration model or random permutations
        # Method: assign each left vertex 3 right vertices, ensuring each right gets 3
        # Simple: random 3-regular bipartite via random permutation decomposition
        # 3-regular bipartite = sum of 3 perfect matchings
        
        perms = []
        for _ in range(3):
            p = list(range(6))
            random.shuffle(p)
            perms.append(p)
        
        adj_m = [[0]*6 for _ in range(6)]
        for p in perms:
            for i in range(6):
                adj_m[i][p[i]] = 1
        
        if not valid_adj(adj_m):
            continue
        
        score = get_score_from_adj(adj_m)
        if score < best_found:
            best_found = score
            best_adj_found = [row[:] for row in adj_m]
            print(f"New best: {score} at trial {num_trials}")
    
    # Use the best found
    H_edges_final = []
    for i in range(6):
        for j in range(6):
            if best_adj_found[i][j]:
                H_edges_final.append((i, 6+j))
    
    G_edges_final = _build_line_graph_edges(H_edges_final, 6, 6)
    final_score = _max_triangle_free_induced(G_edges_final)
    
    # Also try the original K_{6,6,6} as fallback
    parts = [range(0, 6), range(6, 12), range(12, 18)]
    fallback_edges = [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]
    fallback_score = _max_triangle_free_induced(fallback_edges)
    
    if final_score <= fallback_score:
        return G_edges_final
    else:
        return fallback_edges
