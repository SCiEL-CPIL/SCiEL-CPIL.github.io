import time
import random
from itertools import combinations

def _build_adj(edges, n):
    """Build adjacency matrix from edge list."""
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    return adj

def _is_k4_free(adj, n):
    """Check if graph (given by adjacency matrix) is K4-free."""
    for i in range(n):
        for j in range(i + 1, n):
            if not adj[i][j]:
                continue
            for k in range(j + 1, n):
                if not adj[i][k] or not adj[j][k]:
                    continue
                for l in range(k + 1, n):
                    if adj[i][l] and adj[j][l] and adj[k][l]:
                        return False
    return True

def _max_triangle_free_induced(adj, n):
    """
    Find the size of the largest triangle-free induced subgraph.
    Uses branch and bound: we try to find a subset of vertices of size k
    that is triangle-free. We binary search on k or just try from high to low.
    """
    # Precompute triangles
    # A set is triangle-free if it contains no triangle.
    # We want to find the largest subset with no triangles.
    
    # This is equivalent to: find the maximum size of an independent set
    # in the "triangle hypergraph"? No, that's not right.
    # Triangle-free induced subgraph = subset of vertices containing no K3.
    
    # Let's use a simple branch and bound on subsets.
    # Since n=18, we can iterate over all subsets? 2^18 = 262144.
    # For each subset, check if it's triangle-free.
    # Checking if a subset is triangle-free:
    #   Iterate over all triples in the subset.
    #   If any triple forms a triangle, reject.
    
    # Optimization:
    # If a subset S is triangle-free, any subset of S is triangle-free.
    # So we can do a DFS/BFS.
    
    # Actually, 2^18 is small enough to just brute force all subsets if we optimize the check.
    # But checking each subset takes O(|S|^3) or O(E_S).
    # Total work could be high.
    
    # Better: Branch and Bound.
    # Start with all vertices.
    # If the current set has a triangle, we must remove at least one vertex from each triangle.
    # This is the "Hitting Set" problem for triangles, but we want to MINIMIZE removals, i.e., MAXIMIZE remaining.
    
    # Let's implement a recursive search that tries to keep vertices.
    # State: (current_set, candidate_vertices)
    # We try to add vertices one by one.
    # If adding a vertex creates a triangle with two existing vertices, we can't add it (or we must remove one of the existing ones).
    
    # Simpler approach for n=18:
    # Iterate over all subsets in decreasing order of size?
    # Or use the fact that the complement of a triangle-free set is a vertex cover for the triangles.
    
    # Let's try a heuristic first to get a good upper bound, then verify.
    # Greedy: Start with all vertices. While there is a triangle, remove the vertex with highest degree in the current triangle subgraph? Or just any vertex in a triangle.
    
    # Let's do a proper Branch and Bound.
    # We want to find the largest subset S such that no triangle is fully contained in S.
    # This is the same as finding the maximum size of a set S that is an independent set in the 3-uniform hypergraph of triangles.
    
    # Algorithm:
    # 1. Identify all triangles in G.
    # 2. We want to select a maximum number of vertices such that no triangle is fully selected.
    #    This is equivalent to: Minimum Vertex Cover in the hypergraph of triangles? No.
    #    It is the Maximum Independent Set in the hypergraph where hyperedges are triangles.
    
    # Since n=18, we can use a bitset representation.
    # vertices: 0..17
    # Triangles: list of bitmasks.
    
    triangles = []
    for i in range(n):
        for j in range(i + 1, n):
            if not adj[i][j]: continue
            for k in range(j + 1, n):
                if adj[i][k] and adj[j][k]:
                    triangles.append((1 << i) | (1 << j) | (1 << k))
    
    if not triangles:
        return n
    
    # Now, find max subset of {0..17} that does not contain any mask from `triangles`.
    # Let S be the set of selected vertices (bitmask).
    # Condition: For all t in triangles, (S & t) != t.
    
    # We can iterate over all possible S? 2^18 = 262144.
    # For each S, check if it's valid.
    # Check: For each triangle t, if (S & t) == t, invalid.
    # Number of triangles can be up to C(18,3) = 816.
    # 262144 * 816 ~ 200 million operations. Might be too slow in Python if not optimized.
    
    # Optimization:
    # Precompute for each vertex v, the list of triangles containing v.
    # Or, simpler:
    # Iterate S from (2^n - 1) down to 0?
    # We want the largest S.
    # We can iterate by size?
    
    # Let's try iterating all masks from (1<<18)-1 down to 0.
    # Stop at the first valid one.
    # This is worst case 2^18 checks.
    # Each check: loop through triangles.
    
    # We can speed up the check by breaking early.
    
    # Also, we can prune: if a mask S is invalid, any superset of S is invalid.
    # So if we find an invalid mask, we don't need to check its supersets.
    # But iterating from high to low naturally handles this?
    # If we iterate from (2^n-1) down to 0, and we find a valid one, it's the max.
    # If we find an invalid one, we just continue to the next lower number.
    # This doesn't explicitly prune supersets because we are going down.
    # But we know that if S is invalid, all S' > S that are supersets of S are invalid.
    # We can't easily skip them in a simple loop.
    
    # However, 262k iterations is very fast in Python if the inner loop is tight.
    # Let's try it.
    
    max_size = 0
    # To be safe and fast, we can just check all masks.
    # But we want the MAX size.
    # So we can iterate masks in descending order of bit count?
    # Or just iterate all masks, keep track of max bit_count that is valid.
    
    # Actually, iterating from (2^n - 1) down to 0 and returning the first valid one is correct and potentially faster if the answer is close to n.
    # If the answer is small, it's slow.
    
    # Let's assume the answer is not too small.
    
    for mask in range((1 << n) - 1, -1, -1):
        # Check if mask is triangle-free
        valid = True
        for t in triangles:
            if (mask & t) == t:
                valid = False
                break
        if valid:
            return bin(mask).count('1')
            
    return 0

def _get_paley_edges(q):
    """Get edges for Paley graph of order q."""
    residues = set()
    for x in range(1, q):
        residues.add((x * x) % q)
    
    edges = []
    for u in range(q):
        for v in range(u + 1, q):
            diff = (v - u) % q
            if diff in residues:
                edges.append([u, v])
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = 18
    
    # Candidate 1: Paley graph P(17) + isolated vertex
    edges_17 = _get_paley_edges(17)
    adj_17 = _build_adj(edges_17, n)
    if _is_k4_free(adj_17, n):
        score_17 = _max_triangle_free_induced(adj_17, n)
        # print(f"Paley 17 score: {score_17}")
        if time.time() > deadline:
            return edges_17
        # Store best
        best_edges = edges_17
        best_score = score_17
    else:
        best_edges = None
        best_score = float('inf')
        
    # Candidate 2: Two disjoint Paley graphs P(9)
    edges_9_1 = _get_paley_edges(9)
    edges_9_2 = [[u + 9, v + 9] for u, v in edges_9_1]
    edges_18 = edges_9_1 + edges_9_2
    adj_18 = _build_adj(edges_18, n)
    if _is_k4_free(adj_18, n):
        score_18 = _max_triangle_free_induced(adj_18, n)
        # print(f"2x Paley 9 score: {score_18}")
        if score_18 < best_score:
            best_edges = edges_18
            best_score = score_18
            
    # Candidate 3: K6,6,6 (Baseline)
    parts = [range(0, 6), range(6, 12), range(12, 18)]
    edges_k666 = [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]
    adj_k666 = _build_adj(edges_k666, n)
    if _is_k4_free(adj_k666, n):
        score_k666 = _max_triangle_free_induced(adj_k666, n)
        # print(f"K666 score: {score_k666}")
        if score_k666 < best_score:
            best_edges = edges_k666
            best_score = score_k666
            
    # Candidate 4: Randomized Local Search
    # Start with a random K4-free graph or a modified Paley graph.
    # Try to minimize the max triangle-free induced subgraph.
    
    # We need a way to generate random K4-free graphs.
    # Or start with one of the above and perturb it.
    
    # Let's try to improve the best candidate found so far.
    if best_edges is not None:
        current_edges = best_edges[:]
        current_adj = _build_adj(current_edges, n)
        current_score = best_score
        
        # Local search: flip edges (add or remove) if it keeps K4-free and improves score.
        # Score is "largest triangle-free induced subgraph". We want to minimize this.
        # Adding an edge can only decrease or keep the same the size of the largest triangle-free induced subgraph?
        # No. Adding an edge might create a K4 (invalid).
        # If we add an edge, we might create new triangles, which might reduce the size of triangle-free sets.
        # Removing an edge might allow larger triangle-free sets? No, removing edges makes it easier to be triangle-free.
        # So removing edges is bad (increases score). Adding edges is good (decreases score), provided no K4 is formed.
        
        # So we should try to ADD edges.
        
        # List of non-edges
        non_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if not current_adj[u][v]:
                    non_edges.append((u, v))
                    
        random.shuffle(non_edges)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for u, v in non_edges:
                if time.time() > deadline:
                    break
                # Try adding edge (u, v)
                # Check if it creates a K4
                # A K4 containing (u, v) must be {u, v, x, y} where x, y are common neighbors of u and v.
                # And x, y must be connected.
                
                # Find common neighbors
                common = []
                for w in range(n):
                    if w != u and w != v and current_adj[u][w] and current_adj[v][w]:
                        common.append(w)
                        
                # Check if any pair in common forms a K4 with u, v
                creates_k4 = False
                for i in range(len(common)):
                    for j in range(i + 1, len(common)):
                        if current_adj[common[i]][common[j]]:
                            creates_k4 = True
                            break
                    if creates_k4:
                        break
                            
                if not creates_k4:
                    # Add edge
                    current_adj[u][v] = True
                    current_adj[v][u] = True
                    current_edges.append([u, v])
                    
                    # Check new score
                    new_score = _max_triangle_free_induced(current_adj, n)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        # Remove from non_edges? It's now an edge.
                        # We can just break and re-shuffle/re-generate non_edges?
                        # Or mark it.
                        # For simplicity, break and restart loop with updated non_edges.
                        non_edges = []
                        for uu in range(n):
                            for vv in range(uu + 1, n):
                                if not current_adj[uu][vv]:
                                    non_edges.append((uu, vv))
                        random.shuffle(non_edges)
                        break
                    else:
                        # Revert
                        current_adj[u][v] = False
                        current_adj[v][u] = False
                        current_edges.pop()
                        
        if current_score < best_score:
            best_edges = current_edges
            best_score = current_score
            
    if best_edges is None:
        # Fallback
        return edges_k666
        
    return best_edges
