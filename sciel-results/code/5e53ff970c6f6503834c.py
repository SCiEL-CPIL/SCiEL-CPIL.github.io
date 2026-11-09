"""K4-free graph on 18 vertices minimizing largest triangle-free induced subgraph."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    random.seed(42)
    
    n = 18
    
    def is_k4_free(adj):
        """Check if graph is K4-free using adjacency matrix."""
        for a in range(n):
            for b in range(a+1, n):
                if not adj[a][b]:
                    continue
                for c in range(b+1, n):
                    if not adj[a][c] or not adj[b][c]:
                        continue
                    for d in range(c+1, n):
                        if adj[a][d] and adj[b][d] and adj[c][d]:
                            return False
        return True
    
    def max_triangle_free(adj):
        """Find the maximum number of vertices inducing a triangle-free subgraph."""
        # Use branch and bound
        # A set S is triangle-free if no three vertices in S form a triangle
        # We want to find the maximum |S| such that S induces a triangle-free graph
        
        # Precompute: for each vertex, its neighbors
        neighbors = [set() for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if adj[i][j]:
                    neighbors[i].add(j)
        
        # A set S is triangle-free iff for every edge (u,v) in S, no w in S is adjacent to both u and v
        # Equivalently, the subgraph induced by S has no triangles
        
        # Branch and bound: try to include vertices one by one
        # State: (current_set, candidate_vertices, best_size)
        
        best = [0]
        
        def has_triangle_in_set(S):
            """Check if set S contains a triangle."""
            S_list = list(S)
            for i in range(len(S_list)):
                for j in range(i+1, len(S_list)):
                    u, v = S_list[i], S_list[j]
                    if not adj[u][v]:
                        continue
                    for k in range(j+1, len(S_list)):
                        w = S_list[k]
                        if adj[u][w] and adj[v][w]:
                            return True
            return False
        
        def max_tf_recursive(candidates, current_size, current_set):
            if time.time() > deadline:
                return
            if not candidates:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            # Upper bound: current_size + len(candidates)
            if current_size + len(candidates) <= best[0]:
                return
            
            # Try to pick a vertex from candidates
            # Heuristic: try vertices with high degree first (more likely to create triangles, so excluding them might help)
            # Or try vertices with low degree first (more likely to be triangle-free)
            
            # Sort candidates by degree in the induced subgraph on candidates
            # For efficiency, just iterate
            for idx, v in enumerate(candidates):
                # Try including v
                new_candidates = candidates[idx+1:]
                # Check if adding v creates a triangle with existing set
                # v creates a triangle if there exist u, w in current_set such that u-v and v-w and u-w are all edges
                # But actually we need to check if the NEW set (current_set ∪ {v}) is triangle-free
                # This is expensive. Instead, we can check incrementally.
                
                # Actually, let's use a different approach: 
                # A set S is triangle-free iff it does not contain any triangle from the graph.
                # So we can precompute all triangles in the graph, and check if S intersects any triangle in all 3 vertices.
                
                # For now, let's just do a direct check for small sets
                if len(current_set) <= 1:
                    # Can always add v
                    new_set = current_set | {v}
                    max_tf_recursive(new_candidates, current_size + 1, new_set)
                else:
                    # Check if adding v creates a triangle
                    creates_triangle = False
                    for u in current_set:
                        if adj[v][u]:
                            for w in current_set:
                                if w != u and adj[v][w] and adj[u][w]:
                                    creates_triangle = True
                                    break
                        if creates_triangle:
                            break
                    
                    if not creates_triangle:
                        new_set = current_set | {v}
                        max_tf_recursive(new_candidates, current_size + 1, new_set)
                
                # Try excluding v (implicitly by not recursing with v included)
                # The loop continues to next candidate
        
        all_vertices = set(range(n))
        max_tf_recursive(list(all_vertices), 0, set())
        return best[0]
    
    def score_graph(adj):
        """Compute the score (max triangle-free induced subgraph size)."""
        return max_triangle_free(adj)
    
    # Start with a random K4-free graph
    # Use a random 3-partite graph with parts of size 6
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    
    def random_k4_free():
        """Generate a random K4-free graph."""
        # Start with empty graph
        adj = [[False] * n for _ in range(n)]
        
        # Add edges between parts randomly
        # To ensure K4-free, we can use the fact that a complete 3-partite graph is K4-free
        # but we want smaller triangle-free induced subgraphs
        # Let's try a different approach: start with a random graph and remove edges to make it K4-free
        
        # Actually, let's use a structured approach:
        # Assign each vertex to one of 3 parts
        # Add edges between parts with some probability
        # Then check K4-freeness and remove edges if needed
        
        # Simpler: use a random bipartite-like structure
        # Or just start with the seed (complete 3-partite) and modify
        
        # Let's start with a random graph and iteratively remove edges to make it K4-free
        adj = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i][j] = adj[j][i] = True
        
        # Remove edges to make K4-free
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for a in range(n):
                for b in range(a+1, n):
                    if not adj[a][b]:
                        continue
                    for c in range(b+1, n):
                        if not adj[a][c] or not adj[b][c]:
                            continue
                        for d in range(c+1, n):
                            if adj[a][d] and adj[b][d] and adj[c][d]:
                                # Found K4, remove an edge
                                # Remove the edge with highest "cost" (most triangles it's in)
                                # For now, just remove (a,d)
                                adj[a][d] = adj[d][a] = False
                                changed = True
                                break
                        if changed:
                            break
                    if changed:
                        break
                if changed:
                    break
        
        return adj
    
    # Local search: flip edges to reduce the score
    def local_search(adj, max_iters=1000):
        """Perform local search to minimize the score."""
        current_score = score_graph(adj)
        best_adj = [row[:] for row in adj]
        best_score = current_score
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try flipping a random edge
            i, j = random.sample(range(n), 2)
            if i > j:
                i, j = j, i
            
            # Flip edge
            adj[i][j] = adj[j][i] = not adj[i][j]
            
            # Check K4-free
            if not is_k4_free(adj):
                # Revert
                adj[i][j] = adj[j][i] = not adj[i][j]
                continue
            
            new_score = score_graph(adj)
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = [row[:] for row in adj]
            else:
                # Revert
                adj[i][j] = adj[j][i] = not adj[i][j]
        
        return best_adj, best_score
    
    # Try multiple random starts
    best_adj = None
    best_score = n  # worst case
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        adj = random_k4_free()
        if not is_k4_free(adj):
            continue
        
        adj, score = local_search(adj, max_iters=200)
        if score < best_score:
            best_score = score
            best_adj = [row[:] for row in adj]
        
        # If we've found a good score, try to improve
        if best_score <= 10:
            # Do more local search on the best solution
            adj2, score2 = local_search(best_adj, max_iters=500)
            if score2 < best_score:
                best_score = score2
                best_adj = [row[:] for row in adj2]
    
    # Convert adjacency matrix to edge list
    if best_adj is None:
        # Fallback: return the seed solution
        parts = [range(0, 6), range(6, 12), range(12, 18)]
        return [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                edges.append([i, j])
    
    return edges

# EVOLVE-BLOCK-END
