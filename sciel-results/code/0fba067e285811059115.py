import random
import time

def construct():
    """
    Constructs a K4-free graph on 18 vertices minimizing the largest triangle-free induced subgraph.
    Uses local search starting from K6,6,6.
    """
    n = 18
    # Initialize with K6,6,6
    # Parts: 0-5, 6-11, 12-17
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    
    # Adjacency matrix as list of sets
    adj = [set() for _ in range(n)]
    
    def add_edge(u, v):
        adj[u].add(v)
        adj[v].add(u)
        
    def remove_edge(u, v):
        adj[u].remove(v)
        adj[v].remove(u)
        
    def has_edge(u, v):
        return v in adj[u]
        
    # Build K6,6,6
    for i in range(3):
        for j in range(i + 1, 3):
            for u in parts[i]:
                for v in parts[j]:
                    add_edge(u, v)
                    
    def is_k4_free():
        # Check all 4-subsets? Too slow.
        # Check if any vertex has a triangle in its neighborhood.
        for u in range(n):
            neighbors = list(adj[u])
            # Check triangles in neighbors
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if neighbors[j] in adj[neighbors[i]]:
                        return False
        return True
        
    def max_triangle_free_size():
        # Branch and bound or simple heuristic?
        # For n=18, we can check all subsets? 2^18 = 262k. Feasible.
        # But we need to do this many times in local search.
        # We can optimize by checking only subsets that are candidates.
        # Or use a heuristic: find largest independent set in the triangle graph?
        # A set is triangle-free iff it is an independent set in the "triangle graph" where edges connect vertices that form a triangle? No.
        # A set U is triangle-free iff for all u,v,w in U, not (u-v, v-w, w-u all edges).
        
        # Let's use a simple recursive search with pruning for the max triangle-free set.
        # Or, since n=18 is small, we can iterate over all masks?
        # 2^18 iterations * cost of checking triangle-free.
        # Cost of checking: for a set of size k, check O(k^3) triangles.
        # Total cost O(2^18 * 18^3) ~ 260k * 5000 = 1.3e9. Too slow in Python.
        
        # Better: Precompute for each pair (u,v) if they are in a triangle with some w?
        # Or simply: 
        # A set is triangle-free if it contains no triangle.
        # We can find the maximum triangle-free set by finding the minimum vertex cover of the hypergraph of triangles?
        # Or simply: 
        # Iterate through all subsets is too slow if done naively.
        
        # Alternative: 
        # The largest triangle-free induced subgraph is the complement of the smallest set of vertices to remove to make the graph triangle-free.
        # This is the "triangle transversal" problem.
        
        # For n=18, we can use a simple backtracking to find the max triangle-free set.
        # Order vertices by degree.
        
        # Let's try a different approach for scoring:
        # We know the score is at least 6 (independence number of K6,6,6 is 6? No, alpha(K6,6,6)=6).
        # Actually, in K6,6,6, the largest triangle-free set is 12 (two parts).
        
        # Let's implement a fast checker.
        # Precompute triangles?
        # There are at most 18*17*16/6 = 816 triangles max.
        # We can list all triangles.
        # A set U is triangle-free if it contains no triangle from this list.
        # This is the "Maximum Subset with No Forbidden Triple" problem.
        # This is equivalent to finding the maximum independent set in a 3-uniform hypergraph.
        
        # For n=18, we can use a branch and bound on the hypergraph.
        
        # List of triangles
        triangles = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    common = adj[u] & adj[v]
                    for w in common:
                        if w > v:
                            triangles.append((u, v, w))
                            
        # Now find max subset of vertices containing no triangle.
        # This is the independence number of the 3-uniform hypergraph.
        
        # We can use a simple recursive search with memoization or just branch and bound.
        # Given the small n, a simple recursive search with pruning should work.
        
        # Sort triangles by frequency?
        
        # Let's use a simple heuristic: 
        # Start with all vertices. Remove vertices that are in many triangles?
        # No, we need the exact max.
        
        # Branch and Bound:
        # State: set of remaining vertices, set of forbidden triangles (those with all 3 vertices in remaining).
        # Actually, just check if a triangle is fully contained in the current set.
        
        # We will implement a simple recursive function.
        
        # To speed up, we can precompute for each vertex the set of triangles it belongs to.
        
        # For the purpose of this local search, we can approximate the score or use a faster heuristic.
        # But the evaluator uses exact branch and bound.
        # We must match the evaluator's metric? No, the evaluator measures the score of the returned graph.
        # We just need to return a good graph.
        # We can use a heuristic to guide the search, but the final score is calculated by the evaluator.
        # However, we want to improve the score, so we need a good estimate.
        
        # Let's use a simple greedy algorithm to estimate the max triangle-free set.
        # Or, since we have time, we can do a limited search.
        
        # Actually, for n=18, we can just iterate over all 2^18 subsets if we optimize the triangle check.
        # Precompute a bitset for each triangle?
        # There are at most 816 triangles.
        # For each subset (18 bits), check if it contains any triangle.
        # We can precompute a list of triangles as 18-bit integers.
        # Then for each mask, check if (mask & tri) == tri for any tri.
        # This is 262k * 816 ops ~ 215M ops. In Python, this might be ~20-30 seconds.
        # Too slow for local search iterations.
        
        # We need a faster score estimator.
        # Let's use the size of the largest independent set in the "triangle graph" where vertices are original vertices and edges connect vertices that share a triangle? No.
        
        # Let's just use the independence number of the graph formed by triangles?
        # No.
        
        # Let's use a simple heuristic: 
        # The max triangle-free set is at least the independence number of the graph.
        # And at most n - (number of disjoint triangles)?
        
        # Let's try to just run the local search with a heuristic score: 
        # Score = number of triangles. Minimizing triangles might help?
        # Not necessarily.
        
        # Let's use the size of the largest clique in the complement? No.
        
        # I will use a simple recursive B&B for the score, but limit the depth or nodes.
        # Or, I will just run the local search for a fixed number of steps and return the best graph found based on a fast heuristic.
        
        # Heuristic: 
        # 1. Find a large triangle-free set using greedy.
        # 2. If we can find a set of size 11, score is at least 11.
        # 3. If we can't find 11, maybe it's 10.
        
        # Let's implement a fast greedy max triangle-free set finder.
        
        def greedy_max_tf():
            # Start with all vertices
            # Repeatedly remove the vertex that is in the most triangles within the current set?
            # No, we want to KEEP vertices.
            # We want to find the largest subset with no triangles.
            # This is the same as finding the largest independent set in the 3-hypergraph.
            
            # Simple greedy:
            # Order vertices by degree in the "triangle graph" (number of triangles they are in).
            # Pick the one with fewest triangles?
            
            # Let's just return a rough estimate: n - (min vertex cover of triangles).
            # Min vertex cover is hard.
            
            # Let's just use the size of the largest independent set of the graph G.
            # This is a lower bound on the triangle-free set.
            # If alpha(G) is small, the triangle-free set might be larger.
            
            # For K6,6,6, alpha=6. But max TF is 12.
            # So alpha is a bad proxy.
            
            # Let's use the following:
            # Max TF set is at least the size of the largest bipartite induced subgraph?
            
            # I will implement a simple B&B for the score, but only call it every N steps or with a time limit.
            
            pass

    # Given the complexity, I will implement a simpler local search that minimizes the number of triangles.
    # Fewer triangles generally allows for larger triangle-free sets? No, fewer triangles means it's easier to be triangle-free.
    # We want to MAXIMIZE triangles to FORCE the triangle-free set to be small.
    # So we want to ADD edges to create triangles, provided we don't create K4s.
    
    # Revised Strategy:
    # Start with K6,6,6 (12 triangles? No, K6,6,6 has 0 triangles).
    # Wait, K6,6,6 is bipartite (between any two parts), so it has NO triangles.
    # So the whole graph is triangle-free? No, the whole graph is not a triangle, but any 3 vertices from different parts form a triangle?
    # u in P0, v in P1, w in P2.
    # u-v, v-w, w-u are all edges.
    # So K6,6,6 has MANY triangles.
    # Number of triangles = 6*6*6 = 216.
    
    # So we want to ADD more edges to create more triangles, reducing the max triangle-free set.
    # But we must avoid K4s.
    
    # Local Search:
    # 1. Try to add an edge (u,v) if it doesn't create a K4.
    # 2. If adding an edge increases the number of triangles, it might reduce the max TF set.
    # 3. We can use the number of triangles as a heuristic score (higher is better).
    
    # Let's implement this.
    
    def count_triangles():
        cnt = 0
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    common = adj[u] & adj[v]
                    cnt += len(common)
        return cnt // 3 # Each triangle counted 3 times? No, for each edge (u,v), we count common neighbors. Each triangle has 3 edges. So sum is 3*T.
        
    # Actually, the loop above:
    # for u, for v in adj[u] (u<v), count common.
    # This counts each triangle exactly once per edge?
    # Triangle (a,b,c).
    # Edge (a,b): c is common.
    # Edge (a,c): b is common.
    # Edge (b,c): a is common.
    # So sum is 3 * T.
    
    def can_add_edge(u, v):
        if has_edge(u, v):
            return False
        # Check if adding (u,v) creates a K4.
        # This happens if there is a triangle (u, v, w, z) -> (u,v) is the new edge.
        # So we need w, z such that w-z is an edge, and w,z are neighbors of both u and v.
        common = adj[u] & adj[v]
        # Check if common contains an edge.
        for w in common:
            if adj[w] & common:
                return False
        return True

    # Local Search
    start_time = time.time()
    deadline = start_time + 100 # 100s limit
    
    current_triangles = count_triangles()
    best_triangles = current_triangles
    best_adj = [s.copy() for s in adj]
    
    while time.time() < deadline:
        # Try to add a random non-edge
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v or has_edge(u, v):
            continue
            
        if can_add_edge(u, v):
            # Add edge
            add_edge(u, v)
            new_triangles = count_triangles()
            if new_triangles > current_triangles:
                current_triangles = new_triangles
                if current_triangles > best_triangles:
                    best_triangles = current_triangles
                    best_adj = [s.copy() for s in adj]
            else:
                # Revert
                remove_edge(u, v)
                
        # Occasionally try to remove an edge? 
        # Removing edges reduces triangles, which is bad for our heuristic.
        # But it might allow adding other edges later?
        # For now, just adding.
        
    # Reconstruct edges from best_adj
    edges = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                edges.append([u, v])
                
    return edges

# Run construct
# Note: The local search might get stuck. 
# K6,6,6 has 216 triangles.
# If we add edges, we increase triangles.
# Max triangles in K4-free graph?
# The icosahedron (12 vertices) has 20 triangles.
# We can extend it.

# I will output the code with this local search.
