"""
Find the maximum number of edges in a triangle-free, non-3-colorable graph on 16 vertices.

Strategy:
1. We model the problem as finding a set of edges E such that:
   a) The graph (V, E) is triangle-free.
   b) The graph (V, E) is not 3-colorable.
   c) |E| is maximized.

2. Triangle-free constraint: If (u,v) and (v,w) are in E, then (u,w) cannot be in E.
   This is equivalent to saying that the set of edges E is an independent set in the
   line graph L(K_16). However, L(K_16) has C(16,2) = 120 vertices, which is large for
   a direct maximum independent set search.

3. Alternative perspective:
   A triangle-free graph is a subgraph of a bipartite graph? No, triangle-free graphs
   can have odd cycles (e.g., C5).
   
   Key observation: A graph is triangle-free if and only if its edges can be partitioned
   into a matching? No.
   
   Let's use the property: For any vertex v, its neighbors N(v) must form an independent set.
   So for each v, the edges among N(v) must be empty.

4. Search approach:
   We want to maximize |E|. The maximum possible edges in a triangle-free graph on n vertices
   is floor(n^2/4) (Mantel's theorem). For n=16, this is 64.
   However, we also require the graph to be NOT 3-colorable.
   
   Bipartite graphs are 2-colorable, hence 3-colorable. So we need a non-bipartite triangle-free
   graph. The smallest such graph is C5 (5 edges).
   
   We can start with a maximal triangle-free graph (like the complete bipartite graph K_{8,8})
   and check if it's 3-colorable. K_{8,8} is bipartite, so it is 3-colorable.
   We need to modify it to break 3-colorability while maintaining triangle-freeness.
   
   Actually, we want the MAXIMUM number of edges. So we should search for the densest
   triangle-free graph that is not 3-colorable.
   
   Let's use a backtracking search over edges, but with strong pruning.
   Order edges in a fixed order. For each edge, decide include or exclude.
   Pruning:
   - If adding an edge creates a triangle, prune.
   - If the current partial graph is already 3-colorable AND we cannot add more edges
     that would make it non-3-colorable... this is hard to check incrementally.
   
   Better approach:
   We know that a triangle-free graph is 3-colorable if it is bipartite.
   But there are non-bipartite triangle-free graphs that are 3-colorable (e.g., C7).
   And there are non-bipartite triangle-free graphs that are NOT 3-colorable (e.g., the
   Grötzsch graph, which is triangle-free and 4-chromatic).
   
   The Grötzsch graph has 11 vertices and 20 edges. We can embed it in 16 vertices
   and add more edges as long as we maintain triangle-freeness and non-3-colorability.
   
   However, we want the MAXIMUM edges. So we should search for the densest such graph.
   
   Let's use a different approach:
   1. Generate all triangle-free graphs on 16 vertices? Too many.
   2. Use a heuristic to build a dense triangle-free graph, then check 3-colorability.
      If it's 3-colorable, remove edges to make it non-3-colorable while keeping as many
      edges as possible.
   
   Actually, let's think about the structure. The densest triangle-free graphs are
   complete bipartite graphs. But they are 2-colorable.
   To make a triangle-free graph non-3-colorable, we need to add "odd cycles" in a way
   that breaks 3-colorability.
   
   The Grötzsch graph is the unique 4-chromatic triangle-free graph on 11 vertices.
   We can take the Grötzsch graph on vertices 0..10, and then add vertices 11..15
   with as many edges as possible while maintaining triangle-freeness and ensuring
   the whole graph is not 3-colorable.
   
   But wait, if we add isolated vertices, the graph remains not 3-colorable (since the
   Grötzsch part is not 3-colorable). But we want to MAXIMIZE edges, so we should add
   edges to vertices 11..15 as well.
   
   Let's try a constructive approach:
   1. Start with the Grötzsch graph on 11 vertices.
   2. Add vertices 11..15 one by one, connecting each to as many existing vertices as
      possible without creating a triangle.
   3. Check if the resulting graph is still not 3-colorable. It should be, since it
      contains the Grötzsch graph as a subgraph.
   
   But this might not give the maximum number of edges. We might be able to do better
      by choosing a different base or by adding edges differently.
   
   Let's implement a backtracking search that tries to add edges in a smart order,
   aiming for a high edge count, and checks 3-colorability at the end (or incrementally).
   
   Given the time limit (120s), we need an efficient search.
   
   Alternative: Use the fact that the maximum number of edges in a triangle-free graph
   on 16 vertices is 64 (K_{8,8}). We can try to find a triangle-free graph with 64 edges
   that is not 3-colorable. If we can't, try 63, etc.
   
   But K_{8,8} is bipartite. To make it non-3-colorable, we need to modify it.
   However, any triangle-free graph with 64 edges on 16 vertices must be K_{8,8} (by
   Mantel's theorem, equality holds only for complete bipartite graphs with balanced parts).
   So K_{8,8} is the only triangle-free graph with 64 edges. And it is 3-colorable.
   So we need to look for 63 edges.
   
   Graphs with 63 edges that are triangle-free: They must be "almost" complete bipartite.
   We can remove one edge from K_{8,8}. But that graph is still bipartite, hence 3-colorable.
   So we need to add an edge that creates an odd cycle, but this would create a triangle
   unless we remove other edges.
   
   This is getting complex. Let's use a computational approach with a time budget.
   
   Plan:
   1. Use a backtracking search over the 120 possible edges.
   2. Order edges in a way that favors high-degree vertices first (to build dense graphs).
   3. Prune if a triangle is formed.
   4. Keep track of the best (max edge count) non-3-colorable graph found so far.
   5. Use a deadline to stop the search and return the best result.
   
   To check 3-colorability, we use a simple backtracking 3-coloring algorithm.
   
   To speed up the search, we can use the following:
   - If the current partial graph is already not 3-colorable, we can try to add more edges
     (since adding edges preserves non-3-colorability... wait, no! Adding edges can make
     a graph 3-colorable? No, adding edges cannot make a non-3-colorable graph 3-colorable.
     If G is not 3-colorable, then any supergraph of G is also not 3-colorable.
     So once we have a non-3-colorable graph, we can try to add more edges to maximize
     the count, as long as we maintain triangle-freeness.
   
   This is a key insight! So the search can be:
   1. Find any non-3-colorable triangle-free graph (e.g., the Grötzsch graph).
   2. From that point, try to add more edges (in some order) to maximize the count,
      while maintaining triangle-freeness.
   3. Since the graph is already non-3-colorable, we don't need to check 3-colorability
      again for the final graph, only for the initial seed.
   
   But we want the MAXIMUM number of edges. So we should try different starting points
   or different orders of edge addition to see if we can get a denser graph.
   
   However, the densest triangle-free graph is K_{8,8} with 64 edges, which is 3-colorable.
   So we need to find the densest triangle-free graph that is NOT 3-colorable.
   
   Let's implement a search that:
   1. Starts with an empty graph.
   2. Tries to add edges in a fixed order.
   3. If adding an edge creates a triangle, skip it.
   4. If the current graph is not 3-colorable, mark it as a candidate and continue adding
      edges (since the supergraph will also be non-3-colorable).
   5. At the end, return the candidate with the most edges.
   
   But this might not be optimal because the order of edge addition matters.
   
   Better: Use a branch-and-bound approach where we try both including and excluding each edge,
   with pruning based on triangle-freeness and an upper bound on the number of edges we can
   still add.
   
   Given the complexity, let's implement a simpler approach first and see if it works within
   the time limit.
   
   Approach:
   1. Generate the Grötzsch graph (11 vertices, 20 edges) as a seed.
   2. Add vertices 11..15, connecting each to as many existing vertices as possible
      without creating a triangle.
   3. This gives a valid graph. Then try to add more edges among the new vertices or
      between new and old vertices, in a greedy manner, to increase the edge count.
   4. Since the graph contains the Grötzsch graph, it is not 3-colorable.
   
   Let's implement this and see how many edges we get. Then we can try to improve it
   with a local search or a more systematic backtracking.
   
   Actually, let's just implement a backtracking search with a time limit and see what
   we can find. We'll use the Grötzsch graph as a starting point and try to add more edges.
"""

import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s out of 120s
    
    n = 16
    # List of all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # We'll use a backtracking search to find the maximum number of edges
    # in a triangle-free, non-3-colorable graph.
    
    # First, let's get a seed: the Grötzsch graph on vertices 0..10
    # The Grötzsch graph is the Mycielski graph of C5.
    # C5: 0-1-2-3-4-0
    # Mycielski: add vertices 5..9 and 10
    # Edges of C5:
    groetzsch_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # C5
        (0, 5), (1, 5), (1, 6), (2, 6), (2, 7), (3, 7), (3, 8), (4, 8), (4, 9), (5, 9),  # Mycielski connections
        (5, 10), (6, 10), (7, 10), (8, 10), (9, 10)  # connections to the new vertex 10
    ]
    
    # Verify that this is triangle-free and not 3-colorable
    # For now, assume it is.
    
    best_edges = groetzsch_edges[:]
    best_count = len(best_edges)
    
    # Now, try to add more edges to vertices 11..15
    # We'll use a greedy approach: for each new vertex, connect it to as many
    # existing vertices as possible without creating a triangle.
    
    # Build the current graph
    adj = [[False]*n for _ in range(n)]
    for u, v in best_edges:
        adj[u][v] = adj[v][u] = True
    
    # Add vertices 11..15
    for v in range(11, n):
        # Try to connect v to existing vertices 0..v-1
        # To avoid triangles, if v is connected to u and w, then u and w must not be connected.
        # So N(v) must be an independent set in the current graph.
        
        # Greedy: try to add as many neighbors as possible
        # Sort existing vertices by degree (ascending) to try to add more edges
        candidates = list(range(v))
        # We want to find a large independent set in the current graph induced by candidates
        # For simplicity, let's use a greedy approach:
        # Try to add vertices one by one, checking if they can be added to N(v)
        
        N_v = []
        for u in candidates:
            # Check if u can be added to N(v)
            # u can be added if u is not adjacent to any vertex in N_v
            can_add = True
            for w in N_v:
                if adj[u][w]:
                    can_add = False
                    break
            if can_add:
                N_v.append(u)
        
        # Add edges from v to N_v
        for u in N_v:
            best_edges.append((v, u))
            adj[v][u] = adj[u][v] = True
            best_count += 1
    
    # Now, try to add more edges among vertices 11..15 and between 11..15 and 0..10
    # We'll use a local search: try to add each possible edge that doesn't create a triangle
    # and check if the graph remains non-3-colorable (it should, since it contains the Grötzsch graph)
    
    # Actually, since the graph contains the Grötzsch graph, it is guaranteed to be non-3-colorable.
    # So we just need to maximize the number of edges while maintaining triangle-freeness.
    
    # Let's do a greedy pass: try to add each edge in all_edges that is not already present
    # and doesn't create a triangle.
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for u, v in all_edges:
            if not adj[u][v]:
                # Check if adding (u,v) creates a triangle
                creates_triangle = False
                for w in range(n):
                    if w != u and w != v and adj[u][w] and adj[v][w]:
                        creates_triangle = True
                        break
                if not creates_triangle:
                    # Add the edge
                    best_edges.append((u, v))
                    adj[u][v] = adj[v][u] = True
                    best_count += 1
                    changed = True
    
    # Convert to list of lists
    result = [[u, v] for u, v in best_edges]
    return result

# Let's test the function
if __name__ == "__main__":
    edges = construct()
    print(f"Number of edges: {len(edges)}")
    # Verify triangle-free
    n = 16
    adj = [[False]*n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = adj[v][u] = True
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                for k in range(j+1, n):
                    if adj[i][k] and adj[j][k]:
                        print(f"Triangle found: {i}, {j}, {k}")
                        sys.exit(1)
    print("Graph is triangle-free.")
    # Verify not 3-colorable
    # Simple backtracking 3-coloring
    colors = [-1]*n
    def is_3_colorable():
        def backtrack(v):
            if v == n:
                return True
            for c in range(3):
                if all(colors[u] != c for u in range(v) if adj[v][u]):
                    colors[v] = c
                    if backtrack(v+1):
                        return True
                    colors[v] = -1
            return False
        return backtrack(0)
    
    if is_3_colorable():
        print("Graph is 3-colorable. This is invalid.")
    else:
        print("Graph is not 3-colorable. Valid.")
