"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def construct():
    """
    Construct a graph with 40 vertices and 401 edges minimizing the largest book size.
    
    Strategy: Start with K_{20,20} (complete bipartite graph) which has 400 edges.
    Add one extra edge within one partition. The book size for an edge (u,v) is the
    number of common neighbors of u and v.
    
    For K_{20,20} with partitions A={0..19} and B={20..39}:
    - Every edge between A and B has 19 common neighbors (all other vertices in the opposite partition)
    - If we add an intra-partition edge (a1, a2) where a1, a2 in A, then:
      * The new edge (a1, a2) has book size = number of common neighbors = |B| = 20
      * Edges (a1, b) and (a2, b) for b in B now have book size = 19 (unchanged, since a1 and a2 are still connected to all of B)
      * Other edges remain with book size 19
    
    Wait, let me reconsider. In K_{20,20}, each edge (a,b) with a in A, b in B has common neighbors = all other 19 vertices in A that connect to b, which is 19.
    
    When we add edge (a1, a2) within A:
    - New edge (a1, a2): common neighbors are all vertices in B (20 vertices), so book size = 20
    - Edge (a1, b): common neighbors are all other 19 vertices in A connected to b, PLUS a2 (since a2 is now connected to a1 and to b). So book size = 20
    - Similarly edge (a2, b): book size = 20
    - Other edges (a, b) where a not in {a1,a2}: book size remains 19
    
    So the maximum book size becomes 20. That's worse than the baseline!
    
    Let me reconsider the problem. The baseline K_{20,20} + one edge has max book = 20.
    
    Actually, I need to think about this differently. Perhaps we should NOT use K_{20,20} as the base.
    
    Let me think about what graph structure minimizes the maximum book size.
    
    The book size of edge (u,v) is the number of triangles containing (u,v), which equals the number of common neighbors of u and v.
    
    We want to minimize the maximum over all edges of the number of common neighbors.
    
    For a regular graph of degree d, the expected number of common neighbors is roughly d^2/n.
    
    With 40 vertices and 401 edges, the average degree is 401*2/40 = 20.05.
    
    If we had a 20-regular graph on 40 vertices, the expected common neighbors would be about 20^2/40 = 10.
    
    K_{20,20} is 20-regular, and each edge has exactly 19 common neighbors (since it's complete bipartite).
    
    So we need a different structure. Let's try to use a more balanced approach.
    
    Actually, let me try simulated annealing or a local search approach to find a better graph.
    """
    
    n = 40
    target_edges = 401
    
    # Use a time budget
    start_time = time.time()
    time_budget = 140  # seconds, leave 10s margin
    
    # Initialize with a random graph close to the target
    np.random.seed(42)
    
    def compute_max_book(adj):
        """Compute the maximum book size (max common neighbors over all edges)."""
        max_book = 0
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    # Count common neighbors
                    common = np.sum(adj[u] & adj[v])
                    if common > max_book:
                        max_book = common
        return max_book
    
    def compute_all_books(adj):
        """Compute book sizes for all edges."""
        books = {}
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    books[(u, v)] = np.sum(adj[u] & adj[v])
        return books
    
    # Start with K_{20,20} plus one edge as initial guess
    adj = np.zeros((n, n), dtype=bool)
    for a in range(20):
        for b in range(20, 40):
            adj[a, b] = True
            adj[b, a] = True
    # Add one intra-partition edge
    adj[0, 1] = True
    adj[1, 0] = True
    
    # Count edges
    edge_count = np.sum(adj) // 2
    assert edge_count == target_edges, f"Expected {target_edges} edges, got {edge_count}"
    
    best_adj = adj.copy()
    best_score = compute_max_book(adj)
    
    # Local search: try edge swaps to reduce max book
    # An edge swap: remove edge (u1,v1) and add edge (u2,v2)
    
    # Precompute common neighbors for faster evaluation
    def get_common_neighbors_count(adj, u, v):
        return np.sum(adj[u] & adj[v])
    
    # Simulated annealing style local search
    temperature = 10.0
    cooling_rate = 0.999
    min_temperature = 0.01
    
    current_adj = adj.copy()
    current_score = best_score
    
    # Build list of all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # For efficiency, maintain a set of current edges
    current_edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            if current_adj[u, v]:
                current_edges.add((u, v))
    
    # Track best
    best_adj = current_adj.copy()
    best_score = current_score
    
    # Local search with edge swaps
    iteration = 0
    while time.time() - start_time < time_budget:
        # Try a random edge swap
        if len(current_edges) < 2:
            break
            
        # Pick two random edges to swap
        edge_list = list(current_edges)
        idx1 = np.random.randint(len(edge_list))
        idx2 = np.random.randint(len(edge_list))
        
        if idx1 == idx2:
            continue
            
        e1 = edge_list[idx1]
        e2 = edge_list[idx2]
        
        # We want to remove e1 and add e2' where e2' is not currently an edge
        # Let's pick a random non-edge to add
        # Find a non-edge
        non_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) not in current_edges:
                    non_edges.append((u, v))
                    if len(non_edges) > 100:
                        break
            if len(non_edges) > 100:
                break
        
        if not non_edges:
            break
            
        new_edge = non_edges[np.random.randint(len(non_edges))]
        
        # Check if new_edge is the same as e1
        if new_edge == e1:
            continue
        
        # Perform the swap: remove e1, add new_edge
        u1, v1 = e1
        current_adj[u1, v1] = False
        current_adj[v1, u1] = False
        current_edges.remove(e1)
        
        u2, v2 = new_edge
        current_adj[u2, v2] = True
        current_adj[v2, u2] = True
        current_edges.add(new_edge)
        
        # Compute new score
        new_score = compute_max_book(current_adj)
        
        # Accept or reject
        if new_score <= current_score or np.random.random() < np.exp(-(new_score - current_score) / temperature):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = current_adj.copy()
        else:
            # Revert
            current_adj[u1, v1] = True
            current_adj[v1, u1] = True
            current_edges.add(e1)
            
            current_adj[u2, v2] = False
            current_adj[v2, u2] = False
            current_edges.remove(new_edge)
        
        # Cool down
        temperature *= cooling_rate
        if temperature < min_temperature:
            temperature = min_temperature
        
        iteration += 1
    
    # Return the best graph found
    best_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                best_edges.append([u, v])
    
    # Verify edge count
    assert len(best_edges) == target_edges, f"Expected {target_edges} edges, got {len(best_edges)}"
    
    return best_edges

# EVOLVE-BLOCK-END
