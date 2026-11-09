"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    n = 16
    total_edges = n * (n - 1) // 2  # 120
    
    # Precompute Q3 structure
    q3_edges = set()
    for i in range(8):
        for j in range(i+1, 8):
            diff = bin(i ^ j).count('1')
            if diff == 1:
                q3_edges.add((i, j))
    
    # Build Q3 adjacency lists
    q3_adj = [set() for _ in range(8)]
    for (a, b) in q3_edges:
        q3_adj[a].add(b)
        q3_adj[b].add(a)
    
    # Generate all possible embeddings of Q3 into 16 vertices
    # An embedding is a mapping from Q3 vertices {0..7} to graph vertices {0..15}
    # Two Q3 edges map to two graph edges.
    # We'll precompute for each pair of graph edges, which Q3 embeddings they participate in.
    
    # Actually, let's think differently. We want to find a maximum set of edges in K16
    # such that no 8 vertices form a Q3 subgraph.
    
    # Approach: Use a greedy algorithm with a fast Q3 check.
    # Start with a good initial graph, then try to add edges.
    
    # Better approach: Use simulated annealing or local search on the edge set.
    # Represent the graph as a 16x16 adjacency matrix.
    # Objective: maximize edges subject to Q3-free constraint.
    
    # Let's use a different strategy:
    # 1. Start with K_{3,13} (39 edges) or similar bipartite graph
    # 2. Try to add edges greedily
    # 3. Use local search to improve
    
    # Fast Q3 check: Given a graph, check if it contains Q3 as a subgraph.
    # We can use the fact that Q3 is a 3-regular bipartite graph on 8 vertices.
    
    def check_q3_free(adj_sets):
        """Check if graph is Q3-free using backtracking over 8-vertex subsets."""
        # For each 8-vertex subset, check if it induces a supergraph of Q3
        # Actually, we need to check if there's an injective mapping from Q3 to the subset
        # such that all Q3 edges map to edges in the graph.
        
        for subset in itertools.combinations(range(n), 8):
            subset_set = set(subset)
            
            # Backtracking: assign Q3 vertices to subset vertices
            # Order Q3 vertices by degree (all are 3), so any order works
            # Use a smart ordering: try to assign vertices that have many constraints first
            
            assignment = [-1] * 8
            used = set()
            
            # Precompute for each Q3 vertex, which other Q3 vertices it's adjacent to
            # q3_adj already has this
            
            def backtrack(q3_idx):
                if q3_idx == 8:
                    return True
                
                # Try to find a vertex in subset that is compatible
                # Compatibility: for all already-assigned neighbors in Q3,
                # the corresponding graph vertices must be adjacent
                for sv in subset:
                    if sv in used:
                        continue
                    
                    # Check if sv is compatible with all assigned Q3 neighbors
                    compatible = True
                    for prev_q3 in range(q3_idx):
                        if prev_q3 in q3_adj[q3_idx]:
                            prev_sv = assignment[prev_q3]
                            if prev_sv not in adj_sets[sv]:
                                compatible = False
                                break
                    
                    if compatible:
                        used.add(sv)
                        assignment[q3_idx] = sv
                        if backtrack(q3_idx + 1):
                            return True
                        used.remove(sv)
                        assignment[q3_idx] = -1
                
                return False
            
            if backtrack(0):
                return False  # Found a Q3 subgraph
        
        return True  # No Q3 subgraph found
    
    # Start with a good initial graph
    # K_{3,13} has 39 edges
    # Let's try a different starting point: a random graph with high edge density
    
    # Actually, let's use a smarter approach:
    # Start with K_{4,12} (48 edges) and check if it's Q3-free
    # K_{4,12} is bipartite, and Q3 is bipartite, so it could contain Q3
    
    # Let's just start with K_{3,13} and greedily add edges
    
    adj = [[0]*n for _ in range(n)]
    adj_sets = [set() for _ in range(n)]
    edges = []
    
    # Add all edges between part A (0,1,2) and part B (3-15)
    for a in range(3):
        for b in range(3, n):
            adj[a][b] = 1
            adj[b][a] = 1
            adj_sets[a].add(b)
            adj_sets[b].add(a)
            edges.append([a, b])
    
    # Now try to add all other possible edges
    all_possible_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if not adj[i][j]:
                all_possible_edges.append((i, j))
    
    # Sort edges: prioritize edges that are "easier" to add
    # Edges within smaller partitions first
    def edge_priority(edge):
        i, j = edge
        in_a = lambda x: x < 3
        if in_a(i) and in_a(j):
            return 0  # Highest priority
        elif not in_a(i) and not in_a(j):
            return 1  # Lower priority
        else:
            return 2  # Cross edges (shouldn't happen)
    
    all_possible_edges.sort(key=edge_priority)
    
    best_edges = list(edges)
    best_count = len(edges)
    
    for edge in all_possible_edges:
        if time.time() > deadline - 5:
            break
        
        u, v = edge
        # Try adding this edge
        adj[u][v] = 1
        adj[v][u] = 1
        adj_sets[u].add(v)
        adj_sets[v].add(u)
        edges.append([u, v])
        
        # Check if Q3-free
        if check_q3_free(adj_sets):
            best_edges = list(edges)
            best_count = len(edges)
        else:
            # Revert
            adj[u][v] = 0
            adj[v][u] = 0
            adj_sets[u].remove(v)
            adj_sets[v].remove(u)
            edges.pop()
    
    return best_edges
# EVOLVE-BLOCK-END
