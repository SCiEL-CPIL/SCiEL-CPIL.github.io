"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

# EVOLVE-BLOCK-START
def construct():
    # K3,3-free graph on 20 vertices.
    # We use a probabilistic construction: take a random graph G(n, p) and 
    # remove edges to eliminate K3,3 subgraphs.
    # A better approach: use known constructions.
    
    # For n=20, we can try to maximize edges.
    # The Turan graph T(20, 3) has floor(20^2/4) - (20 mod 2) * ... 
    # Actually, let's use a greedy approach with local search.
    
    n = 20
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with a random graph with moderate edge probability
    best_edges = None
    best_score = -1
    
    # Try several random starts
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        random.seed(attempt)
        
        # Start with a random graph
        p = 0.5
        adj = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i][j] = adj[j][i] = True
        
        # Greedy: add edges if they don't create K3,3
        # K3,3 exists if there are two sets of 3 vertices with all 9 cross edges
        # To check if adding edge (u,v) creates K3,3:
        # We need to check if there exist 2 other vertices in N(u) and 2 in N(v) 
        # such that... actually this is complex.
        
        # Simpler: use a known construction.
        # The complete bipartite graph K_{10,10} has 100 edges and is K3,3-free?
        # No! K_{10,10} contains K3,3 as a subgraph (pick any 3 from each side).
        
        # We need a K3,3-free graph. 
        # A graph is K3,3-free if it has no two disjoint sets of 3 vertices with all cross edges.
        
        # Let's use a different approach: start with complete graph and remove edges.
        # Or use a probabilistic method.
        
        # Actually, for small n, we can use integer programming or greedy.
        # Let's try: take the complete graph and greedily remove edges to break all K3,3.
        
        # Better: use a known family. 
        # The complete graph K_n is K3,3-free only if n < 6.
        # For n=20, we need a sparse-ish graph.
        
        # Let's try a random graph and then repair.
        # Check if graph is K3,3-free:
        # For each pair of vertices (u,v), count common neighbors.
        # If there exist 3 vertices a,b,c and 3 vertices d,e,f such that 
        # {a,b,c} and {d,e,f} are disjoint and all cross edges exist, then K3,3 exists.
        
        # Efficient check: 
        # For each triple of vertices, check if they have 3 common neighbors.
        
        def is_k33_free(adj):
            n = len(adj)
            # For each triple of vertices, check common neighbors
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(j + 1, n):
                        common = 0
                        for l in range(n):
                            if l not in (i, j, k) and adj[i][l] and adj[j][l] and adj[k][l]:
                                common += 1
                        if common >= 3:
                            return False
            return True
        
        # This is O(n^4) which is 20^4 = 160000, acceptable.
        
        if is_k33_free(adj):
            edge_count = sum(sum(row) for row in adj) // 2
            if edge_count > best_score:
                best_score = edge_count
                best_edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i][j]]
        
        # Try to improve by adding edges
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                for j in range(i + 1, n):
                    if not adj[i][j]:
                        adj[i][j] = adj[j][i] = True
                        if is_k33_free(adj):
                            improved = True
                            break
                        else:
                            adj[i][j] = adj[j][i] = False
                if improved:
                    break
        
        if is_k33_free(adj):
            edge_count = sum(sum(row) for row in adj) // 2
            if edge_count > best_score:
                best_score = edge_count
                best_edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i][j]]
    
    if best_edges is not None:
        return best_edges
    
    # Fallback: K2,18
    return [[a, b] for a in range(2) for b in range(2, 20)]
# EVOLVE-BLOCK-END
