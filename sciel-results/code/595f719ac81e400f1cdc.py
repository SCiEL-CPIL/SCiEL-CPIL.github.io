"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np

def construct():
    n = 16
    # List of all possible edges
    all_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            all_edges.append((u, v))
    
    edge_index = {}
    for i, (u, v) in enumerate(all_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    
    m = len(all_edges)
    
    # Find all Q3 subgraphs (embeddings)
    # A Q3 is a cube with 8 vertices and 12 edges.
    # Vertices are 3-bit strings. Edges connect vertices differing in exactly 1 bit.
    # We need to find all sets of 8 distinct vertices from 0..15 that can be mapped to Q3.
    # Since Q3 is small, we can enumerate all subsets of 8 vertices? No, C(16,8) = 12870.
    # For each subset, check if it induces a Q3 or contains a Q3?
    # The task says "containing no subgraph isomorphic to the 3-cube Q3".
    # This means we must forbid any set of 8 vertices and 12 edges that form a Q3.
    # It's a subgraph, not necessarily induced. So if there is a mapping from Q3 vertices to G vertices such that all Q3 edges map to edges in G, it's forbidden.
    
    # Let's generate all possible Q3 embeddings.
    # A Q3 embedding is a set of 8 distinct vertices in 0..15 and a bijection to {0,1}^3 such that edges in Q3 correspond to edges in G.
    # Actually, we just need to identify the set of 12 edges that would form a Q3.
    # Two Q3s are the same set of edges if they use the same 12 edges.
    # Let's enumerate all subsets of 8 vertices. For each subset, check if it contains a Q3.
    # Wait, a Q3 has 8 vertices. If we pick 8 vertices, they might form multiple Q3s? No, the structure is fixed.
    # The number of ways to map 8 vertices to Q3 is 8! * 2 (for the isomorphisms of Q3? No, Q3 has 48 automorphisms).
    # Actually, simpler:
    # Iterate over all 8-subsets of vertices.
    # For each subset, check if the induced subgraph contains a Q3.
    # But we are building the graph. We need to know which edges in the full graph constitute a Q3.
    # A set of 12 edges forms a Q3 if its endpoints are 8 vertices and the graph is isomorphic to Q3.
    
    # Let's precompute all Q3 edge sets.
    # There are C(16,8) * (number of Q3s on 8 labeled vertices).
    # Number of Q3s on 8 labeled vertices:
    # Fix vertex 0 at 000. Vertex 1, 2, 3 are neighbors (100, 010, 001).
    # The rest are determined.
    # The number of distinct Q3 subgraphs on a set of 8 vertices is 8! / 48 = 3360.
    # Total Q3s in K16: C(16,8) * 3360 = 12870 * 3360 ~ 43 million. This is too many to store explicitly if we want to be fast, but maybe we can generate them on the fly or use a smarter check.
    
    # Alternative approach:
    # Instead of precomputing all Q3s, we can use a heuristic or a smaller set of constraints.
    # Or, we can just check for Q3s in the current graph during the annealing process.
    # Checking if a graph contains a Q3 is NP-complete in general, but for n=16, we can do it.
    # How to check if G contains a Q3?
    # Iterate over all 8-subsets of vertices.
    # For each subset, check if it contains a Q3.
    # To check if an 8-vertex graph contains a Q3:
    # We can try all mappings of Q3 to the 8 vertices.
    # There are 48 automorphisms of Q3. So we just need to check 48 mappings (fixing one vertex and its neighbors).
    # Actually, fix vertex v0 in G to 000 in Q3.
    # The 3 neighbors of 000 in Q3 must map to 3 neighbors of v0 in G.
    # There are C(deg(v0), 3) ways to choose the 3 neighbors.
    # Once 000 and its 3 neighbors are fixed, the rest of the Q3 is determined.
    # So for each v0, and each choice of 3 neighbors, we can check if the corresponding 8 vertices and 12 edges exist in G.
    
    # This is much faster.
    # For a given graph G:
    # For each v in 0..15:
    #   For each 3-subset of neighbors of v:
    #     Let the neighbors be u1, u2, u3.
    #     The 8 vertices of the potential Q3 are:
    #     v (000)
    #     u1 (100), u2 (010), u3 (001)
    #     u1+u2 (110) -> must be a vertex connected to both u1 and u2? No, in Q3, 110 is connected to 100 and 010.
    #     Wait, the vertices are:
    #     000: v
    #     100: u1
    #     010: u2
    #     001: u3
    #     110: w12 (must be connected to u1 and u2)
    #     101: w13 (must be connected to u1 and u3)
    #     011: w23 (must be connected to u2 and u3)
    #     111: w123 (must be connected to w12, w13, w23)
    #     
    #     So we need to find w12, w13, w23, w123.
    #     w12 must be a common neighbor of u1 and u2 (other than v? No, v is not connected to w12 in Q3).
    #     Actually, in Q3:
    #     110 is connected to 100 and 010.
    #     101 is connected to 100 and 001.
    #     011 is connected to 010 and 001.
    #     111 is connected to 110, 101, 011.
    #     
    #     So:
    #     w12 is a vertex connected to u1 and u2.
    #     w13 is a vertex connected to u1 and u3.
    #     w23 is a vertex connected to u2 and u3.
    #     w123 is a vertex connected to w12, w13, w23.
    #     
    #     And all 8 vertices must be distinct.
    #     
    #     This check is very fast.
    
    def count_q3s(adj):
        """Count the number of Q3 subgraphs in the graph represented by adj (list of sets)."""
        count = 0
        for v in range(n):
            neighbors = list(adj[v])
            # Choose 3 neighbors
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    for k in range(j + 1, len(neighbors)):
                        u1, u2, u3 = neighbors[i], neighbors[j], neighbors[k]
                        
                        # Find w12: common neighbor of u1 and u2
                        common_12 = adj[u1] & adj[u2]
                        # w12 cannot be v (v is not connected to w12 in Q3? Wait, v is 000, w12 is 110. They are not adjacent.)
                        # So w12 != v.
                        common_12.discard(v)
                        
                        # Find w13: common neighbor of u1 and u3
                        common_13 = adj[u1] & adj[u3]
                        common_13.discard(v)
                        
                        # Find w23: common neighbor of u2 and u3
                        common_23 = adj[u2] & adj[u3]
                        common_23.discard(v)
                        
                        # For each combination of w12, w13, w23
                        for w12 in common_12:
                            for w13 in common_13:
                                for w23 in common_23:
                                    # Check distinctness
                                    vertices = {v, u1, u2, u3, w12, w13, w23}
                                    if len(vertices) < 7:
                                        continue
                                    
                                    # Find w123: common neighbor of w12, w13, w23
                                    common_123 = adj[w12] & adj[w13] & adj[w23]
                                    # w123 must be distinct from the other 7
                                    common_123 -= vertices
                                    
                                    for w123 in common_123:
                                        # We found a Q3
                                        # To avoid counting the same Q3 multiple times, we can impose an ordering.
                                        # But for the purpose of counting violations in annealing, we just need a count.
                                        # However, the same Q3 can be found via different (v, u1, u2, u3) choices.
                                        # A Q3 has 8 vertices, each with degree 3.
                                        # So each Q3 will be counted 8 times (once for each vertex as v).
                                        # And for each v, there is only 1 way to choose the 3 neighbors (since the neighbors in Q3 are fixed).
                                        # Wait, if we pick v, the 3 neighbors in Q3 are unique.
                                        # So each Q3 is counted exactly 8 times.
                                        count += 1
        return count // 8
    
    # Simulated Annealing
    # Start with a random graph or a dense graph
    # Let's start with a random graph with p=0.5
    random.seed(42)
    adj = [set() for _ in range(n)]
    edge_set = set()
    
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < 0.5:
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add((u, v))
    
    current_violations = count_q3s(adj)
    
    best_edges = list(edge_set)
    best_violations = current_violations
    
    T = 100.0
    T_min = 1e-3
    alpha = 0.999
    steps = 50000
    
    for step in range(steps):
        T *= alpha
        if T < T_min:
            break
            
        # Pick a random edge to flip
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        # Calculate delta in violations
        if (u, v) in edge_set:
            # Remove edge
            # Check if removing this edge breaks any Q3
            # A Q3 containing edge (u,v) will be broken.
            # How many Q3s contain edge (u,v)?
            # We can't easily know this without enumerating.
            # So we just recompute the count.
            # This is slow.
            
            # Optimization:
            # We can precompute the number of Q3s containing each edge.
            # But this changes as the graph changes.
            
            # Let's just recompute.
            # To speed up, we can only check Q3s that involve u or v.
            
            # For now, let's just recompute the whole count.
            # If it's too slow, we'll optimize.
            
            adj[u].remove(v)
            adj[v].remove(u)
            edge_set.remove((u, v))
            
            new_violations = count_q3s(adj)
            
            delta = new_violations - current_violations
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_violations = new_violations
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_edges = list(edge_set)
            else:
                # Reject
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add((u, v))
        else:
            # Add edge
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))
            
            new_violations = count_q3s(adj)
            
            delta = new_violations - current_violations
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_violations = new_violations
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_edges = list(edge_set)
            else:
                # Reject
                adj[u].remove(v)
                adj[v].remove(u)
                edge_set.remove((u, v))
    
    # Convert best_edges to list of lists
    return [[u, v] for u, v in best_edges]

# EVOLVE-BLOCK-END
