import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 10s buffer
    
    # 1. Generate Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    # Adjacency matrix or set of edges
    # Using a set of frozensets for cliques might be faster for checking containment?
    # Or just a boolean matrix for fast lookup.
    
    n_current = q
    # adjacency[i][j] is True if i and j are connected
    adj = [[False] * n_current for _ in range(n_current)]
    for i in range(n_current):
        for j in range(i + 1, n_current):
            if (j - i) % q in residues:
                adj[i][j] = True
                adj[j][i] = True
                
    def get_edges():
        edges = []
        for i in range(n_current):
            for j in range(i + 1, n_current):
                if adj[i][j]:
                    edges.append((i, j))
        return edges

    # Precompute K4s and Independent 4-sets for the current graph
    # This needs to be updated when a vertex is added.
    # For now, let's just check on the fly or recompute.
    # Recomputing for n=37 is fast. For n=42, it's slower but still doable within seconds if optimized.
    
    def find_k4s(n, adj):
        k4s = []
        # Iterate over all 4-tuples
        # Optimization: only check if the 4 vertices form a clique
        # We can iterate i < j < k < l
        for i in range(n):
            for j in range(i + 1, n):
                if not adj[i][j]: continue
                for k in range(j + 1, n):
                    if not adj[i][k] or not adj[j][k]: continue
                    for l in range(k + 1, n):
                        if adj[i][l] and adj[j][l] and adj[k][l]:
                            k4s.append((i, j, k, l))
        return k4s

    def find_ind4s(n, adj):
        ind4s = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]: continue
                for k in range(j + 1, n):
                    if adj[i][k] or adj[j][k]: continue
                    for l in range(k + 1, n):
                        if not adj[i][l] and not adj[j][l] and not adj[k][l]:
                            ind4s.append((i, j, k, l))
        return ind4s

    # Initial check
    # The Paley graph is (5,5)-free, so it should have no K5 and no Ind5.
    # We are adding vertices, so we just need to ensure the new vertex doesn't complete a K5 or Ind5.
    
    def can_add_vertex(n, adj, target_vertex, k4s, ind4s):
        # target_vertex is the index of the new vertex
        # We need to choose S (neighbors of target_vertex)
        # S must not contain any K4 from k4s
        # V\S must not contain any Ind4 from ind4s
        
        # Randomized search for S
        # Heuristic: S should be roughly half the vertices
        # Let's try random subsets of size ~ n/2
        
        # To make it faster, we can pick a random vertex v and start with S = {v} or S = V\{v}
        # But random subset is safer.
        
        for _ in range(5000): # Try 5000 random configurations
            if time.time() > deadline:
                return None
            
            # Generate a random subset S
            # We can flip each vertex with probability 0.5
            # Or pick a random size
            s_set = set()
            for v in range(n):
                if random.random() < 0.5:
                    s_set.add(v)
            
            # Check K4 constraint: No K4 is fully contained in S
            valid = True
            for k4 in k4s:
                if all(v in s_set for v in k4):
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Check Ind4 constraint: No Ind4 is fully contained in V\S
            # i.e., all vertices in the ind4 are NOT in S
            for ind4 in ind4s:
                if all(v not in s_set for v in ind4):
                    valid = False
                    break
            
            if valid:
                return s_set
        
        return None

    # Loop to add vertices
    for target_n in range(38, 43): # Try to go up to 42
        if time.time() > deadline:
            break
            
        n = target_n - 1
        # Precompute K4s and Ind4s for the current graph of size n
        # Note: This is O(n^4). For n=41, 41^4 ~ 2.8M. Python might be slow.
        # Let's optimize the K4 finding.
        
        # Actually, we only need to check if the NEW vertex creates a K5 or Ind5.
        # A K5 involving the new vertex v_new means there is a K4 in the old graph that is fully connected to v_new.
        # So we need to know all K4s in the old graph.
        # Similarly for Ind4s.
        
        k4s = find_k4s(n, adj)
        ind4s = find_ind4s(n, adj)
        
        s_set = can_add_vertex(n, adj, n, k4s, ind4s)
        
        if s_set is not None:
            # Add the vertex
            adj.append([False] * (n + 1))
            for i in range(n):
                adj.append([False] * (n + 1)) # Wait, adj is a list of lists.
                # Actually, adj is n x n. We need to make it (n+1) x (n+1).
                pass
            
            # Correctly expand adjacency matrix
            new_adj = [[False] * (n + 1) for _ in range(n + 1)]
            for i in range(n):
                for j in range(n):
                    new_adj[i][j] = adj[i][j]
            for v in s_set:
                new_adj[n][v] = True
                new_adj[v][n] = True
            adj = new_adj
            n_current = n + 1
        else:
            # Failed to add vertex
            break

    # Return the final graph
    edges = []
    for i in range(n_current):
        for j in range(i + 1, n_current):
            if adj[i][j]:
                edges.append((i, j))
                
    return n_current, edges
