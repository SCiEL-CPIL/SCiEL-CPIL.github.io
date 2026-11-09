import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 10s safety margin

    def get_paley(q):
        """Generate Paley graph on q vertices (q must be prime congruent to 1 mod 4)."""
        residues = {(x * x) % q for x in range(1, q)}
        adj = [0] * q
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in residues:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def is_valid_5_5(n, adj):
        """
        Checks if the graph contains no K5 and no independent set of size 5.
        """
        # Precompute non-neighbors for each vertex for speed? 
        # Or just check on the fly.
        
        # Check for K5
        # A K5 exists if there is a vertex v such that its neighborhood contains a K4.
        # Or simply iterate all 5-sets.
        # For n <= 42, iterating all 5-sets is feasible.
        
        # Optimization: 
        # Count K5s. If any, invalid.
        # Count Indep5s. If any, invalid.
        
        # Helper to check if a set of 5 vertices is a clique
        def is_clique(verts):
            for i in range(5):
                for j in range(i+1, 5):
                    u, v = verts[i], verts[j]
                    if not (adj[u] & (1 << v)):
                        return False
            return True

        def is_independent(verts):
            for i in range(5):
                for j in range(i+1, 5):
                    u, v = verts[i], verts[j]
                    if (adj[u] & (1 << v)):
                        return False
            return True

        # Iterate all 5-subsets
        # This is slow in pure Python if done naively for many trials.
        # Better: Use the "neighborhood contains K4" logic.
        
        # Let's stick to the 5-subset iteration but optimize the inner loop.
        # Or use a recursive backtracking to find a K5.
        
        # Fastest simple check:
        for v in range(n):
            # Neighbors of v
            neigh = []
            for u in range(n):
                if u != v and (adj[v] & (1 << u)):
                    neigh.append(u)
            
            # Check if neigh contains K4
            # If neigh size < 4, no K5 containing v
            if len(neigh) >= 4:
                # Check all 4-subsets of neigh
                for quad in itertools.combinations(neigh, 4):
                    if is_clique(list(quad)):
                        return False
            
            # Non-neighbors of v
            non_neigh = []
            for u in range(n):
                if u != v and not (adj[v] & (1 << u)):
                    non_neigh.append(u)
            
            # Check if non_neigh contains independent set of size 4
            if len(non_neigh) >= 4:
                for quad in itertools.combinations(non_neigh, 4):
                    if is_independent(list(quad)):
                        return False
        
        return True

    def adj_to_edges(n, adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    edges.append((i, j))
        return edges

    def edges_to_adj(n, edges):
        adj = [0] * n
        for i, j in edges:
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
        return adj

    # 1. Construct 38-vertex graph: Disjoint union of two Paley(19) graphs
    p19_adj = get_paley(19)
    n_38 = 38
    adj_38 = [0] * n_38
    for i in range(19):
        for j in range(19):
            if p19_adj[i] & (1 << j):
                # Copy 1
                adj_38[i] |= (1 << j)
                adj_38[j] |= (1 << i)
                # Copy 2
                adj_38[i + 19] |= (1 << (j + 19))
                adj_38[j + 19] |= (1 << (i + 19))
    
    # Verify 38-vertex graph
    if is_valid_5_5(n_38, adj_38):
        best_n = n_38
        best_adj = adj_38
    else:
        # Fallback to Paley 37 if something went wrong (shouldn't happen)
        p37_adj = get_paley(37)
        best_n = 37
        best_adj = p37_adj

    # 2. Try to find larger graphs (39, 40, 41, 42)
    # Use local search on random graphs
    for target_n in range(best_n + 1, 43):
        if time.time() > deadline - 10:
            break
        
        found = False
        # Try a few random restarts
        for restart in range(50):
            if time.time() > deadline - 5:
                break
            
            # Initialize random graph
            # Density 0.5 is typical for Ramsey graphs
            adj = [0] * target_n
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Local search: Flip edges to remove K5 or Indep5
            # We maintain a set of "conflicting" edges? 
            # Simpler: Just try flipping random edges and check validity.
            # This is slow.
            
            # Better: Identify a K5 or Indep5, and flip an edge within it to break it.
            # But checking validity is the bottleneck.
            
            # Given the time, if we don't find it quickly, we keep the 38-vertex graph.
            # Let's do a limited number of flips.
            
            current_adj = list(adj)
            for _ in range(100): # 100 flips per restart
                if time.time() > deadline - 5:
                    break
                
                if is_valid_5_5(target_n, current_adj):
                    best_n = target_n
                    best_adj = current_adj
                    found = True
                    break
                
                # Find a violation to target for flipping
                # This is complex. Let's just flip a random edge.
                i = random.randint(0, target_n - 1)
                j = random.randint(0, target_n - 1)
                if i == j:
                    continue
                
                current_adj[i] ^= (1 << j)
                current_adj[j] ^= (1 << i)
            
            if found:
                break
    
    return best_n, adj_to_edges(best_n, best_adj)
