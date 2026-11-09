"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a graph on as many vertices as possible with no K5 and no independent 5-set.
    Strategy: Start from the Paley graph on 37 vertices. Try to add vertices one by one,
    optimizing the adjacency of the new vertex to maximize n while maintaining the (5,5)-free property.
    If we reach 42, we return that. If not, we return the best we found (at least 37).
    """
    start_time = time.time()
    deadline = start_time + 290  # 300s limit, keep 10s margin
    
    # 1. Base Paley Graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    
    # Adjacency matrix for the base graph
    # A[i][j] = 1 if (j-i) % q in residues
    A = np.zeros((q, q), dtype=np.uint8)
    for i in range(q):
        for j in range(q):
            if i != j and (j - i) % q in residues:
                A[i][j] = 1
    
    # Helper to check if adding a vertex with a specific connection pattern to the existing graph
    # creates a K5 or an independent 5-set.
    # Existing graph G has n vertices. New vertex v connects to subset S of G.
    # A K5 involving v exists if there is a K4 in G[S].
    # An independent 5-set involving v exists if there is an independent 4-set in G[~S] (vertices not connected to v).
    
    # Pre-compute all K4s and independent 4-sets in the base graph?
    # 37 vertices is small enough.
    # Let's precompute sets of vertices forming K4 and independent 4-sets in the base graph.
    
    def find_k4s(adj):
        n = len(adj)
        k4s = []
        for combo in combinations(range(n), 4):
            i, j, k, l = combo
            if adj[i][j] and adj[i][k] and adj[i][l] and adj[j][k] and adj[j][l] and adj[k][l]:
                k4s.append(frozenset(combo))
        return k4s

    def find_ind_4s(adj):
        n = len(adj)
        ind_4s = []
        # Non-edges
        non_adj = 1 - adj
        np.fill_diagonal(non_adj, 0)
        for combo in combinations(range(n), 4):
            i, j, k, l = combo
            if non_adj[i][j] and non_adj[i][k] and non_adj[i][l] and non_adj[j][k] and non_adj[j][l] and non_adj[k][l]:
                ind_4s.append(frozenset(combo))
        return ind_4s

    # We will maintain the current graph's K4s and Ind4s as we add vertices.
    # Initially:
    current_adj = A
    current_n = q
    
    # We need to track K4s and Ind4s in the *current* graph.
    # Adding a vertex v connected to S:
    # New K4s: Any K3 in S that forms a K4 with v.
    # New Ind4s: Any Ind3 in ~S that forms an Ind4 with v.
    # Also, existing K4s and Ind4s remain.
    
    # Let's precompute K3s and Ind3s of the base graph to speed up updates?
    # Or just recompute K4s/Ind4s from scratch if n is small?
    # For n=37, C(37,4) = 66045. Checking each is fast.
    # For n=42, C(42,4) = 111930. Still fast.
    # So we can just recompute K4s and Ind4s after each addition if needed, or maintain them.
    # Maintaining is better for speed during search.
    
    # Let's store K4s and Ind4s as lists of frozensets.
    # But we need to check if a *new* vertex creates a violation.
    # Violation condition for new vertex v with neighbor set S:
    # 1. Exists K3 in S? -> K4 with v.
    # 2. Exists Ind3 in ~S? -> Ind4 with v.
    
    # So we need the set of all K3s and Ind3s in the current graph.
    # Let's precompute K3s and Ind3s for the base graph.
    
    def find_k3s(adj):
        n = len(adj)
        k3s = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    for k in range(j+1, n):
                        if adj[i][k] and adj[j][k]:
                            k3s.append(frozenset([i, j, k]))
        return k3s

    def find_ind_3s(adj):
        n = len(adj)
        ind_3s = []
        non_adj = 1 - adj
        np.fill_diagonal(non_adj, 0)
        for i in range(n):
            for j in range(i+1, n):
                if non_adj[i][j]:
                    for k in range(j+1, n):
                        if non_adj[i][k] and non_adj[j][k]:
                            ind_3s.append(frozenset([i, j, k]))
        return ind_3s

    base_k3s = find_k3s(current_adj)
    base_ind_3s = find_ind_3s(current_adj)
    
    # We want to add 5 more vertices (to reach 42).
    # Let's try to add vertices one by one.
    # For each new vertex, we try all 2^37 possible connection patterns? No, too many.
    # We can use local search or greedy.
    # Greedy: Try to find a connection pattern S for the new vertex such that:
    # - No K3 is contained in S.
    # - No Ind3 is contained in ~S.
    # This is equivalent to:
    # - S is an independent set in the "K3-hypergraph"? No, S must not contain any K3.
    # - ~S must not contain any Ind3.
    
    # Let's try a randomized local search for the connection pattern of the new vertex.
    # Or, we can just iterate through a set of candidate patterns.
    
    # Actually, there is a known construction for 42 vertices.
    # The Paley graph of order 37 is not the only one.
    # Let's try to extend the Paley graph.
    
    # If we can't extend the Paley graph, we should try other starting points or different search strategies.
    # But the prompt says "The seed is the Paley graph on 37 vertices".
    # Let's stick to extending it.
    
    # Algorithm to add a vertex:
    # 1. Start with a random S (subset of current vertices).
    # 2. Check validity:
    #    - For each K3 in current_k3s: if K3 subset of S -> Invalid.
    #    - For each Ind3 in current_ind_3s: if Ind3 subset of ~S -> Invalid.
    # 3. If valid, accept.
    # 4. If invalid, flip a random bit in S and retry.
    #    - If flipping bit i (vertex i) from 0 to 1 (adding edge):
    #      - Check if any K3 in current_k3s is now subset of S.
    #      - Check if any Ind3 in current_ind_3s is now subset of ~S (i.e., was subset of ~S before, but i was in ~S? No, i is removed from ~S).
    #        Actually, if i is added to S, it is removed from ~S.
    #        So we only need to check if adding i to S creates a K3 subset in S.
    #        The Ind3 condition might be relaxed (fewer vertices in ~S), so it's safer.
    #    - If flipping bit i from 1 to 0 (removing edge):
    #      - Check if any Ind3 in current_ind_3s is now subset of ~S.
    #      - The K3 condition might be relaxed.
    
    # Let's implement this local search for each new vertex.
    
    current_k3s = base_k3s
    current_ind_3s = base_ind_3s
    
    # We will add 5 vertices.
    # We need to update current_k3s and current_ind_3s after each addition.
    # New K3s involving new vertex v and neighbors S:
    # Any edge (u, w) in S such that u, w in S and edge exists.
    # Wait, K3 is a triangle.
    # New triangles involving v:
    # Pairs (u, w) in S such that u~w in current graph.
    # New Ind3s involving v and non-neighbors ~S:
    # Pairs (u, w) in ~S such that u!~w in current graph.
    
    def add_vertex(adj, n, k3s, ind_3s, deadline):
        """
        Try to add a new vertex to the graph defined by adj (n x n).
        Returns (new_adj, new_n, new_k3s, new_ind_3s, success)
        """
        # Initialize S randomly
        S = np.random.randint(0, 2, size=n).astype(np.uint8)
        # ~S is 1-S
        
        # Helper to check validity of S
        def is_valid(S):
            # Check K3s in S
            for k3 in k3s:
                # k3 is a frozenset of vertices
                # Check if all vertices in k3 are in S
                valid = True
                for v in k3:
                    if S[v] == 0:
                        valid = False
                        break
                if valid:
                    return False
            # Check Ind3s in ~S
            for i3 in ind_3s:
                valid = True
                for v in i3:
                    if S[v] == 1: # v is in S, so not in ~S
                        valid = False
                        break
                if valid:
                    return False
            return True
        
        # Local search
        best_S = None
        for step in range(1000):
            if time.time() > deadline:
                return None, n, k3s, ind_3s, False
            if is_valid(S):
                best_S = S.copy()
                break
            # Flip a random bit
            i = np.random.randint(0, n)
            S[i] = 1 - S[i]
            
        if best_S is None:
            # Try a few more strategies?
            # Maybe try to maximize the number of edges or non-edges?
            # For now, return failure.
            return None, n, k3s, ind_3s, False
            
        # Construct new adjacency matrix
        new_n = n + 1
        new_adj = np.zeros((new_n, new_n), dtype=np.uint8)
        new_adj[:n, :n] = adj
        new_adj[:n, n] = best_S
        new_adj[n, :n] = best_S
        new_adj[n, n] = 0
        
        # Update k3s and ind_3s
        new_k3s = list(k3s)
        new_ind_3s = list(ind_3s)
        
        # Find new K3s involving the new vertex n
        # These are triangles (u, w, n) where u, w in S and u~w in adj
        for u in range(n):
            if best_S[u] == 1:
                for w in range(u+1, n):
                    if best_S[w] == 1 and adj[u][w] == 1:
                        new_k3s.append(frozenset([u, w, n]))
                        
        # Find new Ind3s involving the new vertex n
        # These are independent triangles (u, w, n) where u, w in ~S and u!~w in adj
        for u in range(n):
            if best_S[u] == 0:
                for w in range(u+1, n):
                    if best_S[w] == 0 and adj[u][w] == 0:
                        new_ind_3s.append(frozenset([u, w, n]))
                        
        return new_adj, new_n, new_k3s, new_ind_3s, True

    # Try to add 5 vertices
    adj = current_adj.copy()
    n = current_n
    k3s = current_k3s
    ind_3s = current_ind_3s
    
    for _ in range(5):
        if time.time() > deadline:
            break
        new_adj, new_n, new_k3s, new_ind_3s, success = add_vertex(adj, n, k3s, ind_3s, deadline)
        if success:
            adj = new_adj
            n = new_n
            k3s = new_k3s
            ind_3s = new_ind_3s
        else:
            # If we fail to add a vertex, we might try to restart the addition process
            # or just stop.
            # Let's try a few restarts for this specific vertex addition.
            found = False
            for attempt in range(20):
                if time.time() > deadline:
                    break
                # Randomize the initial S in add_vertex? 
                # The add_vertex function already randomizes S.
                # But it only does 1000 steps.
                # Let's just call it again.
                new_adj, new_n, new_k3s, new_ind_3s, success = add_vertex(adj, n, k3s, ind_3s, deadline)
                if success:
                    adj = new_adj
                    n = new_n
                    k3s = new_k3s
                    ind_3s = new_ind_3s
                    found = True
                    break
            if not found:
                break
                
    # If we reached 42, we are done.
    # If not, we return the current graph.
    # We must ensure the graph is valid.
    # The construction guarantees validity if we only add valid vertices.
    # But let's double check with a full scan if n > 37.
    
    if n > 37:
        # Full verification
        def has_k5(adj, n):
            for combo in combinations(range(n), 5):
                i, j, k, l, m = combo
                if adj[i][j] and adj[i][k] and adj[i][l] and adj[i][m] and \
                   adj[j][k] and adj[j][l] and adj[j][m] and \
                   adj[k][l] and adj[k][m] and \
                   adj[l][m]:
                    return True
            return False

        def has_ind_5(adj, n):
            non_adj = 1 - adj
            np.fill_diagonal(non_adj, 0)
            for combo in combinations(range(n), 5):
                i, j, k, l, m = combo
                if non_adj[i][j] and non_adj[i][k] and non_adj[i][l] and non_adj[i][m] and \
                   non_adj[j][k] and non_adj[j][l] and non_adj[j][m] and \
                   non_adj[k][l] and non_adj[k][m] and \
                   non_adj[l][m]:
                    return True
            return False

        if has_k5(adj, n) or has_ind_5(adj, n):
            # Fallback to base graph
            adj = A
            n = q

    # Convert to edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j] == 1:
                edges.append((i, j))
                
    return n, edges

# EVOLVE-BLOCK-END
