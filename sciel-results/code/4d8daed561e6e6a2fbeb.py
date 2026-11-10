"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START
def _is_k23_free(adj, n):
    """Check if graph with adjacency matrix adj (n x n, symmetric, 0/1) is K2,3-free.
    K2,3 exists iff two vertices have >=3 common neighbors."""
    for i in range(n):
        for j in range(i + 1, n):
            # Count common neighbors
            common = 0
            for k in range(n):
                if adj[i][k] and adj[j][k]:
                    common += 1
                    if common >= 3:
                        return False
    return True

def _count_edges(adj):
    return sum(sum(row) for row in adj) // 2

def _common_neighbors_count(adj, i, j):
    c = 0
    for k in range(len(adj)):
        if adj[i][k] and adj[j][k]:
            c += 1
    return c

def _can_add(adj, i, j, n):
    """Check if adding edge (i,j) keeps graph K2,3-free."""
    if i == j or adj[i][j]:
        return False
    # After adding (i,j), for any vertex k, the pair (i,k) and (j,k) must not have 3 common neighbors
    # More precisely: adding edge (i,j) means i and j become neighbors.
    # We need to check: for all pairs (u,v) in the graph, common neighbors < 3.
    # The only new pairs that could be affected are pairs involving i or j that weren't before.
    # Actually, the constraint is: no two vertices have 3 common neighbors.
    # When we add edge (i,j), the adjacency of i and j changes.
    # For any vertex k != i,j: 
    #   - pair (i,k): common neighbors include j now (since j is neighbor of i, and if j is neighbor of k). 
    #   - pair (j,k): similarly.
    #   - pair (i,j): common neighbors are vertices adjacent to both i and j.
    # Let's just check all pairs involving i or j after the addition.
    
    # Check pair (i, j): common neighbors
    common_ij = 0
    for k in range(n):
        if k != i and k != j and adj[i][k] and adj[j][k]:
            common_ij += 1
            if common_ij >= 3:
                return False
    
    # For each k != i, j: check pair (i, k)
    for k in range(n):
        if k == i or k == j:
            continue
        # Pair (i, k): common neighbors are vertices adjacent to both i and k
        # After adding (i,j), j is a neighbor of i. If j is also a neighbor of k, then j is a common neighbor.
        common_ik = 0
        for m in range(n):
            if m == i or m == k:
                continue
            # Is m adjacent to i? (including new edge if m==j)
            adj_im = adj[i][m] or (m == j)
            adj_km = adj[k][m]
            if adj_im and adj_km:
                common_ik += 1
                if common_ik >= 3:
                    return False
        
        # Pair (j, k)
        common_jk = 0
        for m in range(n):
            if m == j or m == k:
                continue
            adj_jm = adj[j][m] or (m == i)
            adj_km = adj[k][m]
            if adj_jm and adj_km:
                common_jk += 1
                if common_jk >= 3:
                    return False
    
    return True

def construct():
    n = 20
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    # Start with the seed: 5 disjoint K4's
    adj = [[0] * n for _ in range(n)]
    for k in range(5):
        for a in range(4):
            for b in range(a + 1, 4):
                u, v = 4 * k + a, 4 * k + b
                adj[u][v] = adj[v][u] = 1
    
    best_adj = [row[:] for row in adj]
    best_edges = _count_edges(adj)
    
    # Greedy phase: try to add edges
    for i in range(n):
        for j in range(i + 1, n):
            if not adj[i][j]:
                if _can_add(adj, i, j, n):
                    adj[i][j] = adj[j][i] = 1
                    edges = _count_edges(adj)
                    if edges > best_edges:
                        best_edges = edges
                        best_adj = [row[:] for row in adj]
    
    # Local search: try to add/remove/swap edges to improve
    random.seed(42)
    for iteration in range(1000):
        if time.time() > deadline:
            break
        
        improved = False
        # Try adding a random missing edge
        for _ in range(200):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j or adj[i][j]:
                continue
            if _can_add(adj, i, j, n):
                adj[i][j] = adj[j][i] = 1
                edges = _count_edges(adj)
                if edges > best_edges:
                    best_edges = edges
                    best_adj = [row[:] for row in adj]
                improved = True
                break
        
        if not improved:
            # Try a random swap: remove edge (a,b), add edge (c,d)
            edges_list = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i][j]]
            if not edges_list:
                break
            a, b = random.choice(edges_list)
            # Try to find a non-edge to add
            found = False
            for _ in range(50):
                c = random.randint(0, n - 1)
                d = random.randint(0, n - 1)
                if c == d or adj[c][d] or (c, d) == (a, b) or (d, c) == (a, b):
                    continue
                # Remove (a,b)
                adj[a][b] = adj[b][a] = 0
                if _can_add(adj, c, d, n):
                    adj[c][d] = adj[d][c] = 1
                    edges = _count_edges(adj)
                    if edges > best_edges:
                        best_edges = edges
                        best_adj = [row[:] for row in adj]
                    found = True
                    break
                # Restore
                adj[a][b] = adj[b][a] = 1
            
            if not found:
                # Try removing an edge to allow more additions later? 
                # For now, just restart with a different random order
                adj = [row[:] for row in best_adj]
    
    # Extract edges from best_adj
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i][j]:
                edges.append([i, j])
    
    return edges
# EVOLVE-BLOCK-END
