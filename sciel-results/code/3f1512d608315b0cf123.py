"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys
from itertools import combinations

# EVOLVE-BLOCK-START
def _is_triangle_free(adj, n):
    """Check if graph with adjacency matrix adj (list of bytearrays) is triangle-free."""
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                # Check if any common neighbor
                # Using bit operations would be faster, but for n=16 this is fine
                ai = adj[i]
                aj = adj[j]
                for k in range(n):
                    if ai[k] and aj[k]:
                        return False
    return True

def _is_3_colorable(adj, n):
    """Check if graph is 3-colorable via backtracking."""
    colors = [-1] * n
    adj_list = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                adj_list[i].append(j)
                adj_list[j].append(i)
    
    # Order vertices by degree (descending) for better pruning
    order = sorted(range(n), key=lambda x: -len(adj_list[x]))
    # Map original vertex to position in order
    pos_in_order = [0] * n
    for i, v in enumerate(order):
        pos_in_order[v] = i
    
    # We color in the order of 'order'
    # But for backtracking, we need to check neighbors that are already colored.
    # It's easier to just iterate 0..n-1 but use the ordering to pick the next uncolored vertex with most constraints.
    
    def backtrack(idx):
        if idx == n:
            return True
        
        # Find the next uncolored vertex with the most colored neighbors (MRV heuristic)
        best_v = -1
        best_score = -1
        for v in range(n):
            if colors[v] == -1:
                # Count colored neighbors
                cnt = 0
                for u in adj_list[v]:
                    if colors[u] != -1:
                        cnt += 1
                if cnt > best_score:
                    best_score = cnt
                    best_v = v
        
        if best_v == -1:
            return True # All colored
            
        v = best_v
        used = [False] * 3
        for u in adj_list[v]:
            if colors[u] != -1:
                used[colors[u]] = True
        
        for c in range(3):
            if not used[c]:
                colors[v] = c
                if backtrack(idx + 1):
                    return True
                colors[v] = -1
        return False
    
    return backtrack(0)

def _edge_count(adj, n):
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                count += 1
    return count

def _adj_to_edges(adj, n):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append([i, j])
    return edges

def construct():
    N = 16
    deadline = time.time() + 140.0  # Safety margin
    
    # Start with a known valid solution: Mycielski of C5 (Grötzsch) + isolated vertices
    # Grötzsch graph has 11 vertices, 20 edges.
    # Vertices 0-4: cycle 0-1-2-3-4-0
    # Vertex 5: connected to 0, 2, 4 (neighbors of 0,1,2? No. Mycielski: v_i' connected to v_{i-1}, v_{i+1} of original? 
    # Standard Mycielski:
    # Original vertices 0..4
    # New vertices 5..9 (copies of 0..4)
    # Copy i (i+5) is connected to neighbors of i in original.
    # Neighbors of 0: 1, 4 -> 5 connected to 1, 4
    # Neighbors of 1: 0, 2 -> 6 connected to 0, 2
    # Neighbors of 2: 1, 3 -> 7 connected to 1, 3
    # Neighbors of 3: 2, 4 -> 8 connected to 2, 4
    # Neighbors of 4: 3, 0 -> 9 connected to 3, 0
    # Vertex 10 connected to all copies 5..9.
    
    start_adj = [bytearray(N) for _ in range(N)]
    
    # Cycle 0-4
    for i in range(5):
        j = (i + 1) % 5
        start_adj[i][j] = 1
        start_adj[j][i] = 1
    
    # Copies
    neighbors = {0: [1, 4], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 0]}
    for i in range(5):
        for nb in neighbors[i]:
            start_adj[5+i][nb] = 1
            start_adj[nb][5+i] = 1
            
    # Top vertex 10
    for i in range(5):
        start_adj[10][5+i] = 1
        start_adj[5+i][10] = 1
        
    # Vertices 11-15 are isolated.
    
    best_adj = [row[:] for row in start_adj]
    best_edges = _edge_count(best_adj, N)
    
    # Verify start
    if not _is_triangle_free(best_adj, N) or _is_3_colorable(best_adj, N):
        # Should not happen, but if it does, return empty or something
        return _adj_to_edges(best_adj, N)
        
    # Local search / Branch and Bound
    # Strategy:
    # 1. Try to add edges greedily? No, we want to maximize.
    # 2. Use a local search with moves:
    #    - Add an edge if it doesn't create a triangle.
    #    - If stuck, try "remove one, add two" or "remove two, add three" moves to escape local optima.
    #    - Or use simulated annealing / hill climbing with restarts.
    
    # Since N=16 is small, we can afford a more thorough search.
    # Let's use a simple stochastic local search with backtracking for 3-colorability check.
    
    # We will maintain a set of candidate edges.
    # Actually, let's just manipulate the adjacency matrix directly.
    
    def try_improve(adj, time_limit):
        current_edges = _edge_count(adj, N)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try adding any possible edge
            for i in range(N):
                for j in range(i+1, N):
                    if not adj[i][j]:
                        # Check if adding (i,j) creates a triangle
                        creates_triangle = False
                        for k in range(N):
                            if adj[i][k] and adj[j][k]:
                                creates_triangle = True
                                break
                        if not creates_triangle:
                            # Check 3-colorability
                            adj[i][j] = 1
                            adj[j][i] = 1
                            if not _is_3_colorable(adj, N):
                                current_edges += 1
                                improved = True
                                # Restart scanning from beginning? Or just continue.
                                # Continuing is fine, we'll loop again.
                                break
                            else:
                                # Revert
                                adj[i][j] = 0
                                adj[j][i] = 0
                if improved:
                    break
            if time.time() > time_limit:
                break
        
        # If we couldn't add any edge, try more complex moves
        # Move: Remove edge (a,b), add (c,d) and (e,f)
        # This is expensive. Let's try a simpler "swap": Remove (a,b), add (c,d)
        # But removing an edge reduces count, adding one increases it. Net 0.
        # We need net positive.
        # "Remove 1, Add 2":
        # Pick an edge (a,b). Remove it.
        # Try to add two edges (c,d), (e,f) that are not present, don't create triangles (with each other or existing), and result in non-3-colorable graph.
        
        # Let's implement a limited version of this.
        if time.time() < time_limit:
            edges_list = []
            for i in range(N):
                for j in range(i+1, N):
                    if adj[i][j]:
                        edges_list.append((i, j))
                        
            random.shuffle(edges_list)
            
            for (a, b) in edges_list:
                if time.time() > time_limit:
                    break
                # Remove (a, b)
                adj[a][b] = 0
                adj[b][a] = 0
                
                # Now try to add two edges
                # Candidates for new edges are non-edges
                candidates = []
                for i in range(N):
                    for j in range(i+1, N):
                        if not adj[i][j] and not (i==a and j==b) and not (i==b and j==a):
                            candidates.append((i, j))
                
                # Limit candidates to those that don't immediately create triangles with existing edges?
                # Or just check on the fly.
                # To keep it fast, limit the number of candidate pairs we try.
                random.shuffle(candidates)
                
                found = False
                # Try pairs of candidates
                # This is O(M^2) which might be slow if M is large.
                # Let's just try adding one edge at a time in a nested loop but break early.
                
                for idx1 in range(min(20, len(candidates))): # Limit search space
                    c, d = candidates[idx1]
                    # Check if (c,d) can be added (no triangle)
                    ok1 = True
                    for k in range(N):
                        if adj[c][k] and adj[d][k]:
                            ok1 = False
                            break
                    if not ok1:
                        continue
                    
                    adj[c][d] = 1
                    adj[d][c] = 1
                    
                    for idx2 in range(idx1 + 1, min(20, len(candidates))):
                        e, f = candidates[idx2]
                        # Check if (e,f) can be added given (c,d) is now present
                        ok2 = True
                        for k in range(N):
                            if adj[e][k] and adj[f][k]:
                                ok2 = False
                                break
                        if not ok2:
                            continue
                        
                        adj[e][f] = 1
                        adj[f][e] = 1
                        
                        # Check 3-colorability
                        if not _is_3_colorable(adj, N):
                            # Success! We removed 1, added 2. Net +1 edge.
                            found = True
                            break
                        
                        # Revert (e,f)
                        adj[e][f] = 0
                        adj[f][e] = 0
                    
                    if found:
                        break
                    
                    # Revert (c,d)
                    adj[c][d] = 0
                    adj[d][c] = 0
                
                if found:
                    improved = True
                    break
                else:
                    # Revert (a,b)
                    adj[a][b] = 1
                    adj[b][a] = 1
                    
        return adj, _edge_count(adj, N)

    # Run local search multiple times with restarts
    # Since the space is small, we can do several iterations.
    
    current_adj = [row[:] for row in best_adj]
    
    # Main loop
    while time.time() < deadline:
        current_adj, current_edges = try_improve(current_adj, deadline)
        
        if current_edges > best_edges:
            best_adj = [row[:] for row in current_adj]
            best_edges = current_edges
        else:
            # If no improvement, restart from a perturbed state or the best state
            # Let's restart from best_adj with some random perturbations
            current_adj = [row[:] for row in best_adj]
            # Randomly remove a few edges to allow different paths
            for _ in range(3):
                i = random.randint(0, N-1)
                j = random.randint(i+1, N-1)
                if current_adj[i][j]:
                    current_adj[i][j] = 0
                    current_adj[j][i] = 0
                    
            # If we've been stuck for a while, maybe try a completely different random start?
            # But the Grötzsch base is strong.
            # Let's just continue.
            
    return _adj_to_edges(best_adj, N)
# EVOLVE-BLOCK-END
