import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Fallback: The seed solution (n=28)
    def get_seed_graph():
        n = 28
        edges = []
        for b in range(7):
            for i in range(4):
                for j in range(i + 1, 4):
                    edges.append([4 * b + i, 4 * b + j])
        return {'n': n, 'edges': edges}

    # Helper: Check if graph is C5-free
    def is_c5_free(n, adj):
        # adj is n x n boolean numpy array
        # A graph is C5-free if it has no simple cycle of length 5.
        # We can check for any 5-cycle by looking at paths of length 4.
        # If there is a path v1-v2-v3-v4-v5 and v5-v1 is an edge, it's a C5.
        # To avoid backtracking (v1-v2-v1-v2-v1), we need simple paths.
        # Efficient way: 
        # For each pair (u, v) not connected, check if there is a path of length 4.
        # Actually, just check if A^5 has any off-diagonal elements? No.
        # Let's use a simple DFS/BFS for small n (<=35).
        # Or use the property: C5 exists iff there is a cycle of length 5.
        # We can iterate over all 5-tuples? 35^5 is too big.
        # Iterate over edges (u, v). Check if there is a path of length 4 from v to u in G - (u,v).
        # This is O(E * N^2) roughly. 35^2 * 35^2 ~ 1.5M ops. Very fast.
        
        # Optimization: Precompute paths of length 2 and 3?
        # Let's just do a quick check using numpy for small n.
        # A^2 counts paths of length 2.
        # A^4 counts paths of length 4.
        # If (A^4)[u, v] > 0 and adj[u, v] == 1, it might be a C5 or a shorter cycle with backtracking.
        # Since we are removing edges to break C5s, we can just check for existence.
        
        # Simpler: Just use a library-free check for n <= 35.
        # For each node u, do BFS up to depth 4?
        # Actually, we can just check if the graph contains a cycle of length 5.
        
        # Let's use a simple recursive check for C5 for the specific n.
        # But in a loop, this might be slow.
        # Let's use the matrix power trick with a caveat:
        # If we ensure the graph is triangle-free and C4-free, then any closed walk of length 5 is a C5.
        # But we don't require triangle-free.
        
        # Let's do a direct check for C5.
        # For each pair (i, j) with i < j and adj[i, j] == 1:
        #   Check if there is a path of length 4 from j to i in the graph with edge (i, j) removed.
        #   This can be done by checking if (A_with_edge_removed^4)[j, i] > 0.
        #   But A^4 is expensive to recompute.
        
        # Alternative:
        # We can maintain the graph and just check for C5 when adding/removing edges.
        # For the search, we will use a probabilistic approach.
        
        # Let's implement a fast C5 checker using bitsets.
        # For each node u, let N(u) be its neighbors.
        # A C5 is u-v-w-x-y-u.
        # We can check for each pair (u, v) in E, if there is a path v->...->u of length 4.
        
        # Given n is small, we can just use a simple DFS for each edge.
        # But we will call this many times.
        
        # Let's use a heuristic: 
        # If we are in the "remove edge to break C5" phase, we know which C5 to break.
        # So we only need to check if the graph is C5-free after removal.
        # We can just re-run the check.
        
        # For n=32, O(N^5) is 33M. In Python, this is too slow if done many times.
        # We need a faster check.
        
        # Let's use the fact that C5-free is equivalent to:
        # For all u, v, if (u, v) in E, then there is no path of length 4 from v to u in G-(u,v).
        # We can precompute paths of length 2 (A^2) and 3 (A^3).
        # Path of length 4 from v to u is sum over w of (A^3)[v, w] * A[w, u]?
        # (A^4)[v, u] = sum_w (A^3)[v, w] A[w, u].
        # We can compute A^4 in O(N^3).
        # For N=32, 32^3 = 32768. Very fast.
        
        A4 = np.dot(np.dot(adj, adj), np.dot(adj, adj))
        # A4[v, u] > 0 means there is a walk of length 4 from v to u.
        # If adj[v, u] == 1, it could be a C5, or a C3 with a backtrack, or a C4 with a backtrack, etc.
        # However, if we are trying to find *any* C5, and we know the graph might have C3, C4,
        # this check is not exact.
        
        # To be exact, we can just use the seed graph as a safe fallback and try to improve.
        # If the check is not exact, we might return an invalid graph.
        # The evaluator checks "exactly". So we must be exact.
        
        # Let's write an exact C5 checker for N <= 35.
        # It will be called a limited number of times.
        
        def has_c5(n, adj):
            # adj is list of lists or numpy array
            # We check for each edge (u, v) if there is a path of length 4 from v to u.
            # We use DFS with depth limit 4.
            # To speed up, we only explore from v.
            
            # Precompute neighbors
            neighbors = [np.where(adj[i] == 1)[0] for i in range(n)]
            
            for u in range(n):
                for v in neighbors[u]:
                    if u >= v: continue # check each edge once
                    # Check if there is a path v -> x -> y -> z -> u
                    # x in N(v), x != u
                    # y in N(x), y != v, y != u
                    # z in N(y), z != x, z != u
                    # u in N(z)
                    
                    found = False
                    for x in neighbors[v]:
                        if x == u: continue
                        for y in neighbors[x]:
                            if y == v or y == u: continue
                            for z in neighbors[y]:
                                if z == x or z == u: continue
                                if adj[z][u] == 1:
                                    return True
                    # If no path found for this edge, continue
            return False
        
        return has_c5(n, adj)

    # Helper: Check if alpha < 9
    def has_independent_set_of_size_9(n, adj):
        # Greedy + Backtracking
        # For n=32, we can use a simple backtracking.
        
        # Convert to bitmasks for speed
        masks = [0] * n
        for i in range(n):
            mask = 0
            for j in range(n):
                if adj[i][j] == 1:
                    mask |= (1 << j)
            masks[i] = mask | (1 << i) # Self mask? No, independent set doesn't include self-edge logic in this way.
            # Actually, mask[i] should be the set of neighbors of i.
            # If we pick i, we cannot pick any neighbor.
            
        # Let's use a recursive function
        def backtrack(start, current_set, size):
            if size == 9:
                return True
            if start == n:
                return False
            
            # Option 1: Exclude vertex 'start'
            if backtrack(start + 1, current_set, size):
                return True
            
            # Option 2: Include vertex 'start'
            # Check if 'start' is compatible with current_set
            # current_set is a bitmask of chosen vertices
            # We need to ensure no edge between 'start' and any in current_set
            # And no edge between 'start' and any future chosen vertices (handled by pruning)
            
            # Simple check:
            compatible = True
            for v in range(start):
                if (current_set >> v) & 1:
                    if adj[start][v] == 1:
                        compatible = False
                        break
            if not compatible:
                return False
            
            # Pruning: If we can't reach size 9 with remaining vertices, return False
            remaining = n - start - 1
            if size + 1 + remaining < 9:
                return False
            
            if backtrack(start + 1, current_set | (1 << start), size + 1):
                return True
            
            return False
        
        return backtrack(0, 0, 0)

    # Search for a better graph
    best_n = 28
    best_graph = get_seed_graph()
    
    # Try to find a graph with n > 28
    # We will try n = 32, 31, 30, 29
    for target_n in [32, 31, 30, 29]:
        if time.time() > deadline - 20:
            break
            
        found = False
        # Try multiple random restarts
        for restart in range(50):
            if time.time() > deadline - 10:
                break
                
            # Initialize random graph
            # Use a moderate edge probability
            p = 0.5 
            adj = np.zeros((target_n, target_n), dtype=int)
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < p:
                        adj[i][j] = 1
                        adj[j][i] = 1
            
            # Local search
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                
                # Check C5
                if is_c5_free(target_n, adj):
                    # Check Alpha
                    if not has_independent_set_of_size_9(target_n, adj):
                        # Valid!
                        edges = [[i, j] for i in range(target_n) for j in range(i + 1, target_n) if adj[i][j] == 1]
                        best_n = target_n
                        best_graph = {'n': target_n, 'edges': edges}
                        found = True
                        break
                    else:
                        # Alpha >= 9, need to add edges
                        # Find a random independent set of size 9? 
                        # Just add a random edge that doesn't create C5
                        added = False
                        for _ in range(100):
                            i = random.randint(0, target_n - 1)
                            j = random.randint(0, target_n - 1)
                            if i != j and adj[i][j] == 0:
                                # Check if adding (i, j) creates C5
                                # This is hard to check quickly.
                                # Let's just add it and if C5 appears, we'll remove it in next iteration.
                                adj[i][j] = 1
                                adj[j][i] = 1
                                added = True
                                break
                        if added:
                            improved = True
                else:
                    # Has C5, need to remove edges
                    # Find a C5 and remove a random edge from it
                    # This requires finding a C5.
                    # For now, just remove a random edge.
                    # This is slow but might work.
                    i = random.randint(0, target_n - 1)
                    j = random.randint(0, target_n - 1)
                    if i != j and adj[i][j] == 1:
                        adj[i][j] = 0
                        adj[j][i] = 0
                        improved = True
                        
            if found:
                break
                
    return best_graph

construct()
