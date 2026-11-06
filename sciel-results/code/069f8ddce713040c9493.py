import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = 18
    # Graph representation: list of sets of neighbors
    # Or bitmask for speed? 18 bits fits in int.
    # Let's use a list of 18 ints, where bit i is set if edge (v, i) exists.
    
    def is_k4_free(adj, u, v):
        # Check if adding edge (u, v) creates a K4
        # Common neighbors of u and v
        common = adj[u] & adj[v]
        # If there is an edge between any two common neighbors, K4 is formed
        # Iterate through common neighbors
        c = common
        while c:
            # Get lowest set bit
            lsb = c & -c
            w = lsb.bit_length() - 1
            c ^= lsb
            # Check if w is connected to any other common neighbor
            # We need to check if adj[w] & (common & ~(1 << w)) is not 0
            # Actually, just check if adj[w] & common has any bit other than w
            # Since w is in common, adj[w] & common will have w set if loop? No, adj[w] doesn't have self loop.
            # So check if (adj[w] & common) != 0
            if (adj[w] & common) != 0:
                return False
        return True

    def get_triangles(adj):
        triangles = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    # Common neighbors of i and j
                    common = adj[i] & adj[j]
                    # Filter for k > j to avoid duplicates? 
                    # Or just store (i,j,k) with i<j<k
                    c = common
                    while c:
                        lsb = c & -c
                        k = lsb.bit_length() - 1
                        c ^= lsb
                        if i < j < k:
                            triangles.append((i, j, k))
                        elif i < k < j:
                            triangles.append((i, k, j))
                        elif k < i < j:
                            triangles.append((k, i, j))
        return triangles

    def min_vertex_cover(triangles):
        if not triangles:
            return 0
        
        # Branch and bound
        best = n  # Upper bound
        
        # Sort triangles to pick the one with fewest remaining options? 
        # All have 3.
        
        def solve(triangles, used_mask, depth):
            nonlocal best
            if depth >= best:
                return
            if not triangles:
                if depth < best:
                    best = depth
                return
            
            # Pick a triangle
            t = triangles[0]
            rest = triangles[1:]
            
            # Option 1: Pick t[0]
            solve(rest, used_mask | (1 << t[0]), depth + 1)
            # Option 2: Pick t[1]
            solve(rest, used_mask | (1 << t[1]), depth + 1)
            # Option 3: Pick t[2]
            solve(rest, used_mask | (1 << t[2]), depth + 1)
            
        # Optimization: Remove triangles that are already covered?
        # The above solve doesn't remove covered triangles.
        # We need to filter triangles that are fully covered by used_mask?
        # No, if a triangle is fully covered, it's satisfied.
        # We only need to cover triangles that are NOT fully covered.
        
        def solve_opt(triangles, used_mask, depth):
            nonlocal best
            if depth >= best:
                return
            
            # Filter triangles: keep only those not fully covered
            active = []
            for t in triangles:
                if not (used_mask & (1 << t[0])) and not (used_mask & (1 << t[1])) and not (used_mask & (1 << t[2])):
                    active.append(t)
            
            if not active:
                if depth < best:
                    best = depth
                return
            
            t = active[0]
            rest = active[1:]
            
            # Branch on t[0]
            solve_opt(rest, used_mask | (1 << t[0]), depth + 1)
            # Branch on t[1]
            solve_opt(rest, used_mask | (1 << t[1]), depth + 1)
            # Branch on t[2]
            solve_opt(rest, used_mask | (1 << t[2]), depth + 1)

        solve_opt(triangles, 0, 0)
        return best

    def evaluate(adj):
        triangles = get_triangles(adj)
        cover = min_vertex_cover(triangles)
        return n - cover  # This is the score (largest triangle-free induced subgraph)

    # Initialize with a random K4-free graph
    # Start with empty graph and add edges
    adj = [0] * n
    
    # Try to build a dense graph
    random.seed(42)
    for _ in range(1000):
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        if adj[u] & (1 << v):
            continue
        if is_k4_free(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
    current_score = evaluate(adj)
    best_score = current_score
    best_adj = adj[:]
    
    # Local Search
    while time.time() < deadline:
        # Pick a random edge to remove or non-edge to add
        if random.random() < 0.5:
            # Add edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v or (adj[u] & (1 << v)):
                continue
            if is_k4_free(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                new_score = evaluate(adj)
                if new_score < best_score:
                    best_score = new_score
                    best_adj = adj[:]
                else:
                    # Revert
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
        else:
            # Remove edge
            # Pick a random edge
            edges = []
            for i in range(n):
                c = adj[i]
                while c:
                    lsb = c & -c
                    j = lsb.bit_length() - 1
                    c ^= lsb
                    if i < j:
                        edges.append((i, j))
            if not edges:
                continue
            u, v = random.choice(edges)
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            new_score = evaluate(adj)
            if new_score < best_score:
                best_score = new_score
                best_adj = adj[:]
            else:
                # Revert
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
    # Convert best_adj to list of edges
    result_edges = []
    for i in range(n):
        c = best_adj[i]
        while c:
            lsb = c & -c
            j = lsb.bit_length() - 1
            c ^= lsb
            if i < j:
                result_edges.append([i, j])
                
    return result_edges
# EVOLVE-BLOCK-END
