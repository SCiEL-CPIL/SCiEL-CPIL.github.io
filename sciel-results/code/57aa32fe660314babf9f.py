import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5 second margin for safety
    
    best_n = 0
    best_edges = []
    
    # We know R(3,10) <= 41, so max n is 40.
    # Known largest is 39.
    # We try to find a valid graph for n from 40 down to 30.
    # If we find one, we return it.
    # If we don't find a valid one for a specific n within time, we move to n-1.
    
    # To maximize score, we should prioritize finding high n.
    # However, random search might fail for high n.
    # We can use a hybrid: try a few fast heuristics for high n, then fallback to lower n if needed.
    
    # Strategy:
    # 1. Try to construct a valid graph for n=40, 39, 38... using local search.
    # 2. Local search: Start with random dense graph.
    #    - While triangle exists: remove an edge from a random triangle.
    #    - This ensures triangle-free.
    #    - Check alpha. If alpha >= 10, try to add edges?
    #      Adding edges to a triangle-free graph is hard (must maintain triangle-free).
    #      Alternative: Start with a specific known structure or random regular graph.
    
    # A better strategy for triangle-free graphs:
    # Use a "random Turan-like" or "random blow-up" of a small triangle-free graph?
    # Or simply: Random graph G(n, p), then iteratively remove edges to kill triangles.
    # If we start with high p, we get many triangles. Removing edges reduces density.
    # We want high density to keep alpha low.
    
    # Let's try:
    # 1. Generate random graph G(n, 0.5).
    # 2. While triangle exists: pick a random triangle, remove one of its edges randomly.
    # 3. This results in a triangle-free graph.
    # 4. Check if alpha < 10.
    # 5. If alpha >= 10, we can try to "repair" by adding edges?
    #    Adding an edge (u,v) is valid if no common neighbor exists.
    #    So, we can scan for non-adjacent pairs (u,v) with no common neighbor and add them.
    #    This increases density, potentially reducing alpha.
    
    def has_triangle(adj):
        # Fast check using bitsets or numpy
        # For n<=40, simple check is fast enough if called sparingly
        # But we call it in a loop.
        # Let's use a bitset approach for the repair loop.
        pass

    def remove_random_triangle(adj, n):
        # Find a triangle and remove one edge
        # Returns True if a triangle was found and removed
        # Use random sampling of triples? O(n^3) is 40^3 = 64000, which is fast.
        # But we can do better.
        # Let's just pick 3 random vertices and check if they form a triangle.
        # Repeat until found.
        # Since graph is dense initially, probability is high.
        
        # To be robust, we can iterate through all edges and check neighbors.
        # For simplicity and speed in Python, let's use a random sampling approach first.
        # If it takes too long, fall back to systematic.
        
        # Actually, a systematic scan is O(n^3) worst case, which is ~64k ops.
        # Doing this 100 times is 6.4M ops. Very fast in C/NumPy, but Python loop is slow.
        # Let's use a bitset to find triangles faster.
        
        # Precompute neighbors as bitsets
        # This is complex to maintain dynamically.
        
        # Alternative: Just use the fact that n is small.
        # We'll do a random check.
        for _ in range(1000):
            u, v, w = random.sample(range(n), 3)
            if adj[u, v] and adj[v, w] and adj[w, u]:
                # Remove one edge randomly
                r = random.random()
                if r < 1/3:
                    adj[u, v] = False
                    adj[v, u] = False
                elif r < 2/3:
                    adj[v, w] = False
                    adj[w, v] = False
                else:
                    adj[w, u] = False
                    adj[u, w] = False
                return True
        return False

    def check_alpha_less_than_10(adj, n):
        # Exact branch and bound for alpha
        # Returns True if alpha < 10
        
        neighbors = [0] * n
        for i in range(n):
            mask = 0
            for j in range(n):
                if adj[i, j]:
                    mask |= (1 << j)
            neighbors[i] = mask
        
        found_large = [False]
        
        def dfs(vertices_mask, current_size):
            if found_large[0]:
                return
            if current_size >= 10:
                found_large[0] = True
                return
            
            if vertices_mask == 0:
                return
            
            # Pruning
            v_count = bin(vertices_mask).count('1')
            if current_size + v_count < 10:
                return
            
            # Pick vertex with min degree in subgraph?
            # For speed, just pick first set bit
            lsb = vertices_mask & -vertices_mask
            v = lsb.bit_length() - 1
            
            # Branch 1: Include v
            new_mask = vertices_mask & ~(neighbors[v] | lsb)
            dfs(new_mask, current_size + 1)
            
            if found_large[0]:
                return
            
            # Branch 2: Exclude v
            new_mask = vertices_mask & ~lsb
            dfs(new_mask, current_size)
        
        dfs((1 << n) - 1, 0)
        return not found_large[0]

    def add_valid_edges(adj, n):
        # Try to add edges to increase density
        # An edge (u,v) can be added if adj[u,v] is False and they have no common neighbor.
        # common_neighbor(u,v) = neighbors[u] & neighbors[v]
        
        # Precompute neighbor bitsets for speed
        neighbors = [0] * n
        for i in range(n):
            mask = 0
            for j in range(n):
                if adj[i, j]:
                    mask |= (1 << j)
            neighbors[i] = mask
            
        added = 0
        # Iterate over all non-edges
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u, v]:
                    # Check common neighbors
                    if (neighbors[u] & neighbors[v]) == 0:
                        # Add edge
                        adj[u, v] = True
                        adj[v, u] = True
                        # Update neighbors
                        neighbors[u] |= (1 << v)
                        neighbors[v] |= (1 << u)
                        added += 1
        return added

    def get_edges_from_adj(adj, n):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    edges.append([i, j])
        return edges

    # Main search loop
    # We prioritize higher n.
    # For each n, we try several random instances.
    
    for n in range(40, 29, -1):
        if time.time() > deadline:
            break
            
        # Try a few attempts for this n
        attempts = 20 if n > 35 else 50
        
        for _ in range(attempts):
            if time.time() > deadline:
                break
                
            # Initialize random graph
            adj = np.zeros((n, n), dtype=bool)
            p = 0.5
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < p:
                        adj[i, j] = True
                        adj[j, i] = True
                        
            # Phase 1: Remove triangles
            # Limit iterations to avoid infinite loops if stuck
            # But random removal should eventually clear triangles.
            for _ in range(10000):
                if not remove_random_triangle(adj, n):
                    break
                    
            # Verify no triangles (sanity check, maybe skip for speed if confident)
            # If we failed to remove all triangles, this graph is invalid.
            # The remove function returns False if no triangle found in 1000 tries.
            # It's possible triangles remain.
            # Let's do a quick check.
            has_tri = False
            for u in range(n):
                for v in range(u+1, n):
                    if adj[u, v]:
                        # Check common neighbors
                        # Using bitsets would be faster, but let's do a simple check
                        # This is O(n^3) again.
                        # For n=40, this is 64k checks.
                        # We do this once per candidate.
                        for w in range(v+1, n):
                            if adj[v, w] and adj[w, u]:
                                has_tri = True
                                break
                        if has_tri:
                            break
                if has_tri:
                    break
            
            if has_tri:
                continue
                
            # Phase 2: Maximize density (add valid edges)
            # This helps reduce alpha
            add_valid_edges(adj, n)
            
            # Check alpha
            if check_alpha_less_than_10(adj, n):
                best_n = n
                best_edges = get_edges_from_adj(adj, n)
                return {'n': n, 'edges': best_edges}
                
    # Fallback: Return a known small valid graph if we didn't find a good one
    # We need a valid graph.
    # A complete bipartite graph K_{5,5} has n=10, alpha=5.
    # K_{9,9} has n=18, alpha=9.
    # Let's construct K_{9,9} as a safe fallback.
    # Vertices 0-8 in A, 9-17 in B.
    # Edges between A and B.
    n = 18
    edges = []
    for u in range(9):
        for v in range(9, 18):
            edges.append([u, v])
            
    return {'n': n, 'edges': edges}

# Helper to ensure we don't crash if time runs out
# The main loop has a deadline.
# If we exit the loop without returning, we hit the fallback.

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Found n={res['n']}, edges={len(res['edges'])}")
