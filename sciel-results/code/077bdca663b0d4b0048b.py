import random
from itertools import combinations

def construct():
    n = 39
    random.seed(42)
    
    # Use bitsets for fast operations
    # adj[i] is an integer where bit j is set if (i,j) is an edge
    
    def create_random_triangle_free(n):
        adj = [0] * n
        # Start with a random graph, then remove edges to break triangles? 
        # Or add edges carefully.
        # Adding edges carefully is O(n^3) if not optimized.
        # Let's start with a random bipartite graph (triangle free) and try to add intra-part edges.
        # Split vertices into two sets A and B.
        # Random edges between A and B.
        # Then try to add edges within A and within B, ensuring no triangles.
        
        # Actually, a better start is a random "blow-up" of C5 or similar.
        # But let's stick to simple random triangle-free generation.
        
        # Generate random permutation
        perm = list(range(n))
        random.shuffle(perm)
        
        # We will build the graph by adding edges in a random order.
        # To be triangle-free, if we add (u,v), we must ensure no common neighbor.
        # This is equivalent to building a graph where the neighborhood of any vertex is an independent set.
        
        # Strategy: Assign each vertex a "color" from a large palette? No.
        
        # Let's use the "random greedy" triangle-free graph generation.
        # Order vertices randomly.
        # For each vertex v, connect to a random subset of previous vertices such that no triangle is formed.
        # This ensures triangle-freeness.
        
        adj = [0] * n
        for i in range(n):
            # Connect i to some j < i
            # If we connect i to j, we must not connect i to any neighbor of j that is also connected to i?
            # No, if we process i, and connect to j < i, we must ensure that i is not connected to any k < i such that (j,k) is an edge.
            # So, for a fixed i, the set of neighbors among {0..i-1} must be an independent set in the subgraph induced by {0..i-1}.
            
            # This is complex to do randomly with high density.
            pass
        
        # Alternative: Start with empty graph, add random edges if triangle-free.
        # This is what the current code does, but slow.
        # Optimization: Use bitsets.
        
        adj = [0] * n
        for _ in range(n * n): # Try to add many edges
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v or (adj[u] >> v) & 1:
                continue
            # Check if adding (u,v) creates a triangle
            if (adj[u] & adj[v]) == 0:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def has_independent_set_of_size(adj, n, k=10):
        # Bitset based branch and bound
        # candidates is a bitmask of available vertices
        # current is the size of the current independent set
        def dfs(candidates, current_size):
            if current_size >= k:
                return True
            if candidates == 0:
                return False
            
            # Pick a vertex, e.g., the one with lowest index or random
            # Heuristic: Pick vertex with fewest neighbors in candidates to prune faster?
            # Or just pick the first one.
            v = (candidates & -candidates).bit_length() - 1
            # Remove v from candidates
            cand_without_v = candidates ^ (1 << v)
            
            # Branch 1: Include v
            # New candidates are those in cand_without_v that are NOT neighbors of v
            new_candidates = cand_without_v & ~adj[v]
            if dfs(new_candidates, current_size + 1):
                return True
            
            # Branch 2: Exclude v
            if dfs(cand_without_v, current_size):
                return True
            
            return False
        
        full_mask = (1 << n) - 1
        return dfs(full_mask, 0)

    # Local Search
    # Start with a random triangle-free graph
    adj = create_random_triangle_free(n)
    
    # Check if it's already good
    if not has_independent_set_of_size(adj, n, 10):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}
    
    # Try to improve
    # We want to reduce alpha.
    # Adding edges reduces alpha (or keeps it same), removing edges increases alpha.
    # So we should try to ADD edges.
    
    for step in range(10000):
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v or ((adj[u] >> v) & 1):
            continue
        if (adj[u] & adj[v]) == 0:
            # Add edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if not has_independent_set_of_size(adj, n, 10):
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if (adj[i] >> j) & 1:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
                
            # If it doesn't work, maybe revert? 
            # Adding edges is monotonic for alpha (non-increasing). 
            # So if we add an edge and alpha is still >= 10, we are not worse off.
            # We can just keep it.
    
    # Fallback: Return a known valid graph (e.g., n=26 Andrásfai)
    # But we want to maximize n.
    # If we fail to find n=39, we should try n=38, 37...
    # The current code returns 26 as fallback.
    # I will return the best found or the fallback.
    
    # Since the random search might not find 39, and we don't have a hardcoded 39,
    # and the time limit is 120s, the local search might run out of time.
    
    # Let's ensure we return *something* valid.
    # The 26-vertex Andrásfai graph is valid.
    n_fallback = 26
    edges_fallback = []
    for i in range(n_fallback):
        for j in range(i+1, n_fallback):
            if (j - i) % n_fallback % 3 == 1:
                edges_fallback.append([i, j])
                
    return {'n': n_fallback, 'edges': edges_fallback}
