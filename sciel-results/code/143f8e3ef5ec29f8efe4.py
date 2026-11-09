import sys
import random
import time

def has_independent_set_of_size_k(adj, n, k):
    """
    Checks if there is an independent set of size k in the graph.
    Uses branch and bound with greedy coloring upper bound.
    """
    # adj is a list of sets or lists of neighbors
    # For speed, use bitmasks if n <= 64? n=40 fits in int.
    # Let's use Python integers as bitmasks for adjacency.
    
    # Convert adj to bitmask representation for faster intersection
    # adj_mask[i] is an integer with bits set for neighbors of i
    
    def color_graph(candidates_mask):
        """
        Greedy coloring of the subgraph induced by candidates_mask.
        Returns the number of colors used.
        """
        colors = 0
        remaining = candidates_mask
        while remaining:
            colors += 1
            # Find a maximal independent set in the remaining graph
            # This is the same problem, but greedy is fast enough for upper bound
            # Actually, for coloring, we pick a vertex, remove it and its neighbors.
            # Wait, greedy coloring picks a vertex, assigns a color, removes it and its neighbors?
            # No, greedy coloring: 
            # 1. Pick a vertex v.
            # 2. Assign color.
            # 3. Remove v and all its neighbors from 'remaining' for this color class?
            # Yes, that forms an independent set (color class).
            
            # To be fast, pick the vertex with highest degree in the remaining graph?
            # Or just the first one.
            v = (remaining & -remaining).bit_length() - 1
            remaining &= ~(1 << v)
            # Remove neighbors of v
            remaining &= ~adj_mask[v]
            
        return colors

    # Precompute adj_mask
    adj_mask = [0] * n
    for i in range(n):
        for j in adj[i]:
            adj_mask[i] |= (1 << j)

    def backtrack(candidates_mask, current_size):
        if current_size == k:
            return True
        
        # Pruning: if current_size + number of bits in candidates < k
        if current_size + bin(candidates_mask).count('1') < k:
            return False
            
        # Upper bound using greedy coloring
        # If the chromatic number of the induced subgraph is less than (k - current_size), we can't fit
        # Actually, if we color the candidates, the max independent set size is <= number of colors?
        # No, max independent set is <= chromatic number? No.
        # Max independent set is >= n / chromatic_number.
        # We need an upper bound on the size of the independent set.
        # A simple bound is the number of vertices.
        # A better bound: Greedy coloring gives an upper bound on the clique number, not independent set.
        # For independent set, we can use the fact that if we color the graph, 
        # the largest independent set is at least n/chromatic.
        # This doesn't help for upper bound.
        
        # Better upper bound: 
        # If we take a maximal independent set greedily, its size is a lower bound.
        # We need an upper bound to prune.
        # A common bound is: if we order vertices, and for each vertex v, 
        # we count how many "later" vertices are non-adjacent to v.
        
        # Let's stick to simple pruning: 
        # If current_size + count(candidates) < k: return False
        
        # Try vertices
        # Order candidates by degree in the induced subgraph?
        # High degree vertices are less likely to be in a large independent set?
        # Actually, if a vertex has high degree, it conflicts with many others.
        # Let's try vertices in order of index for simplicity, or random.
        
        candidates = []
        m = candidates_mask
        while m:
            v = (m & -m).bit_length() - 1
            candidates.append(v)
            m &= ~(1 << v)
            
        # Sort by degree in the remaining graph (descending)
        # Degree of v in candidates is count(candidates_mask & adj_mask[v])
        # Wait, we want to pick vertices that "block" many others?
        # If we pick a high-degree vertex, we remove many candidates.
        # This might lead to a smaller set faster.
        
        # Actually, standard MIS heuristics sort by degree descending.
        candidates.sort(key=lambda v: bin(candidates_mask & adj_mask[v]).count('1'), reverse=True)
        
        for v in candidates:
            if not (candidates_mask & (1 << v)):
                continue
            new_mask = candidates_mask & ~(1 << v) & ~adj_mask[v]
            if backtrack(new_mask, current_size + 1):
                return True
            
        return False

    return backtrack((1 << n) - 1, 0)

def construct():
    # Target n=40
    # Use local search to find a triangle-free graph with alpha < 10
    
    start_time = time.time()
    time_limit = 100  # seconds
    
    for n in range(40, 25, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try to find a graph on n vertices
        found = False
        seed = 42
        attempts = 0
        
        while not found and time.time() - start_time < time_limit:
            attempts += 1
            random.seed(seed)
            
            # Initialize with a random bipartite graph
            # Split vertices into two sets A and B
            mid = n // 2
            A = list(range(mid))
            B = list(range(mid, n))
            
            adj = [set() for _ in range(n)]
            
            # Add random edges between A and B
            # Start with p=0.5
            for u in A:
                for v in B:
                    if random.random() < 0.5:
                        adj[u].add(v)
                        adj[v].add(u)
            
            # Local Search
            # Goal: No triangles, alpha < 10
            # Step 1: Remove triangles
            # Step 2: Add edges to reduce alpha
            
            # We will iterate
            for _ in range(1000):
                # 1. Remove triangles
                changed = False
                # Check all triples? O(n^3) is 40^3 = 64000, very fast.
                # But we can be smarter.
                # Iterate through all pairs (u, v) that are edges.
                # If they have a common neighbor w, remove one edge.
                # To be fast, we can just check for triangles in the current graph.
                
                # Efficient triangle removal:
                # For each u, for each v in adj[u], for each w in adj[v]:
                # if w > u and w > v and adj[u][w]: triangle.
                
                # Let's just do a simple pass:
                # Collect all triangles
                triangles_to_remove = []
                for u in range(n):
                    for v in adj[u]:
                        if v > u:
                            for w in adj[v]:
                                if w > v and w in adj[u]:
                                    # Triangle u, v, w
                                    # Remove edge (v, w) arbitrarily? Or (u,v)?
                                    # Let's remove (v, w)
                                    triangles_to_remove.append((v, w))
                
                if triangles_to_remove:
                    for u, v in triangles_to_remove:
                        if v in adj[u]:
                            adj[u].remove(v)
                            adj[v].remove(u)
                            changed = True
                            
                # 2. Check alpha
                # Convert adj to list of lists for the checker
                adj_list = [list(adj[i]) for i in range(n)]
                
                if has_independent_set_of_size_k(adj_list, n, 10):
                    # Alpha >= 10, need to add edges
                    # Find a pair (u, v) not adjacent, with no common neighbors
                    # Prefer pairs that are "central"
                    
                    candidates = []
                    for u in range(n):
                        for v in range(u+1, n):
                            if v not in adj[u]:
                                # Check if adding (u,v) creates a triangle
                                # i.e., if adj[u] and adj[v] intersect
                                common = adj[u] & adj[v]
                                if not common:
                                    # Safe to add
                                    # Score this edge: 
                                    # We want to reduce alpha.
                                    # Adding an edge between two non-adjacent vertices 
                                    # reduces the potential independent sets containing both.
                                    # Let's just pick one randomly or by degree.
                                    candidates.append((u, v))
                    
                    if candidates:
                        # Pick one randomly
                        u, v = random.choice(candidates)
                        adj[u].add(v)
                        adj[v].add(u)
                        changed = True
                    else:
                        # No edges can be added without creating triangles
                        # And alpha is still too high
                        # Break to restart with new random seed
                        break
                else:
                    # Valid graph found!
                    found = True
                    break
                    
            if found:
                break
            
            # If not found, try next seed
            seed += 1
            
        if found:
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
            
    # Fallback
    n = 26
    edges = [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % 3 == 1]
    return {'n': n, 'edges': edges}
