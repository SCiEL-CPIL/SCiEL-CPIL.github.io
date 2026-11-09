import random
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290  # 10s safety margin

    def count_k4_in_set(n, adj, vertex_set):
        """
        Count K4s in the induced subgraph on vertex_set.
        vertex_set is a bitmask.
        """
        # Get list of vertices in the set
        verts = [v for v in range(n) if vertex_set & (1 << v)]
        count = 0
        for i in range(len(verts)):
            for j in range(i+1, len(verts)):
                for k in range(j+1, len(verts)):
                    u, v, w = verts[i], verts[j], verts[k]
                    # Check if u,v,w form a triangle
                    if (adj[u] & (1 << v)) and (adj[u] & (1 << w)) and (adj[v] & (1 << w)):
                        # For each x in vertex_set not in {u,v,w}, check if x connects to all three
                        for x in verts:
                            if x not in (u, v, w):
                                if (adj[x] & (1 << u)) and (adj[x] & (1 << v)) and (adj[x] & (1 << w)):
                                    count += 1
        # Each K4 is counted 4 times (once per vertex)
        return count // 4

    def count_indep4_in_set(n, adj, vertex_set):
        """
        Count independent 4-sets in the induced subgraph on vertex_set.
        vertex_set is a bitmask.
        """
        verts = [v for v in range(n) if vertex_set & (1 << v)]
        count = 0
        for i in range(len(verts)):
            for j in range(i+1, len(verts)):
                for k in range(j+1, len(verts)):
                    u, v, w = verts[i], verts[j], verts[k]
                    # Check if u,v,w form an independent triple
                    if not (adj[u] & (1 << v)) and not (adj[u] & (1 << w)) and not (adj[v] & (1 << w)):
                        # For each x in vertex_set not in {u,v,w}, check if x is non-adjacent to all three
                        for x in verts:
                            if x not in (u, v, w):
                                if not (adj[x] & (1 << u)) and not (adj[x] & (1 << v)) and not (adj[x] & (1 << w)):
                                    count += 1
        # Each independent 4-set is counted 4 times
        return count // 4

    def is_valid_5_5(n, adj):
        """
        Checks if the graph contains no K5 and no independent set of size 5.
        Uses efficient counting: a K5 exists iff there's a vertex v such that
        its neighborhood contains a K4. Similarly for independent sets.
        """
        for v in range(n):
            # Count K4s in the neighborhood of v
            if count_k4_in_set(n, adj, adj[v]) > 0:
                return False
            # Count independent 4-sets in the non-neighborhood of v
            non_neighbors = 0
            for u in range(n):
                if u != v and not (adj[v] & (1 << u)):
                    non_neighbors |= (1 << u)
            if count_indep4_in_set(n, adj, non_neighbors) > 0:
                return False
        return True

    def get_paley_37():
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
        return q, edges

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

    def local_search(n, adj, max_iters=1000):
        """
        Try to improve the graph by flipping edges.
        Returns (improved_adj, improved) where improved is True if we found a valid graph.
        """
        current_adj = list(adj)
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Try flipping a random edge
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j:
                continue
            
            # Flip the edge
            old_val = (current_adj[i] >> j) & 1
            current_adj[i] ^= (1 << j)
            current_adj[j] ^= (1 << i)
            
            # Check if valid
            if is_valid_5_5(n, current_adj):
                return current_adj, True
            
            # Revert if not valid (we want to maintain validity during search)
            # Actually, for local search we should allow invalid states but that's complex.
            # Let's just try to find a valid graph by random restarts instead.
            current_adj[i] ^= (1 << j)
            current_adj[j] ^= (1 << i)
        
        return current_adj, False

    # Strategy: 
    # 1. Start with Paley 37 as baseline
    # 2. Try to extend to 38, 39, 40, 41, 42 vertices
    # 3. For each n, use random circulant search + local improvement
    
    best_n, best_edges = get_paley_37()
    
    # Try to find larger valid graphs
    for n in range(38, 43):
        if time.time() > deadline - 10:
            break
        
        # Try random circulant graphs
        max_dist = n // 2
        possible_dists = list(range(1, max_dist + 1))
        
        # Try various sizes of connection sets
        for size in range(n//5, n//2):
            if time.time() > deadline - 5:
                break
            
            for _ in range(50):  # 50 trials per size
                if time.time() > deadline:
                    break
                
                S = set(random.sample(possible_dists, min(size, len(possible_dists))))
                
                adj = [0] * n
                for i in range(n):
                    for d in S:
                        j1 = (i + d) % n
                        j2 = (i - d) % n
                        adj[i] |= (1 << j1)
                        adj[i] |= (1 << j2)
                
                if is_valid_5_5(n, adj):
                    best_n, best_edges = n, adj_to_edges(n, adj)
                    break
            if best_n >= n:
                break
    
    # If we found something >= 38, try to push further with more random searches
    if best_n < 42 and time.time() < deadline - 15:
        for n in range(best_n + 1, 43):
            if time.time() > deadline - 5:
                break
            
            max_dist = n // 2
            possible_dists = list(range(1, max_dist + 1))
            
            for _ in range(200):  # More trials
                if time.time() > deadline:
                    break
                
                size = random.randint(n//5, n//2)
                S = set(random.sample(possible_dists, min(size, len(possible_dists))))
                
                adj = [0] * n
                for i in range(n):
                    for d in S:
                        j1 = (i + d) % n
                        j2 = (i - d) % n
                        adj[i] |= (1 << j1)
                        adj[i] |= (1 << j2)
                
                if is_valid_5_5(n, adj):
                    best_n, best_edges = n, adj_to_edges(n, adj)
                    break
    
    return best_n, best_edges
