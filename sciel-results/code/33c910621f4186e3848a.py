import hashlib
import struct

def construct(k, g):
    # For k=5, g=9: use n = 2 * 5^4 = 1250 vertices
    # Build a 5-regular graph using hash-based permutations
    # Verify girth >= 9
    
    n = 2 * (5 ** 4)  # 1250
    
    # Build 5 permutations of [0, n-1] using SHA-256
    # Each vertex i connects to perm[j](i) for j=0..4
    # But we need to ensure the graph is undirected and simple
    
    # Better approach: use a deterministic pseudo-random permutation
    # based on hashing (i, j) to get a unique neighbor
    
    # For a 5-regular graph, each vertex has exactly 5 neighbors
    # We can define: for vertex v, its neighbors are determined by
    # hashing (v, j) for j in 0..4 and mapping to [0, n-1]
    # But this won't guarantee regularity or symmetry
    
    # Alternative: use circulant-like structure with hash-derived offsets
    # Or: build a bipartite graph with two sets of n/2 vertices
    
    # Let's use a bipartite construction:
    # Left side: 0..n/2-1, Right side: n/2..n-1
    # Each left vertex connects to 5 right vertices
    # Each right vertex connects to 5 left vertices
    # Total edges: n/2 * 5 = n * 5 / 2
    
    # Use hash to determine connections
    half = n // 2
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    edges = []
    
    for left in range(half):
        # Get 5 distinct right neighbors using hash
        neighbors = set()
        while len(neighbors) < k:
            h = hashlib.sha256(struct.pack('>II', left, len(neighbors))).digest()
            right = int.from_bytes(h[:4], 'big') % half
            neighbors.add(right)
        for right in neighbors:
            right_node = half + right
            adj[left].append(right_node)
            adj[right_node].append(left)
            edges.append((left, right_node))
    
    # Verify it's 5-regular
    for v in range(n):
        if len(adj[v]) != k:
            raise ValueError(f"Vertex {v} has degree {len(adj[v])}, not {k}")
    
    # Verify no duplicate edges or loops
    edge_set = set()
    for u, v in edges:
        if u == v:
            raise ValueError("Loop found")
        e = (min(u, v), max(u, v))
        if e in edge_set:
            raise ValueError("Duplicate edge")
        edge_set.add(e)
    
    # Verify girth >= g using BFS from each vertex
    # For efficiency, we can stop early if we find a cycle shorter than g
    
    for start in range(n):
        # BFS with parent tracking to detect cycles
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        # Girth is too small, need to retry with different seed
                        # For now, raise an error
                        raise ValueError(f"Cycle of length {cycle_len} < {g} found")
    
    return n, edges
