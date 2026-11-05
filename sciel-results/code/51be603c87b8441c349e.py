import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    sys.setrecursionlimit(100000)
    
    # Helper to check if adding edge (u, v) creates a C5
    # Graph is represented as a list of sets for neighbors
    def creates_c5(graph, u, v):
        # We check if there is a path of length 4 between u and v
        # Path: u - a - b - c - v
        # a in N(u), c in N(v)
        # b is a common neighbor of a and c
        nu = graph[u]
        nv = graph[v]
        
        for a in nu:
            if a == v: continue
            for c in nv:
                if c == u: continue
                if a == c: continue
                # Check if N(a) and N(c) have a common neighbor b
                # b must not be u, v, a, c
                common = graph[a] & graph[c]
                common -= {u, v, a, c}
                if common:
                    return True
        return False

    def has_independent_set_size_9(graph, n):
        # Use a branch and bound with bitsets for speed
        # Represent available vertices as a bitmask
        # For n <= 64, we can use Python integers as bitsets
        
        # Precompute neighbor masks
        neighbor_mask = [0] * n
        for u in range(n):
            for v in graph[u]:
                neighbor_mask[u] |= (1 << v)
        
        all_mask = (1 << n) - 1
        
        def dfs(available, size):
            if size >= 9:
                return True
            if available == 0:
                return False
            
            # Count bits to check if we can possibly reach 9
            # available.bit_count() is available in Python 3.10+
            # For compatibility, use bin().count()
            if size + bin(available).count('1') < 9:
                return False
            
            # Pick a vertex: use the least significant bit
            v = (available & -available).bit_length() - 1
            rest = available & ~(1 << v)
            
            # Branch 1: Include v
            # Remove v and its neighbors
            new_available = rest & ~neighbor_mask[v]
            if dfs(new_available, size + 1):
                return True
            
            # Branch 2: Exclude v
            if dfs(rest, size):
                return True
            
            return False
        
        return dfs(all_mask, 0)

    # Initialize graph
    n = 33
    graph = [set() for _ in range(n)]
    
    # Start with 8 K4s
    for b in range(8):
        base = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                u = base + i
                v = base + j
                graph[u].add(v)
                graph[v].add(u)
    
    best_graph = None
    best_n = 32  # Fallback
    
    # Try to build a valid 33-vertex graph
    while time.time() < deadline:
        # Reset vertex 32 connections
        graph[32] = set()
        for u in range(32):
            graph[u].discard(32)
        
        # Try to connect 32 to vertices in a smart order
        # First try connecting to one vertex from each K4 block
        # This helps reduce alpha without creating C5
        
        # Shuffle vertices
        vertices = list(range(32))
        random.shuffle(vertices)
        
        # Try adding edges until alpha < 9
        for v in vertices:
            if not creates_c5(graph, 32, v):
                graph[32].add(v)
                graph[v].add(32)
            
            # Check alpha periodically (not every time for speed)
            if not has_independent_set_size_9(graph, n):
                # Valid candidate found!
                best_graph = [set(s) for s in graph]
                best_n = 33
                break
        
        if best_graph is not None:
            break
    
    # If we didn't find a 33-vertex graph, try 34
    if best_graph is None and time.time() < deadline - 10:
        n = 34
        graph = [set() for _ in range(n)]
        
        # Start with 8 K4s (32 vertices)
        for b in range(8):
            base = b * 4
            for i in range(4):
                for j in range(i + 1, 4):
                    u = base + i
                    v = base + j
                    graph[u].add(v)
                    graph[v].add(u)
        
        # Vertices 32 and 33 are isolated initially
        # We need to connect them to reduce alpha from 10 to < 9
        
        attempts = 0
        while time.time() < deadline - 10 and attempts < 50:
            attempts += 1
            
            # Reset vertices 32 and 33
            graph[32] = set()
            graph[33] = set()
            for u in range(32):
                graph[u].discard(32)
                graph[u].discard(33)
            
            # Connect 32 to random vertices
            v32_neighbors = set()
            for v in random.sample(range(32), 16):  # Try 16 random vertices
                if not creates_c5(graph, 32, v):
                    graph[32].add(v)
                    graph[v].add(32)
                    v32_neighbors.add(v)
            
            # Connect 33 to random vertices, avoiding C5
            v33_neighbors = set()
            for v in random.sample(range(32), 16):
                if not creates_c5(graph, 33, v):
                    graph[33].add(v)
                    graph[v].add(33)
                    v33_neighbors.add(v)
            
            # Also consider connecting 32-33
            if not creates_c5(graph, 32, 33):
                graph[32].add(33)
                graph[33].add(32)
            
            # Check alpha
            if not has_independent_set_size_9(graph, n):
                best_graph = [set(s) for s in graph]
                best_n = 34
                break
    
    if best_graph is None:
        # Fallback to 32 vertices
        n = 32
        graph = [set() for _ in range(n)]
        for b in range(8):
            base = b * 4
            for i in range(4):
                for j in range(i + 1, 4):
                    u = base + i
                    v = base + j
                    graph[u].add(v)
                    graph[v].add(u)
        best_graph = graph
        best_n = 32

    # Final verification
    n = len(best_graph)
    
    # Strict C5 Check
    def strict_has_c5(g, n):
        for u in range(n):
            nu = g[u]
            for v in nu:
                if v <= u: continue
                nv = g[v]
                for w in nv:
                    if w == u: continue
                    nw = g[w]
                    for x in nw:
                        if x == u or x == v: continue
                        nx = g[x]
                        for y in nx:
                            if y == u or y == v or y == w: continue
                            if u in g[y]:
                                return True
        return False

    # Strict Alpha Check
    def strict_alpha_at_least_9(g, n):
        neighbor_mask = [0] * n
        for u in range(n):
            for v in g[u]:
                neighbor_mask[u] |= (1 << v)
        
        all_mask = (1 << n) - 1
        
        def dfs(available, size):
            if size >= 9:
                return True
            if available == 0:
                return False
            
            if size + bin(available).count('1') < 9:
                return False
            
            v = (available & -available).bit_length() - 1
            rest = available & ~(1 << v)
            
            new_available = rest & ~neighbor_mask[v]
            if dfs(new_available, size + 1):
                return True
            
            if dfs(rest, size):
                return True
            
            return False
        
        return dfs(all_mask, 0)

    if strict_has_c5(best_graph, n) or strict_alpha_at_least_9(best_graph, n):
        # Invalid, fallback to 32
        n = 32
        best_graph = [set() for _ in range(n)]
        for b in range(8):
            base = b * 4
            for i in range(4):
                for j in range(i + 1, 4):
                    u = base + i
                    v = base + j
                    best_graph[u].add(v)
                    best_graph[v].add(u)

    edges = []
    for u in range(n):
        for v in best_graph[u]:
            if u < v:
                edges.append([u, v])
                
    return {'n': n, 'edges': edges}
