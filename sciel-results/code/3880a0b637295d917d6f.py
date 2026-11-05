import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin

    def seed_graph():
        n = 32
        edges = []
        for b in range(8):
            base = b * 4
            for i in range(4):
                for j in range(i + 1, 4):
                    edges.append([base + i, base + j])
        return {'n': n, 'edges': edges}

    best_result = seed_graph()

    # Helper functions
    def build_adj(n, edge_set):
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def has_c5(n, adj):
        """Check if graph has a 5-cycle using adjacency sets"""
        for a in range(n):
            for b in adj[a]:
                if b <= a:
                    continue
                for c in adj[b]:
                    if c <= b or c == a:
                        continue
                    for d in adj[c]:
                        if d <= c or d in (a, b):
                            continue
                        for e in adj[d]:
                            if e > d and e != a and e not in (b, c) and a in adj[e]:
                                return True
        return False

    def has_independent_set(n, adj, k):
        """Check if graph has an independent set of size k using backtracking"""
        # Order vertices by degree for better pruning
        def backtrack(remaining, current_size):
            if current_size == k:
                return True
            if len(remaining) < k - current_size:
                return False
            # Upper bound: can we still reach k?
            if current_size + len(remaining) < k:
                return False
            
            # Pick vertex with highest degree in remaining (branch on it)
            v = max(remaining, key=lambda x: len(adj[x] & remaining))
            remaining_list = list(remaining)
            remaining_list.remove(v)
            
            # Option 1: Include v (remove v and its neighbors from remaining)
            new_remaining = set(remaining_list)
            for u in adj[v]:
                if u in new_remaining:
                    new_remaining.remove(u)
            if backtrack(new_remaining, current_size + 1):
                return True
            
            # Option 2: Don't include v
            if backtrack(set(remaining_list), current_size):
                return True
            
            return False
        
        return backtrack(set(range(n)), 0)

    def is_valid(n, edge_set):
        """Check if graph is C5-free and has no independent set of size 9"""
        adj = build_adj(n, edge_set)
        if has_c5(n, adj):
            return False
        if has_independent_set(n, adj, 9):
            return False
        return True

    # Structural approach: Try to construct n=33 graph
    # Start with 8 K4s and add vertex 32 with strategic connections
    # Key insight: To avoid C5, vertex 32 can connect to at most 1 vertex per K4
    # But that gives alpha >= 9. So we need to modify the K4 structure.
    
    # Alternative: Use a different base structure
    # Try: 7 K4s + 1 K5 (but K5 has C5, so no)
    # Try: Mix of K4s and other C5-free components
    
    # Let's try a systematic search for n=33
    def search_n33():
        """Search for n=33 C5-free graph with alpha < 9"""
        n = 33
        random.seed(123)
        
        # Start with 8 K4s
        base_edges = set()
        for b in range(8):
            base = b * 4
            for i in range(4):
                for j in range(i + 1, 4):
                    base_edges.add((base + i, base + j))
        
        # Try adding vertex 32 with various connections
        # and modifying some K4 edges to break C5s while keeping alpha < 9
        
        # Strategy: Remove some edges from K4s to allow more connections for v32
        # while maintaining C5-free property
        
        for attempt in range(200):
            if time.time() > deadline - 10.0:
                break
            
            edges = set(base_edges)
            
            # Randomly remove some edges from K4s to create "slots" for v32
            removed = set()
            for _ in range(random.randint(0, 3)):
                b = random.randint(0, 7)
                base = b * 4
                i, j = random.sample(range(4), 2)
                edge = (base + i, base + j)
                if edge in edges:
                    edges.remove(edge)
                    removed.add(edge)
            
            # Connect v32 to vertices in the modified K4s
            for b in range(8):
                base = b * 4
                # Connect to 1-2 vertices per K4
                for i in range(4):
                    if random.random() < 0.4:
                        edges.add((32, base + i))
            
            # Check validity
            if is_valid(n, list(edges)):
                edge_list = [[u, v] for u, v in sorted(edges)]
                return {'n': n, 'edges': edge_list}
        
        return None

    # Try the search
    result = search_n33()
    if result is not None:
        best_result = result
    else:
        # Fall back to seed
        best_result = seed_graph()

    return best_result
