"""Improved C6-free graph construction on 40 vertices."""
import time
import random
import numpy as np

def construct():
    """
    Construct a C6-free graph on 40 vertices maximizing edge count.
    Uses a greedy randomized approach with C6 detection via bitsets.
    """
    n = 40
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    def has_c6(adj, edge_set):
        """Check if adding an edge creates a C6. Uses adjacency bitsets."""
        # adj is a list of int bitsets for each vertex
        # We check if there's a path of length 4 between u and v
        # (which combined with edge (u,v) would form a C6)
        # Actually, we need to check if there's a C6 in the current graph.
        # For efficiency during greedy construction, we check if adding (u,v) creates a C6.
        # A C6 is formed if there's a path u - a - b - c - d - v of length 4.
        # This is expensive. Instead, let's just check the whole graph periodically.
        pass
    
    def count_c6s(adj_bitsets):
        """Count C6s in the graph using bitset operations."""
        n = len(adj_bitsets)
        c6_count = 0
        # For each pair (u,v) not adjacent, check if there's a path of length 4
        # This is O(n^2 * n) with bitsets... let's use a different approach.
        # Count 6-cycles by looking at paths of length 3 between non-adjacent vertices
        # Actually, a C6: u-v-w-x-y-z-u. 
        # Alternative: for each edge (u,v), count paths of length 4 from u to v that don't use (u,v)
        # and don't repeat vertices. This is complex.
        
        # Simpler: just do a BFS/DFS based check for each potential cycle
        # But that's slow. Let's use the standard approach:
        # For C6-free, we can use the fact that in a C6-free graph,
        # the number of paths of length 3 between any two non-adjacent vertices is 0.
        # Wait, that's not quite right. A C6 contains paths of length 3 between vertices at distance 3.
        
        # Let's just check: for each pair (u,v) where u<v and (u,v) not in edge set,
        # count paths of length 4 from u to v. If >0, there's a C6.
        # Path of length 4: u-a-b-c-v
        # We can compute this using matrix multiplication of adjacency matrix.
        # A^4[u,v] gives number of walks of length 4 from u to v.
        # But walks can repeat vertices. For simple paths, we need more care.
        
        # For a greedy approach, let's use a simpler check:
        # When adding edge (u,v), check if there's a path of length 4 between u and v
        # in the current graph. If yes, adding (u,v) creates a C6.
        
        pass
    
    # Let's use a cleaner approach with numpy for matrix operations
    
    def check_c6_free(adj_matrix):
        """Check if graph is C6-free using matrix powers."""
        n = adj_matrix.shape[0]
        # A^2 counts walks of length 2
        # A^4 counts walks of length 4
        # For C6-free: for any non-adjacent pair (u,v), the number of simple paths of length 4 should be 0
        # But A^4 includes walks with repeated vertices.
        # 
        # Alternative approach: use the fact that a graph is C6-free iff
        # for every pair of vertices at distance >= 2, there's no path of length 4.
        # 
        # Let's use a direct check: enumerate all 6-tuples and check.
        # O(n^6) is too slow for n=40.
        
        # Better: for each pair (u,v) with u<v and not adjacent,
        # count paths of length 4 from u to v.
        # Use dynamic programming with bitsets.
        
        # Let's just do it with integer adjacency matrix and numpy
        A = adj_matrix.astype(np.int64)
        # Compute number of walks of length 4 from u to v
        # A2 = A @ A
        # A4 = A2 @ A2
        A2 = A @ A
        A4 = A2 @ A2
        # For non-adjacent pairs, A4[u,v] should be 0 if C6-free?
        # No, A4 counts walks, not simple paths. A walk of length 4 can repeat vertices.
        # For example, u-v-u-v-w is a walk of length 4 from u to w.
        
        # Let's use a different strategy: since n=40 is small, we can do
        # a more careful check. For each pair (u,v) not adjacent, check if there
        # exists a path u-a-b-c-d-v with all distinct vertices.
        
        # This is still expensive. Let's use a heuristic: if A4[u,v] > 0 for non-adjacent (u,v),
        # it might indicate a C6, but it's not exact.
        
        # For the greedy construction, let's use a simpler local check.
        return True  # Placeholder
    
    # Let's restart with a cleaner, more practical approach.
    
    def build_greedy_c6_free(n, seed=42, time_limit=None):
        """Build a C6-free graph greedily."""
        rng = np.random.RandomState(seed)
        
        # Adjacency as list of sets for fast membership
        adj = [set() for _ in range(n)]
        edge_list = []
        
        # All possible edges
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        rng.shuffle(all_edges)
        
        def would_create_c6(u, v):
            """Check if adding edge (u,v) creates a C6."""
            # A C6 is u-v-...-u where the path from v back to u has length 4
            # So we need to check if there's a path of length 4 from v to u
            # that doesn't use the edge (u,v) [which isn't there yet]
            # and has all distinct internal vertices.
            
            # Path: v - a - b - c - d - u, with a,b,c,d all distinct and not u,v
            # Let's do a BFS/DFS to find paths of length 4 from v to u.
            
            # Use a set to track visited (vertex, depth) pairs
            # Actually, let's just enumerate:
            # For each a in adj[v], for each b in adj[a] (b != v), 
            # for each c in adj[b] (c != a), for each d in adj[c] (d != b),
            # check if u in adj[d] and d != v and u != a,b,c
            
            found = False
            for a in adj[v]:
                if a == u:
                    continue
                for b in adj[a]:
                    if b == v or b == u:
                        continue
                    for c in adj[b]:
                        if c == a or c == v or c == u:
                            continue
                        for d in adj[c]:
                            if d == b or d == v or d == u or d == a:
                                continue
                            if u in adj[d]:
                                found = True
                                return True
            return found
        
        for u, v in all_edges:
            if time_limit is not None and time.time() > time_limit:
                break
            if u not in adj[v]:
                if not would_create_c6(u, v):
                    adj[u].add(v)
                    adj[v].add(u)
                    edge_list.append([u, v])
        
        return edge_list
    
    # Try multiple seeds and keep the best
    best_edges = None
    best_count = 0
    
    for seed in range(200):
        if time.time() > deadline:
            break
        edges = build_greedy_c6_free(40, seed=seed, time_limit=deadline - 2.0)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
    
    if best_edges is None:
        # Fallback: use the original construction
        edges = []
        for k in range(8):
            edges += [[5 * k + a, 5 * k + b] for a in range(5) for b in range(a + 1, 5)]
        return edges
    
    return best_edges
