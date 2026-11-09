import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Helper to check if adding edge (u, v) creates a triangle
    def can_add_edge(adj, u, v):
        # adj is a list of sets
        if v in adj[u]:
            return False
        # Check common neighbors
        # For speed, iterate over smaller set
        if len(adj[u]) < len(adj[v]):
            for w in adj[u]:
                if w in adj[v]:
                    return False
        else:
            for w in adj[v]:
                if w in adj[u]:
                    return False
        return True

    def is_triangle_free(adj):
        for u in range(len(adj)):
            for v in adj[u]:
                if v > u:
                    # Check if u and v share a neighbor
                    # This is O(E^2) worst case, but with sets it's okay for small n
                    # Actually, simpler: for each edge (u,v), check intersection of adj[u] and adj[v]
                    pass
        # Better check:
        for u in range(len(adj)):
            for v in adj[u]:
                if v > u:
                    if adj[u].intersection(adj[v]):
                        return False
        return True

    def max_independent_set_size(adj):
        # Exact branch and bound for alpha
        n = len(adj)
        # Use a simple recursive search with pruning
        # Order vertices by degree?
        
        # For n up to 40, a simple backtracking with bitsets is fast enough.
        # Let's use a bitset representation for speed.
        # adj_bits[i] is an integer where bit j is 1 if i~j
        
        # Convert adj (list of sets) to adj_bits
        adj_bits = [0] * n
        for i in range(n):
            for j in adj[i]:
                adj_bits[i] |= (1 << j)
        
        # We want to find the max set of vertices with no edges between them.
        # This is the same as finding the max clique in the complement graph.
        # Or just use the standard MIS algorithm.
        
        # Simple recursive search with pruning
        # candidates: set of vertices that can still be added
        # current_set: set of vertices currently in the independent set
        # bound: upper bound on size of independent set in candidates
        
        best = 0
        
        # Sort vertices by degree in descending order for better pruning?
        # Or just use the standard Bron-Kerbosch on the complement?
        # Let's use a simple recursive MIS finder.
        
        def mis_search(candidates, current_size):
            nonlocal best
            if not candidates:
                if current_size > best:
                    best = current_size
                return
            
            # Pruning: if current_size + len(candidates) <= best, return
            if current_size + len(candidates) <= best:
                return
            
            # Pick a vertex v from candidates
            # Heuristic: pick vertex with highest degree in the induced subgraph on candidates?
            # Or just the first one.
            v = next(iter(candidates))
            
            # Branch 1: Include v
            # New candidates: candidates \ (N(v) U {v})
            # In independent set, if we pick v, we cannot pick its neighbors.
            # So new candidates = candidates - {v} - N(v)
            new_candidates = set(candidates)
            new_candidates.discard(v)
            for nb in adj[v]:
                if nb in new_candidates:
                    new_candidates.discard(nb)
            
            mis_search(new_candidates, current_size + 1)
            
            # Branch 2: Exclude v
            new_candidates2 = set(candidates)
            new_candidates2.discard(v)
            mis_search(new_candidates2, current_size)

        # To speed up, we can use a greedy upper bound.
        # But for n=40, the simple recursion might be too slow if not pruned well.
        # Let's use a better algorithm: Max Clique on complement.
        
        # Actually, for n=40, a well-implemented MIS is fast.
        # Let's try a bitset-based approach.
        
        # Re-implementation with bitsets for speed
        def mis_bitset(candidates_mask, current_size):
            nonlocal best
            if not candidates_mask:
                if current_size > best:
                    best = current_size
                return
            
            if current_size + bin(candidates_mask).count('1') <= best:
                return
            
            # Pick a vertex
            # Find the lowest set bit
            v_bit = candidates_mask & -candidates_mask
            v = v_bit.bit_length() - 1
            
            # Branch 1: Include v
            # Remove v and its neighbors from candidates
            new_mask = candidates_mask & ~(adj_bits[v] | v_bit)
            mis_bitset(new_mask, current_size + 1)
            
            # Branch 2: Exclude v
            new_mask2 = candidates_mask & ~v_bit
            mis_bitset(new_mask2, current_size)

        full_mask = (1 << n) - 1
        mis_bitset(full_mask, 0)
        return best

    # Andrásfai graph And(3) on 3k vertices
    def andrasfai(k):
        n = 3 * k
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % 3 == 1:
                    adj[i].add(j)
                    adj[j].add(i)
        return adj

    # Local Search to reduce alpha
    def local_search(adj, max_time=5):
        n = len(adj)
        start = time.time()
        current_alpha = max_independent_set_size(adj)
        
        if current_alpha <= 9:
            return adj, current_alpha
        
        # Try to add edges to reduce alpha
        # Candidate edges: non-edges that don't create triangles
        # We will randomly pick a non-edge, check if it can be added, and if it reduces alpha, keep it.
        
        while time.time() - start < max_time:
            # Generate list of potential edges
            potential = []
            for u in range(n):
                for v in range(u + 1, n):
                    if v not in adj[u]:
                        if can_add_edge(adj, u, v):
                            potential.append((u, v))
            
            if not potential:
                break
            
            # Randomly shuffle and try a few
            random.shuffle(potential)
            improved = False
            for u, v in potential[:100]: # Try up to 100 random edges
                # Add edge
                adj[u].add(v)
                adj[v].add(u)
                
                new_alpha = max_independent_set_size(adj)
                if new_alpha < current_alpha:
                    current_alpha = new_alpha
                    improved = True
                    if current_alpha <= 9:
                        return adj, current_alpha
                    break # Restart loop to regenerate potential edges
                else:
                    # Remove edge
                    adj[u].remove(v)
                    adj[v].remove(u)
            
            if not improved:
                # Stuck, maybe try removing an edge?
                # Or just break and try a different seed/n
                break
        
        return adj, current_alpha

    # Try n from 35 down to 27
    best_graph = None
    best_n = 0
    
    # Start with Andrásfai 27 as a fallback
    adj_27 = andrasfai(9)
    alpha_27 = max_independent_set_size(adj_27)
    if alpha_27 <= 9:
        best_graph = adj_27
        best_n = 27

    for n in range(35, 26, -1):
        if time.time() > deadline:
            break
        
        # Initialize random triangle-free graph
        adj = [set() for _ in range(n)]
        # Add random edges
        # To ensure it's not too sparse, we can start with a Turán graph T2(n) (bipartite)
        # T2(n) has no triangles.
        # Split vertices into two sets A and B
        mid = n // 2
        A = list(range(mid))
        B = list(range(mid, n))
        for u in A:
            for v in B:
                if random.random() < 0.5:
                    adj[u].add(v)
                    adj[v].add(u)
        
        # Check alpha
        alpha = max_independent_set_size(adj)
        
        # Run local search
        adj, alpha = local_search(adj, max_time=10)
        
        if alpha <= 9:
            best_graph = adj
            best_n = n
            break # Found a valid graph for this n, which is better than lower n
    
    if best_graph is None:
        # Fallback to Andrásfai 27
        best_graph = adj_27
        best_n = 27
        
    # Convert adj to edge list
    edges = []
    for u in range(best_n):
        for v in best_graph[u]:
            if u < v:
                edges.append([u, v])
                
    return {'n': best_n, 'edges': edges}
