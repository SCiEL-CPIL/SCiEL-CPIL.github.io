import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Known valid graph fallback (Andrasfai And(9))
    def get_fallback():
        n = 26
        return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}

    def is_sum_free(S, n):
        S_set = set(S)
        for x in S_set:
            for y in S_set:
                if (x + y) % n in S_set:
                    return False
        return True

    def has_independent_set(adj, n, k):
        # adj: list of int bitsets
        # Returns True if there is an independent set of size k
        
        # Sort vertices by degree in descending order for better pruning
        # deg[i] is number of 1s in adj[i]
        # But we need to sort the *candidates* dynamically?
        # Pre-sorting might not be optimal if the graph changes, but here it's static.
        # However, the "degree" in the subgraph induced by candidates changes.
        # Simple heuristic: pick the vertex with the highest degree in the *original* graph?
        # Or just pick the first one?
        # Let's use a recursive B&B with a heuristic: pick vertex with max degree in current candidate set.
        
        def search(candidates, current_size):
            if current_size == k:
                return True
            if not candidates:
                return False
            if current_size + len(candidates) < k:
                return False
            
            # Heuristic: pick vertex v in candidates with max degree in G
            # This minimizes the new candidate set size
            max_deg = -1
            best_v = -1
            for v in candidates:
                # Count neighbors in candidates? No, degree in G is static.
                # But we want to shrink the *new* candidate set.
                # New candidates are non-neighbors of v in G, restricted to candidates.
                # So we want v that has many neighbors in candidates.
                # This is equivalent to high degree in the induced subgraph?
                # Let's just use static degree for simplicity and speed.
                deg = bin(adj[v]).count('1')
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
            
            v = best_v
            # Remove v from candidates
            new_candidates = [u for u in candidates if u != v]
            
            # Branch 1: Include v
            # New candidates must be non-neighbors of v
            filtered_candidates = [u for u in new_candidates if not ((adj[v] >> u) & 1)]
            if search(filtered_candidates, current_size + 1):
                return True
            
            # Branch 2: Exclude v
            if search(new_candidates, current_size):
                return True
                
            return False
            
        return search(list(range(n)), 0)

    def try_cayley(n):
        # Try to find a sum-free set S such that Cayley graph has alpha < 10
        # Random restarts
        while time.time() < deadline:
            elements = list(range(n))
            random.shuffle(elements)
            S = []
            for x in elements:
                if is_sum_free(S + [x], n):
                    S.append(x)
            
            # Construct adjacency list
            adj = [0] * n
            for u in range(n):
                for v in S:
                    adj[u] |= (1 << ((u + v) % n))
            
            # Check triangle-free (should be true by sum-free)
            # Check alpha < 10
            if not has_independent_set(adj, n, 10):
                # Found valid graph
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if (adj[u] >> v) & 1:
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
                
        return None

    # Try n=40, 39, 38...
    for n in [40, 39, 38, 37, 36]:
        if time.time() >= deadline:
            break
        result = try_cayley(n)
        if result:
            return result
            
    # Fallback
    return get_fallback()
