import random
import itertools
import time

def construct():
    # Approach: Efficient Random Search with Bitmask Validation
    # This replaces the slow O(k^2) subset check with O(k) bitmask operations.
    # This allows exhaustive checking of all subsets for n=20 within time limits,
    # making random search viable for finding large valid graphs.
    
    def is_valid_graph_fast(n, edges):
        # Build adjacency bitmasks
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Check all subsets of size >= 6
        for k in range(6, n + 1):
            for subset in itertools.combinations(range(n), k):
                # Calculate mask for the subset
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                
                # Check if induced subgraph is regular
                # Degree of first vertex in subset
                first_v = subset[0]
                target_deg = (adj[first_v] & mask).bit_count()
                
                is_reg = True
                for v in subset[1:]:
                    deg = (adj[v] & mask).bit_count()
                    if deg != target_deg:
                        is_reg = False
                        break
                if is_reg:
                    return False
        return True

    def generate_random_graph(n, p):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < p:
                    edges.append([u, v])
        return edges

    def try_find(n, max_attempts=500):
        # Try a range of probabilities to cover different graph densities
        # Regular induced subgraphs are less likely in graphs that are not too dense or too sparse
        for p in [0.3, 0.4, 0.5, 0.6, 0.7]:
            for _ in range(max_attempts):
                edges = generate_random_graph(n, p)
                if is_valid_graph_fast(n, edges):
                    return edges
        return None

    # Try n=20 first
    start_time = time.time()
    edges = try_find(20, max_attempts=200)
    if edges is not None:
        return {'n': 20, 'edges': edges}
        
    if time.time() - start_time > 90:
        # If 20 is taking too long, prioritize finding a valid smaller graph
        pass

    edges = try_find(19, max_attempts=200)
    if edges is not None:
        return {'n': 19, 'edges': edges}

    edges = try_find(18, max_attempts=200)
    if edges is not None:
        return {'n': 18, 'edges': edges}

    edges = try_find(15, max_attempts=500)
    if edges is not None:
        return {'n': 15, 'edges': edges}

    # Fallback to a very small valid graph if all else fails
    # n=5 is always valid (no subset of size 6 exists)
    return {'n': 5, 'edges': []}
