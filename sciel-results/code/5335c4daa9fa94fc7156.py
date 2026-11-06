import itertools
import random
import time

def construct():
    start_time = time.time()
    time_limit = 145  # Leave 5s margin before 150s kill

    # Hardcoded known valid 20-vertex graph
    known_20_edges = [
        [0, 1], [0, 2], [0, 5], [0, 6], [0, 9], [0, 10], [0, 13], [0, 14],
        [1, 3], [1, 4], [1, 7], [1, 8], [1, 11], [1, 12], [1, 15], [1, 16],
        [2, 3], [2, 6], [2, 8], [2, 10], [2, 13], [2, 16], [2, 18],
        [3, 5], [3, 9], [3, 12], [3, 14], [3, 17],
        [4, 6], [4, 7], [4, 10], [4, 15], [4, 18],
        [5, 7], [5, 11], [5, 14], [5, 17],
        [6, 8], [6, 12], [6, 15],
        [7, 9], [7, 13], [7, 16],
        [8, 11], [8, 14], [8, 19],
        [9, 12], [9, 15], [9, 18],
        [10, 13], [10, 16], [10, 19],
        [11, 14], [11, 17],
        [12, 15], [12, 18],
        [13, 16], [13, 19],
        [14, 17],
        [15, 18],
        [16, 19],
    ]

    def build_adj_masks(n, edges):
        adj_masks = [0] * n
        for u, v in edges:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
        return adj_masks

    def get_edges_from_masks(n, adj_masks):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_masks[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def count_violations_size_6(n, adj_masks):
        """Count the number of 6-subsets that induce a regular subgraph."""
        count = 0
        for subset in itertools.combinations(range(n), 6):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            
            # Check if regular
            first_deg = (adj_masks[subset[0]] & mask).bit_count()
            is_regular = True
            for v in subset[1:]:
                deg = (adj_masks[v] & mask).bit_count()
                if deg != first_deg:
                    is_regular = False
                    break
            if is_regular:
                count += 1
        return count

    def is_valid_full(n, adj_masks):
        """Check if no subset of size >= 6 induces a regular subgraph."""
        # Check size 6 first
        if count_violations_size_6(n, adj_masks) > 0:
            return False
        
        # Check larger sizes if necessary
        # Note: If no 6-subset is regular, it's possible a larger one is.
        # However, usually 6 is the bottleneck.
        # For safety, check 7..n
        for k in range(7, n + 1):
            for subset in itertools.combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                first_deg = (adj_masks[subset[0]] & mask).bit_count()
                is_regular = True
                for v in subset[1:]:
                    deg = (adj_masks[v] & mask).bit_count()
                    if deg != first_deg:
                        is_regular = False
                        break
                if is_regular:
                    return False
        return True

    # Step 1: Verify 20-vertex graph
    adj_20 = build_adj_masks(20, known_20_edges)
    if not is_valid_full(20, adj_20):
        # Fallback
        return {'n': 5, 'edges': []}

    # Step 2: Try to find a 21-vertex valid graph using Simulated Annealing
    n = 21
    deadline = start_time + time_limit - 5
    
    # Initialize random graph
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    current_cost = count_violations_size_6(n, adj)
    best_adj = list(adj)
    best_cost = current_cost
    
    # Simulated Annealing Parameters
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    
    while time.time() < deadline and T > min_T:
        # Pick a random edge to flip
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
            
        # Flip edge
        if (adj[u] >> v) & 1:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        new_cost = count_violations_size_6(n, adj)
        delta = new_cost - current_cost
        
        if delta < 0:
            # Improvement
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_adj = list(adj)
                if best_cost == 0:
                    # Check if fully valid (sizes > 6)
                    if is_valid_full(n, adj):
                        edges = get_edges_from_masks(n, adj)
                        return {'n': 21, 'edges': edges}
        else:
            # Accept with probability exp(-delta/T)
            if random.random() < pow(2.718281828, -delta / T):
                current_cost = new_cost
        
        T *= cooling_rate
        
        # Early exit if we find a perfect graph
        if current_cost == 0:
            if is_valid_full(n, adj):
                edges = get_edges_from_masks(n, adj)
                return {'n': 21, 'edges': edges}

    # If we found a graph with 0 size-6 violations, check if it's fully valid
    if best_cost == 0:
        if is_valid_full(n, best_adj):
            edges = get_edges_from_masks(n, best_adj)
            return {'n': 21, 'edges': edges}

    # Step 3: Return the known 20-vertex graph
    return {'n': 20, 'edges': known_20_edges}
