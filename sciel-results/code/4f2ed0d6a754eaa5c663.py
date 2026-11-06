import itertools
import random
import time

def construct():
    start_time = time.time()
    time_limit = 140

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

    def is_regular_subset_fast(subset, adj_masks):
        mask = 0
        for v in subset:
            mask |= (1 << v)
        first_deg = (adj_masks[subset[0]] & mask).bit_count()
        for v in subset[1:]:
            deg = (adj_masks[v] & mask).bit_count()
            if deg != first_deg:
                return False
        return True

    def count_violations_size_6(n, adj_masks):
        count = 0
        for subset in itertools.combinations(range(n), 6):
            if is_regular_subset_fast(subset, adj_masks):
                count += 1
        return count

    def is_valid_full(n, adj_masks):
        # Check size 6
        if count_violations_size_6(n, adj_masks) > 0:
            return False
        
        # Check larger sizes
        for k in range(7, n + 1):
            for subset in itertools.combinations(range(n), k):
                if is_regular_subset_fast(subset, adj_masks):
                    return False
        return True

    # Verify 20-vertex graph
    adj_20 = build_adj_masks(20, known_20_edges)
    if not is_valid_full(20, adj_20):
        return {'n': 5, 'edges': []}

    # Try to find 21-vertex graph
    n_target = 21
    deadline = start_time + time_limit - 5
    
    # Precompute all 6-subsets
    all_6_subsets = list(itertools.combinations(range(n_target), 6))
    
    # Mapping edge to subsets
    edge_to_subsets = {}
    for subset in all_6_subsets:
        for i in range(6):
            for j in range(i + 1, 6):
                u, v = subset[i], subset[j]
                edge = (u, v) if u < v else (v, u)
                if edge not in edge_to_subsets:
                    edge_to_subsets[edge] = []
                edge_to_subsets[edge].append(subset)

    def local_search(adj, deadline):
        current_cost = count_violations_size_6(n_target, adj)
        
        if current_cost == 0:
            if is_valid_full(n_target, adj):
                return adj

        improved = True
        while improved and time.time() < deadline:
            improved = False
            violations = []
            for subset in all_6_subsets:
                if is_regular_subset_fast(subset, adj):
                    violations.append(subset)
            
            if not violations:
                if is_valid_full(n_target, adj):
                    return adj
                break

            # Find best edge flip
            best_edge = None
            best_impact = 0
            
            # Sample a subset of edges to speed up? Or check all?
            # There are n*(n-1)/2 edges. For n=21, 210 edges.
            # Each edge is in ~3876 subsets.
            # Total checks: 210 * 3876 * 2 (before/after) * 6 (deg calc) ~ 10M ops.
            # This is too slow for Python in a tight loop.
            
            # Strategy: Only check a random subset of edges or use a heuristic.
            # Heuristic: Pick an edge from a random violating subset.
            # This is much faster.
            
            random.shuffle(violations)
            for subset in violations[:10]: # Check edges in first 10 violations
                for i in range(6):
                    for j in range(i + 1, 6):
                        u, v = subset[i], subset[j]
                        edge = (u, v) if u < v else (v, u)
                        
                        # Calculate impact of flipping this edge
                        # Only check subsets containing this edge
                        # We need to know which of the *current* violations are fixed/created
                        # This is complex.
                        
                        # Simpler: Just flip and check new cost?
                        # Too slow.
                        
                        pass
            
            # Alternative: Random flip with acceptance (Simulated Annealing)
            # Pick random edge, flip, calculate new cost.
            # If new cost < old cost, accept.
            # If new cost == old cost, accept with prob p.
            # If new cost > old cost, reject.
            
            # To speed up cost calculation, we can update it incrementally?
            # Hard to do with bitmasks.
            
            # Let's try a "Greedy" approach:
            # For each violating subset, try to break it by flipping one of its edges.
            # Check if flipping that edge reduces the total number of violations.
            
            # We'll iterate through a few random violating subsets
            for _ in range(5):
                if time.time() > deadline:
                    return None
                subset = random.choice(violations)
                # Try flipping each edge in this subset
                for i in range(6):
                    for j in range(i + 1, 6):
                        u, v = subset[i], subset[j]
                        edge = (u, v) if u < v else (v, u)
                        
                        # Flip
                        if (adj[u] >> v) & 1:
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                        else:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                        
                        new_cost = count_violations_size_6(n_target, adj)
                        
                        if new_cost < current_cost:
                            current_cost = new_cost
                            improved = True
                            if new_cost == 0:
                                if is_valid_full(n_target, adj):
                                    return adj
                            break
                        else:
                            # Undo
                            if (adj[u] >> v) & 1:
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                            else:
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)
                    if improved:
                        break
                if improved:
                    break
        
        return None

    # Initializations
    initializations = []
    
    # 1. Random p=0.5
    adj_rand = [0] * n_target
    for i in range(n_target):
        for j in range(i + 1, n_target):
            if random.random() < 0.5:
                adj_rand[i] |= (1 << j)
                adj_rand[j] |= (1 << i)
    initializations.append(adj_rand)
    
    # 2. Random Regular (from K22 matchings) -> remove vertex
    # Generate 21 random perfect matchings of K22
    # This is complex to generate correctly in a few lines.
    # Let's just use random graphs for now and rely on the search.
    
    for restart in range(100):
        if time.time() > deadline:
            break
            
        # Create new random graph
        adj = [0] * n_target
        p = random.uniform(0.3, 0.7)
        for i in range(n_target):
            for j in range(i + 1, n_target):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        result = local_search(adj, deadline)
        if result:
            edges = get_edges_from_masks(n_target, result)
            return {'n': 21, 'edges': edges}

    # Fallback
    return {'n': 20, 'edges': known_20_edges}
