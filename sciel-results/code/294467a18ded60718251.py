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

    def is_regular_subset(subset, adj_masks):
        """Check if the given subset induces a regular subgraph."""
        mask = 0
        for v in subset:
            mask |= (1 << v)
        first_deg = (adj_masks[subset[0]] & mask).bit_count()
        for v in subset[1:]:
            deg = (adj_masks[v] & mask).bit_count()
            if deg != first_deg:
                return False
        return True

    def find_violating_6_subsets(n, adj_masks):
        """Find all 6-subsets that induce regular subgraphs."""
        violations = []
        for subset in itertools.combinations(range(n), 6):
            if is_regular_subset(subset, adj_masks):
                violations.append(subset)
        return violations

    def count_violations_size_6(n, adj_masks):
        """Count the number of 6-subsets that induce a regular subgraph."""
        return len(find_violating_6_subsets(n, adj_masks))

    def is_valid_full(n, adj_masks):
        """Check if no subset of size >= 6 induces a regular subgraph."""
        # Check size 6 first
        if find_violating_6_subsets(n, adj_masks):
            return False
        
        # Check larger sizes if necessary
        for k in range(7, n + 1):
            for subset in itertools.combinations(range(n), k):
                if is_regular_subset(subset, adj_masks):
                    return False
        return True

    def edge_flip_impact(n, adj_masks, u, v):
        """
        Estimate how many violations would be fixed/created by flipping edge (u,v).
        Returns (delta_violations, affected_subsets_count)
        """
        # A 6-subset is affected by edge (u,v) if it contains both u and v
        # We need to check all 4-subsets of the remaining n-2 vertices
        affected_count = 0
        delta = 0
        
        # Current state: edge (u,v) exists or not
        current_exists = (adj_masks[u] >> v) & 1
        
        for other4 in itertools.combinations([x for x in range(n) if x != u and x != v], 4):
            subset = tuple(sorted((u, v) + tuple(other4)))
            if len(set(subset)) != 6:
                continue
                
            # Check if this subset is currently regular
            was_regular = is_regular_subset(subset, adj_masks)
            
            # Simulate flip
            if current_exists:
                # Remove edge
                adj_masks[u] &= ~(1 << v)
                adj_masks[v] &= ~(1 << u)
            else:
                # Add edge
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
            
            is_regular_now = is_regular_subset(subset, adj_masks)
            
            # Restore
            if current_exists:
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
            else:
                adj_masks[u] &= ~(1 << v)
                adj_masks[v] &= ~(1 << u)
            
            if was_regular and not is_regular_now:
                delta -= 1  # Fixed a violation
                affected_count += 1
            elif not was_regular and is_regular_now:
                delta += 1  # Created a violation
                affected_count += 1
        
        return delta, affected_count

    # Step 1: Verify 20-vertex graph
    adj_20 = build_adj_masks(20, known_20_edges)
    if not is_valid_full(20, adj_20):
        # Fallback
        return {'n': 5, 'edges': []}

    # Step 2: Try to find a 21-vertex valid graph using targeted search
    n = 21
    deadline = start_time + time_limit - 5
    
    # Try multiple random restarts
    best_adj = None
    best_cost = float('inf')
    
    for restart in range(20):
        if time.time() > deadline:
            break
            
        # Initialize random graph with bias toward sparse
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.4:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        current_cost = count_violations_size_6(n, adj)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_adj = list(adj)
        
        if current_cost == 0:
            if is_valid_full(n, adj):
                edges = get_edges_from_masks(n, adj)
                return {'n': 21, 'edges': edges}
        
        # Local search: iteratively flip edges that reduce violations
        improved = True
        while improved and time.time() < deadline:
            improved = False
            violations = find_violating_6_subsets(n, adj)
            
            if not violations:
                if is_valid_full(n, adj):
                    edges = get_edges_from_masks(n, adj)
                    return {'n': 21, 'edges': edges}
                break
            
            # For each violating subset, try flipping edges within it
            # Count how many times each edge appears in violations
            edge_violation_count = {}
            for subset in violations:
                for i in range(6):
                    for j in range(i + 1, 6):
                        u, v = subset[i], subset[j]
                        if u > v:
                            u, v = v, u
                        key = (u, v)
                        edge_violation_count[key] = edge_violation_count.get(key, 0) + 1
            
            # Try flipping the most "important" edges first
            candidate_edges = sorted(edge_violation_count.items(), key=lambda x: -x[1])
            
            for (u, v), count in candidate_edges[:10]:  # Try top 10
                if time.time() > deadline:
                    break
                
                # Try flipping this edge
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                
                new_cost = count_violations_size_6(n, adj)
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_adj = list(adj)
                    
                    if new_cost == 0:
                        if is_valid_full(n, adj):
                            edges = get_edges_from_masks(n, adj)
                            return {'n': 21, 'edges': edges}
                        break
                else:
                    # Undo flip
                    if (adj[u] >> v) & 1:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        
        # If we made progress, continue with improved graph
        if best_cost < float('inf') and best_adj is not None:
            if best_cost == 0:
                if is_valid_full(n, best_adj):
                    edges = get_edges_from_masks(n, best_adj)
                    return {'n': 21, 'edges': edges}
    
    # If we found a graph with 0 size-6 violations, check if it's fully valid
    if best_cost == 0 and best_adj is not None:
        if is_valid_full(n, best_adj):
            edges = get_edges_from_masks(n, best_adj)
            return {'n': 21, 'edges': edges}

    # Step 3: Return the known 20-vertex graph
    return {'n': 20, 'edges': known_20_edges}
