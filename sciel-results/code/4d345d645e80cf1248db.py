import random
import time
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # 5 seconds safety margin before 150s kill

    def is_regular_subset(subset, adj):
        """Check if a subset induces a regular subgraph."""
        k = len(subset)
        if k < 2:
            return True
        # Compute degrees within the induced subgraph
        idx_to_vertex = {i: subset[i] for i in range(k)}
        degrees = [0] * k
        for i in range(k):
            u = subset[i]
            for j in range(k):
                if i != j and adj[u, subset[j]] == 1:
                    degrees[i] += 1
        first_deg = degrees[0]
        for d in degrees[1:]:
            if d != first_deg:
                return False
        return True

    def compute_all_6sets(n):
        """Precompute all 6-vertex subsets."""
        return list(combinations(range(n), 6))

    def get_subsets_containing_edge(u, v, all_6sets):
        """Get all 6-sets that contain both u and v."""
        result = []
        for s in all_6sets:
            if u in s and v in s:
                result.append(s)
        return result

    def count_regular_6sets(adj, n, all_6sets=None):
        """Count how many 6-sets induce regular subgraphs."""
        if all_6sets is None:
            all_6sets = compute_all_6sets(n)
        count = 0
        bad_sets = []
        for s in all_6sets:
            if is_regular_subset(s, adj):
                count += 1
                bad_sets.append(s)
        return count, bad_sets, all_6sets

    def verify_no_regular_induced(adj, n):
        """Verify no induced subgraph of size >= 6 is regular."""
        for k in range(6, n + 1):
            for subset in combinations(range(n), k):
                if is_regular_subset(subset, adj):
                    return False
        return True

    def score_edge_flip(adj, n, u, v, all_6sets, bad_sets_cache=None):
        """
        Score an edge flip by counting how many regular 6-sets containing (u,v)
        would become non-regular, minus how many non-regular ones would become regular.
        Positive score = good flip.
        """
        # Get all 6-sets containing both u and v
        affected_sets = [s for s in all_6sets if u in s and v in s]
        
        if not affected_sets:
            return 0
        
        score = 0
        current_val = adj[u, v]
        
        for s in affected_sets:
            is_reg_now = is_regular_subset(s, adj)
            
            # Simulate the flip
            adj[u, v] = 1 - current_val
            adj[v, u] = adj[u, v]
            is_reg_after = is_regular_subset(s, adj)
            adj[u, v] = current_val
            adj[v, u] = adj[u, v]
            
            if is_reg_now and not is_reg_after:
                score += 1  # Resolves a violation
            elif not is_reg_now and is_reg_after:
                score -= 1  # Creates a new violation
            # else: no change in regularity status
        
        return score

    def solve_for_n(n, time_limit, all_6sets=None):
        """
        Local search with constraint propagation:
        - Precompute all 6-sets
        - Track violations
        - Flip edge that maximizes net conflict resolution
        """
        if all_6sets is None:
            all_6sets = compute_all_6sets(n)
        
        # Initialize random graph
        adj = np.random.randint(0, 2, (n, n))
        adj = (adj + adj.T) % 2
        np.fill_diagonal(adj, 0)
        
        best_adj = adj.copy()
        best_bad_count = float('inf')
        
        # Precompute which 6-sets contain each edge for faster lookup
        edge_to_sets = {}
        for i in range(n):
            for j in range(i + 1, n):
                edge_to_sets[(i, j)] = [s for s in all_6sets if i in s and j in s]
        
        while time.time() < start_time + time_limit:
            # Count current violations
            bad_count, bad_sets, _ = count_regular_6sets(adj, n, all_6sets)
            
            if bad_count < best_bad_count:
                best_bad_count = bad_count
                best_adj = adj.copy()
            
            if bad_count == 0:
                if verify_no_regular_induced(adj, n):
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i, j] == 1:
                                edges.append([i, j])
                    return n, edges
                else:
                    # Restart with new random graph
                    adj = np.random.randint(0, 2, (n, n))
                    adj = (adj + adj.T) % 2
                    np.fill_diagonal(adj, 0)
                    continue
            
            if not bad_sets:
                break
            
            # Score all possible edge flips
            best_score = -float('inf')
            best_edge = None
            best_flip_val = None
            
            # Sample a subset of edges to consider for speed
            all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            
            # Prioritize edges that appear in many bad sets
            edge_in_bad_count = {}
            for bs in bad_sets:
                for i in range(6):
                    for j in range(i + 1, 6):
                        e = tuple(sorted((bs[i], bs[j])))
                        edge_in_bad_count[e] = edge_in_bad_count.get(e, 0) + 1
            
            # Consider edges in bad sets first, then sample others
            candidate_edges = list(edge_in_bad_count.keys())
            if len(candidate_edges) < 20:
                # Add some random edges
                other_edges = [e for e in all_edges if e not in edge_in_bad_count]
                if other_edges:
                    extra = random.sample(other_edges, min(10, len(other_edges)))
                    candidate_edges.extend(extra)
            
            random.shuffle(candidate_edges)
            # Limit candidates for speed
            candidate_edges = candidate_edges[:50]
            
            for (u, v) in candidate_edges:
                # Try both flip directions
                for flip_val in [0, 1]:
                    if flip_val == adj[u, v]:
                        continue  # No change
                    s = score_edge_flip(adj, n, u, v, all_6sets)
                    if s > best_score:
                        best_score = s
                        best_edge = (u, v)
                        best_flip_val = flip_val
            
            if best_edge is None or best_score <= 0:
                # No good flip found, try random edge
                u, v = random.choice(all_edges)
                adj[u, v] = 1 - adj[u, v]
                adj[v, u] = adj[u, v]
            else:
                u, v = best_edge
                adj[u, v] = best_flip_val
                adj[v, u] = best_flip_val
        
        # Return best found
        if best_bad_count < float('inf'):
            # Check if best is valid
            if verify_no_regular_induced(best_adj, n):
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if best_adj[i, j] == 1:
                            edges.append([i, j])
                return n, edges
        
        return None, None

    # Try n=21 first (ambitious)
    time_left = deadline - time.time()
    if time_left > 40:
        result_n, result_edges = solve_for_n(21, time_left - 5)
        if result_n is not None:
            return {'n': result_n, 'edges': result_edges}

    # Try n=20 (known feasible)
    time_left = deadline - time.time()
    if time_left > 20:
        result_n, result_edges = solve_for_n(20, time_left - 5)
        if result_n is not None:
            return {'n': result_n, 'edges': result_edges}

    # Try n=19
    time_left = deadline - time.time()
    if time_left > 10:
        result_n, result_edges = solve_for_n(19, time_left - 5)
        if result_n is not None:
            return {'n': result_n, 'edges': result_edges}

    # Try n=18
    time_left = deadline - time.time()
    if time_left > 5:
        result_n, result_edges = solve_for_n(18, time_left - 2)
        if result_n is not None:
            return {'n': result_n, 'edges': result_edges}

    # Fallback
    return {'n': 5, 'edges': []}
