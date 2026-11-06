import random
import time
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = 143.0  # 7 seconds safety margin before 150s kill

    def fast_is_regular(adj, subset):
        """Check if induced subgraph on subset is regular using NumPy."""
        idx = np.array(subset)
        sub = adj[np.ix_(idx, idx)]
        degs = sub.sum(axis=1)
        return np.all(degs == degs[0])

    def count_violations_size6(adj, n):
        """Count and collect all regular 6-vertex induced subgraphs."""
        bads = []
        for subset in combinations(range(n), 6):
            if fast_is_regular(adj, subset):
                bads.append(subset)
        return bads

    def verify_full(adj, n):
        """Check all subsets of size >= 6 for regularity."""
        for k in range(6, n + 1):
            for subset in combinations(range(n), k):
                if fast_is_regular(adj, subset):
                    return False
        return True

    def get_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 1:
                    edges.append([i, j])
        return edges

    def random_graph(n, rng):
        adj = rng.integers(0, 2, (n, n))
        adj = (adj + adj.T) % 2
        np.fill_diagonal(adj, 0)
        return adj

    def solve_for_n(n, time_budget, rng):
        """Simulated annealing with random restarts to find a graph with no regular induced subgraph of size >= 6."""
        best_adj = None
        best_score = float('inf')
        
        while time.time() - start_time < deadline - 3.0:
            # Random restart
            adj = random_graph(n, rng)
            
            # Local search / simulated annealing
            for step in range(500):
                if time.time() - start_time > deadline - 3.0:
                    break
                
                bads = count_violations_size6(adj, n)
                score = len(bads)
                
                if score < best_score:
                    best_score = score
                    best_adj = adj.copy()
                
                if score == 0:
                    if verify_full(adj, n):
                        return n, get_edges(adj, n)
                    else:
                        break  # restart
                
                if bads:
                    # Pick a random bad subset and flip a random edge within it
                    bad_subset = rng.choice(bads)
                    u, v = rng.choice(list(combinations(bad_subset, 2)))
                    adj[u, v] = 1 - adj[u, v]
                    adj[v, u] = adj[u, v]
            
            # Occasionally do a larger perturbation
            if time.time() - start_time > deadline - 3.0:
                break
        
        # If we found a zero-size6-violation graph, verify fully
        if best_adj is not None and best_score == 0:
            if verify_full(best_adj, n):
                return n, get_edges(best_adj, n)
        
        return None, None

    rng = np.random.default_rng(42)
    
    # Try n=21 first (ambitious)
    time_left = deadline - time.time()
    if time_left > 40:
        result_n, result_edges = solve_for_n(21, time_left - 5, rng)
        if result_n is not None:
            return {'n': result_n, 'edges': result_edges}

    # Try n=20 (known feasible)
    time_left = deadline - time.time()
    if time_left > 15:
        result_n, result_edges = solve_for_n(20, time_left - 5, rng)
        if result_n is not None:
            return {'n': result_n, 'edges': result_edges}

    # Try n=19
    time_left = deadline - time.time()
    if time_left > 5:
        result_n, result_edges = solve_for_n(19, time_left - 2, rng)
        if result_n is not None:
            return {'n': result_n, 'edges': result_edges}

    # Fallback: small graph that trivially satisfies the condition
    return {'n': 5, 'edges': []}
