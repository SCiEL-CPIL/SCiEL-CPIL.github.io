import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # We know n=20 works. Let's try to find one.
    # If we can't find one quickly, we return n=19 or lower.
    
    def is_valid(n, adj):
        """
        Check if the graph defined by adjacency matrix adj (n x n numpy array)
        has no regular induced subgraph on k >= 6 vertices.
        """
        # We only need to check k=6..n.
        # If a graph has no regular induced subgraph on 6 vertices, 
        # it might still have one on 7? Yes.
        # However, for random graphs, violations are mostly at small k.
        # We must check all k.
        
        # Optimization: 
        # If we find a regular subset, we return False.
        
        # Precompute degrees? 
        # For a subset S, the degree of v in S is sum_{u in S} A[v,u].
        
        # We will iterate k from 6 to n.
        # For each k, we iterate all subsets.
        # To speed up, we use a helper that checks a specific subset.
        
        def is_regular_subset(verts):
            # verts: list of vertex indices
            k = len(verts)
            # Extract submatrix
            sub_adj = adj[np.ix_(verts, verts)]
            # Degrees
            degrees = sub_adj.sum(axis=1)
            # Check if all equal
            return np.all(degrees == degrees[0])

        for k in range(6, n + 1):
            for subset in combinations(range(n), k):
                if is_regular_subset(subset):
                    return False
        return True

    def count_violations(n, adj, k):
        """
        Count the number of regular induced subgraphs of size k.
        Returns the count.
        """
        count = 0
        for subset in combinations(range(n), k):
            verts = list(subset)
            sub_adj = adj[np.ix_(verts, verts)]
            degrees = sub_adj.sum(axis=1)
            if np.all(degrees == degrees[0]):
                count += 1
        return count

    def find_violations(n, adj, k, limit=1000):
        """
        Find up to 'limit' regular induced subgraphs of size k.
        Returns list of tuples (subset, degrees).
        """
        violations = []
        for subset in combinations(range(n), k):
            verts = list(subset)
            sub_adj = adj[np.ix_(verts, verts)]
            degrees = sub_adj.sum(axis=1)
            if np.all(degrees == degrees[0]):
                violations.append((subset, degrees[0]))
                if len(violations) >= limit:
                    break
        return violations

    def local_search(n, adj, max_iters=500):
        """
        Perform local search to eliminate regular subgraphs.
        Focuses on k=6 first, then larger if needed.
        """
        current_adj = adj.copy()
        
        for i in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Check validity fully? Too slow.
            # Instead, check for violations at k=6.
            # If no violations at k=6, check k=7, etc.
            
            found_violation = False
            for k in range(6, n + 1):
                # Just find one violation
                violations = find_violations(n, current_adj, k, limit=1)
                if violations:
                    found_violation = True
                    subset, _ = violations[0]
                    
                    # Try to fix this violation by flipping an edge
                    # Which edge to flip?
                    # We want to break the regularity.
                    # Flipping an edge (u,v) changes degrees of u and v in this subset by +/- 1.
                    # Let's try flipping random edges within the subset or connecting to outside.
                    # Actually, flipping an edge inside the subset is most direct.
                    
                    # Try all edges within the subset
                    fixed = False
                    edges_to_try = []
                    for u_idx in range(len(subset)):
                        for v_idx in range(u_idx + 1, len(subset)):
                            u = subset[u_idx]
                            v = subset[v_idx]
                            edges_to_try.append((u, v))
                    
                    random.shuffle(edges_to_try)
                    
                    for u, v in edges_to_try:
                        # Flip edge
                        old_val = current_adj[u, v]
                        current_adj[u, v] = 1 - old_val
                        current_adj[v, u] = 1 - old_val
                        
                        # Check if this specific violation is fixed
                        # And if we didn't create a new one at k=6 (or same k)
                        # Simple heuristic: just check if this specific subset is still regular
                        verts = list(subset)
                        sub_adj = current_adj[np.ix_(verts, verts)]
                        degrees = sub_adj.sum(axis=1)
                        if not np.all(degrees == degrees[0]):
                            # Violation fixed. Accept move.
                            fixed = True
                            break
                        else:
                            # Revert
                            current_adj[u, v] = old_val
                            current_adj[v, u] = old_val
                            
                    if not fixed:
                        # Try flipping an edge connecting a vertex in subset to one outside?
                        # Or just accept a random move that reduces total violations?
                        # For now, if we can't fix this specific one easily, try next violation or break.
                        # Let's try a random flip anywhere and see if total violations decrease.
                        # This is expensive to check.
                        # Let's just break and try a different initialization or strategy.
                        pass
                    
                    break # Only handle one k at a time, restart loop to re-evaluate
                else:
                    # No violation at this k, try next k
                    continue
            
            if not found_violation:
                # No violations found for any k >= 6
                return current_adj, True

        return current_adj, False

    def try_random_graph(n, seed):
        random.seed(seed)
        # Initialize with p=0.5
        adj = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i, j] = 1
                    adj[j, i] = 1
        return adj

    best_n = 5
    best_edges = []
    
    # Try to find a valid graph on 20 vertices
    # Use local search with multiple seeds
    for seed in range(50):
        if time.time() > deadline - 10:
            break
            
        adj = try_random_graph(20, seed)
        
        # Run local search
        adj, valid = local_search(20, adj, max_iters=100)
        
        if valid:
            # Verify with full check just in case
            if is_valid(20, adj):
                edges = []
                for i in range(20):
                    for j in range(i + 1, 20):
                        if adj[i, j] == 1:
                            edges.append([i, j])
                return {'n': 20, 'edges': edges}

    # If 20 fails, try 19
    for seed in range(50):
        if time.time() > deadline - 5:
            break
        adj = try_random_graph(19, seed)
        adj, valid = local_search(19, adj, max_iters=100)
        if valid:
            if is_valid(19, adj):
                edges = []
                for i in range(19):
                    for j in range(i + 1, 19):
                        if adj[i, j] == 1:
                            edges.append([i, j])
                return {'n': 19, 'edges': edges}

    # Fallback: Return a small valid graph
    # n=5 is always valid (no k>=6)
    # Let's just return n=5 with no edges or some edges.
    # Actually, any graph on n<6 is valid.
    # Let's return n=19 with random edges if we have time? No, might be invalid.
    # Safe bet: n=5.
    return {'n': 5, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4]]}

# Test locally? No, just output.
