"""Find a graph on up to 22 vertices with no regular induced subgraph of size >= 6."""
import time
import random
import numpy as np

DEADLINE = time.time() + 118  # 120s limit, 2s safety
SAFETY = 2.0


def _count_violations_fast(adj, n):
    """Count regular induced subgraphs of size >= 6.
    adj: numpy array of shape (n, n) with 0/1 entries.
    Returns (count, list_of_violating_subsets).
    """
    count = 0
    violations = []
    
    # For each subset of size k >= 6, check if induced subgraph is regular
    # Use numpy for speed where possible
    for k in range(6, n + 1):
        # Iterate over all k-combinations
        # For n=20, k=6: C(20,6)=38760, k=7: 77520, etc.
        # Total for n=20 is about 1M subsets - too slow for pure iteration
        # But we can early-exit and use smarter checks
        
        # Generate combinations using itertools for memory efficiency
        import itertools
        for subset in itertools.combinations(range(n), k):
            # Extract adjacency submatrix
            sub = adj[np.ix_(subset, subset)]
            # Compute degrees (sum of each row, minus diagonal if self-loops, but no self-loops)
            degs = sub.sum(axis=1)
            # Check if all degrees equal
            if np.all(degs == degs[0]):
                count += 1
                if len(violations) < 10:
                    violations.append(list(subset))
                if count > 100:
                    return count, violations
    return count, violations


def _count_violations_optimized(adj, n):
    """Optimized violation counting with early pruning.
    adj: list of numpy arrays or use integer bitmasks for speed.
    """
    # Use integer bitmasks for maximum speed
    # adj_mask[v] is an integer where bit u is set if edge (v,u) exists
    # Convert from numpy adjacency if needed
    
    # Precompute: for each vertex, its neighbors as a bitmask
    # We'll use Python integers for bit operations
    
    count = 0
    violations = []
    
    # For small subsets (size 6), we can check quickly
    # For larger subsets, we need to be more careful
    
    import itertools
    
    for k in range(6, n + 1):
        for subset in itertools.combinations(range(n), k):
            # Compute degree of each vertex in the induced subgraph
            degs = []
            for v in subset:
                d = 0
                for u in subset:
                    if u != v and (adj[v] >> u) & 1:
                        d += 1
                degs.append(d)
            
            if all(d == degs[0] for d in degs):
                count += 1
                if len(violations) < 10:
                    violations.append(list(subset))
                if count > 100:
                    return count, violations
    
    return count, violations


def _local_search_bitmask(adj, n, max_iters=10000):
    """Local search using bitmask adjacency.
    adj: list of integers, each is a bitmask of neighbors.
    Returns (adj, violations_count).
    """
    import itertools
    
    def count_violations(adj_local):
        """Count violations for bitmask adjacency."""
        count = 0
        violations = []
        for k in range(6, n + 1):
            for subset in itertools.combinations(range(n), k):
                degs = []
                for v in subset:
                    d = 0
                    for u in subset:
                        if u != v and (adj_local[v] >> u) & 1:
                            d += 1
                    degs.append(d)
                if all(d == degs[0] for d in degs):
                    count += 1
                    if len(violations) < 5:
                        violations.append(list(subset))
                    if count > 50:
                        break
            if count > 50:
                break
        return count, violations
    
    current_count, current_subs = count_violations(adj)
    if current_count == 0:
        return adj, 0
    
    best_adj = adj[:]
    best_count = current_count
    
    for it in range(max_iters):
        if time.time() > DEADLINE - SAFETY:
            break
        
        if not current_subs:
            break
        
        # Try to fix one violating subset
        subset = current_subs[0]
        improved = False
        
        # Try flipping edges within the subset
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                u, v = subset[i], subset[j]
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                
                new_count, _ = count_violations(adj)
                
                if new_count < best_count:
                    best_count = new_count
                    best_adj = adj[:]
                    improved = True
                    if new_count == 0:
                        return adj, 0
                else:
                    # Revert
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                
                if time.time() > DEADLINE - SAFETY:
                    return best_adj, best_count
        
        # Try flipping edges incident to subset vertices
        if not improved:
            for u in subset:
                for v in range(n):
                    if v in subset:
                        continue
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                    
                    new_count, _ = count_violations(adj)
                    
                    if new_count < best_count:
                        best_count = new_count
                        best_adj = adj[:]
                        improved = True
                        if new_count == 0:
                            return adj, 0
                    else:
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                    
                    if time.time() > DEADLINE - SAFETY:
                        return best_adj, best_count
        
        if not improved:
            # Random restart
            for _ in range(5):
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u != v:
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
            current_count, current_subs = count_violations(adj)
            if current_count < best_count:
                best_count = current_count
                best_adj = adj[:]
    
    return best_adj, best_count


def construct():
    """Construct a graph with no regular induced subgraph of size >= 6.
    Maximize n.
    """
    start_time = time.time()
    
    best_n = 5
    best_edges = []
    
    # Try to build graphs of increasing size
    for n in range(10, 23):
        if time.time() > DEADLINE - 5:
            break
        
        found = False
        max_restarts = 30 if n <= 18 else 20
        
        for restart in range(max_restarts):
            if time.time() > DEADLINE - SAFETY:
                break
            
            # Initialize with random graph using bitmasks
            adj = [0] * n
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.5:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            # Local search
            adj, final_violations = _local_search_bitmask(adj, n, max_iters=5000)
            
            if final_violations == 0:
                found = True
                break
        
        if found:
            best_n = n
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        edges.append([u, v])
            best_edges = edges
    
    # Verify the result
    if best_n >= 6 and best_edges:
        adj = [0] * best_n
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Quick verification
        import itertools
        violations = 0
        for k in range(6, best_n + 1):
            for subset in itertools.combinations(range(best_n), k):
                degs = []
                for v in subset:
                    d = 0
                    for u in subset:
                        if u != v and (adj[v] >> u) & 1:
                            d += 1
                    degs.append(d)
                if all(d == degs[0] for d in degs):
                    violations += 1
                    break
            if violations > 0:
                break
        
        if violations > 0:
            best_n = 5
            best_edges = []
    
    return {'n': best_n, 'edges': best_edges}
