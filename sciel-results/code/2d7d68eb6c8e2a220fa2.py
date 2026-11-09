import random
import time
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    def is_regular_subset(adj_bits, subset_mask, n):
        """Check if induced subgraph on subset_mask is regular.
        adj_bits: list of int bitmasks, adj_bits[i] has bit j set if edge i-j
        subset_mask: int bitmask of vertices in subset
        """
        # Count bits in subset
        k = bin(subset_mask).count('1')
        if k < 6:
            return False
        
        # For each vertex in subset, count degree within subset
        degrees = []
        for i in range(n):
            if (subset_mask >> i) & 1:
                deg = bin(adj_bits[i] & subset_mask).count('1')
                degrees.append(deg)
        
        # Check if all degrees equal
        if len(set(degrees)) == 1:
            return True
        return False
    
    def verify_graph(n, adj_bits):
        """Check no subset of size >= 6 induces a regular subgraph."""
        # Check all subsets of size 6 (if any size >= 6 is regular, 
        # we can focus on size 6 since a regular subgraph on k>6 vertices
        # has all vertices degree d, and any 6-vertex subset would have
        # degrees that are... actually not necessarily regular.
        # But we must check all sizes >= 6.
        # Optimization: if a subset of size k is regular with degree d,
        # then d < k. For k=6, d in {0,1,2,3,4,5}.
        # We need to check all subset sizes from 6 to n.
        
        # For efficiency, check size 6 first (most common violation),
        # then larger sizes
        for size in range(6, n + 1):
            for subset in itertools.combinations(range(n), size):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                if is_regular_subset(adj_bits, mask, n):
                    return False
        return True
    
    def verify_graph_fast(n, adj_bits):
        """Faster verification using precomputed degree arrays."""
        # Precompute for each vertex, the bitmask of neighbors
        # For a subset, degree of vertex i = popcount(adj_bits[i] & subset_mask)
        
        # Check size 6 subsets
        for subset in itertools.combinations(range(n), 6):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            # Compute degrees
            degs = []
            for i in subset:
                degs.append(bin(adj_bits[i] & mask).count('1'))
            if len(set(degs)) == 1:
                return False
        
        # Check size 7+ subsets (less likely to be regular)
        for size in range(7, n + 1):
            for subset in itertools.combinations(range(n), size):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                degs = []
                for i in subset:
                    degs.append(bin(adj_bits[i] & mask).count('1'))
                if len(set(degs)) == 1:
                    return False
        return True
    
    def random_graph(n):
        """Generate a random graph on n vertices."""
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj
    
    def local_search(n, adj, max_iter=5000):
        """Local search: flip edges to eliminate regular induced subgraphs."""
        current_adj = adj[:]
        current_valid = verify_graph_fast(n, current_adj)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            if current_valid:
                return current_adj, True
            
            # Find a violating subset
            violation = None
            for size in range(6, n + 1):
                for subset in itertools.combinations(range(n), size):
                    mask = 0
                    for v in subset:
                        mask |= (1 << v)
                    degs = []
                    for i in subset:
                        degs.append(bin(current_adj[i] & mask).count('1'))
                    if len(set(degs)) == 1:
                        violation = (subset, mask, degs[0])
                        break
                if violation:
                    break
            
            if violation is None:
                return current_adj, True
            
            subset, mask, target_deg = violation
            # Try flipping an edge to break regularity
            # Pick a random vertex in subset and try adding/removing an edge
            random.shuffle(subset)
            improved = False
            for i in subset:
                for j in range(n):
                    if j == i:
                        continue
                    # Try flipping edge (i,j)
                    if (current_adj[i] >> j) & 1:
                        # Remove edge
                        current_adj[i] &= ~(1 << j)
                        current_adj[j] &= ~(1 << i)
                    else:
                        # Add edge
                        current_adj[i] |= (1 << j)
                        current_adj[j] |= (1 << i)
                    
                    # Check if this broke the violation
                    degs_new = []
                    for v in subset:
                        degs_new.append(bin(current_adj[v] & mask).count('1'))
                    if len(set(degs_new)) > 1:
                        # Violation broken, check full validity
                        if verify_graph_fast(n, current_adj):
                            return current_adj, True
                        # Partial improvement, keep it
                        improved = True
                        break
                    else:
                        # Revert
                        if (current_adj[i] >> j) & 1:
                            current_adj[i] &= ~(1 << j)
                            current_adj[j] &= ~(1 << i)
                        else:
                            current_adj[i] |= (1 << j)
                            current_adj[j] |= (1 << i)
                if improved:
                    break
            
            if not improved:
                # Random perturbation
                for _ in range(3):
                    i = random.randint(0, n - 1)
                    j = random.randint(0, n - 1)
                    if i == j:
                        continue
                    if (current_adj[i] >> j) & 1:
                        current_adj[i] &= ~(1 << j)
                        current_adj[j] &= ~(1 << i)
                    else:
                        current_adj[i] |= (1 << j)
                        current_adj[j] |= (1 << i)
        
        return current_adj, verify_graph_fast(n, current_adj)
    
    best_n = 5
    best_adj = [0] * 5
    
    # Try to find valid graphs for increasing n
    for n in range(6, 21):  # Try up to 20 (known achievable)
        if time.time() > deadline - 30:
            break
        
        found = False
        for attempt in range(200):
            if time.time() > deadline - 10:
                break
            
            adj = random_graph(n)
            valid = verify_graph_fast(n, adj)
            if valid:
                best_n = n
                best_adj = adj
                found = True
                break
            
            # Local search
            adj2, valid2 = local_search(n, adj, max_iter=2000)
            if valid2:
                best_n = n
                best_adj = adj2
                found = True
                break
        
        if found:
            # Try to extend to n+1
            continue
        else:
            # If we can't find n, try n-1 with more attempts
            # Actually, let's try harder with local search
            pass
    
    # If we found a valid graph, return it
    if best_n >= 6:
        edges = []
        for i in range(best_n):
            for j in range(i + 1, best_n):
                if (best_adj[i] >> j) & 1:
                    edges.append([i, j])
        return {'n': best_n, 'edges': edges}
    else:
        return {'n': 5, 'edges': []}
