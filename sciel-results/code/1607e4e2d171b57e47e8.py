import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    best_result = {'n': 39, 'edges': []}
    
    def check_c4(adj, n):
        """Check if adjacency matrix has a C4. Returns (has_c4, c4_vertices) or (False, None)."""
        # A C4 exists if there are two vertices with at least 2 common neighbors
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i, j]:
                    common = np.sum(adj[i] & adj[j])
                    if common >= 2:
                        return True, (i, j)
        return False, None
    
    def count_c4s(adj, n):
        """Count number of C4s (each counted once)."""
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i, j]:
                    common = np.sum(adj[i] & adj[j])
                    if common >= 2:
                        count += 1
        return count
    
    def get_c4s(adj, n, max_c4s=100):
        """Get up to max_c4s C4s."""
        c4s = []
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i, j]:
                    common = np.where(adj[i] & adj[j])[0]
                    if len(common) >= 2:
                        # Pick two common neighbors
                        idx = random.sample(range(len(common)), 2)
                        c4s.append((i, common[idx[0]], j, common[idx[1]]))
                        if len(c4s) >= max_c4s:
                            return c4s
        return c4s
    
    def min_degree(adj, n):
        return int(np.min(adj.sum(axis=1)))
    
    def try_construct(n, num_trials=50, max_iters=20000):
        """Try to construct a C4-free graph on n vertices with min degree >= n-39."""
        min_deg_required = n - 39
        if min_deg_required < 1:
            min_deg_required = 1
        
        for trial in range(num_trials):
            if time.time() > deadline - 10:
                return None
            
            # Initialize: random graph with appropriate edge probability
            # Expected degree ~ min_deg_required
            p = min_deg_required / (n - 1) * 1.2  # slightly above target
            p = min(p, 0.9)
            
            adj = np.zeros((n, n), dtype=np.int8)
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < p:
                        adj[i, j] = 1
                        adj[j, i] = 1
            
            # Ensure minimum degree
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    return None
                
                # Check and fix minimum degree
                degrees = adj.sum(axis=1)
                min_deg = int(np.min(degrees))
                
                if min_deg < min_deg_required:
                    # Find vertex with low degree and add edges
                    low_v = int(np.argmin(degrees))
                    # Find non-neighbors
                    non_neigh = np.where(adj[low_v] == 0)[0]
                    non_neigh = non_neigh[non_neigh != low_v]
                    if len(non_neigh) == 0:
                        break
                    # Add an edge to a random non-neighbor
                    target = random.choice(non_neigh)
                    adj[low_v, target] = 1
                    adj[target, low_v] = 1
                    continue
                
                # Check for C4s
                has_c4, _ = check_c4(adj, n)
                if not has_c4:
                    # Success!
                    edges = []
                    for i in range(n):
                        for j in range(i+1, n):
                            if adj[i, j]:
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
                
                # Remove an edge from a C4
                c4s = get_c4s(adj, n, max_c4s=10)
                if not c4s:
                    break
                i, a, j, b = random.choice(c4s)
                # Remove one of the edges: (i,a), (a,j), (j,b), (b,i)
                edge_to_remove = random.choice([(i,a), (a,j), (j,b), (b,i)])
                u, v = edge_to_remove
                if adj[u, v]:
                    adj[u, v] = 0
                    adj[v, u] = 0
                    # Check if this breaks min degree
                    new_deg_u = int(adj[u].sum())
                    new_deg_v = int(adj[v].sum())
                    if new_deg_u < min_deg_required or new_deg_v < min_deg_required:
                        # Revert
                        adj[u, v] = 1
                        adj[v, u] = 1
                        # Try removing a different edge from the C4
                        for edge in [(i,a), (a,j), (j,b), (b,i)]:
                            u2, v2 = edge
                            if adj[u2, v2]:
                                du = int(adj[u2].sum()) - 1
                                dv = int(adj[v2].sum()) - 1
                                if du >= min_deg_required and dv >= min_deg_required:
                                    adj[u2, v2] = 0
                                    adj[v2, u2] = 0
                                    break
                        else:
                            # Can't remove any edge without violating min degree
                            # Try a different C4 or give up this trial
                            if random.random() < 0.1:
                                break
        
        return None
    
    # Try from largest n down
    for n in range(47, 38, -1):
        if time.time() > deadline - 5:
            break
        
        result = try_construct(n)
        if result is not None:
            best_result = result
            break
    
    return best_result
