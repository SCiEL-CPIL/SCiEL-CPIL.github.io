import time
import math
import numpy as np
from itertools import combinations

def construct() -> dict:
    start_time = time.time()
    deadline = start_time + 135.0

    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    def make_paley_adjacency(p):
        residues = set()
        for x in range(1, p):
            residues.add((x * x) % p)
        adjacency = []
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in residues:
                    adjacency.append(0)
                else:
                    adjacency.append(1)
        return adjacency

    def build_adjacency_matrix(p, adjacency):
        adj = np.zeros((p, p), dtype=bool)
        idx = 0
        for i in range(p):
            for j in range(i + 1, p):
                if adjacency[idx] == 0:
                    adj[i, j] = True
                else:
                    adj[j, i] = True
                idx += 1
        return adj

    def count_bad_4sets(adj, n):
        """Count 4-sets that lack a common dominator."""
        in_neighbors = adj.T
        bad_count = 0
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            # Check if any vertex dominates all four
            dominated = in_neighbors[a] & in_neighbors[b] & in_neighbors[c] & in_neighbors[d]
            if not dominated.any():
                bad_count += 1
        return bad_count

    def try_local_search(n, max_iter, time_limit):
        """Try to find a valid tournament on n vertices via local search."""
        # Start with random tournament
        adj = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                if np.random.random() < 0.5:
                    adj[i, j] = True
                else:
                    adj[j, i] = True
        
        current_bad = count_bad_4sets(adj, n)
        if current_bad == 0:
            return adj
        
        best_adj = adj.copy()
        best_bad = current_bad
        
        for iteration in range(max_iter):
            if time.time() > time_limit:
                break
            
            # Try flipping each edge and keep if it reduces bad count
            improved = False
            for i in range(n):
                if time.time() > time_limit:
                    break
                for j in range(i + 1, n):
                    if time.time() > time_limit:
                        break
                    
                    # Flip edge
                    if adj[i, j]:
                        adj[i, j] = False
                        adj[j, i] = True
                    else:
                        adj[i, j] = True
                        adj[j, i] = False
                    
                    new_bad = count_bad_4sets(adj, n)
                    
                    if new_bad < current_bad:
                        current_bad = new_bad
                        improved = True
                        if current_bad == 0:
                            return adj
                    else:
                        # Revert
                        if adj[i, j]:
                            adj[i, j] = False
                            adj[j, i] = True
                        else:
                            adj[i, j] = True
                            adj[j, i] = False
            
            if not improved:
                # Try random restarts
                for _ in range(5):
                    if time.time() > time_limit:
                        break
                    adj = np.zeros((n, n), dtype=bool)
                    for i in range(n):
                        for j in range(i + 1, n):
                            if np.random.random() < 0.5:
                                adj[i, j] = True
                            else:
                                adj[j, i] = True
                    current_bad = count_bad_4sets(adj, n)
                    if current_bad < best_bad:
                        best_bad = current_bad
                        best_adj = adj.copy()
                    if current_bad == 0:
                        return adj
            
            if current_bad < best_bad:
                best_bad = current_bad
                best_adj = adj.copy()
        
        return best_adj if best_bad == 0 else None

    def adjacency_to_list(adj, n):
        """Convert adjacency matrix to list format."""
        result = []
        for i in range(n):
            for j in range(i + 1, n):
                result.append(0 if adj[i, j] else 1)
        return result

    # First, try local search for smaller n values
    for n in range(31, 45):
        if time.time() > deadline - 30.0:
            break
        
        # Try multiple random restarts
        for attempt in range(3):
            if time.time() > deadline - 25.0:
                break
            
            adj = try_local_search(n, max_iter=200, time_limit=deadline - 20.0)
            if adj is not None:
                return {'n': n, 'orientation': adjacency_to_list(adj, n)}
    
    # Fall back to Paley tournament
    primes_to_try = []
    for p in range(31, 70):
        if is_prime(p) and p % 4 == 3:
            primes_to_try.append(p)
    primes_to_try.sort()
    
    for p in primes_to_try:
        if time.time() > deadline - 15.0:
            break
        adj_list = make_paley_adjacency(p)
        adj = build_adjacency_matrix(p, adj_list)
        if count_bad_4sets(adj, p) == 0:
            return {'n': p, 'orientation': adj_list}
    
    adj_67 = make_paley_adjacency(67)
    return {'n': 67, 'orientation': adj_67}
