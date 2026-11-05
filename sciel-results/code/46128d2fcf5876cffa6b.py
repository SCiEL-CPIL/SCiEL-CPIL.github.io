"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def construct():
    n = 40
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def is_c6_free(A, u, v):
        """Check if adding edge (u, v) to graph with adjacency matrix A creates a C6."""
        if A[u, v] != 0:
            return False
            
        # Calculate A^5
        # A^2 = A @ A
        # A^3 = A^2 @ A
        # A^5 = A^3 @ A^2
        A2 = A @ A
        A3 = A2 @ A
        A5 = A3 @ A2
        
        return A5[u, v] == 0

    best_edges = []
    best_count = 0
    start_time = time.time()
    time_limit = 115.0 # Leave 5s margin

    # Strategy 1: Degree-based Greedy
    def run_degree_greedy(seed):
        random.seed(seed)
        np.random.seed(seed)
        A = np.zeros((n, n), dtype=np.int32)
        edges = []
        edge_set = set()
        deg = np.zeros(n, dtype=np.int32)
        remaining = set(all_edges)
        
        while remaining:
            # Find edge with max degree sum
            best_e = None
            best_val = -1
            for u, v in remaining:
                val = deg[u] + deg[v]
                if val > best_val:
                    best_val = val
                    best_e = (u, v)
            
            if best_e is None:
                break
            
            u, v = best_e
            remaining.remove(best_e)
            
            if A[u, v] == 0 and is_c6_free(A, u, v):
                A[u, v] = 1
                A[v, u] = 1
                edges.append((u, v))
                edge_set.add((u, v))
                deg[u] += 1
                deg[v] += 1
                
        return edges

    # Strategy 2: Random Greedy + 2-for-3 Local Search
    def run_random_greedy_2x3(seed, local_search_iters=5000):
        random.seed(seed)
        np.random.seed(seed)
        A = np.zeros((n, n), dtype=np.int32)
        edges = []
        edge_set = set()
        
        # Greedy Phase
        candidates = list(all_edges)
        random.shuffle(candidates)
        for u, v in candidates:
            if A[u, v] == 0:
                if is_c6_free(A, u, v):
                    A[u, v] = 1
                    A[v, u] = 1
                    edges.append((u, v))
                    edge_set.add((u, v))
        
        current_count = len(edges)
        
        # Local Search Phase: 2-remove, 3-add
        for _ in range(local_search_iters):
            if time.time() - start_time > time_limit:
                break
            if len(edges) < 2:
                break
            
            # Pick 2 edges to remove
            idx1 = random.randint(0, len(edges)-1)
            idx2 = random.randint(0, len(edges)-1)
            if idx1 == idx2:
                continue
                
            e1 = edges[idx1]
            e2 = edges[idx2]
            
            # Temporarily remove
            A[e1[0], e1[1]] = 0
            A[e1[1], e1[0]] = 0
            A[e2[0], e2[1]] = 0
            A[e2[1], e2[0]] = 0
            
            # Try to add 3 edges
            added_edges = []
            success = True
            for _ in range(3):
                # Find a valid non-edge
                u, v = -1, -1
                for _attempt in range(50):
                    ru = random.randint(0, n-1)
                    rv = random.randint(0, n-1)
                    if ru == rv:
                        continue
                    if A[ru, rv] == 1:
                        continue
                    if ru > rv:
                        ru, rv = rv, ru
                    if (ru, rv) in edge_set:
                        continue
                    u, v = ru, rv
                    break
                
                if u == -1:
                    success = False
                    break
                    
                if is_c6_free(A, u, v):
                    A[u, v] = 1
                    A[v, u] = 1
                    added_edges.append((u, v))
                else:
                    success = False
                    break
            
            if success and len(added_edges) == 3:
                # Accept
                # Remove e1, e2 from list
                # Careful with indices
                edges.remove(e1)
                edges.remove(e2)
                edge_set.discard(e1)
                edge_set.discard(e2)
                
                for e in added_edges:
                    edges.append(e)
                    edge_set.add(e)
                    
                current_count += 1
            else:
                # Revert
                A[e1[0], e1[1]] = 1
                A[e1[1], e1[0]] = 1
                A[e2[0], e2[1]] = 1
                A[e2[1], e2[0]] = 1
                for e in added_edges:
                    A[e[0], e[1]] = 0
                    A[e[1], e[0]] = 0

        return edges

    # Run strategies
    # 1. Degree Greedy
    edges = run_degree_greedy(42)
    if len(edges) > best_count:
        best_count = len(edges)
        best_edges = edges.copy()
        
    # 2. Random Greedy + 2x3 Local Search (multiple restarts)
    for i in range(5):
        if time.time() - start_time > time_limit:
            break
        edges = run_random_greedy_2x3(100 + i, local_search_iters=5000)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges.copy()

    return best_edges
# EVOLVE-BLOCK-END
