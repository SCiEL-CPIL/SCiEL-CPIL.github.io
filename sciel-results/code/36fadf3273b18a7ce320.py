import time
import numpy as np
import itertools

def paley_graph(q):
    """Construct the Paley graph on q vertices."""
    residues = {(x * x) % q for x in range(1, q)}
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                edges.append((i, j))
    return q, edges

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # Fallback: Paley graph on 37 vertices
    n_seed, edges_seed = paley_graph(37)
    best_n = n_seed
    best_edges = edges_seed
    
    # Helper to get indices for all 5-sets
    def get_5set_indices(n):
        combos = list(itertools.combinations(range(n), 5))
        N = len(combos)
        if N == 0:
            return np.empty((0, 10), dtype=np.int32), np.empty((0, 10), dtype=np.int32)
        rows = np.zeros((N, 10), dtype=np.int32)
        cols = np.zeros((N, 10), dtype=np.int32)
        # Vectorized generation of pairs is hard, loop is okay for N < 1M
        for k, comb in enumerate(combos):
            pairs = list(itertools.combinations(comb, 2))
            for j, (u, v) in enumerate(pairs):
                rows[k, j] = u
                cols[k, j] = v
        return rows, cols

    def count_violations(adj, rows, cols):
        if len(rows) == 0:
            return 0
        vals = adj[rows, cols]
        k5 = np.all(vals, axis=1)
        ind = np.all(~vals, axis=1)
        return np.sum(k5) + np.sum(ind)

    def search(n, time_limit):
        # Initialize adjacency matrix
        adj = np.zeros((n, n), dtype=bool)
        
        # Seed with Paley(37) if n >= 37
        if n >= 37:
            _, seed_edges = paley_graph(37)
            for u, v in seed_edges:
                adj[u, v] = True
                adj[v, u] = True
            # Randomize connections for new vertices
            for i in range(37, n):
                for j in range(i):
                    # If j < 37, it's a new connection. If j >= 37, it's between new vertices.
                    # We just randomize all edges involving vertices >= 37
                    adj[i, j] = np.random.rand() < 0.5
                    adj[j, i] = adj[i, j]
        else:
            for i in range(n):
                for j in range(i+1, n):
                    adj[i, j] = np.random.rand() < 0.5
                    adj[j, i] = adj[i, j]
        
        rows, cols = get_5set_indices(n)
        
        current_viol = count_violations(adj, rows, cols)
        best_adj = adj.copy()
        best_viol = current_viol
        
        if best_viol == 0:
            return best_adj, best_viol
            
        T = 1.0
        min_T = 0.01
        cooling = 0.995 # Slower cooling to explore more
        
        # Stagnation counter
        stagnation = 0
        max_stagnation = 5000
        
        while time.time() < time_limit:
            # Pick random edge to flip
            u = np.random.randint(0, n)
            v = np.random.randint(0, n)
            if u == v:
                continue
            
            # Flip
            old_val = adj[u, v]
            adj[u, v] = not old_val
            adj[v, u] = adj[u, v]
            
            new_viol = count_violations(adj, rows, cols)
            
            delta = new_viol - current_viol
            
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                current_viol = new_viol
                if new_viol < best_viol:
                    best_viol = new_viol
                    best_adj = adj.copy()
                    stagnation = 0
                    if best_viol == 0:
                        return best_adj, best_viol
                else:
                    stagnation += 1
            else:
                # Revert
                adj[u, v] = old_val
                adj[v, u] = old_val
                stagnation += 1
                
            T = max(min_T, T * cooling)
            
            # Restart logic: if stuck for too long, re-randomize the new vertices
            if stagnation > max_stagnation:
                # Re-initialize edges involving vertices >= 37
                if n >= 37:
                    for i in range(37, n):
                        for j in range(i):
                            adj[i, j] = np.random.rand() < 0.5
                            adj[j, i] = adj[i, j]
                    current_viol = count_violations(adj, rows, cols)
                    T = 1.0
                    stagnation = 0
                else:
                    # Full random restart
                    for i in range(n):
                        for j in range(i+1, n):
                            adj[i, j] = np.random.rand() < 0.5
                            adj[j, i] = adj[i, j]
                    current_viol = count_violations(adj, rows, cols)
                    T = 1.0
                    stagnation = 0

        return best_adj, best_viol

    # Try to find a valid graph for n > 37
    # We prioritize larger n, but we must ensure we have time.
    # If we find a valid graph for n=38, we should try 39, etc.
    # But if we don't find 38 quickly, we should move to 39?
    # No, finding 38 is easier than 39.
    # Let's allocate time based on remaining budget.
    
    for n in range(38, 43):
        if time.time() > deadline:
            break
        
        # Estimate time remaining
        remaining = deadline - time.time()
        if remaining < 10:
            break
            
        # Allocate time for this n
        # Simple heuristic: give each n a chunk of time
        # If we find a solution, we update best and continue to next n
        time_for_n = min(60, remaining)
        
        adj, viol = search(n, time.time() + time_for_n)
        
        if viol == 0:
            best_n = n
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i, j]:
                        edges.append((i, j))
            best_edges = edges
            # Continue to try larger n
            
    return best_n, best_edges
