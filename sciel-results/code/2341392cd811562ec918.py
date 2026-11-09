import time
import numpy as np
import itertools
import random

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
    deadline = start_time + 280  # Safety margin for 330s limit
    
    # Fallback: Paley graph on 37 vertices
    n_seed, edges_seed = paley_graph(37)
    best_n = n_seed
    best_edges = edges_seed
    
    def build_5set_structure(n):
        """Pre-compute all 5-sets and their edge pairs for fast violation counting."""
        combos = list(itertools.combinations(range(n), 5))
        N = len(combos)
        if N == 0:
            return np.empty((0, 10), dtype=np.int32), np.empty((0, 10), dtype=np.int32), combos
        
        rows = np.zeros((N, 10), dtype=np.int32)
        cols = np.zeros((N, 10), dtype=np.int32)
        # Also store edge lists for each 5-set for incremental updates?
        # Actually, for incremental updates in SA, we need to know which 5-sets contain a specific edge.
        # So we need a mapping: edge (u,v) -> list of 5-set indices.
        
        return rows, cols, combos

    def count_violations_full(adj, rows, cols):
        if len(rows) == 0:
            return 0
        vals = adj[rows, cols]
        k5 = np.all(vals, axis=1)
        ind = np.all(~vals, axis=1)
        return int(np.sum(k5) + np.sum(ind))

    def search_circulant(n, time_limit):
        """
        Search for a valid circulant graph on n vertices.
        """
        if time.time() > time_limit:
            return None
        
        rows, cols, combos = build_5set_structure(n)
        
        m = n // 2
        
        best_viol = float('inf')
        best_mask = None
        
        trials = 0
        while time.time() < time_limit:
            trials += 1
            mask = np.random.rand(m) < 0.5
            
            # Build adjacency matrix efficiently
            row0 = np.zeros(n, dtype=bool)
            for d in range(1, m + 1):
                if mask[d-1]:
                    row0[d] = True
                    row0[n-d] = True
            
            # Construct full matrix
            # adj[i, j] = row0[(j-i)%n]
            # We can construct this using broadcasting or indexing
            # indices = np.arange(n)
            # adj = row0[(indices[None, :] - indices[:, None]) % n]
            
            # Let's try the broadcasting way, it's fast for n=42
            indices = np.arange(n)
            diff = (indices[None, :] - indices[:, None]) % n
            adj = row0[diff]
            
            # Symmetrize? Circulant is symmetric if row0 is symmetric.
            # row0[d] = row0[n-d]. So yes.
            
            viol = count_violations_full(adj, rows, cols)
            
            if viol == 0:
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i, j]:
                            edges.append((i, j))
                return n, edges
            
            if viol < best_viol:
                best_viol = viol
                best_mask = mask.copy()

        return None

    def search_general(n, time_limit, seed_adj=None):
        """
        Simulated Annealing search for general graph.
        """
        if time.time() > time_limit:
            return None, float('inf')
            
        rows, cols, combos = build_5set_structure(n)
        N_combos = len(combos)
        
        if N_combos == 0:
            return np.zeros((n, n), dtype=bool), 0

        # Build edge_to_5sets mapping
        # This is crucial for incremental updates
        edge_to_5sets = {}
        for k, comb in enumerate(combos):
            for u, v in itertools.combinations(comb, 2):
                key = (min(u,v), max(u,v))
                if key not in edge_to_5sets:
                    edge_to_5sets[key] = []
                edge_to_5sets[key].append(k)
        
        # Initialize adjacency
        if seed_adj is not None:
            adj = seed_adj.copy()
        elif n >= 37:
            adj = np.zeros((n, n), dtype=bool)
            _, seed_edges = paley_graph(37)
            for u, v in seed_edges:
                adj[u, v] = True
                adj[v, u] = True
            # Randomize new vertices
            for i in range(37, n):
                for j in range(i):
                    adj[i, j] = np.random.rand() < 0.5
                    adj[j, i] = adj[i, j]
        else:
            adj = np.zeros((n, n), dtype=bool)
            for i in range(n):
                for j in range(i+1, n):
                    adj[i, j] = np.random.rand() < 0.5
                    adj[j, i] = adj[i, j]
        
        # Pre-compute edge counts for each 5-set
        # edge_counts[k] = number of edges in combo k
        edge_counts = np.zeros(N_combos, dtype=np.int32)
        for k, comb in enumerate(combos):
            count = 0
            for u, v in itertools.combinations(comb, 2):
                if adj[u, v]:
                    count += 1
            edge_counts[k] = count
        
        # Count violations (0 or 10 edges)
        current_viol = int(np.sum((edge_counts == 0) | (edge_counts == 10)))
        best_viol = current_viol
        best_adj = adj.copy()
        
        if best_viol == 0:
            return best_adj, 0
            
        T = 1.0
        min_T = 0.01
        cooling = 0.999
        stagnation = 0
        max_stagnation = 5000
        
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        # Pre-compute the list of 5-set indices for each edge to avoid dict lookup overhead?
        # edge_to_5sets is a dict of lists. Accessing it is fast enough.
        
        while time.time() < time_limit:
            idx = np.random.randint(0, len(all_edges))
            u, v = all_edges[idx]
            
            old_val = adj[u, v]
            adj[u, v] = not old_val
            adj[v, u] = adj[u, v]
            
            affected = edge_to_5sets.get((u, v), [])
            
            # Update edge counts and violation count incrementally
            # delta_viol = change in number of sets with 0 or 10 edges
            delta_viol = 0
            if old_val:
                # Edge removed: counts decrease by 1
                for k in affected:
                    old_count = edge_counts[k]
                    new_count = old_count - 1
                    edge_counts[k] = new_count
                    
                    # Check if status changed
                    # Bad if 0 or 10
                    if old_count == 1 or old_count == 11: # Wait, max is 10.
                        # If old was 1, new is 0. 1 is good, 0 is bad. Viol increases by 1.
                        if old_count == 1:
                            delta_viol += 1
                    elif old_count == 10:
                        # If old was 10, new is 9. 10 is bad, 9 is good. Viol decreases by 1.
                        delta_viol -= 1
            else:
                # Edge added: counts increase by 1
                for k in affected:
                    old_count = edge_counts[k]
                    new_count = old_count + 1
                    edge_counts[k] = new_count
                    
                    # Check if status changed
                    if old_count == 0:
                        # 0 is bad, 1 is good. Viol decreases by 1.
                        delta_viol -= 1
                    elif old_count == 9:
                        # 9 is good, 10 is bad. Viol increases by 1.
                        delta_viol += 1
            
            new_viol = current_viol + delta_viol
            delta = new_viol - current_viol # This is just delta_viol
            
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                current_viol = new_viol
                if new_viol < best_viol:
                    best_viol = new_viol
                    best_adj = adj.copy()
                    stagnation = 0
                    if best_viol == 0:
                        return best_adj, 0
                else:
                    stagnation += 1
            else:
                # Revert
                adj[u, v] = old_val
                adj[v, u] = adj[u, v]
                
                # Revert edge counts
                if old_val:
                    for k in affected:
                        edge_counts[k] += 1
                else:
                    for k in affected:
                        edge_counts[k] -= 1
                # current_viol remains unchanged
                stagnation += 1
                
            T = max(min_T, T * cooling)
            
            if stagnation > max_stagnation:
                # Restart with random perturbation
                if n >= 37:
                    for i in range(37, n):
                        for j in range(i):
                            adj[i, j] = np.random.rand() < 0.5
                            adj[j, i] = adj[i, j]
                else:
                    for i in range(n):
                        for j in range(i+1, n):
                            adj[i, j] = np.random.rand() < 0.5
                            adj[j, i] = adj[i, j]
                
                # Rebuild edge_counts
                for k in range(N_combos):
                    count = 0
                    for u, v in itertools.combinations(combos[k], 2):
                        if adj[u, v]:
                            count += 1
                    edge_counts[k] = count
                current_viol = int(np.sum((edge_counts == 0) | (edge_counts == 10)))
                T = 1.0
                stagnation = 0

        return best_adj, best_viol

    # Try n from 42 down to 38
    # First, try circulant search (fast)
    for n in range(42, 37, -1):
        if time.time() > deadline - 20:
            break
        time_for_circ = min(5, deadline - time.time())
        result = search_circulant(n, time.time() + time_for_circ)
        if result:
            best_n, best_edges = result
            return best_n, best_edges

    # If no circulant graph found, use general search
    for n in range(42, 37, -1):
        if time.time() > deadline - 10:
            break
        remaining = deadline - time.time()
        if remaining < 10:
            break
        
        time_for_n = min(40, remaining)
        
        adj, viol = search_general(n, time.time() + time_for_n)
        
        if viol == 0:
            best_n = n
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i, j]:
                        edges.append((i, j))
            best_edges = edges
            return best_n, best_edges

    return best_n, best_edges
