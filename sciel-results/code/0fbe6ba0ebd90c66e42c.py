"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import random

def construct():
    """
    Constructs a graph with no K5 and no independent set of size 5.
    Attempts to find a graph on 42 vertices (Exoo's bound) via local search.
    Falls back to Paley graph on 37 vertices if search fails.
    """
    
    def get_paley_graph(q):
        """Constructs the Paley graph on q vertices."""
        residues = {(x * x) % q for x in range(1, q)}
        edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
        return q, edges

    def verify_graph(n, edges):
        """
        Verifies if the graph has no K5 and no independent set of size 5.
        Returns True if valid.
        """
        # Build adjacency matrix
        adj = np.zeros((n, n), dtype=np.int8)
        for u, v in edges:
            adj[u, v] = 1
            adj[v, u] = 1
            
        # Generate all 5-combinations
        # For n=42, this is ~850k combinations.
        # We check each one.
        # Optimization: Use bitsets? 
        # For verification, a simple loop with early exit is fine if it's invalid.
        # If it's valid, we have to check all.
        
        # We can use the same vectorized check as in the search, 
        # but for a single graph, a simple check is okay if we just need Yes/No.
        # However, to be safe and fast, let's use the vectorized method.
        
        combos = np.array(list(itertools.combinations(range(n), 5)), dtype=np.int32)
        if len(combos) == 0:
            return True
            
        # Indices of pairs in the 5-set
        # Pairs: (0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
        pair_indices = [(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
        
        # Check if any is clique or independent
        # We can sum the adjacency values for the 10 pairs.
        # If sum == 10, it's a clique.
        # If sum == 0, it's independent.
        
        # Vectorized sum
        # adj_flat = adj.flatten()
        # For each combo, we need to index adj.
        # This is tricky with advanced indexing in a single vectorized op for all combos at once 
        # without creating a huge intermediate array.
        
        # Alternative: Loop over combos in chunks or just loop.
        # 850k loops in Python might take 1-2 seconds. Acceptable for verification.
        
        for c in combos:
            c = c.tolist()
            is_clique = True
            is_indep = True
            for i in range(5):
                for j in range(i+1, 5):
                    if adj[c[i], c[j]] == 1:
                        is_indep = False
                    else:
                        is_clique = False
            if is_clique or is_indep:
                return False
        return True

    def find_graph_local_search(n, time_limit=100):
        """
        Attempts to find a (5,5)-free graph on n vertices using local search.
        Returns (n, edges) if found, else None.
        """
        start_time = time.time()
        
        # Precompute all 5-sets
        all_combos = list(itertools.combinations(range(n), 5))
        M = len(all_combos)
        
        # Map each edge (u,v) to the indices of 5-sets containing it
        # edge_to_combos: dict (u,v) -> list of indices
        edge_to_combos = {}
        for idx, c in enumerate(all_combos):
            c_list = list(c)
            for i in range(5):
                for j in range(i+1, 5):
                    u, v = c_list[i], c_list[j]
                    key = (min(u,v), max(u,v))
                    if key not in edge_to_combos:
                        edge_to_combos[key] = []
                    edge_to_combos[key].append(idx)
        
        # Initialize random graph
        # Start with p=0.5
        adj = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i, j] = 1
                    adj[j, i] = 1
        
        # Calculate initial energy
        # We need to know for each combo if it is clique/indep
        # is_clique[idx], is_indep[idx]
        is_clique = np.zeros(M, dtype=bool)
        is_indep = np.zeros(M, dtype=bool)
        
        # Helper to update status of a combo
        def update_combo_status(idx):
            c = all_combos[idx]
            c_list = list(c)
            edges_count = 0
            for i in range(5):
                for j in range(i+1, 5):
                    if adj[c_list[i], c_list[j]]:
                        edges_count += 1
            is_clique[idx] = (edges_count == 10)
            is_indep[idx] = (edges_count == 0)
            
        for idx in range(M):
            update_combo_status(idx)
            
        current_energy = np.sum(is_clique) + np.sum(is_indep)
        
        if current_energy == 0:
            # Convert adj to edges
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i, j]:
                        edges.append((i, j))
            return n, edges

        # Simulated Annealing
        T = 10.0
        cooling_rate = 0.999
        min_T = 0.01
        
        # List of all edges for random selection
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        while time.time() - start_time < time_limit:
            if current_energy == 0:
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if adj[i, j]:
                            edges.append((i, j))
                return n, edges
                
            # Pick random edge
            u, v = random.choice(all_edges)
            key = (min(u,v), max(u,v))
            
            # Get affected combos
            affected_idxs = edge_to_combos.get(key, [])
            
            # Calculate delta energy
            # For each affected combo, check if it changes status
            delta = 0
            old_statuses = []
            for idx in affected_idxs:
                old_statuses.append((is_clique[idx], is_indep[idx]))
                
            # Temporarily flip edge
            old_val = adj[u, v]
            adj[u, v] = 1 - old_val
            adj[v, u] = 1 - old_val
            
            for i, idx in enumerate(affected_idxs):
                was_clique, was_indep = old_statuses[i]
                update_combo_status(idx)
                is_clique_now, is_indep_now = is_clique[idx], is_indep[idx]
                
                # Energy contribution change
                old_contrib = (1 if was_clique else 0) + (1 if was_indep else 0)
                new_contrib = (1 if is_clique_now else 0) + (1 if is_indep_now else 0)
                delta += new_contrib - old_contrib
                
            new_energy = current_energy + delta
            
            # Accept?
            if new_energy < current_energy or random.random() < np.exp(-delta / T):
                current_energy = new_energy
            else:
                # Revert
                adj[u, v] = old_val
                adj[v, u] = old_val
                for i, idx in enumerate(affected_idxs):
                    was_clique, was_indep = old_statuses[i]
                    is_clique[idx] = was_clique
                    is_indep[idx] = was_indep
                    
            T *= cooling_rate
            if T < min_T:
                T = min_T
                
        return None

    # Try n=42
    result = find_graph_local_search(42, time_limit=100)
    if result:
        n, edges = result
        if verify_graph(n, edges):
            return n, edges
            
    # Try n=41
    result = find_graph_local_search(41, time_limit=100)
    if result:
        n, edges = result
        if verify_graph(n, edges):
            return n, edges

    # Fallback to Paley 37
    return get_paley_graph(37)

# EVOLVE-BLOCK-END
