import numpy as np

def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    adj = np.asarray(adjacency, dtype=np.float64)
    # Precompute neighbor indices for faster updates
    # Actually, A[v] is a row. non-zero indices are neighbors.
    
    best_labels = None
    best_cut = -1.0
    
    rng = np.random.default_rng(42)
    
    n_restarts = 20
    n_iterations = 50000
    
    for restart in range(n_restarts):
        # Initialize spins
        # Use random init
        spins = rng.integers(0, 2, size=n) * 2 - 1 # +1/-1
        
        # Calculate initial fields h = A @ spins
        h = adj @ spins
        
        # Calculate initial cut
        # C = M - 0.25 * (spins @ (adj @ spins))
        # spins @ h is s^T A s
        s_as = spins @ h
        M = adj.sum() / 2.0
        current_cut = M - 0.25 * s_as
        
        best_local_cut = current_cut
        best_local_spins = spins.copy()
        
        T_start = 10.0
        T_end = 0.01
        cooling_rate = (T_end / T_start) ** (1.0 / n_iterations)
        current_T = T_start
        
        for it in range(n_iterations):
            v = rng.integers(0, n)
            
            # Delta C = s_v * h_v
            delta_c = spins[v] * h[v]
            
            if delta_c > 0:
                # Always accept improvement
                accept = True
            else:
                # Accept with probability exp(delta_c / T)
                # Note: delta_c is negative here
                if current_T > 1e-9:
                    prob = np.exp(delta_c / current_T)
                    accept = rng.random() < prob
                else:
                    accept = False
            
            if accept:
                # Update cut
                current_cut += delta_c
                
                # Update spin
                old_spin = spins[v]
                spins[v] = -old_spin
                
                # Update fields for neighbors
                # h_u -= 2 * A_uv * old_spin
                # A[v] is the row.
                # We only need to update neighbors where A[v, u] != 0
                # Since adj is 0/1, A[v, u] is 1 for neighbors.
                # h[neighbors] -= 2 * old_spin
                # Get neighbors
                neighbors = np.nonzero(adj[v])[0]
                if len(neighbors) > 0:
                    h[neighbors] -= 2.0 * old_spin
                
                # Track best
                if current_cut > best_local_cut:
                    best_local_cut = current_cut
                    best_local_spins = spins.copy()
            
            current_T *= cooling_rate
            
        # Local Search Polish
        # Greedy: flip if delta > 0
        improved = True
        while improved:
            improved = False
            for v in range(n):
                delta_c = spins[v] * h[v]
                if delta_c > 0:
                    # Flip
                    current_cut += delta_c
                    old_spin = spins[v]
                    spins[v] = -old_spin
                    neighbors = np.nonzero(adj[v])[0]
                    if len(neighbors) > 0:
                        h[neighbors] -= 2.0 * old_spin
                    improved = True
        
        if best_local_cut > best_cut:
            best_cut = best_local_cut
            best_labels = best_local_spins.copy()
            
    # Convert spins (+1/-1) to labels (0/1)
    # 1 -> 1, -1 -> 0
    return (best_labels + 1) // 2
