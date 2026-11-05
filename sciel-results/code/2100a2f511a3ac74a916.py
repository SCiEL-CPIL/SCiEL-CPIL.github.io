"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def partition(adjacency):
    """
    Solve Maximum Cut using Simulated Annealing with Spin Glass formulation.
    
    Args:
        adjacency: bool matrix of shape (n, n)
        
    Returns:
        labels: array of 0/1 labels per vertex
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array
    # Use float for calculations, int8 for storage if possible, but float is safer for sums
    adj = np.asarray(adjacency, dtype=np.float32)
    
    # Precompute neighbors list for fast updates
    neighbors = [np.where(adj[i] > 0)[0] for i in range(n)]
    
    best_labels = None
    best_cut = -1
    
    # Parameters
    n_restarts = 50
    n_iterations = 2000
    T_start = 10.0
    T_end = 0.01
    cooling_rate = (T_end / T_start) ** (1.0 / n_iterations)
    
    rng = np.random.default_rng(42)
    
    for restart in range(n_restarts):
        # Initialize spins: +1 or -1
        if restart == 0:
            # Random
            spins = rng.choice([-1, 1], size=n)
        else:
            spins = rng.choice([-1, 1], size=n)
            
        # Calculate initial M_v = sum_u A_vu s_u
        # M = adj @ spins
        M = adj @ spins
        
        # Calculate current Cut
        # Cut = 0.5 * sum_{i<j} A_ij (1 - s_i s_j) ?
        # Sum_{i,j} A_ij (1 - s_i s_j) / 2?
        # Let's just compute it directly for initialization check, or track it.
        # Energy H = sum_{i<j} A_ij s_i s_j = 0.5 * (spins @ adj @ spins)
        # Cut = (E_total - H) / 2 ?
        # E_total = sum_{i<j} A_ij = 0.5 * sum_{i,j} A_ij
        # H = sum_{same} A - sum_{diff} A
        # Cut = sum_{diff} A
        # E_total = sum_{same} A + sum_{diff} A
        # H = E_total - 2 * Cut
        # Cut = (E_total - H) / 2
        
        E_total = 0.5 * np.sum(adj)
        
        # Current H
        H = 0.5 * (spins @ (adj @ spins))
        current_cut = (E_total - H) / 2
        
        best_local_cut = current_cut
        best_local_spins = spins.copy()
        
        current_T = T_start
        
        for it in range(n_iterations):
            # Pick random vertex
            v = rng.integers(0, n)
            
            # Delta H = -2 * s_v * M_v
            delta_H = -2.0 * spins[v] * M[v]
            
            # Accept if delta_H < 0 (improvement) or with prob exp(-delta_H/T)
            if delta_H < 0:
                accept = True
            else:
                # If T is very small, exp is 0
                if current_T > 1e-6:
                    prob = np.exp(-delta_H / current_T)
                    accept = rng.random() < prob
                else:
                    accept = False
            
            if accept:
                # Flip spin v
                old_spin = spins[v]
                new_spin = -old_spin
                spins[v] = new_spin
                
                # Update Cut
                # H changes by delta_H
                # Cut changes by -delta_H / 2
                current_cut -= delta_H / 2.0
                
                # Update M for neighbors of v
                # M_u = sum_w A_uw s_w
                # s_v changed from old_spin to new_spin
                # Change in s_v is 2 * new_spin (since new = -old, new - old = 2new)
                # Actually, change is -2 * old_spin
                # M_u += A_uv * (new_spin - old_spin)
                # M_u += A_uv * (-2 * old_spin)
                
                neigh = neighbors[v]
                if len(neigh) > 0:
                    M[neigh] -= 2.0 * old_spin * adj[v, neigh]
                
                # Track best
                if current_cut > best_local_cut:
                    best_local_cut = current_cut
                    best_local_spins = spins.copy()
            
            # Update temperature
            current_T *= cooling_rate
            
        # Greedy Polish
        # We need to recompute M if we modified spins in SA, but we kept best_local_spins.
        # Let's run greedy on best_local_spins.
        s = best_local_spins.copy()
        M_greedy = adj @ s
        cut_greedy = (E_total - 0.5 * (s @ (adj @ s))) / 2.0
        
        improved = True
        while improved:
            improved = False
            for v in range(n):
                # Check if flipping v improves cut
                # Delta H = -2 s_v M_v
                # If Delta H < 0, flip.
                delta_H = -2.0 * s[v] * M_greedy[v]
                if delta_H < 0:
                    # Flip
                    old_spin = s[v]
                    s[v] = -old_spin
                    cut_greedy -= delta_H / 2.0
                    # Update M for neighbors
                    neigh = neighbors[v]
                    if len(neigh) > 0:
                        M_greedy[neigh] -= 2.0 * old_spin * adj[v, neigh]
                    improved = True
        
        if cut_greedy > best_cut:
            best_cut = cut_greedy
            best_labels = s.copy()
            
    # Convert spins (+1/-1) to labels (0/1)
    # +1 -> 1, -1 -> 0
    return (best_labels + 1) // 2
# EVOLVE-BLOCK-END
