"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def partition(adjacency):
    """
    Solve Maximum Cut using Simulated Annealing with vectorized spin glass formulation.
    
    Args:
        adjacency: bool matrix of shape (n, n)
        
    Returns:
        labels: array of 0/1 labels per vertex
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array for efficient operations
    adj = np.asarray(adjacency, dtype=np.int8)
    
    # Best solution tracking
    best_labels = None
    best_cut = -1
    
    # Parameters for Simulated Annealing
    n_restarts = 30
    n_iterations = 50000
    T_start = 5.0
    T_end = 0.01
    cooling_rate = (T_end / T_start) ** (1.0 / n_iterations)
    
    rng = np.random.default_rng(42)
    
    for restart in range(n_restarts):
        # Initialization
        if restart == 0:
            # Start with alternating pattern
            labels = np.arange(n) % 2
        else:
            # Random initialization
            labels = rng.integers(0, 2, size=n)
            
        # Convert labels to spins: +1 for 1, -1 for 0
        spins = 2 * labels - 1
        
        # Compute the "field" acting on each spin
        # field[i] = sum_j adj[i,j] * spin[j]
        field = adj @ spins
        
        # Initial cut calculation
        mask0 = labels == 0
        mask1 = labels == 1
        current_cut = adj[np.ix_(mask0, mask1)].sum()
        
        best_local_cut = current_cut
        best_local_labels = labels.copy()
        
        current_T = T_start
        
        for it in range(n_iterations):
            # Pick a random vertex
            v = rng.integers(0, n)
            
            # Calculate delta in cut size if we flip v
            # In the spin glass model, Energy E = -sum_{i<j} J_{ij} s_i s_j
            # Max Cut is equivalent to minimizing E.
            # Change in E when flipping v: dE = -2 * s_v * field[v]
            # If dE < 0, we improve the cut.
            delta_E = -2 * spins[v] * field[v]
            
            # Acceptance criterion
            if delta_E < 0:
                # Improvement in cut (decrease in energy)
                # Flip spin
                spins[v] = -spins[v]
                labels[v] = 1 - labels[v]
                
                # Update field for all neighbors of v
                # field[u] += 2 * spins[v] * adj[v, u]  (since spin[v] changed)
                neighbors = np.nonzero(adj[v])[0]
                if len(neighbors) > 0:
                    field[neighbors] += 2 * spins[v] * adj[v, neighbors]
                
                # Update current_cut
                # delta_cut = -delta_E / 2
                current_cut -= delta_E / 2
                
            else:
                # Deterioration
                if current_T > 1e-9:
                    prob = np.exp(delta_E / current_T)
                    if rng.random() < prob:
                        # Flip spin
                        spins[v] = -spins[v]
                        labels[v] = 1 - labels[v]
                        
                        # Update field
                        neighbors = np.nonzero(adj[v])[0]
                        if len(neighbors) > 0:
                            field[neighbors] += 2 * spins[v] * adj[v, neighbors]
                        
                        current_cut -= delta_E / 2
            
            # Track best local
            if current_cut > best_local_cut:
                best_local_cut = current_cut
                best_local_labels = labels.copy()
                
            # Update temperature
            current_T *= cooling_rate
            
        # Greedy local search polish
        # Re-calculate field to be sure it's consistent
        spins = 2 * best_local_labels - 1
        field = adj @ spins
        
        improved = True
        while improved:
            improved = False
            for v in range(n):
                delta_E = -2 * spins[v] * field[v]
                if delta_E < 0:
                    # Flip
                    spins[v] = -spins[v]
                    best_local_labels[v] = 1 - best_local_labels[v]
                    
                    # Update field
                    neighbors = np.nonzero(adj[v])[0]
                    if len(neighbors) > 0:
                        field[neighbors] += 2 * spins[v] * adj[v, neighbors]
                    
                    best_local_cut -= delta_E / 2
                    improved = True
        
        # Update global best
        if best_local_cut > best_cut:
            best_cut = best_local_cut
            best_labels = best_local_labels.copy()
            
    return best_labels
# EVOLVE-BLOCK-END
