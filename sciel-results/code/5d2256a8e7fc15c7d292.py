"""
Heuristic Maximum Cut solver using Multi-Start Simulated Annealing.
Strategy:
1. Initialize with multiple random configurations.
2. Use Simulated Annealing to escape local optima.
3. Use a vectorized "best-first" local search (hill climbing) to polish solutions.
4. Maintain the best solution found across all restarts within the time budget.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def partition(adjacency):
    """
    Solve Maximum Cut using Multi-Start Simulated Annealing.
    Returns a 0/1 label per vertex.
    """
    start_time = time.time()
    DEADLINE = 115.0  # Leave 5s safety margin for Python overhead/return
    
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array for fast access
    # Ensure symmetric
    adj = np.array(adjacency, dtype=np.int32)
    adj = (adj + adj.T) // 2 # Ensure symmetry just in case, though input should be bool matrix
    
    # Precompute degrees for quick cut calculation
    # Cut(s) = 0.5 * (Sum_Degrees - s^T A s)
    # where s is in {-1, 1}
    degrees = adj.sum(axis=1)
    total_degree = degrees.sum()
    
    def get_cut(s):
        """Calculate cut size for spin vector s (values -1, 1)."""
        return 0.5 * (total_degree - s @ (adj @ s))
    
    def best_first_local_search(s, max_iters=1000):
        """
        Greedy local search: repeatedly flip the vertex that gives the largest
        positive gain in cut size until no improvement is possible.
        """
        # s is a numpy array of -1/1
        # Gain from flipping vertex v:
        # new_cut - old_cut = s[v] * (adj @ s)[v]
        # We want to maximize this gain.
        
        best_s = s.copy()
        best_cut = get_cut(best_s)
        
        for _ in range(max_iters):
            # Calculate gains for all vertices
            # adj_s = adj @ best_s
            # gains = best_s * adj_s
            # If max gain > 0, flip the vertex with max gain
            
            adj_s = adj @ best_s
            gains = best_s * adj_s
            
            max_gain = np.max(gains)
            
            if max_gain <= 0:
                break
            
            # Find vertex to flip
            v = np.argmax(gains)
            best_s[v] *= -1
            # Update best_cut incrementally? 
            # new_cut = old_cut + max_gain
            best_cut += max_gain
            
        return best_s, best_cut

    # Simulated Annealing
    def simulated_annealing(s_init, max_iters=5000, start_temp=10.0, end_temp=0.1):
        """
        Run simulated annealing starting from s_init.
        Returns best state found during the run.
        """
        s = s_init.copy()
        current_cut = get_cut(s)
        best_s = s.copy()
        best_cut = current_cut
        
        # Cooling schedule
        cooling_factor = (end_temp / start_temp) ** (1.0 / max_iters)
        temp = start_temp
        
        for i in range(max_iters):
            if temp <= end_temp:
                break
            
            # Select random vertex to flip
            v = np.random.randint(n)
            
            # Calculate gain
            adj_s = adj @ s
            gain = s[v] * adj_s[v]
            
            # Metropolis criterion
            if gain > 0 or np.random.rand() < np.exp(gain / temp):
                s[v] *= -1
                current_cut += gain
                
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_s = s.copy()
            
            temp *= cooling_factor
            
        return best_s, best_cut

    # Main loop: Multi-start
    rng = np.random.default_rng(42)
    
    best_global_s = None
    best_global_cut = -1
    
    # We can run multiple restarts. Each restart consists of:
    # 1. Random init
    # 2. Simulated Annealing
    # 3. Local Search polish
    
    # Estimate time per restart to determine how many we can fit.
    # For n=100, SA with 5000 iters + Local Search is very fast (< 0.1s).
    # We can run many restarts.
    
    restarts = 0
    while time.time() < DEADLINE:
        # Random initialization
        s = rng.choice([-1, 1], size=n)
        
        # Run SA
        s_sa, cut_sa = simulated_annealing(s, max_iters=2000, start_temp=5.0, end_temp=0.1)
        
        # Polish with local search
        s_final, cut_final = best_first_local_search(s_sa, max_iters=200)
        
        if cut_final > best_global_cut:
            best_global_cut = cut_final
            best_global_s = s_final.copy()
            
        restarts += 1
        
        # Check time again before next restart
        if time.time() > DEADLINE:
            break
            
        # Adaptive: if we are improving, maybe do more SA iterations?
        # For now, keep it simple.
        
    # Fallback if no restarts completed (should not happen)
    if best_global_s is None:
        s = np.ones(n, dtype=np.int8) * -1
        best_global_s, _ = best_first_local_search(s, max_iters=100)
        best_global_cut = get_cut(best_global_s)
        
    # Convert spin vector (-1, 1) back to labels (0, 1)
    # 0 -> -1, 1 -> 1 ? 
    # Let's say label 0 corresponds to spin -1, label 1 corresponds to spin 1.
    # labels = (s + 1) // 2
    labels = (best_global_s + 1) // 2
    labels = labels.astype(int)
    
    return labels
# EVOLVE-BLOCK-END
