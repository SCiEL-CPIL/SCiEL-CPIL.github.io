"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    """
    Assign a priority score to each vertex for a greedy Maximum Independent Set decoder.
    
    Strategy: Combinatorial Ordering Search with Simulated Annealing.
    1. Initialize the vertex order using a strong structural heuristic.
    2. Use Simulated Annealing to swap vertices in the ordering, accepting worse solutions
       with a decaying probability to escape local optima.
    3. Return the scores corresponding to the best ordering found (inverse rank).
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    adj = np.asarray(adjacency)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([], dtype=float)
    
    adj_bool = adj.astype(bool)
    degree = adj_bool.sum(axis=1).astype(float)
    
    # Fast Greedy Simulator
    def simulate_greedy(order):
        selected = 0
        available = np.ones(n, dtype=bool)
        
        for v in order:
            if not available[v]:
                continue
            selected += 1
            available[v] = False
            available[adj_bool[v]] = False
            
            if not available.any():
                break
                
        return selected

    # Structural features for initial scoring
    neighbor_degree_sum = adj_bool @ degree
    
    if degree.max() > 0:
        deg_norm = degree / degree.max()
    else:
        deg_norm = degree.copy()
    
    if neighbor_degree_sum.max() > 0:
        nbs_norm = neighbor_degree_sum / neighbor_degree_sum.max()
    else:
        nbs_norm = neighbor_degree_sum.copy()
    
    # Initial priority: primarily degree, secondarily neighborhood density
    initial_priority = deg_norm + 0.3 * nbs_norm
    
    # Sort by initial priority ascending (low priority value = selected early)
    current_order = np.argsort(initial_priority, kind='stable')
    current_size = simulate_greedy(current_order)
    
    best_order = current_order.copy()
    best_size = current_size
    
    # Simulated Annealing on the permutation
    def simulated_annealing(order, size, deadline):
        current_order = order.copy()
        current_size = size
        
        # Temperature schedule
        t_start = 1.0
        t_end = 0.001
        alpha = 0.995  # Cooling rate
        
        t = t_start
        no_improve = 0
        max_no_improve = 5000
        
        while time.time() < deadline and t > t_end:
            # Pick two random positions to swap
            i, j = np.random.randint(0, n, 2)
            if i == j:
                continue
            
            new_order = current_order.copy()
            new_order[i], new_order[j] = new_order[j], new_order[i]
            
            new_size = simulate_greedy(new_order)
            
            delta = new_size - current_size
            
            if delta > 0:
                # Accept improvement
                current_order = new_order
                current_size = new_size
                no_improve = 0
            else:
                # Accept worsening with probability exp(delta / t)
                prob = np.exp(delta / t)
                if np.random.random() < prob:
                    current_order = new_order
                    current_size = new_size
                else:
                    no_improve += 1
            
            # Update best
            if current_size > best_size:
                best_size = current_size
                best_order = current_order.copy()
            
            # Cool down
            t *= alpha
            
            # Early stop if stuck too long
            if no_improve > max_no_improve:
                break
        
        return current_order, current_size
    
    # Run simulated annealing
    sa_order, sa_size = simulated_annealing(best_order, best_size, deadline - 5)
    
    if sa_size > best_size:
        best_size = sa_size
        best_order = sa_order
    
    # Final local search polish: try all single swaps
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(n):
                if time.time() > deadline - 5:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline - 5:
                        break
                    
                    new_order = best_order.copy()
                    new_order[i], new_order[j] = new_order[j], new_order[i]
                    new_size = simulate_greedy(new_order)
                    
                    if new_size > best_size:
                        best_order = new_order
                        best_size = new_size
                        improved = True
    
    # Convert best_order to scores
    scores = np.zeros(n)
    scores[best_order] = n - np.arange(n)
    
    return scores
# EVOLVE-BLOCK-END
