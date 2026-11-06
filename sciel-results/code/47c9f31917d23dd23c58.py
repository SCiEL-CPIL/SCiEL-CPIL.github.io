"""
Strategy: Multi-Start Local Search (Hill Climbing) on Vertex Order.

The current program uses Simulated Annealing on a continuous score vector.
This program treats the problem as finding an optimal permutation (ordering)
of vertices. We define the "state" as the order of vertices. The objective is
the size of the independent set produced by the greedy decoder on this order.

Moves: Swap adjacent elements in the order.
This allows us to perform fine-grained local search. Since the greedy decoder
is deterministic given the order, we can efficiently evaluate swaps.
We run multiple restarts with different initial heuristics (degree-based,
random, etc.) and keep the best result.
"""

# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    """
    Assign a priority score to each vertex for a greedy Maximum Independent Set decoder.
    
    Strategy: Multi-Start Local Search (Hill Climbing) on the vertex ordering.
    1. Initialize with several different heuristics (inverse degree, random, etc.).
    2. For each initialization, perform local search by swapping adjacent vertices
       in the ordering if it increases the independent set size.
    3. Return the scores corresponding to the best ordering found.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    adj = np.asarray(adjacency)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([], dtype=float)
    
    adj_bool = adj.astype(bool)
    degree = adj_bool.sum(axis=1).astype(int)
    
    # Precompute neighbors as lists of indices for faster access in simulation
    # adj_bool is n x n boolean. 
    # For simulation, we need to quickly check if a vertex is available.
    
    def simulate_greedy(order):
        """
        Simulate greedy MIS given a specific order of vertices.
        Returns the size of the independent set.
        """
        selected = 0
        available = np.ones(n, dtype=bool)
        
        for v in order:
            if not available[v]:
                continue
            # Select v
            selected += 1
            # Remove v and its neighbors from available
            available[v] = False
            # Vectorized update: available[neighbors] = False
            # adj_bool[v] is a boolean mask of length n
            available[adj_bool[v]] = False
            
            # Early exit
            if not available.any():
                break
                
        return selected

    def local_search(order):
        """
        Perform hill climbing on the order by swapping adjacent elements.
        Returns the best order found.
        """
        current_order = order.copy()
        current_size = simulate_greedy(current_order)
        best_order = current_order.copy()
        best_size = current_size
        
        no_improvement = 0
        max_no_improvement = 50 # Stop if no improvement for 50 swaps
        
        # We iterate through the list and try swapping i, i+1
        # To be efficient, we can just scan the whole list repeatedly
        while no_improvement < max_no_improvement:
            improved = False
            no_improvement += 1
            
            for i in range(n - 1):
                # Check time
                if time.time() > deadline:
                    return best_order, best_size
                
                # Try swapping i and i+1
                # Create a new order temporarily? 
                # It's faster to just simulate the change if we can, 
                # but simulating the whole greedy from scratch is O(n^2) worst case.
                # n=200, so O(n^2) = 40k ops. Very fast.
                
                # Swap in a copy? No, just create a view or small copy?
                # Actually, creating a new array of 200 ints is cheap.
                # But we can be smarter: 
                # The greedy result only changes if the relative order of 
                # "available" vertices changes in a way that affects selection.
                # However, for simplicity and correctness, let's just simulate.
                
                # Optimization: Only simulate if the two vertices are "close" 
                # in the greedy process? Hard to know without simulating.
                
                # Let's just do a full simulation for the swapped order.
                # To avoid copying the whole array every time, we can 
                # construct the order on the fly? 
                # No, `simulate_greedy` takes an array.
                
                # Efficient swap simulation:
                # We can copy the order, swap, simulate, and revert.
                # But copying 200 elements is trivial.
                
                new_order = current_order.copy()
                new_order[i], new_order[i+1] = new_order[i+1], new_order[i]
                
                new_size = simulate_greedy(new_order)
                
                if new_size > current_size:
                    current_order = new_order
                    current_size = new_size
                    if current_size > best_size:
                        best_size = current_size
                        best_order = current_order.copy()
                    improved = True
                    no_improvement = 0
                    # Break and restart scan? Or continue? 
                    # Continuing might find more improvements in the same pass.
                    # But the state changed, so indices > i are shifted.
                    # It's safer to restart the scan or just continue and 
                    # rely on the next pass.
                    # Let's restart the scan to be safe and thorough.
                    break
            
            if not improved:
                break
                
        return best_order, best_size

    # Multi-Start: Try different initializations
    best_global_order = None
    best_global_size = 0
    
    # Heuristic 1: Inverse Degree (Low degree first)
    # Add small noise to break ties
    init1 = np.argsort(-degree + np.random.normal(0, 0.1, n), kind='stable')
    
    # Heuristic 2: Random
    init2 = np.random.permutation(n)
    
    # Heuristic 3: Degree (High degree first) - sometimes helps?
    init3 = np.argsort(degree - np.random.normal(0, 0.1, n), kind='stable')
    
    # Heuristic 4: "Core" based? Low degree but high degree neighbors?
    # Score: -degree + sum(degrees of neighbors)
    # This is O(n^2) to compute but n=200 is small.
    # Let's compute a "potential" score
    # potential[v] = sum(degrees of neighbors of v)
    # We want to pick vertices that "block" many high-degree neighbors?
    # Or vertices that are "easy" to pick (low degree) but don't block much?
    
    # Let's try a few diverse starts.
    starts = [init1, init2, init3]
    
    # Add more random starts if time permits
    num_extra_starts = 10
    for _ in range(num_extra_starts):
        starts.append(np.random.permutation(n))
    
    for start_order in starts:
        if time.time() > deadline:
            break
            
        # Run local search
        found_order, found_size = local_search(start_order)
        
        if found_size > best_global_size:
            best_global_size = found_size
            best_global_order = found_order
            
        # If we hit the theoretical max (n), we can stop
        if best_global_size == n:
            break
            
    # If no local search was run (e.g. n=0 or timeout immediately), fallback
    if best_global_order is None:
        best_global_order = init1
        best_global_size = simulate_greedy(best_global_order)
        
    # Convert the best order back to scores.
    # The greedy decoder picks vertices in descending score order.
    # So the vertex at index 0 in `best_global_order` should have the highest score.
    # The vertex at index n-1 should have the lowest score.
    scores = np.zeros(n, dtype=float)
    # Rank: 0 is highest priority (largest score)
    # scores[vertex] = n - rank
    for rank, vertex in enumerate(best_global_order):
        scores[vertex] = n - rank
        
    return scores
# EVOLVE-BLOCK-END
