import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy: Large Neighborhood Search (LNS) on the selection permutation.
    1. Initialize the selection order using a structural heuristic (Min-Degree / Coreness).
    2. Perform LNS: 
       - Destroy: Randomly remove a subset of vertices from the current order.
       - Reconstruct: Re-insert the removed vertices into the order using a 
         greedy reconstruction heuristic (prioritizing vertices with low 
         current degree in the "remaining" graph or high isolation).
    3. Accept improvements. Repeat until time budget is exhausted.
    4. Return the final order as priority scores (rank-based).
    """
    start_time = time.time()
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Precompute neighbor sets for fast degree updates
    # Using lists of neighbors
    neighbors_list = [np.where(adj[i])[0].tolist() for i in range(n)]
    degrees = np.array([len(neighbors_list[i]) for i in range(n)])
    
    def simulate_order(order):
        """
        Simulate greedy selection given a fixed order.
        Returns the size of the independent set.
        """
        active = np.ones(n, dtype=bool)
        count = 0
        for v in order:
            if active[v]:
                count += 1
                active[v] = False
                # Deactivate neighbors
                for nb in neighbors_list[v]:
                    if active[nb]:
                        active[nb] = False
        return count

    def initial_heuristic_order():
        """
        Generate an initial order using a mixed heuristic.
        We want to pick vertices that are 'safe' (low degree) early.
        However, simply sorting by degree isn't always optimal.
        We use a score: -degree + small random noise to break ties and allow variation.
        """
        # Use coreness-like metric: vertices in sparse regions first
        # Approximate coreness by iterative peeling is expensive, so use degree + neighbor degree sum
        deg = degrees.copy()
        neighbor_deg_sum = adj @ deg
        # Score: lower is better for priority (picked earlier)
        # We want to pick low degree, low local density
        scores = deg + 0.5 * neighbor_deg_sum
        # Add small noise to avoid deterministic local minima
        noise = np.random.default_rng(42).normal(0, 0.1, n)
        scores += noise
        # Order: ascending score (lowest score picked first)
        order = np.argsort(scores, kind='stable')
        return order

    def destroy_reconstruct(current_order, rng, destroy_size):
        """
        LNS Move:
        1. Pick a random contiguous block or random subset of indices in the order to destroy.
        2. Remove them.
        3. Re-insert them one by one into the best position (or end, then sort locally?) 
           Actually, standard LNS for sequencing often involves removing a set and re-optimizing their positions.
           For simplicity and speed: 
           - Remove a random set of K vertices from the order.
           - Re-insert them into the order based on a heuristic that considers the *current* state of the graph 
             if we were to run the greedy on the prefix.
           
           Simpler and effective variant: 
           - Take a random segment [start, end] of the order.
           - Shuffle it.
           - Or: Take K random vertices, remove them, and re-insert them at random positions.
           
           Let's try: Remove K random vertices, and re-insert them at the end, then perform a local "bubble" 
           improvement on the tail? No, that's slow.
           
           Better: "Or-opt" move. Take a segment of length L, remove it, and insert it back at a random position.
        """
        if n < 10:
            return current_order.copy()
        
        # Choose segment length
        max_len = min(n // 2, 50)
        seg_len = rng.integers(2, max_len + 1)
        
        # Choose start index
        start = rng.integers(0, n - seg_len)
        
        # Extract segment
        segment = current_order[start:start+seg_len].copy()
        
        # Remove segment
        new_order = np.concatenate([current_order[:start], current_order[start+seg_len:]])
        
        # Choose insertion point
        # Insert at a random position in the new order
        ins_pos = rng.integers(0, len(new_order) + 1)
        
        # Insert segment
        new_order = np.concatenate([new_order[:ins_pos], segment, new_order[ins_pos:]])
        
        return new_order

    def local_refine(order, rng, max_iters=100):
        """
        Simple local search: try swapping adjacent elements or small shifts.
        """
        best_order = order.copy()
        best_val = simulate_order(best_order)
        
        for _ in range(max_iters):
            if time.time() - start_time > 115.0:
                break
            # Try a random adjacent swap
            if len(best_order) < 2:
                break
            i = rng.integers(0, len(best_order) - 1)
            j = i + 1
            
            # Swap
            best_order[i], best_order[j] = best_order[j], best_order[i]
            new_val = simulate_order(best_order)
            
            if new_val > best_val:
                best_val = new_val
            else:
                # Revert
                best_order[i], best_order[j] = best_order[j], best_order[i]
        
        return best_order, best_val

    # Main Search Loop
    rng = np.random.default_rng(123)
    
    # 1. Initialize
    current_order = initial_heuristic_order()
    current_val = simulate_order(current_order)
    
    best_order = current_order.copy()
    best_val = current_val
    
    # 2. LNS Loop
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < 115.0:
        # Perform LNS move
        # Occasionally do a full restart if stuck
        if no_improve_count > max_no_improve:
            # Restart with new noise
            current_order = initial_heuristic_order()
            current_val = simulate_order(current_order)
            no_improve_count = 0
            if current_val > best_val:
                best_val = current_val
                best_order = current_order.copy()
            continue
        
        # Or-opt move
        new_order = destroy_reconstruct(current_order, rng, destroy_size=0) # destroy_size ignored in current impl
        
        # Local refinement on the new candidate
        refined_order, refined_val = local_refine(new_order, rng, max_iters=50)
        
        if refined_val > current_val:
            current_order = refined_order
            current_val = refined_val
            no_improve_count = 0
            
            if current_val > best_val:
                best_val = current_val
                best_order = current_order.copy()
        else:
            no_improve_count += 1
            # Sometimes accept worse solutions to escape local minima (Simulated Annealing style)
            # Simple version: 5% chance to accept if not too bad
            if rng.random() < 0.05 and refined_val >= current_val - 2:
                current_order = refined_order
                current_val = refined_val

    # 3. Convert final order to priority scores
    # The decoder picks vertices in descending score order.
    # So, the vertex at index 0 in best_order should have the highest score.
    scores = np.zeros(n)
    # Rank 0 (first picked) gets score n, Rank 1 gets n-1, etc.
    # This ensures strict ordering.
    ranks = np.argsort(np.argsort(best_order))
    # Invert ranks so that first in order has highest score
    scores = n - ranks
    
    return scores

