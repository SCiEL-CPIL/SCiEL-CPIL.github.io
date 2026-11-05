import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    n = adjacency.shape[0]
    
    # Convert to adjacency lists for faster neighbor lookups
    adj_list = [np.nonzero(adjacency[i])[0] for i in range(n)]
    degrees = np.array([len(neighbors) for neighbors in adj_list])
    
    # 1. Initial Construction: Greedy by degree (Low degree first)
    # This is a strong baseline for Maximum Independent Set (MIS)
    current_order = np.argsort(degrees)
    selected = np.zeros(n, dtype=bool)
    
    for v in current_order:
        if selected[v]:
            continue
        selected[v] = True
        for u in adj_list[v]:
            if not selected[u]:
                selected[u] = False # Just marking, but we need to ensure we don't select neighbors
        
    # Wait, the greedy decoder in the task description is:
    # "picks vertices in descending score order, skipping neighbours of picked vertices"
    # My function `priority` returns scores. The evaluator runs the greedy decoder.
    # I cannot change the decoder. I must return scores that lead to a good greedy outcome.
    
    # However, the prompt says: "Write a Python priority(adjacency) function that gives every vertex a score; a fixed greedy decoder picks vertices in descending score order..."
    # The current program returns `counts` from a simulation.
    # The evaluator takes these scores, sorts descending, and greedily picks.
    
    # So I need to find scores S such that the greedy result of S is good.
    # The "local search" idea from inspiration 2 suggests optimizing the ordering.
    
    # Let's implement a local search on the ORDER itself, or on the scores.
    # Actually, optimizing the order directly is equivalent to optimizing scores if we can map order to scores.
    # But the greedy decoder is fixed: Sort by score desc, pick if not neighbor of picked.
    
    # Strategy: 
    # 1. Generate an initial ordering (e.g., low degree first).
    # 2. Evaluate the greedy set size for this ordering.
    # 3. Perform 1-exchange local search on the ordering:
    #    - Swap two vertices in the ordering.
    #    - If the greedy set size improves, keep the swap.
    #    - Repeat until no improvement or time limit.
    # 4. Convert the final ordering to scores (e.g., rank or inverse rank).
    
    # Optimization: 
    # Evaluating the full greedy process for every swap is O(n^2) or O(n + E).
    # With n=200, this is very fast. We can do many iterations.
    
    # Initial Order: Low degree first (standard heuristic for MIS)
    order = np.argsort(degrees)
    
    def evaluate_order(ord_arr):
        """Returns the size of the independent set produced by the greedy decoder on this order."""
        # The decoder picks in descending score order.
        # If we define score such that lower index in `ord_arr` means higher score,
        # then the decoder picks from index 0 to n-1.
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in ord_arr:
            if not selected[v]:
                selected[v] = True
                count += 1
                for u in adj_list[v]:
                    selected[u] = True # Block neighbors
        return count

    current_best_size = evaluate_order(order)
    
    # Local Search: 1-Swap
    # To make it faster, we can try random swaps or hill-climbing.
    # Given n=200, a full pass of all pairs is 200*199/2 ~ 20k swaps.
    # Each eval is ~200 steps. 20k * 200 = 4M ops. Very fast.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Randomize the start of the search to avoid local minima? 
        # Or systematic? Systematic is better for reproducibility and convergence.
        # Let's do a systematic pass of all pairs.
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                # Swap i and j
                order[i], order[j] = order[j], order[i]
                new_size = evaluate_order(order)
                if new_size > current_best_size:
                    current_best_size = new_size
                    improved = True
                else:
                    # Revert
                    order[i], order[j] = order[j], order[i]
            if time.time() > deadline:
                break
        
        # If no improvement in a full pass, we are in a local optimum.
        # We can try to escape by doing a random perturbation (Simulated Annealing style or Random Restart).
        # But for now, let's just stop if no improvement, or do a few random restarts if time permits.
        
        # Actually, if we do a full systematic pass and find no improvement, we are stuck.
        # Let's add a simple random restart mechanism if we have time.
        if not improved:
            # Try a few random restarts
            restarts = 5
            for _ in range(restarts):
                if time.time() > deadline:
                    break
                # Perturb the current order
                new_order = order.copy()
                # Randomly swap 10% of pairs
                for _ in range(n // 10):
                    i, j = np.random.choice(n, 2, replace=False)
                    new_order[i], new_order[j] = new_order[j], new_order[i]
                
                new_size = evaluate_order(new_order)
                if new_size > current_best_size:
                    order = new_order
                    current_best_size = new_size
                    improved = True
                    break # Restart the main loop? Or just continue?
                    # Let's break to restart the systematic pass with the new order
            if not improved:
                # If even restarts don't help, we might be at a global optimum for this heuristic.
                # Or we can try a different initial order.
                # Let's try one more restart with a completely random order.
                if time.time() < deadline - 10:
                    rand_order = np.random.permutation(n)
                    rand_size = evaluate_order(rand_order)
                    if rand_size > current_best_size:
                        order = rand_order
                        current_best_size = rand_size
                        improved = True
    
    # Convert final order to scores.
    # The decoder sorts by descending score.
    # So the vertex at index 0 in `order` should have the highest score.
    # Score can be simply the rank: n - index.
    scores = np.zeros(n)
    for idx, v in enumerate(order):
        scores[v] = n - idx
        
    return scores
