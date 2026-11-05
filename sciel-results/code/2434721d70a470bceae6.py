import numpy as np
import time
import random

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Precompute bitsets for fast neighbor blocking
    # Use Python integers for arbitrary precision bitsets (n=200 fits easily)
    # This is much faster than numpy boolean array updates for small n in Python loops
    # if we do it right, but actually numpy is fine if we use vectorized ops.
    # However, for the greedy simulation loop, we need to check "is v blocked?".
    # A blocked mask is a boolean array.
    
    # Let's optimize the greedy simulation.
    # Standard greedy:
    # 1. Sort by score
    # 2. Iterate, if not blocked, select, block neighbors.
    
    # To speed up simulation, we can represent the blocked set as a numpy boolean array.
    # Updating neighbors: blocked[neighbors] = True.
    
    # Even better: Precompute neighbor indices as lists for faster access in Python loop?
    # np.where(adj[v]) is O(n). For n=200, it's fast.
    
    # Let's try a different optimization: 
    # Instead of simulating from scratch for every swap, we can't easily do incremental updates 
    # for arbitrary swaps in a greedy set cover/max independent set context easily without 
    # complex bookkeeping. 
    # So we rely on fast simulation.
    
    # Optimization: Use a single boolean array for blocked, and reuse it? 
    # No, each simulation is independent.
    
    # Let's write a highly optimized simulation function.
    
    def simulate_greedy_fast(order):
        """
        order: numpy array of indices sorted by descending priority.
        Returns: size of independent set.
        """
        blocked = np.zeros(n, dtype=bool)
        count = 0
        # Local variables for speed
        adj_local = adj
        blocked_local = blocked
        n_local = n
        
        for v in order:
            if not blocked_local[v]:
                count += 1
                # Block v and its neighbors
                # adj_local[v] is a bool array of length n
                # We can do: blocked_local |= adj_local[v]
                # But we must also block v itself.
                # adj_local[v, v] is False (no self-loops).
                # So blocked_local[v] = True is needed.
                # Or: blocked_local[adj_local[v]] = True; blocked_local[v] = True
                
                # Vectorized update:
                # This creates a copy or views? 
                # blocked_local[adj_local[v]] = True is fast.
                neighbors = adj_local[v]
                blocked_local[neighbors] = True
                blocked_local[v] = True
                
        return count

    # Initialize order with a good heuristic: Low degree first (Greedy MIS heuristic)
    # Low degree vertices are "cheaper" to keep.
    degrees = adj.sum(axis=1)
    # Sort by ascending degree. Ties broken randomly? Or by index?
    # Let's use a mix: primarily degree, but we will search.
    
    # Strategy:
    # 1. Start with a few good initial heuristics (low degree, low coreness).
    # 2. Perform Hill Climbing on the permutation.
    #    Move: Swap two adjacent elements? Or swap any two?
    #    Swap any two is a larger search space but harder to implement efficiently for evaluation.
    #    Adjacent swaps are local.
    #    Actually, for greedy decoding, the relative order of non-adjacent vertices matters.
    #    A "swap" move: Pick two random vertices i, j in the permutation, swap them.
    #    Evaluate new score. If better, accept.
    
    # To make this faster, we can limit the number of evaluations per second.
    
    # Let's estimate speed of simulate_greedy_fast.
    # n=200. Loop 200 times. Inside: check bool, if true, vectorized update.
    # Vectorized update on 200 bools is very fast.
    # Python loop overhead is the bottleneck.
    # 200 iterations * ~1-2 microsecond per iteration? Maybe 10-20 us total per sim?
    # 1000 sims / second is conservative.
    # 115 seconds -> 100,000 simulations.
    
    # Let's refine the search:
    # 1. Generate initial order: Sort by degree (ascending).
    # 2. Evaluate.
    # 3. Loop while time < deadline:
    #    a. Pick a random pair (i, j) of positions in the permutation.
    #    b. Swap elements at i and j.
    #    c. Evaluate.
    #    d. If better, keep swap. Else revert.
    #    e. Occasionally (e.g., every 1000 steps), try a larger move or restart if stuck?
    #       Simple hill climbing can get stuck. 
    #       Let's add "Simulated Annealing" style: accept worse moves with probability exp(-delta/T).
    #       Or simpler: Random restarts if no improvement for K steps.
    
    # Given the "Inspiration" mentioned Simulated Annealing improved a score, let's implement a simple SA or Hill Climbing with restarts.
    
    # Let's stick to Hill Climbing with Random Restarts for simplicity and robustness.
    # Or, better: Use the "Best Improvement" hill climbing? No, that's too slow (O(n^2) evals per step).
    # "First Improvement" or "Random Swap" hill climbing is better.
    
    # Let's try:
    # 1. Initial: Sort by degree.
    # 2. Current Order, Current Score.
    # 3. Stagnation counter.
    # 4. While time:
    #    - Randomly pick 2 indices i, j.
    #    - Swap order[i], order[j].
    #    - Eval new score.
    #    - If new > current: Accept, reset stagnation.
    #    - Else: Revert, increment stagnation.
    #    - If stagnation > threshold (e.g., 1000): Restart from a new random heuristic or random shuffle?
    #      Restarting from random shuffle is bad. Restart from "Low Degree" or "Random" but keep best found so far?
    #      Actually, just continue. Hill climbing on permutations is effective.
    
    # Let's add a bit of randomness to the initial order to break ties.
    
    # Heuristic 1: Degree
    order1 = np.argsort(degrees) # Ascending degree (low degree first)
    # Heuristic 2: Random
    order2 = np.random.permutation(n)
    # Heuristic 3: Coreness (low coreness first)
    # Compute coreness quickly? 
    # For now, let's just use Degree and Random.
    
    # Let's run a quick benchmark of simulate_greedy_fast to see how many we can do.
    # But we can't benchmark inside the function easily without wasting time.
    # We'll just run the loop and break on time.
    
    best_order = order1
    best_score = simulate_greedy_fast(order1)
    
    current_order = order1.copy()
    current_score = best_score
    
    stagnation = 0
    max_stagnation = 500  # Restart if stuck for a while
    
    # To speed up random swap selection, we can pre-generate random pairs? No, just use np.random.
    
    # Optimization: 
    # If we use `order` as a numpy array, swapping is fast.
    # `order[i], order[j] = order[j], order[i]`
    
    # Let's also try to bias the swaps towards "important" vertices?
    # No, uniform random is standard.
    
    # One issue: The greedy decoder depends on the *entire* order.
    # Swapping two vertices far apart might have little effect if they are both blocked early or both selected late.
    # But it's a valid search.
    
    # Let's implement.
    
    # Pre-compute a random permutation for restarts if needed?
    
    # To ensure we explore different regions, let's do:
    # 1. Start with Degree-based order.
    # 2. Hill climb.
    # 3. If stuck, try a new initial order (e.g., Random, or Degree + noise) and hill climb from there.
    # 4. Keep track of global best.
    
    # Let's manage multiple "runs" of hill climbing.
    
    run_count = 0
    while time.time() < deadline:
        # Determine initial order for this run
        if run_count == 0:
            # Start with low degree
            init_order = np.argsort(degrees)
        elif run_count == 1:
            # Start with random
            init_order = np.random.permutation(n)
        else:
            # Start with low degree + random noise to break ties
            # Or just random
            init_order = np.random.permutation(n)
        
        current_order = init_order.copy()
        current_score = simulate_greedy_fast(current_order)
        
        if current_score > best_score:
            best_score = current_score
            best_order = current_order.copy()
        
        stagnation = 0
        
        # Inner loop: Hill Climbing
        while time.time() < deadline:
            # Pick random swap
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            
            # Swap
            current_order[i], current_order[j] = current_order[j], current_order[i]
            
            new_score = simulate_greedy_fast(current_order)
            
            if new_score > current_score:
                current_score = new_score
                stagnation = 0
                if new_score > best_score:
                    best_score = new_score
                    best_order = current_order.copy()
            else:
                # Revert
                current_order[i], current_order[j] = current_order[j], current_order[i]
                stagnation += 1
            
            if stagnation > max_stagnation:
                break
        
        run_count += 1
        # Maybe increase max_stagnation for later runs? Or keep it fixed.
        # Fixed is fine.
        
        # If we have time, we can do more runs.
        # The while loop structure handles this.
        
        # Safety: if we just broke out of inner loop, check outer condition.
        if time.time() > deadline - 5.0:
            break

    # Convert best_order (permutation) to scores
    # The greedy decoder sorts by descending score.
    # So, the vertex at position 0 in best_order should have the highest score.
    # The vertex at position n-1 in best_order should have the lowest score.
    
    scores = np.zeros(n)
    # scores[best_order[k]] = n - k
    # This gives distinct scores, which is good for tie-breaking stability (though greedy is deterministic on ties if indices are used, distinct scores remove ambiguity).
    for k, v in enumerate(best_order):
        scores[v] = n - k
        
    return scores
