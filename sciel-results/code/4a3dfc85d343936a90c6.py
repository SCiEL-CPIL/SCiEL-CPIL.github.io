import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy: Stochastic Gradient Ascent (Policy Gradient) on continuous priority scores.
    - Instead of permuting vertices, we optimize a continuous score vector s.
    - The greedy decoder's output (independent set size) is a function of the ordering induced by s.
    - We use a REINFORCE-style gradient estimator: 
        grad_s E[size] ≈ E[ (size - baseline) * grad_s log P(order | s) ]
    - To make this tractable and fast, we use a deterministic approximation where we 
      compute the gradient of the "smoothed" objective or simply use finite differences 
      on the score vector, but more efficiently, we use a local search on the score vector 
      guided by the change in independent set size when perturbing scores.
    
    However, a more robust structural change is to use **Multi-Start Local Search on Scores** 
    with a sophisticated move operator: 
    1. Start with multiple random or heuristic initial score vectors.
    2. In each step, pick a random vertex v and a random neighbor u, and swap their relative 
       priority by a small amount, or better, perform a "2-opt" like move on the induced ordering.
    
    Actually, the directive suggests an LP relaxation approach. Let's try that.
    The LP relaxation for Max Independent Set is:
    Maximize sum(x_v)
    Subject to: x_u + x_v <= 1 for all (u,v) in E
                0 <= x_v <= 1
    
    The dual is:
    Minimize sum(y_e)
    Subject to: sum_{e incident to v} y_e >= 1 for all v
                y_e >= 0
    
    The dual variables y_e correspond to edges. The vertex score can be derived from the 
    slack of the dual constraints or the primal variables. 
    Specifically, in the optimal dual solution, the value of the independent set is the 
    dual objective. But we need a priority order.
    
    A common heuristic is to use the primal LP solution x_v as the priority. 
    Vertices with higher x_v are more "independent" and should be picked first.
    We can solve the LP relaxation using scipy.optimize.linprog.
    Then, to improve, we can use the LP solution as an initial guess and refine with local search.
    
    However, solving LP for n=200 and m~2000 edges might be slow if done repeatedly. 
    But we only need to do it once. Then we can use the resulting scores for the greedy decoder.
    To beat the current SA score, we might need to combine LP with local search.
    
    Let's try:
    1. Solve LP relaxation to get initial scores x.
    2. Use these scores to generate an initial ordering.
    3. Perform local search on the ordering (or scores) to improve the greedy IS size.
    
    Wait, the task is to return scores. The decoder is fixed.
    If we use LP scores directly, it might not be optimal for the greedy decoder.
    The greedy decoder is sensitive to the exact ordering, not just the values.
    
    Alternative Structural Change: **Tabu Search on Orderings**
    Tabu search is a metaheuristic that explores the neighborhood of orderings but avoids cycling 
    by keeping a short-term memory (tabu list) of recent moves. This is structurally different 
    from SA (which uses temperature) and simple local search (which gets stuck in local optima).
    
    Let's implement Tabu Search on the permutation.
    - Neighborhood: Swap two vertices in the permutation.
    - Tabu List: Keep a list of recent swaps (i, j). A swap is tabu if (i, j) or (j, i) is in the list.
    - Aspiration Criterion: Accept a tabu move if it leads to a solution better than the best found so far.
    - Evaluation: Greedy IS size.
    
    This is a different design family (Tabu vs SA) and may explore the landscape differently.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Precompute neighbors
    neighbors_list = [np.where(adj[i])[0].tolist() for i in range(n)]
    
    def evaluate_order(order):
        """
        Evaluate the greedy independent set size for a given permutation.
        """
        selected = np.zeros(n, dtype=bool)
        blocked = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v] and not blocked[v]:
                selected[v] = True
                count += 1
                for u in neighbors_list[v]:
                    if not selected[u]:
                        blocked[u] = True
        return count
    
    # Initial ordering: Random or Degree-based?
    # Let's start with degree ascending as it's a good heuristic.
    deg = adj.sum(axis=1)
    order = np.argsort(deg, kind='stable')
    
    best_count = evaluate_order(order)
    best_order = order.copy()
    current_order = order.copy()
    current_count = best_count
    
    # Tabu Search Parameters
    tabu_tenure = 10  # Number of iterations a move is tabu
    tabu_list = []    # List of (i, j) swaps
    rng = np.random.default_rng(42)
    
    # We will perform a fixed number of iterations or until time limit
    max_iters = 20000
    
    for it in range(max_iters):
        if time.time() - start_time > deadline:
            break
            
        # Generate a random neighbor (swap two positions)
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i == j:
            continue
            
        # Check if move is tabu
        is_tabu = False
        for k in range(tabu_tenure):
            if len(tabu_list) > k:
                ti, tj = tabu_list[len(tabu_list) - 1 - k]
                if (ti == i and tj == j) or (ti == j and tj == i):
                    is_tabu = True
                    break
        
        # Perform the swap
        current_order[i], current_order[j] = current_order[j], current_order[i]
        new_count = evaluate_order(current_order)
        
        # Aspiration criterion
        if new_count > best_count:
            # Accept even if tabu
            best_count = new_count
            best_order = current_order.copy()
            # Update tabu list
            tabu_list.append((i, j))
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)
        elif not is_tabu and new_count >= current_count:
            # Accept non-tabu move if it's not worse
            current_count = new_count
            tabu_list.append((i, j))
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)
        else:
            # Reject: swap back
            current_order[i], current_order[j] = current_order[j], current_order[i]
            # Even if rejected, do we add to tabu? Usually no, only accepted moves are tabu.
            # But sometimes we add to avoid immediate reversal. Let's stick to standard: only accepted moves.
            
    # Convert best_order to scores
    scores = np.zeros(n)
    for k, v in enumerate(best_order):
        scores[v] = n - k
        
    return scores
