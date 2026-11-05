import numpy as np
import time
import random

def _greedy_set_size_from_order(adj, order):
    """
    Run the fixed greedy decoder given a specific order of vertices.
    Returns the size of the independent set.
    """
    n = adj.shape[0]
    chosen = set()
    blocked = np.zeros(n, dtype=bool)
    count = 0
    for v in order:
        v = int(v)
        if not blocked[v]:
            chosen.add(v)
            count += 1
            # Block v and its neighbors
            blocked[v] = True
            nbrs = np.nonzero(adj[v])[0]
            if len(nbrs) > 0:
                blocked[nbrs] = True
    return count

def priority(adjacency):
    start_time = time.time()
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Deadline guard: 140s out of 150s limit
    deadline = start_time + 140.0
    
    # Initial solution: Low degree first (heuristic start)
    deg = adj.sum(axis=1)
    keys = np.random.rand(n)
    current_order = np.lexsort((keys, deg))
    current_score = _greedy_set_size_from_order(adj, current_order)
    
    best_order = current_order.copy()
    best_score = current_score
    
    # Simulated Annealing Parameters
    # Temperature schedule: Geometric cooling is standard and robust
    # T0 = 10.0, alpha = 0.9999 (approx 10000 steps to halve)
    # We will adjust alpha based on time budget if necessary, but fixed is safer for determinism-ish behavior
    T = 10.0
    alpha = 0.9998
    
    # Precompute neighbor lists for faster blocking? 
    # The greedy decoder is the bottleneck. 
    # For n=200, it's O(n^2) per evaluation. 140s allows ~100,000-500,000 evaluations depending on overhead.
    # Python loop in _greedy_set_size_from_order is slow.
    # Optimization: Vectorize the greedy step? 
    # Actually, let's just make sure the SA loop is tight.
    
    # To speed up evaluation, we can precompute the neighbor sets as lists.
    neighbor_lists = [np.nonzero(adj[i])[0] for i in range(n)]
    
    def _fast_greedy(order):
        blocked = [False] * n
        count = 0
        for v in order:
            v = int(v)
            if not blocked[v]:
                count += 1
                blocked[v] = True
                for u in neighbor_lists[v]:
                    blocked[u] = True
        return count

    # Re-evaluate initial with fast greedy
    current_score = _fast_greedy(current_order)
    best_score = current_score
    best_order = current_order.copy()
    
    # SA Loop
    # We want to do as many iterations as possible.
    # Iteration cost: 1 swap + 1 evaluation.
    # Evaluation is ~200 steps * python overhead.
    
    while time.time() < deadline:
        # Generate neighbor: Swap two random positions
        # Adjacent swaps are often better for permutation landscapes, but random swaps allow larger jumps.
        # Let's mix: 70% adjacent, 30% random.
        if random.random() < 0.7:
            i = random.randint(0, n - 2)
            j = i + 1
        else:
            i, j = random.sample(range(n), 2)
        
        # Apply swap
        current_order[i], current_order[j] = current_order[j], current_order[i]
        new_score = _fast_greedy(current_order)
        
        delta = new_score - current_score
        
        if delta > 0:
            # Accept improvement
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_order = current_order.copy()
        else:
            # Accept worsening with probability exp(delta / T)
            # If T is small, this is rarely accepted.
            # Avoid log(0) if T is very small, but T starts high.
            if T > 0:
                prob = np.exp(delta / T)
                if random.random() < prob:
                    current_score = new_score
                else:
                    # Reject: swap back
                    current_order[i], current_order[j] = current_order[j], current_order[i]
        
        # Cool down
        T *= alpha
        
        # If T gets too small, it effectively becomes hill climbing.
        # We can reheat if stuck? 
        # Stagnation detection: If no improvement for X iterations, reheat.
        # Simple reheat: If T < 0.1 and no improvement in last 1000 steps, reset T.
        # To keep it simple and robust, let's just let it cool. 
        # However, pure cooling might get stuck early.
        # Let's add a simple reheat mechanism:
        # If we haven't improved best_score in the last 5000 steps, and T < 1.0, set T = 1.0.
        
        # Tracking steps for reheat
        if not hasattr(priority, '_sa_state'):
            priority._sa_state = {'last_improve_iter': 0, 'iter': 0}
        
        priority._sa_state['iter'] += 1
        if new_score == best_score and new_score > 0: # If we just improved, this block is skipped
            pass # The check above handles the update. We need to know when best_score changed.
            
        # Actually, let's track it properly in the loop above.
        # Refactor:
        # We already updated best_score above.
        # Let's track 'iter_since_improve'
        if 'iter_since_improve' not in priority._sa_state:
            priority._sa_state['iter_since_improve'] = 0
            
        # Wait, the logic above updates best_score but doesn't reset the counter easily.
        # Let's just use a local variable for simplicity, but we need to persist across the while loop? 
        # No, priority is called once per graph. So local state is fine.
        
        # Reset local state for this specific call? 
        # The `priority` function is called once. So `iter_since_improve` should be local.
        # I will move the state tracking to local variables.
        
        # (Self-Correction): The `if not hasattr` block is bad for multiple calls or concurrent execution, 
        # but here it's a single call. However, local variables are cleaner.
        
        # Let's restart the loop logic with local variables for clarity and correctness.
        pass # The code above is getting messy. I will rewrite the function body cleanly.

    # Convert best_order to scores
    scores = np.zeros(n)
    for rank, v in enumerate(best_order):
        scores[v] = n - rank
        
    return scores

# The previous draft had messy state management. Here is the clean, complete implementation.

def priority(adjacency):
    start_time = time.time()
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Deadline guard: 140s out of 150s limit
    deadline = start_time + 140.0
    
    # Precompute neighbor lists for fast greedy evaluation
    neighbor_lists = [np.nonzero(adj[i])[0] for i in range(n)]
    
    def _fast_greedy(order):
        blocked = [False] * n
        count = 0
        for v in order:
            v = int(v)
            if not blocked[v]:
                count += 1
                blocked[v] = True
                for u in neighbor_lists[v]:
                    blocked[u] = True
        return count

    # Initial solution: Low degree first
    deg = adj.sum(axis=1)
    keys = np.random.rand(n)
    current_order = np.lexsort((keys, deg))
    current_score = _fast_greedy(current_order)
    
    best_order = current_order.copy()
    best_score = current_score
    
    # Simulated Annealing Parameters
    T = 5.0  # Initial temperature
    alpha = 0.9999  # Cooling rate
    
    iter_since_improve = 0
    max_iters_without_improve = 5000
    
    while time.time() < deadline:
        # Generate neighbor: Swap two positions
        # Mix of adjacent and random swaps
        if random.random() < 0.6:
            i = random.randint(0, n - 2)
            j = i + 1
        else:
            i, j = random.sample(range(n), 2)
        
        # Apply swap
        current_order[i], current_order[j] = current_order[j], current_order[i]
        new_score = _fast_greedy(current_order)
        
        delta = new_score - current_score
        
        if delta > 0:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_order = current_order.copy()
                iter_since_improve = 0
            else:
                iter_since_improve += 1
        else:
            iter_since_improve += 1
            # Accept worsening with probability exp(delta / T)
            if T > 0.001:
                # Optimization: if delta is very negative, exp(delta/T) is ~0
                # Calculate log(prob) = delta / T
                # If random() < exp(delta/T)
                # Use log for numerical stability if T is small
                prob = np.exp(delta / T)
                if random.random() < prob:
                    current_score = new_score
                else:
                    # Reject: swap back
                    current_order[i], current_order[j] = current_order[j], current_order[i]
            else:
                # T is effectively 0, reject all worsening moves
                current_order[i], current_order[j] = current_order[j], current_order[i]
        
        # Cool down
        T *= alpha
        
        # Reheating strategy:
        # If stuck for a long time and T is low, reheat to escape local minima.
        if iter_since_improve > max_iters_without_improve and T < 1.0:
            T = 1.0
            iter_since_improve = 0
            # Optional: Randomize current solution slightly? 
            # No, just reheat temperature to allow larger jumps.
        
        # If T is extremely small, we are doing hill climbing.
        # If we are stuck in hill climbing, reheat is crucial.
        
    # Convert best_order to scores
    scores = np.zeros(n)
    for rank, v in enumerate(best_order):
        scores[v] = n - rank
        
    return scores
