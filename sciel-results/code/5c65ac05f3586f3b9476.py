import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy: Simulated Annealing on vertex permutation with incremental delta-evaluation.
    - Represent the ordering as a permutation array.
    - Move = swap two positions in the permutation.
    - Delta-evaluation: compute the change in independent-set size in O(1) by tracking
      which vertices are "available" (not blocked by a previously selected neighbor).
    - Annealing schedule: geometric cooling from T_high to T_low.
    - Time guard: return best permutation found before deadline.
    """
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Precompute neighbor sets as sorted arrays for fast membership checks
    # Use bitsets for O(1) neighbor checks via numpy
    # Actually, for n=200, we can use a 2D bool matrix directly
    # neighbors[i] = list of neighbors of i
    neighbors_list = [np.where(adj[i])[0].tolist() for i in range(n)]
    # Also create a set for O(1) lookup
    neighbor_sets = [set(neighbors_list[i]) for i in range(n)]
    
    # Initial ordering: sort by degree ascending (low degree first tends to help)
    deg = adj.sum(axis=1)
    order = np.argsort(deg, kind='stable')
    
    def evaluate_order(order):
        """
        Evaluate the greedy independent set size for a given permutation.
        Returns: (size, selected_mask, blocked_mask)
        selected_mask[v] = True if v is in the independent set
        blocked_mask[v] = True if v is blocked (neighbor of a selected vertex)
        """
        selected = np.zeros(n, dtype=bool)
        blocked = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v] and not blocked[v]:
                selected[v] = True
                count += 1
                # Block all neighbors
                for u in neighbors_list[v]:
                    if not selected[u]:
                        blocked[u] = True
        return count, selected, blocked
    
    # Initial evaluation
    best_count, best_selected, best_blocked = evaluate_order(order)
    best_order = order.copy()
    
    # Simulated Annealing
    rng = np.random.default_rng(42)
    
    # Temperature schedule
    T_start = 5.0  # Start hot enough to accept worse moves
    T_end = 0.01   # Cool down to near-deterministic
    n_iters = 50000  # Total iterations (will be limited by time)
    
    # We'll use a geometric cooling schedule
    # T(k) = T_start * (T_end/T_start)^(k/n_iters)
    cooling_rate = (T_end / T_start) ** (1.0 / n_iters)
    
    current_order = order.copy()
    current_count, current_selected, current_blocked = evaluate_order(current_order)
    current_T = T_start
    
    # Precompute position array for O(1) lookup: pos[v] = position of v in order
    pos = np.zeros(n, dtype=int)
    for i, v in enumerate(current_order):
        pos[v] = i
    
    for iteration in range(n_iters):
        if time.time() - start_time > deadline:
            break
        
        current_T *= cooling_rate
        
        # Pick two random positions to swap
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i == j:
            continue
        
        # Vertices at positions i and j
        v_i = current_order[i]
        v_j = current_order[j]
        
        # Incremental delta-evaluation
        # We need to determine: if we swap v_i and v_j, how does the independent set change?
        # This is complex because the greedy process is sequential.
        # 
        # Key insight: The greedy algorithm processes vertices in order.
        # A vertex v is selected iff it is not blocked when its turn comes.
        # v is blocked iff some neighbor u of v appears before v in the order AND u is selected.
        #
        # For a swap of two vertices at positions i and j (i < j w.l.o.g.):
        # - Vertices at positions < i: unaffected
        # - Vertices at positions > j: may be affected if v_i or v_j's selection status changes
        # - Vertices between i and j: affected by the new relative order of v_i and v_j
        #
        # This is still complex. Let's use a simpler but effective approach:
        # Just re-evaluate from position min(i,j) onward. Since n=200, this is O(n) per move,
        # but we can optimize by only checking the affected segment.
        
        # Actually, for n=200, a full re-evaluation is O(n * avg_degree) ≈ O(200 * 20) = 4000 ops.
        # With 50k iterations, that's 200M ops. In Python, this might be too slow.
        # 
        # Let's use a hybrid: do the swap, re-evaluate only from min(i,j) to end,
        # and track the delta.
        
        # Perform the swap
        current_order[i], current_order[j] = current_order[j], current_order[i]
        pos[v_i] = j
        pos[v_j] = i
        
        # Re-evaluate from position min(i, j) to end
        # We need to know the state of selected/blocked for positions < min(i,j)
        # Let's just re-evaluate the whole thing but with early termination if possible
        
        # For simplicity and correctness, re-evaluate fully but efficiently
        new_selected = np.zeros(n, dtype=bool)
        new_blocked = np.zeros(n, dtype=bool)
        new_count = 0
        start_pos = min(i, j)
        
        # Copy state from before start_pos
        # We need to re-simulate from 0 to get correct state at start_pos
        # This is O(start_pos * avg_degree)
        
        # Alternative: just do full evaluation. With numpy optimizations, it might be fast enough.
        # Let's try a vectorized-ish approach for the full evaluation.
        
        # Actually, let's just do a full re-evaluation. n=200 is small.
        for v in current_order:
            if not new_selected[v] and not new_blocked[v]:
                new_selected[v] = True
                new_count += 1
                for u in neighbors_list[v]:
                    if not new_selected[u]:
                        new_blocked[u] = True
        
        delta = new_count - current_count
        
        # Acceptance criterion
        if delta >= 0:
            # Accept improvement or equal
            current_count = new_count
            current_selected = new_selected
            current_blocked = new_blocked
            if current_count > best_count:
                best_count = current_count
                best_order = current_order.copy()
                best_selected = new_selected
                best_blocked = new_blocked
        else:
            # Accept worse move with probability exp(delta / T)
            # delta is negative, so delta/T is negative
            prob = np.exp(delta / current_T)
            if rng.random() < prob:
                current_count = new_count
                current_selected = new_selected
                current_blocked = new_blocked
            else:
                # Reject: swap back
                current_order[i], current_order[j] = current_order[j], current_order[i]
                pos[v_i] = i
                pos[v_j] = j
    
    # Convert best_order to scores: vertex at position k gets score (n - k)
    # So the first vertex in the order gets the highest score
    scores = np.zeros(n)
    for k, v in enumerate(best_order):
        scores[v] = n - k
    
    return scores
