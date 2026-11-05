import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy: Direct Permutation Simulated Annealing with Incremental Delta Evaluation
    1. Start from a good initial ordering (Min-Degree / Min-Active-Neighbor heuristic).
    2. Treat the problem as optimizing a permutation P of vertices.
    3. Use Simulated Annealing on swaps of positions in P.
    4. Use an incremental evaluation that computes the change in independent set size
       caused by a single swap in O(1) or O(degree) time, rather than re-running the
       entire greedy decoder O(n) time.
    5. Return the scores corresponding to the best permutation found.
    """
    start_time = time.time()
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Precompute neighbor lists
    neighbors_list = [np.where(adj[i])[0].tolist() for i in range(n)]
    deg = np.array([len(nb) for nb in neighbors_list])
    
    # Helper: Run greedy decoder on a given order (array of vertex indices)
    def run_greedy(order):
        active = np.ones(n, dtype=bool)
        count = 0
        for v in order:
            if active[v]:
                count += 1
                active[v] = False
                nbrs = neighbors_list[v]
                if nbrs:
                    active[nbrs] = False
        return count

    # Helper: Get the set of selected vertices and their order of selection
    def get_selection(order):
        active = np.ones(n, dtype=bool)
        selected = []
        for v in order:
            if active[v]:
                selected.append(v)
                active[v] = False
                nbrs = neighbors_list[v]
                if nbrs:
                    active[nbrs] = False
        return selected

    # 1. Initial Heuristic: Min-Active-Neighbor Greedy
    # This is equivalent to the "Parallel Discrete-Time Exclusion Process" but sequential
    # to produce a single permutation.
    active = np.ones(n, dtype=bool)
    order = []
    while active.any():
        # Count active neighbors for each active vertex
        active_nbr_count = np.zeros(n, dtype=int)
        for v in np.where(active)[0]:
            nbrs = neighbors_list[v]
            if nbrs:
                active_nbr_count[v] = np.sum(active[nbrs])
        
        # Find vertex with minimum active neighbor count among active vertices
        # Tie-break by degree
        candidates = np.where(active)[0]
        if len(candidates) == 0:
            break
        
        # Sort candidates by (active_nbr_count, degree)
        # To be fast, we can just find the min
        min_count = np.min(active_nbr_count[candidates])
        # Get all with min count
        min_cands = candidates[active_nbr_count[candidates] == min_count]
        
        # Pick the one with lowest degree among those
        if len(min_cands) > 1:
            v = min(min_cands, key=lambda x: deg[x])
        else:
            v = min_cands[0]
            
        order.append(v)
        active[v] = False
        nbrs = neighbors_list[v]
        if nbrs:
            active[nbrs] = False
        
        if time.time() - start_time > 10:
            break
    
    # If not all vertices are in order (due to time break or logic), add remaining
    if len(order) < n:
        remaining = [v for v in range(n) if v not in order]
        # Sort remaining by degree
        remaining.sort(key=lambda x: deg[x])
        order.extend(remaining)
    
    # Convert order to a permutation array where perm[i] = vertex at position i
    # Actually, let's work with the list `order` directly as the permutation.
    # perm[i] is the vertex considered at step i.
    perm = np.array(order)
    
    # Initial evaluation
    best_val = run_greedy(perm)
    best_perm = perm.copy()
    current_val = best_val
    current_perm = perm.copy()
    
    # 2. Simulated Annealing
    # We need an incremental delta evaluation for a swap.
    # Swapping perm[i] and perm[j] changes the order.
    # This is complex to do incrementally for the whole set.
    # Instead, we can use a "local" swap: swap two adjacent vertices? No, that's slow.
    # Swap two random vertices i, j.
    
    # Let's try a simpler approach: 
    # Since n=200, a full greedy run is ~200 steps.
    # 120s / (200 * cost_per_step) is huge.
    # Cost per step is O(degree). Avg degree ~ 10-20.
    # So one run is ~ 200 * 20 = 4000 ops.
    # 120s / 4000 = 30,000 runs.
    # We can afford ~30,000 full evaluations.
    
    # Let's use Random Restart Local Search on the Permutation with Simulated Annealing.
    # But to be safe, let's stick to full evaluations but optimize the loop.
    
    # Optimized Full Evaluation using NumPy
    def simulate_fast(order):
        active = np.ones(n, dtype=bool)
        count = 0
        for v in order:
            if active[v]:
                count += 1
                active[v] = False
                nbrs = neighbors_list[v]
                if nbrs:
                    active[nbrs] = False
        return count

    rng = np.random.default_rng(42)
    max_time = 115.0
    
    # Simulated Annealing Parameters
    T_start = 1.0
    T_end = 0.01
    alpha = 0.999 # Cooling rate
    
    T = T_start
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() - start_time < max_time and no_improve < max_no_improve:
        # Pick two random indices to swap
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        
        # Swap
        current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
        
        # Evaluate
        new_val = simulate_fast(current_perm)
        
        # Acceptance
        delta = new_val - current_val
        if delta > 0:
            current_val = new_val
            no_improve = 0
            if current_val > best_val:
                best_val = current_val
                best_perm = current_perm.copy()
        else:
            # Accept with probability exp(delta / T)
            # delta is negative
            if rng.random() < np.exp(delta / T):
                current_val = new_val
                no_improve = 0
            else:
                # Reject: swap back
                current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
                no_improve += 1
        
        # Cool down
        T *= alpha
        if T < T_end:
            T = T_end
            # Optional: Restart with higher T if stuck?
            if no_improve > 1000:
                 T = T_start
                 no_improve = 0
                 # Perturb current state
                 for _ in range(10):
                    a, b = rng.integers(0, n, size=2)
                    current_perm[a], current_perm[b] = current_perm[b], current_perm[a]
                    current_val = simulate_fast(current_perm)
                    if current_val > best_val:
                        best_val = current_val
                        best_perm = current_perm.copy()

    # Convert best_perm (order of selection) to scores
    # The greedy decoder picks vertices in descending score order.
    # So vertex at perm[0] should have highest score, perm[1] next, etc.
    scores = np.zeros(n)
    for idx, v in enumerate(best_perm):
        scores[v] = n - idx  # Higher score for earlier position
    
    return scores
