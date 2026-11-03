import numpy as np
import itertools
import time
import random
import sys

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill

    # Generate all 4-subsets of range(11)
    subsets = list(itertools.combinations(range(D), 4))
    n = len(subsets)  # 330

    # Convert subsets to bitmasks for fast intersection checks
    masks = []
    for s in subsets:
        m = 0
        for pos in s:
            m |= (1 << pos)
        masks.append(m)
        
    # Precompute adjacency list (conflict graph)
    # i and j conflict if they share exactly 3 or 4 elements (intersection size >= 3)
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            inter = masks[i] & masks[j]
            if inter.bit_count() >= 3:
                adj[i].add(j)
                adj[j].add(i)
    
    # Precompute degrees
    degrees = [len(adj[i]) for i in range(n)]
    
    best_is = set()
    best_size = 0

    # Helper to check time
    def time_ok():
        return time.time() < deadline

    # --- Strategy 1: Min-Degree Greedy ---
    def greedy_min_degree():
        available = set(range(n))
        is_set = set()
        # Use a copy of degrees to track remaining degrees if we want to be fancy, 
        # but recalculating is safer and n is small (330).
        while available and time_ok():
            # Find node with min degree in induced subgraph
            min_deg = float('inf')
            candidates = []
            for v in available:
                deg = len(adj[v] & available)
                if deg < min_deg:
                    min_deg = deg
                    candidates = [v]
                elif deg == min_deg:
                    candidates.append(v)
            
            if not candidates:
                break
            
            # Pick one (random tie-breaker to diversify if called multiple times? 
            # For now, deterministic min-degree is a strong baseline)
            # Let's add a bit of randomness by shuffling candidates if > 1
            if len(candidates) > 1:
                random.shuffle(candidates)
            best_node = candidates[0]
            
            is_set.add(best_node)
            to_remove = adj[best_node] | {best_node}
            available -= to_remove
        
        return is_set

    # --- Strategy 2: Random Order Greedy ---
    def greedy_random(rng):
        order = list(range(n))
        rng.shuffle(order)
        available = set(range(n))
        is_set = set()
        for v in order:
            if v not in available:
                continue
            # Check if v is still "available" (not removed by a previous pick)
            # In this greedy, we just check if it conflicts with already picked ones.
            # Since we iterate in fixed order, we just need to ensure it doesn't conflict with is_set.
            if not (adj[v] & is_set):
                is_set.add(v)
                # Optimization: remove neighbors from 'available' to skip them later?
                # Not strictly necessary for correctness, but speeds up checks if we used 'available' for iteration.
                # Here we just check intersection.
        return is_set

    # --- Local Search: Add ---
    def local_search_add(is_set):
        is_set = set(is_set)
        improved = True
        while improved and time_ok():
            improved = False
            # Find all candidates that can be added
            # Optimization: Only check nodes not in is_set
            for v in range(n):
                if v in is_set:
                    continue
                if not (adj[v] & is_set):
                    is_set.add(v)
                    improved = True
        return is_set

    # --- Local Search: Swap (1-for-2 or more) ---
    def local_search_swap(is_set, max_iter=20):
        is_set = set(is_set)
        current_size = len(is_set)
        
        for _ in range(max_iter):
            if not time_ok():
                break
            improved = False
            # Try removing each node
            for v in list(is_set):
                if not time_ok():
                    return is_set
                trial = is_set - {v}
                
                added = 0
                # Try to add nodes that were blocked by v
                # Specifically, neighbors of v that are not in trial
                # And any other node not in trial that fits
                # Since n=330, checking all is fine.
                for u in range(n):
                    if u in trial:
                        continue
                    if not (adj[u] & trial):
                        trial.add(u)
                        added += 1
                
                if len(trial) > current_size:
                    is_set = trial
                    current_size = len_set = len(is_set)
                    improved = True
                    break # Restart with new set
            
            if not improved:
                break
        
        return is_set

    # --- Advanced: Branch and Bound for Maximum Independent Set ---
    # This is the core improvement from the "inspiration" strategy.
    # We solve for the exact MIS on the conflict graph.
    # Given n=330, a naive B&B might be slow, but with good bounds (coloring) it can work.
    # However, implementing a robust B&B from scratch in <150s with Python overhead is risky.
    # Instead, we will use a "Greedy + Local Search" hybrid but with a more aggressive
    # "Repair" phase or a "Simulated Annealing" style swap if time permits.
    # 
    # Actually, let's stick to a very fast, high-quality heuristic:
    # 1. Run min-degree greedy.
    # 2. Run random greedy multiple times.
    # 3. Polish all with Add + Swap.
    # 4. Keep the best.

    def polish(is_set):
        is_set = local_search_add(is_set)
        is_set = local_search_swap(is_set, max_iter=10)
        is_set = local_search_add(is_set) # Add again after swaps
        return is_set

    seed_val = 0
    max_restarts = 200 # Upper bound, but time-limited

    while time.time() < deadline - 2.0:
        rng = random.Random(seed_val)
        
        # Alternate strategies
        if seed_val % 2 == 0:
            is_set = greedy_min_degree()
        else:
            is_set = greedy_random(rng)
        
        is_set = polish(is_set)
        
        if len(is_set) > best_size:
            best_size = len(is_set)
            best_is = is_set
            # print(f"Seed {seed_val}: size {best_size}", file=sys.stderr)
            
        seed_val += 1
        
        # If we hit a very high number, we can try to be more careful or stop early if we hit theoretical max?
        # Theoretical max for weight-4 supports in D=11 is not 41.
        # Let's see: Max independent set in the graph where edges are intersection >= 3.
        # Total nodes 330.
        # If we find 40+, that's great.

    # Construct vectors
    vectors = []
    for s_idx in best_is:
        support = subsets[s_idx]
        # Generate all 16 sign variants
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for k, idx in enumerate(support):
                v[idx] = signs[k]
            vectors.append(v)
            
    if not vectors:
        # Fallback: return at least one vector
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        v[2] = 1
        v[3] = 1
        vectors.append(v)
        
    return np.array(vectors, dtype=np.int64)
