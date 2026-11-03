import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s margin (5s safety before 120s limit)
    
    # Generate all supports (combinations of 4 indices)
    supports = list(itertools.combinations(range(D), 4))
    n_supports = len(supports)
    
    # Precompute conflict matrix: conflict[i][j] = True if supports[i] and supports[j] share >= 3 indices
    conflict = np.zeros((n_supports, n_supports), dtype=bool)
    for i in range(n_supports):
        si = set(supports[i])
        for j in range(i + 1, n_supports):
            sj = set(supports[j])
            inter = len(si.intersection(sj))
            if inter >= 3:
                conflict[i][j] = True
                conflict[j][i] = True
    
    # Precompute degree (number of conflicts) for each node
    degree = np.sum(conflict, axis=1)
    
    # Precompute for each node, the list of conflicting nodes (for fast lookup)
    # Use bitmasks for fast intersection checks
    # n_supports = 330, so we can use a 330-bit mask per node
    # Actually, let's use numpy boolean arrays for speed
    
    best_mis = []
    best_size = 0
    
    # Greedy MIS with multiple strategies and random restarts
    num_restarts = 20000
    rng = np.random.default_rng(42)
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Choose strategy based on restart number
        # Strategy 1: Minimum degree greedy (prefer nodes with fewest conflicts)
        # Strategy 2: Random order
        # Strategy 3: Weighted random with degree-based bias
        
        strategy = r % 3
        
        in_mis = np.zeros(n_supports, dtype=bool)
        remaining = np.ones(n_supports, dtype=bool)
        current_mis = []
        
        if strategy == 0:
            # Minimum remaining degree greedy
            while np.any(remaining):
                if time.time() > deadline:
                    break
                
                # Get remaining node indices
                rem_idx = np.where(remaining)[0]
                if len(rem_idx) == 0:
                    break
                
                # For each remaining node, count conflicts with other remaining nodes
                # This is the expensive part - use matrix multiplication
                # conflict_rem = conflict[rem_idx][:, rem_idx] gives conflicts among remaining
                # But we need to exclude self-conflicts (diagonal is False)
                
                # Count conflicts for each remaining node with other remaining nodes
                # Use matrix-vector product: for each node i in rem_idx,
                # count = sum(conflict[i, j] for j in rem_idx, j != i)
                
                # Efficient: create a mask of remaining, multiply conflict by mask
                # For each node, count = sum(conflict[node] & remaining) - conflict[node][node]
                # Since diagonal is False, just sum(conflict[node] & remaining)
                
                # Vectorized: for all nodes at once
                # counts = np.sum(conflict & remaining[None, :], axis=1)
                # But we only care about remaining nodes
                counts_all = np.sum(conflict & remaining[None, :], axis=1)
                counts = counts_all[rem_idx]
                
                # Add small random noise to break ties
                noise = rng.exponential(0.1, size=len(rem_idx))
                scores = counts + noise
                
                # Pick node with minimum score
                min_local_idx = np.argmin(scores)
                chosen = rem_idx[min_local_idx]
                
                current_mis.append(chosen)
                in_mis[chosen] = True
                remaining[chosen] = False
                # Also remove all neighbors of chosen from remaining
                remaining[conflict[chosen]] = False
        
        elif strategy == 1:
            # Random order greedy
            order = rng.permutation(n_supports)
            for node in order:
                if time.time() > deadline:
                    break
                if not remaining[node]:
                    continue
                if np.any(conflict[node] & in_mis):
                    continue
                current_mis.append(node)
                in_mis[node] = True
                remaining[node] = False
                remaining[conflict[node]] = False
        
        else:
            # Degree-biased random: prefer low-degree nodes with randomness
            while np.any(remaining):
                if time.time() > deadline:
                    break
                
                rem_idx = np.where(remaining)[0]
                if len(rem_idx) == 0:
                    break
                
                # Score: inverse degree (lower degree = higher priority) + random
                degs = degree[rem_idx]
                # Avoid division by zero
                safe_degs = np.maximum(degs, 1)
                scores = 1.0 / safe_degs + rng.exponential(0.05, size=len(rem_idx))
                
                # Pick node with maximum score (most likely to be chosen)
                max_local_idx = np.argmax(scores)
                chosen = rem_idx[max_local_idx]
                
                current_mis.append(chosen)
                in_mis[chosen] = True
                remaining[chosen] = False
                remaining[conflict[chosen]] = False
        
        # Local improvement: try to add more nodes or swap
        if len(current_mis) > best_size:
            best_size = len(current_mis)
            best_mis = list(current_mis)
        
        # Local improvement: try adding nodes that conflict with fewest in_mis
        # This is a simple local search
        if len(current_mis) > 0 and time.time() < deadline - 5:
            # Try to improve by swapping: for each node in MIS, try replacing with a non-MIS node
            # that conflicts with fewer MIS nodes
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                mis_set = set(current_mis)
                for i, node in enumerate(current_mis):
                    if time.time() > deadline:
                        break
                    # Try to find a replacement
                    # A replacement node must not conflict with any other MIS node (except this one)
                    # and should have lower degree or be "better" in some way
                    # For simplicity, try all non-MIS nodes
                    for candidate in range(n_supports):
                        if candidate in mis_set:
                            continue
                        # Check if candidate conflicts with any MIS node other than node
                        conflicts_with_mis = np.sum(conflict[candidate] & in_mis)
                        # in_mis includes node, so conflicts_with_mis >= 1 if candidate conflicts with node
                        # We need conflicts_with_mis == 1 (only with node) for a valid swap
                        if conflicts_with_mis == 1:
                            # Check that candidate actually conflicts with node
                            if conflict[candidate][node]:
                                # Swap: remove node, add candidate
                                # But we need to verify no other conflicts
                                # Already checked above
                                current_mis[i] = candidate
                                in_mis[node] = False
                                in_mis[candidate] = True
                                improved = True
                                break
                    if improved:
                        break
        
        if len(current_mis) > best_size:
            best_size = len(current_mis)
            best_mis = list(current_mis)
    
    # Construct vectors
    vectors = []
    for s_idx in best_mis:
        support = supports[s_idx]
        # Generate all 16 sign variants
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for k, idx in enumerate(support):
                v[idx] = signs[k]
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
