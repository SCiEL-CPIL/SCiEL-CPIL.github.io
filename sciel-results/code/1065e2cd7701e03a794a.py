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
    
    # Precompute intersection sizes
    # intersection[i][j] = len(set(supports[i]) & set(supports[j]))
    # We need intersection <= 2. So conflict if >= 3.
    # 330x330 matrix is small.
    conflict = np.zeros((n_supports, n_supports), dtype=bool)
    for i in range(n_supports):
        si = set(supports[i])
        for j in range(i + 1, n_supports):
            sj = set(supports[j])
            inter = len(si.intersection(sj))
            if inter >= 3:
                conflict[i][j] = True
                conflict[j][i] = True
    
    # Precompute degree for each node
    degree = np.sum(conflict, axis=1)
    
    best_mis = []
    
    # Greedy MIS with multiple strategies and random restarts
    num_restarts = 5000
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Choose strategy: weighted greedy with random perturbation
        # Strategy 1: Pick node with fewest conflicts among remaining
        # Strategy 2: Random order
        # Strategy 3: Degree-based ordering with noise
        
        current_mis = []
        in_mis = np.zeros(n_supports, dtype=bool)
        remaining = set(range(n_supports))
        
        # Use a priority-based greedy: at each step, pick the node in remaining
        # that has the fewest conflicts with other remaining nodes
        # Add random noise to break ties and explore different solutions
        
        while remaining:
            if time.time() > deadline:
                break
            
            # For each remaining node, count conflicts with other remaining nodes
            # This is expensive, so we'll use a heuristic:
            # Count conflicts with nodes already in MIS (should be 0 for valid nodes)
            # and use degree as a proxy
            
            # Filter to nodes that don't conflict with any in current_mis
            valid_nodes = []
            for node in remaining:
                if not np.any(conflict[node] & in_mis):
                    valid_nodes.append(node)
            
            if not valid_nodes:
                break
            
            # Score each valid node: prefer nodes with fewer total conflicts
            # (they're more "central" in the independent set sense)
            # Add random noise to explore different solutions
            scores = []
            for node in valid_nodes:
                # Count conflicts with remaining valid nodes
                c = 0
                for other in valid_nodes:
                    if other != node and conflict[node][other]:
                        c += 1
                scores.append(c + np.random.exponential(0.5))
            
            # Pick the node with minimum score
            min_idx = np.argmin(scores)
            chosen = valid_nodes[min_idx]
            
            current_mis.append(chosen)
            in_mis[chosen] = True
            remaining.discard(chosen)
        
        if len(current_mis) > len(best_mis):
            best_mis = current_mis
            # Print progress for debugging
            # print(f"Restart {r}: found MIS of size {len(current_mis)}")

    # Construct vectors
    vectors = []
    for s_idx in best_mis:
        support = supports[s_idx]
        # Generate all 16 sign variants
        # Indices in support are 4.
        # Signs are +/-1 for each.
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for k, idx in enumerate(support):
                v[idx] = signs[k]
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
