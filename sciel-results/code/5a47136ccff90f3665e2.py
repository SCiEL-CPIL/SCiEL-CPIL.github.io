# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 118

    D = 11
    
    # Generate W4 vectors (weight 4, entries in {-1, 0, 1}, norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate W1 vectors (weight 2, entries in {-2, 0, 2}, norm^2=4)
    # Note: In the previous code, W1 was weight 1 with entry 2, which is norm 2, not 4.
    # The task says "min pairwise distance >= max vector norm".
    # If all vectors have norm 2, distance >= 2 implies dot product <= 0.
    # Wait, the task says "Construct nonzero integer vectors ... minimum pairwise distance is at least the maximum vector norm".
    # If we normalize to norm 2, the condition is distance >= 2.
    # Distance^2 = |u|^2 + |v|^2 - 2u.v.
    # If |u|=|v|=2, dist^2 = 8 - 2u.v.
    # dist >= 2 => dist^2 >= 4 => 8 - 2u.v >= 4 => 2u.v <= 4 => u.v <= 2.
    # So we need an independent set in the graph where edges connect vectors with dot product > 2.
    
    # The previous code used W1 as weight 1 with entry 2 (norm 2).
    # If we use vectors of norm 2, the condition is u.v <= 0?
    # Let's re-read carefully.
    # "Normalizing each vector to norm 2 certifies a kissing configuration..."
    # This implies the final vectors should have norm 2? Or just that the configuration is valid?
    # "Construct nonzero integer vectors ... minimum pairwise distance is at least the maximum vector norm".
    # If I return vectors of norm 2, then max norm is 2. Min dist >= 2.
    # If I return vectors of norm 4 (squared), i.e., norm 2.
    # The previous code returned W4 vectors (norm 2) and W1 vectors (norm 2).
    # Wait, the previous code's W1 was `v[pos] = 2 * sign`. Norm is 2.
    # So all vectors have norm 2.
    # Condition: dist(u, v) >= 2.
    # dist^2(u, v) = |u|^2 + |v|^2 - 2 u.v = 4 + 4 - 2 u.v = 8 - 2 u.v.
    # We need 8 - 2 u.v >= 4 => 2 u.v <= 4 => u.v <= 2.
    # So we need u.v <= 2.
    # Conflict if u.v > 2.
    
    # The previous code used:
    # v4: weight 4, entries +/-1. Norm^2 = 4. Norm = 2.
    # v1: weight 1, entry +/-2. Norm^2 = 4. Norm = 2.
    # So both have norm 2.
    # Conflict condition: dot > 2.
    
    # My new code will do the same but include W1 (weight 2, entries +/-2? No, weight 1 entry 2 is norm 2. Weight 2 entries +/-1 is norm sqrt(2). Weight 2 entries +/-2 is norm 4.
    # To have norm 2, we can have:
    # - 4 entries of +/-1 (Norm^2 = 4)
    # - 1 entry of +/-2 (Norm^2 = 4)
    # - 2 entries of +/-1? No, Norm^2 = 2.
    # So the only integer vectors of norm 2 are:
    # 1. One coordinate is +/-2, rest 0. (22 vectors)
    # 2. Four coordinates are +/-1, rest 0. (5280 vectors)
    
    # The previous code generated exactly these.
    # v4_candidates: 5280 vectors.
    # v1_candidates: 22 vectors.
    
    # Total candidates: 5302.
    # We need Max Independent Set on this graph.
    
    all_vectors = []
    # W4
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            all_vectors.append(v)
            
    # W1 (Weight 1, entry 2)
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            all_vectors.append(v)
            
    V = np.array(all_vectors, dtype=np.int64)
    N = len(V)
    
    # Compute Dot Products
    # N is ~5300. N^2 is ~28 million. This is feasible in numpy.
    dots = V @ V.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Convert to adjacency list for faster local search
    # We will use a bitset or boolean array for the independent set.
    
    rng = np.random.default_rng(42)
    
    def get_neighbors(i):
        return np.where(conflicts[i])[0]
    
    # Precompute neighbor lists
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    
    # Heuristic Initialization: Greedy
    in_set = np.zeros(N, dtype=bool)
    alive = np.ones(N, dtype=bool)
    
    # Sort by degree (number of conflicts) to pick low degree nodes first
    degrees = np.array([len(n) for n in neighbors])
    order = np.argsort(degrees)
    
    for i in order:
        if time.time() > deadline - 10:
            break
        if not alive[i]:
            continue
        # If no conflicts with current set, add it
        # Check conflicts with in_set
        # This is O(N) per node, so O(N^2) total. Might be slow.
        # Better: maintain a count of conflicts for each node.
        
    # Let's use a faster greedy:
    in_set = np.zeros(N, dtype=bool)
    conflict_count = np.zeros(N, dtype=np.int32)
    
    # Random shuffle order
    order = np.random.permutation(N)
    
    for i in order:
        if time.time() > deadline - 10:
            break
        if conflict_count[i] == 0:
            in_set[i] = True
            # Update conflict counts for neighbors
            for j in neighbors[i]:
                if not in_set[j]: # Only update if not already in set? 
                    # Actually, if j is in set, we shouldn't have added i.
                    # But we are checking conflict_count[i] == 0 before adding.
                    # So if we add i, it means no neighbor is in set.
                    # So we just need to mark neighbors as "conflicting" for future additions.
                    conflict_count[j] += 1
            # Mark i as not available (though conflict_count handles it if we check >0)
            # But we need to ensure we don't add i again? We iterate once.
            
    current_size = in_set.sum()
    best_size = current_size
    best_in_set = in_set.copy()
    
    # Local Search: Add/Remove
    # We try to add a node v if it has 0 conflicts, or if we remove k nodes and add v, and 1-k > 0.
    # Since we want to maximize size, we only accept if size increases.
    # This is a "steepest ascent" local search.
    
    # Precompute neighbor lists as arrays for vectorization
    # To speed up, we can use a "candidate" list.
    
    sa_deadline = deadline - 5
    T = 1.0
    cooling = 0.999
    
    # Maintain conflict counts
    # conflict_count[v] = number of neighbors of v that are in in_set
    
    conflict_count = np.zeros(N, dtype=np.int32)
    for i in np.where(in_set)[0]:
        conflict_count[neighbors[i]] += 1
        
    stuck = 0
    while time.time() < sa_deadline:
        # Find candidates to add
        # Nodes with conflict_count == 0 can be added immediately.
        # Nodes with conflict_count == 1 can be added if we remove 1 node (delta 0, accept with prob).
        # Nodes with conflict_count == k can be added if we remove k nodes (delta 1-k).
        
        # For simplicity and speed, we only do:
        # 1. Add nodes with conflict_count == 0.
        # 2. Swap: Remove 1 node, Add 1 node (if net gain 0, accept with prob).
        # 3. Remove 1 node, Add 2 nodes? (Complex).
        
        # Let's stick to Add/Remove of single nodes.
        # If we add v, we must remove all its neighbors in in_set.
        # Gain = 1 - (# neighbors in in_set).
        
        not_in_set = np.where(~in_set)[0]
        if len(not_in_set) == 0:
            break
            
        # Calculate gain for each node not in set
        # gain[v] = 1 - conflict_count[v]
        gains = 1 - conflict_count[not_in_set]
        
        # Find max gain
        max_gain = np.max(gains)
        
        if max_gain < 0:
            # No improving move. Try random perturbation or break.
            stuck += 1
            if stuck > 1000:
                # Perturbation
                # Remove 10 random nodes
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) > 10:
                    remove_cands = in_set_indices[rng.choice(len(in_set_indices), 10, replace=False)]
                    for u in remove_cands:
                        in_set[u] = False
                        conflict_count[neighbors[u]] -= 1
                    current_size -= 10
                stuck = 0
                T = 2.0
                continue
            else:
                # Accept random neutral/worse move?
                # For now, just break or accept random.
                # Let's try a random node.
                v = not_in_set[rng.integers(len(not_in_set))]
                k = conflict_count[v]
                # Accept if random < exp(-k/T) ?
                if rng.random() < np.exp(-k / T):
                    to_remove = np.where(conflicts[v] & in_set)[0]
                    for u in to_remove:
                        in_set[u] = False
                        conflict_count[neighbors[u]] -= 1
                    in_set[v] = True
                    conflict_count[neighbors[v]] += 1
                    current_size += 1 - len(to_remove)
        else:
            # Find all nodes with max gain
            max_cands = not_in_set[gains == max_gain]
            v = max_cands[rng.integers(len(max_cands))]
            
            to_remove = np.where(conflicts[v] & in_set)[0]
            
            # Apply move
            for u in to_remove:
                in_set[u] = False
                conflict_count[neighbors[u]] -= 1
                
            in_set[v] = True
            conflict_count[neighbors[v]] += 1
            
            current_size += 1 - len(to_remove)
            
            if current_size > best_size:
                best_size = current_size
                best_in_set = in_set.copy()
                stuck = 0
            else:
                stuck += 1
                
        T *= cooling
        if T < 0.1:
            T = 0.1
            
    # Extract best set
    final_indices = np.where(best_in_set)[0]
    result = V[final_indices]
    
    return result

# EVOLVE-BLOCK-END
