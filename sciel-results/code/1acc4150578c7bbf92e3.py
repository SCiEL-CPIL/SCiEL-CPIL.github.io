"""Combined program for Kissing Number in dimension 11."""
import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 150s kill

    # Generate W4 vectors (weight 4, entries in {-1,0,1}, norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    N4 = len(vectors_v4)
    
    # Compute conflict matrix: conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Convert to neighbor lists for faster local search updates
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    # Precompute conflict counts for each node (same as neighbor_counts)
    # But we need fast updates, so use the conflicts matrix
    
    rng = np.random.default_rng(42)
    
    best_in_set = np.zeros(N4, dtype=bool)
    best_size = 0

    def greedy_min_degree():
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set

    def greedy_fill_fast(in_set, counts, candidate_set):
        """
        Add all possible nodes from candidate_set to in_set, updating counts.
        candidate_set is a boolean array marking which nodes are candidates.
        Returns number of nodes added.
        """
        added = 0
        # Process candidates in order
        candidates = np.where(candidate_set & (counts == 0) & (~in_set))[0]
        
        for v in candidates:
            if counts[v] == 0 and not in_set[v]:
                in_set[v] = True
                added += 1
                for u in neighbors[v]:
                    counts[u] += 1
        return added

    def local_search(in_set):
        """
        Iterated local search with efficient candidate tracking.
        """
        in_set = in_set.copy()
        counts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            counts[neighbors[i]] += 1
            
        current_size = in_set.sum()
        
        # First, greedy fill to maximize current set
        candidate_set = ~in_set
        added = greedy_fill_fast(in_set, counts, candidate_set)
        current_size += added
        
        # Local search loop
        while time.time() < deadline - 8:
            improved = False
            # Randomize order of nodes in set to avoid bias
            in_set_indices = np.where(in_set)[0]
            rng.shuffle(in_set_indices)
            
            for v in in_set_indices:
                if not in_set[v]:
                    continue
                    
                # Try removing v
                in_set[v] = False
                removed_nbrs = neighbors[v]
                for u in removed_nbrs:
                    counts[u] -= 1
                    
                # Greedy fill: find all nodes with count 0 not in set
                candidate_set = ~in_set
                added_nodes = []
                candidates = np.where(candidate_set & (counts == 0))[0]
                for u in candidates:
                    if counts[u] == 0 and not in_set[u]:
                        in_set[u] = True
                        added_nodes.append(u)
                        for w in neighbors[u]:
                            counts[w] += 1
                            
                new_size = in_set.sum()
                if new_size > current_size:
                    current_size = new_size
                    improved = True
                else:
                    # Revert: undo the additions and re-add v
                    for u in added_nodes:
                        in_set[u] = False
                        for w in neighbors[u]:
                            counts[w] -= 1
                    in_set[v] = True
                    for u in removed_nbrs:
                        counts[u] += 1
            
            if not improved:
                # Try a simulated annealing step to escape local optimum
                if time.time() < deadline - 8:
                    # Randomly remove 2-4 nodes and re-fill
                    in_set_indices = np.where(in_set)[0]
                    if len(in_set_indices) > 4:
                        num_remove = rng.integers(2, 5)
                        to_remove = rng.choice(in_set_indices, size=num_remove, replace=False)
                        saved_in_set = in_set.copy()
                        for v in to_remove:
                            in_set[v] = False
                        # Rebuild counts efficiently
                        counts = np.zeros(N4, dtype=np.int32)
                        for i in np.where(in_set)[0]:
                            counts[neighbors[i]] += 1
                        # Greedy fill
                        candidate_set = ~in_set
                        greedy_fill_fast(in_set, counts, candidate_set)
                        new_size = in_set.sum()
                        if new_size > current_size:
                            current_size = new_size
                            improved = True
                        else:
                            # Revert to saved state
                            in_set[:] = saved_in_set
                            counts = np.zeros(N4, dtype=np.int32)
                            for i in np.where(in_set)[0]:
                                counts[neighbors[i]] += 1

                if not improved:
                    break
                    
        return in_set

    # Main Loop: Multiple Restarts
    n_restarts = 0
    max_restarts = 50  # More restarts with faster local search
    
    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1
        
        # 1. Greedy Init
        in_set = greedy_min_degree()
        
        # 2. Local Search
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 2:
            break
            
        in_set = local_search(in_set)
        
        current_size = in_set.sum()
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # 3. Try perturbation of the best set found so far
        if n_restarts > 1 and time.time() < deadline - 5:
            perturbed = best_in_set.copy()
            # Remove 2-3 random elements
            indices = np.where(perturbed)[0]
            if len(indices) > 3:
                num_remove = rng.integers(2, 4)
                to_remove = rng.choice(indices, size=num_remove, replace=False)
                perturbed[to_remove] = False
            perturbed = local_search(perturbed)
            current_size = perturbed.sum()
            if current_size > best_size:
                best_size = current_size
                best_in_set = perturbed.copy()

    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    
    # Add W1 vectors (norm 2, weight 1) - they don't conflict with W4 vectors
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    # Try to add W2 vectors (entries in {-2,0,2}, weight 2, norm^2=8)
    # W2 vector has two non-zero entries of magnitude 2.
    # Dot product with W4 vector: max is 2*1 + 2*1 = 4 (if both positions match)
    # But we need dot <= max_norm = 4 (since W4 has norm 2, W2 has norm 2*sqrt(2) ≈ 2.83)
    # Actually, the condition is: min pairwise distance >= max vector norm
    # Distance squared = ||u-v||^2 = ||u||^2 + ||v||^2 - 2*u·v
    # For W4 (norm 2) and W2 (norm 2*sqrt(2)):
    # dist^2 = 4 + 8 - 2*u·v = 12 - 2*u·v
    # We need dist >= max_norm. What is max_norm?
    # If we have vectors of different norms, max_norm is the largest norm.
    # W2 has norm 2*sqrt(2) ≈ 2.83, W4 has norm 2.
    # So max_norm = 2*sqrt(2).
    # Need dist >= 2*sqrt(2), so dist^2 >= 8.
    # 12 - 2*u·v >= 8 => u·v <= 2.
    # W2·W4: W2 has entries ±2 at two positions, W4 has entries ±1 at four positions.
    # Max dot = 2*1 + 2*1 = 4 (if both W2 positions are in W4 with same sign)
    # So W2 and W4 CAN conflict (dot=4 > 2).
    # W2·W2: max dot = 2*2 + 2*2 = 8 (if same positions, same signs)
    # Need dot <= 2 for W2-W2 pairs too.
    # This is complex. Let's try adding W2 vectors that don't conflict with selected W4 and each other.
    
    # Generate W2 candidates
    v2_candidates = []
    for pos1 in range(D):
        for pos2 in range(pos1+1, D):
            for sign1 in [1, -1]:
                for sign2 in [1, -1]:
                    v = [0] * D
                    v[pos1] = 2 * sign1
                    v[pos2] = 2 * sign2
                    v2_candidates.append(v)
    vectors_v2 = np.array(v2_candidates, dtype=np.int64)
    N2 = len(vectors_v2)
    
    # Check which W2 vectors can be added
    # They must not conflict with any selected W4 vector or with each other
    # Conflict condition: dot > 2 (since we need min distance >= max norm)
    # Actually, let's be more careful about the exact condition.
    
    # The condition is: for all pairs u, v in the set, ||u - v|| >= max_{w in set} ||w||
    # Let M = max_{w in set} ||w||
    # Then for all u, v: ||u - v||^2 >= M^2
    # ||u - v||^2 = ||u||^2 + ||v||^2 - 2*u·v >= M^2
    
    # For two vectors with norms n1, n2:
    # n1^2 + n2^2 - 2*u·v >= M^2
    # u·v <= (n1^2 + n2^2 - M^2) / 2
    
    # If we include W2 (norm^2 = 8), then M^2 >= 8, so M >= 2*sqrt(2)
    # For W4-W4 pairs (norm^2 = 4): u·v <= (4+4-8)/2 = 0. But we already have u·v <= 2 from before!
    # Wait, that's wrong. If M^2 = 8, then for W4-W4: u·v <= (4+4-8)/2 = 0.
    # But our W4 independent set has u·v <= 2 for all pairs. So if we add W2, we need to recheck W4-W4 pairs!
    # This means we might need to remove some W4 vectors.
    
    # This is getting complex. Let's just try adding W2 vectors that don't conflict with W4 (dot <= 2)
    # and with each other (dot <= 2), and see if the overall condition holds.
    # If it doesn't, we'll need to be more careful.
    
    # For now, let's add W1 vectors (which definitely don't conflict) and try a few W2 vectors.
    
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
