"""Rewritten program for kissing number in dimension 11."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # Generate W4 vectors
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate W1 vectors
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)

    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1

    # Precompute conflict masks as Python integers
    # Dot product matrix
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Convert to bitsets
    conflict_masks = [0] * N4
    neighbors = [None] * N4
    for i in range(N4):
        nbrs = np.where(conflicts[i])[0]
        neighbors[i] = nbrs
        mask = 0
        for j in nbrs:
            mask |= (1 << j)
        conflict_masks[i] = mask

    rng = np.random.default_rng(42)

    def greedy_init():
        """Randomized Greedy Independent Set."""
        mask = 0
        size = 0
        order = rng.permutation(N4)
        for i in order:
            if (mask & conflict_masks[i]) == 0:
                mask |= (1 << i)
                size += 1
        return mask, size

    def soft_sa(init_mask, init_size, max_time, lambda_pen=10.0):
        """
        Soft-constraint Simulated Annealing using Bitsets.
        Energy E = -size + lambda * num_conflicts
        """
        mask = init_mask
        size = init_size
        
        # Initialize conflict counts
        # conf_counts[i] = number of neighbors of i currently in the set
        conf_counts = np.zeros(N4, dtype=np.int32)
        # Iterate over set bits to populate conf_counts
        # This is O(N) per init, but N=5280, so it's fine.
        # Or we can compute it faster?
        # For each i in set, for each j in neighbors[i], conf_counts[j] += 1
        # This is O(size * degree).
        active_bits = []
        temp_mask = mask
        while temp_mask:
            # Find least significant set bit
            lsb = temp_mask & -temp_mask
            idx = lsb.bit_length() - 1
            active_bits.append(idx)
            temp_mask ^= lsb
            
        for i in active_bits:
            for j in neighbors[i]:
                conf_counts[j] += 1
        
        num_conflicts = np.sum(conf_counts) // 2
        current_E = -size + lambda_pen * num_conflicts
        best_E = current_E
        best_mask = mask
        best_size = size
        
        T = 5.0
        T_min = 0.01
        cooling = 0.999 # Slow cooling for more exploration
        
        sa_deadline = time.time() + max_time
        
        while time.time() < sa_deadline:
            i = rng.integers(N4)
            bit = 1 << i
            
            if mask & bit:
                # Flip 1 -> 0 (Remove)
                # Size decreases by 1
                # Conflicts decrease by conf_counts[i]
                delta_E = 1 - lambda_pen * conf_counts[i]
                accepted = (delta_E < 0) or (rng.random() < np.exp(-delta_E / T))
                if accepted:
                    mask ^= bit
                    size -= 1
                    current_E += delta_E
                    # Update conf_counts for neighbors
                    for j in neighbors[i]:
                        conf_counts[j] -= 1
                    if current_E < best_E:
                        best_E = current_E
                        best_mask = mask
                        best_size = size
            else:
                # Flip 0 -> 1 (Add)
                # Size increases by 1
                # Conflicts increase by number of neighbors in set
                # num_conf_new = (mask & conflict_masks[i]).bit_count()
                num_conf_new = (mask & conflict_masks[i]).bit_count()
                delta_E = -1 + lambda_pen * num_conf_new
                accepted = (delta_E < 0) or (rng.random() < np.exp(-delta_E / T))
                if accepted:
                    mask |= bit
                    size += 1
                    current_E += delta_E
                    # Update conf_counts for neighbors
                    for j in neighbors[i]:
                        conf_counts[j] += 1
                    if current_E < best_E:
                        best_E = current_E
                        best_mask = mask
                        best_size = size
            
            T *= cooling
            if T < T_min:
                T = T_min

        return best_mask, best_size

    best_size = 0
    best_mask = 0

    # Run multiple restarts
    # Estimate time per restart
    # Greedy init: ~0.1s?
    # SA: 0.5s?
    # Total ~0.6s per restart.
    # 115s / 0.6s ~ 190 restarts.
    
    while time.time() < deadline - 2:
        # Greedy Init
        m, s = greedy_init()
        
        # SA
        sa_time = 0.4
        m_res, s_res = soft_sa(m, s, sa_time)
        
        if s_res > best_size:
            best_size = s_res
            best_mask = m_res

    # Extract vectors from best_mask
    final_indices = []
    temp_mask = best_mask
    while temp_mask:
        lsb = temp_mask & -temp_mask
        idx = lsb.bit_length() - 1
        final_indices.append(idx)
        temp_mask ^= lsb
        
    result_v4 = vectors_v4[final_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])

    return final_vectors

# EVOLVE-BLOCK-END
