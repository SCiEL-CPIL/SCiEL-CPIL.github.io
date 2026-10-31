import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    
    # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate all weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
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
    
    # Compute conflict graph: conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Convert to bitset representation for fast operations
    # Each row i: bitset of nodes conflicting with i
    conflict_bitsets = []
    for i in range(N4):
        mask = 0
        row = conflicts[i]
        # Use numpy to find indices and build bitset
        idxs = np.where(row)[0]
        for idx in idxs:
            mask |= (1 << idx)
        conflict_bitsets.append(mask)
    
    def greedy_mis(seed):
        """Greedy MIS using bitsets for fast neighbor operations."""
        rng = np.random.default_rng(seed)
        alive_mask = (1 << N4) - 1  # All nodes alive
        selected = []
        alive_count = N4
        
        while alive_count > 0:
            # Find min-degree alive node
            # Degree of node i = popcount(conflict_bitsets[i] & alive_mask)
            # To find min degree, we need to check all alive nodes
            # This is O(N4) per step, but with bit operations
            
            # Find all alive indices
            alive_indices = []
            m = alive_mask
            while m:
                # Get lowest set bit
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                alive_indices.append(idx)
                m ^= lsb
            
            if not alive_indices:
                break
            
            # Find min degree among alive nodes
            min_deg = float('inf')
            min_cands = []
            for idx in alive_indices:
                deg = bin(conflict_bitsets[idx] & alive_mask).count('1')
                if deg < min_deg:
                    min_deg = deg
                    min_cands = [idx]
                elif deg == min_deg:
                    min_cands.append(idx)
            
            # Randomly pick from min-degree candidates
            chosen = min_cands[rng.integers(len(min_cands))]
            selected.append(chosen)
            
            # Remove chosen and all its alive neighbors
            # Neighbors of chosen that are alive: conflict_bitsets[chosen] & alive_mask
            # Plus chosen itself
            to_remove = conflict_bitsets[chosen] | (1 << chosen)
            removed_count = bin(to_remove & alive_mask).count('1')
            alive_mask &= ~to_remove
            alive_count -= removed_count
            
            if time.time() > deadline - 10:
                break
        
        return selected
    
    # Try multiple seeds to find best independent set
    best_selected = []
    best_size = 0
    
    # First try deterministic (seed=0)
    selected_0 = greedy_mis(0)
    best_selected = selected_0
    best_size = len(selected_0)
    
    # Try more seeds if time permits
    for seed in range(1, 50):
        if time.time() > deadline - 15:
            break
        selected = greedy_mis(seed)
        if len(selected) > best_size:
            best_selected = selected
            best_size = len(selected)
    
    result_v4 = vectors_v4[best_selected]
    
    # All V1 vectors are mutually compatible (verified in analysis)
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors
