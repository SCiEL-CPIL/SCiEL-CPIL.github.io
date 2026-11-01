"""Combined program for Kissing Number in dimension 11."""
import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # 1. Generate W4 vectors (weight 4, entries in {-1,0,1}, norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    N4 = len(vectors_v4)
    
    # 2. Compute conflict matrix: conflict if dot > 2
    # Distance^2 = |u|^2 + |v|^2 - 2*u.v = 4 + 4 - 2*u.v = 8 - 2*u.v
    # Max norm squared is 4 (for W4) or 4 (for W1). Wait, W1 norm is 2, so norm^2=4.
    # If we mix W4 and W1:
    # dist^2(W4, W4) = 8 - 2 dot. Need dist^2 >= max_norm^2.
    # If max norm is from W1 (norm 2, sq 4) or W4 (norm 2, sq 4).
    # So we need 8 - 2 dot >= 4 => 4 >= 2 dot => dot <= 2.
    # Conflict if dot > 2.
    
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Convert to neighbor lists for faster local search updates
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
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

    def greedy_fill(in_set, counts):
        """
        Add all possible nodes to in_set, updating counts.
        """
        added = 0
        candidates = np.where((counts == 0) & (~in_set))[0]
        
        for v in candidates:
            if counts[v] == 0 and not in_set[v]:
                in_set[v] = True
                added += 1
                for u in neighbors[v]:
                    counts[u] += 1
        return added

    def local_search(in_set):
        """
        Local search with 1-for-1 and 2-for-1 swaps.
        """
        in_set = in_set.copy()
        counts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            counts[neighbors[i]] += 1
            
        current_size = in_set.sum()
        
        # First, greedy fill to maximize current set
        added = greedy_fill(in_set, counts)
        current_size += added
        
        # Local search loop
        while time.time() < deadline - 5:
            improved = False
            in_set_indices = np.where(in_set)[0]
            rng.shuffle(in_set_indices)
            
            # 1-for-1 swaps
            for v in in_set_indices:
                if not in_set[v]:
                    continue
                    
                # Try removing v
                in_set[v] = False
                for u in neighbors[v]:
                    counts[u] -= 1
                    
                # Greedy fill
                candidates = np.where((counts == 0) & (~in_set))[0]
                added = 0
                for u in candidates:
                    if counts[u] == 0 and not in_set[u]:
                        in_set[u] = True
                        added += 1
                        for w in neighbors[u]:
                            counts[w] += 1
                            
                new_size = in_set.sum()
                if new_size > current_size:
                    current_size = new_size
                    improved = True
                else:
                    # Revert
                    in_set[v] = True
                    # Rebuild counts for safety
                    counts = np.zeros(N4, dtype=np.int32)
                    for i in np.where(in_set)[0]:
                        counts[neighbors[i]] += 1
            
            if improved:
                continue
            
            # 2-for-1 swaps if 1-for-1 didn't improve
            # Sample a few pairs to try
            if len(in_set_indices) >= 2:
                # Try a limited number of random pairs to avoid O(N^2) cost
                num_pairs_to_try = min(50, len(in_set_indices) * (len(in_set_indices) - 1) // 2)
                for _ in range(num_pairs_to_try):
                    if time.time() > deadline - 5:
                        break
                    v1, v2 = rng.choice(in_set_indices, size=2, replace=False)
                    if not in_set[v1] or not in_set[v2]:
                        continue
                    
                    # Remove v1, v2
                    in_set[v1] = False
                    in_set[v2] = False
                    for u in neighbors[v1]:
                        counts[u] -= 1
                    for u in neighbors[v2]:
                        counts[u] -= 1
                        
                    # Greedy fill
                    candidates = np.where((counts == 0) & (~in_set))[0]
                    added = 0
                    for u in candidates:
                        if counts[u] == 0 and not in_set[u]:
                            in_set[u] = True
                            added += 1
                            for w in neighbors[u]:
                                counts[w] += 1
                                
                    new_size = in_set.sum()
                    if new_size > current_size:
                        current_size = new_size
                        improved = True
                        break
                    else:
                        # Revert
                        in_set[v1] = True
                        in_set[v2] = True
                        counts = np.zeros(N4, dtype=np.int32)
                        for i in np.where(in_set)[0]:
                            counts[neighbors[i]] += 1
            
            if not improved:
                break
                
        return in_set

    # Main Loop: Multiple Restarts
    n_restarts = 0
    max_restarts = 1000 # Increased restarts as per inspiration
    
    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1
        
        # 1. Greedy Init
        in_set = greedy_min_degree()
        
        # 2. Local Search
        remaining_time = deadline - 10 - time.time()
        if remaining_time < 2:
            break
            
        in_set = local_search(in_set)
        
        current_size = in_set.sum()
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # 3. Perturbation: Remove 4 random elements and local search
        if n_restarts > 1 and time.time() < deadline - 5:
            perturbed = best_in_set.copy()
            indices = np.where(perturbed)[0]
            if len(indices) > 4:
                to_remove = rng.choice(indices, size=4, replace=False)
                perturbed[to_remove] = False
            perturbed = local_search(perturbed)
            current_size = perturbed.sum()
            if current_size > best_size:
                best_size = current_size
                best_in_set = perturbed.copy()

    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    
    # Add W1 vectors (norm 2, weight 1)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    # Check if W1 vectors conflict with each other?
    # W1: (2,0,..0), (0,2,..0). Dot = 0. Dist^2 = 4+4-0=8 >= 4. OK.
    # W1 with W4: Dot <= 2. Dist^2 >= 4. OK.
    
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

if __name__ == "__main__":
    construct()
