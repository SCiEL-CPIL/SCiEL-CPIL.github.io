import numpy as np
import itertools
import time
from collections import defaultdict

D = 11

def generate_weight_4_vectors():
    """Generate all vectors of weight 4 in {-1, 0, 1}^D with a canonical form to reduce duplicates."""
    candidates = []
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(combo, signs):
                v[idx] = s
            # Canonical form: ensure the first non-zero entry is positive
            # This reduces the search space by half (sign symmetry)
            first_nonzero = next(i for i in range(D) if v[i] != 0)
            if v[first_nonzero] < 0:
                v = [-x for x in v]
            candidates.append(v)
    return np.array(candidates, dtype=np.int8)

def compute_conflicts(vectors):
    """
    Compute conflict matrix efficiently using bitwise operations.
    Two vectors conflict if their dot product > 2.
    For weight-4 vectors in {-1,0,1}, dot product = (# same sign) - (# opposite sign)
    where we only count positions where both have non-zero entries.
    """
    N = len(vectors)
    # Convert to bitmasks for positions and signs
    # pos_mask[i]: bitmask of positions where vector i is non-zero
    # sign_mask[i]: bitmask of positions where vector i is +1
    
    pos_masks = np.zeros(N, dtype=np.int32)
    sign_masks = np.zeros(N, dtype=np.int32)
    
    for i in range(N):
        v = vectors[i]
        pm = 0
        sm = 0
        for j in range(D):
            if v[j] != 0:
                pm |= (1 << j)
                if v[j] == 1:
                    sm |= (1 << j)
        pos_masks[i] = pm
        sign_masks[i] = sm
    
    # For each pair (i, j):
    # shared = pos_masks[i] & pos_masks[j]  (positions where both non-zero)
    # same_sign = shared & sign_masks[i] & sign_masks[j]  (both +1)
    # opposite_sign = shared & sign_masks[i] & ~sign_masks[j]  (i is +1, j is -1)
    #               + shared & ~sign_masks[i] & sign_masks[j]  (i is -1, j is +1)
    # dot = popcount(same_sign) - popcount(opposite_sign)
    # conflict if dot > 2
    
    conflicts = np.zeros((N, N), dtype=bool)
    
    # Vectorized computation
    for i in range(N):
        pi = pos_masks[i]
        si = sign_masks[i]
        
        # Shared positions
        shared = pos_masks & pi
        # Same sign: both +1 at shared positions
        same = shared & si
        # Opposite sign: i is +1, j is -1 OR i is -1, j is +1
        # i is +1, j is -1: si & ~sign_masks & shared
        # i is -1, j is +1: ~si & sign_masks & shared
        opp1 = si & (~sign_masks) & shared
        opp2 = (~si) & sign_masks & shared
        opposite = opp1 | opp2
        
        # Count bits
        same_count = np.array([bin(x).count('1') for x in same], dtype=np.int32)
        opp_count = np.array([bin(x).count('1') for x in opposite], dtype=np.int32)
        
        dots = same_count - opp_count
        conflicts[i] = dots > 2
    
    # Symmetrize (should already be symmetric, but ensure it)
    conflicts = conflicts | conflicts.T
    
    return conflicts

def greedy_mis_improved(conflicts, deadline, rng, strategy="min_degree"):
    """
    Improved Greedy Maximum Independent Set.
    strategy: "min_degree", "max_potential", "random"
    """
    N = len(conflicts)
    alive = np.ones(N, dtype=bool)
    degrees = conflicts.sum(axis=1).astype(np.int32)
    
    selected_indices = []
    
    while np.any(alive):
        if time.time() > deadline:
            break
        
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
            
        if strategy == "min_degree":
            # Min degree heuristic
            min_deg = np.min(degrees[alive_indices])
            candidates = alive_indices[degrees[alive_indices] == min_deg]
            chosen = candidates[rng.integers(len(candidates))]
        elif strategy == "max_potential":
            # Choose node that eliminates the fewest future options
            # Approximate by min degree
            min_deg = np.min(degrees[alive_indices])
            candidates = alive_indices[degrees[alive_indices] == min_deg]
            chosen = candidates[rng.integers(len(candidates))]
        else:  # random
            chosen = alive_indices[rng.integers(len(alive_indices))]
        
        selected_indices.append(chosen)
        
        # Find neighbors to remove
        neighbors = np.where(conflicts[chosen])[0]
        to_remove = np.unique(np.concatenate(([chosen], neighbors)))
        
        # Update alive
        alive[to_remove] = False
        
        # Update degrees for remaining alive nodes
        remaining_alive = np.where(alive)[0]
        if len(remaining_alive) > 0:
            removed_conflicts = conflicts[remaining_alive][:, to_remove].sum(axis=1)
            degrees[remaining_alive] -= removed_conflicts
    
    return selected_indices

def local_improve(selected, vectors, conflicts, deadline, rng):
    """
    Try to improve the independent set by adding more vertices or swapping.
    """
    if not selected:
        return selected
    
    N = len(vectors)
    selected_set = set(selected)
    selected_arr = np.array(selected)
    
    # Try to add vertices not in the set
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        
        for i in range(N):
            if i in selected_set:
                continue
            # Check if i can be added (no conflicts with any selected)
            if not np.any(conflicts[i][selected_arr]):
                selected_arr = np.append(selected_arr, i)
                selected_set.add(i)
                improved = True
                break
    
    # Try swaps: remove one, add two
    # This is more expensive, so do it less frequently
    if time.time() < deadline:
        for _ in range(50):  # Limited swap attempts
            if time.time() > deadline:
                break
            
            if len(selected_arr) < 2:
                break
            
            # Try removing a random element and adding others
            remove_idx = rng.integers(len(selected_arr))
            remove_node = selected_arr[remove_idx]
            
            # Temporarily remove
            remaining = np.delete(selected_arr, remove_idx)
            
            # Find candidates that can be added
            candidates = []
            for i in range(N):
                if i in selected_set:
                    continue
                if i == remove_node:
                    continue
                # Check compatibility with remaining
                if not np.any(conflicts[i][remaining]):
                    candidates.append(i)
            
            if len(candidates) >= 2:
                # Try adding two candidates
                for c1_idx in range(len(candidates)):
                    if time.time() > deadline:
                        break
                    c1 = candidates[c1_idx]
                    for c2_idx in range(c1_idx + 1, len(candidates)):
                        c2 = candidates[c2_idx]
                        # Check if c1 and c2 are compatible
                        if not conflicts[c1][c2]:
                            # This swap improves by 1
                            new_set = np.append(remaining, c1)
                            new_set = np.append(new_set, c2)
                            return new_set.tolist()
    
    return selected_arr.tolist()

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 150s limit
    
    # Generate candidates
    vectors = generate_weight_4_vectors()
    N = len(vectors)
    
    if N == 0:
        return np.empty((0, D), dtype=np.int64)
    
    # Compute conflicts
    conflicts = compute_conflicts(vectors)
    
    best_set = []
    rng = np.random.default_rng(42)
    
    strategies = ["min_degree", "max_potential", "random"]
    
    # Run multiple restarts with different strategies
    restart_count = 0
    while time.time() < deadline:
        strategy = strategies[restart_count % len(strategies)]
        current_set = greedy_mis_improved(conflicts, deadline, rng, strategy)
        
        # Local improvement
        if time.time() < deadline:
            current_set = local_improve(current_set, vectors, conflicts, deadline, rng)
        
        if len(current_set) > len(best_set):
            best_set = current_set
            # Update deadline if we made progress (optional)
        
        restart_count += 1
        # If we've done many restarts without improvement, try different seeds
        if restart_count % 10 == 0:
            rng = np.random.default_rng(restart_count * 12345)
    
    if not best_set:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
    
    result_vectors = vectors[best_set]
    return result_vectors.astype(np.int64)
