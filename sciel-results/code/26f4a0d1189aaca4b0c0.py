import numpy as np
import time
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Generate Candidates
    # Weight 2
    v2 = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                v2.append(v)
    
    # Weight 3
    v3 = []
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i], v[j], v[k] = si, sj, sk
                    v3.append(v)
                    
    # Weight 4
    v4 = []
    for i, j, k, l in itertools.combinations(range(D), 4):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = [0] * D
                        v[i], v[j], v[k], v[l] = si, sj, sk, sl
                        v4.append(v)
                        
    candidates = np.array(v2 + v3 + v4, dtype=np.int64)
    N = len(candidates)
    
    # 2. Compute Conflict Matrix
    # Condition: 2 * u.v > min(||u||^2, ||v||^2)
    dots = candidates @ candidates.T
    norms_sq = np.sum(candidates**2, axis=1)
    min_norms_sq = np.minimum(norms_sq[:, None], norms_sq[None, :])
    
    # Conflict if 2 * dot > min_norm
    # Note: For u=v, dot=norm, 2*norm > norm is True. So self-conflict is True.
    # We don't want to add the same vector twice, so self-conflict is fine.
    conflict = (2 * dots > min_norms_sq).astype(np.int8)
    
    # 3. Greedy MIS with Degree Ordering
    degrees = conflict.sum(axis=1)
    order = np.argsort(degrees)[::-1]
    
    in_set = np.zeros(N, dtype=bool)
    # conf_count[i] = number of vectors in in_set that conflict with i
    conf_count = np.zeros(N, dtype=np.int32)
    
    for idx in order:
        if in_set[idx]:
            continue
        if conf_count[idx] == 0:
            in_set[idx] = True
            # Update conf_count for neighbors
            neighbors = np.where(conflict[idx, :] == 1)[0]
            conf_count[neighbors] += 1
            
    current_size = in_set.sum()
    
    # 4. Local Search
    # We will try to improve by removing one and greedily adding.
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # Pick a random vector in set to remove
        set_indices = np.where(in_set)[0]
        if len(set_indices) == 0:
            break
        remove_idx = set_indices[rng.integers(0, len(set_indices))]
        
        # Remove it
        in_set[remove_idx] = False
        neighbors = np.where(conflict[remove_idx, :] == 1)[0]
        conf_count[neighbors] -= 1
        
        # Greedy Add
        # Find all candidates with conf_count == 0 and not in set
        # We can just iterate through a shuffled list of all candidates
        # Or find indices where conf_count == 0 and in_set == False
        available_indices = np.where((conf_count == 0) & (~in_set))[0]
        
        if len(available_indices) > 0:
            # Shuffle available indices
            rng.shuffle(available_indices)
            
            for add_idx in available_indices:
                if in_set[add_idx]:
                    continue
                if conf_count[add_idx] != 0:
                    continue
                
                # Add it
                in_set[add_idx] = True
                add_neighbors = np.where(conflict[add_idx, :] == 1)[0]
                conf_count[add_neighbors] += 1
                
                # Stop adding if we've added enough or time is up
                # Just let it run until no more available?
                # The list `available_indices` is static for this step.
                # Newly added vectors might block others in the list.
                # So we check `conf_count[add_idx] != 0` again?
                # No, we check before adding.
                # If we add A, it might block B (later in list).
                # So when we reach B, `conf_count[B]` will be > 0, so we skip.
                # This is correct.
        
        # Check if improved
        new_size = in_set.sum()
        if new_size < current_size:
            # Revert?
            # This is complex. Instead, we can just keep the best solution found so far.
            # Let's store the best set.
            pass
        else:
            current_size = new_size
            
    # Return the best set found
    # We need to return the actual vectors
    final_indices = np.where(in_set)[0]
    return candidates[final_indices]

# EVOLVE-BLOCK-END
