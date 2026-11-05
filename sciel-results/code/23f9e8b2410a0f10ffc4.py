import time
import numpy as np
import random

def construct(q, n, radius):
    t_start = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    space_size = q ** n
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i-1] * q

    def word_to_idx(w):
        idx = 0
        for d in w:
            idx = idx * q + d
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n-1-pos]) % q
        return w

    # Precompute neighbors for radius=1
    nb_count = 1 + n * (q - 1)
    neighbors = np.empty((space_size, nb_count), dtype=np.int32)
    for idx in range(space_size):
        w = idx_to_word(idx)
        nb_list = [idx]
        for pos in range(n):
            for new_d in range(q):
                if new_d == w[pos]:
                    continue
                delta = (new_d - w[pos]) * q_powers[n-1-pos]
                nb_list.append(idx + delta)
        neighbors[idx] = nb_list

    # Build incidence matrix A[v, u] = 1 if u in N(v)
    A = np.zeros((space_size, space_size), dtype=np.int32)
    for v in range(space_size):
        A[v, neighbors[v]] = 1

    # --- Phase 1: Fast Greedy Cover ---
    uncovered = np.ones(space_size, dtype=np.int32)
    total_uncovered = space_size
    selected_mask = np.zeros(space_size, dtype=bool)
    code_indices = []

    while total_uncovered > 0 and (time.time() - t_start) < deadline - 30.0:
        # score_v = number of uncovered points covered by v
        scores = A @ uncovered
        scores[selected_mask] = -1
        best_idx = int(np.argmax(scores))
        if scores[best_idx] <= 0:
            break
        code_indices.append(best_idx)
        selected_mask[best_idx] = True
        nb = neighbors[best_idx]
        gain = int(np.sum(uncovered[nb]))
        uncovered[nb] = 0
        total_uncovered -= gain

    # Fallback
    while total_uncovered > 0 and (time.time() - t_start) < deadline - 25.0:
        scores = A @ uncovered
        scores[selected_mask] = -1
        best_idx = int(np.argmax(scores))
        if scores[best_idx] <= 0:
            break
        code_indices.append(best_idx)
        selected_mask[best_idx] = True
        nb = neighbors[best_idx]
        gain = int(np.sum(uncovered[nb]))
        uncovered[nb] = 0
        total_uncovered -= gain

    # --- Phase 2: Greedy Removal (Redundancy) ---
    cov = np.zeros(space_size, dtype=np.int32)
    for c_idx in code_indices:
        cov[neighbors[c_idx]] += 1

    active = [True] * len(code_indices)
    changed = True
    while changed and (time.time() - t_start) < deadline - 20.0:
        changed = False
        for i in range(len(code_indices)):
            if not active[i]:
                continue
            c_idx = code_indices[i]
            if np.min(cov[neighbors[c_idx]]) >= 2:
                active[i] = False
                cov[neighbors[c_idx]] -= 1
                changed = True

    current_indices = [code_indices[i] for i in range(len(code_indices)) if active[i]]
    best_size = len(current_indices)
    best_indices = current_indices[:]

    # --- Phase 3: Simulated Annealing on Fixed-Size Selection ---
    # We work with a binary vector x of length space_size, with exactly K ones.
    # Objective: minimize uncovered_count = sum_u [cov_u == 0]
    # Moves: swap a selected codeword with an unselected one.
    
    rng = np.random.default_rng(12345)
    
    # Determine initial K: start with current size, allow it to decrease
    K = best_size
    x = np.zeros(space_size, dtype=np.int32)
    for c in best_indices:
        x[c] = 1
    
    # Precompute cov for current x
    cov = np.zeros(space_size, dtype=np.int32)
    for c in best_indices:
        cov[neighbors[c]] += 1
    uncovered_count = int(np.sum(cov == 0))
    
    best_x = x.copy()
    best_uncovered = uncovered_count
    best_K = K
    
    # Annealing schedule
    t_max = deadline - 5.0
    t_current = t_start + 1.0  # Start after setup
    t_end = t_max
    alpha = 0.9995  # Cooling rate
    
    # We'll iterate until time runs out
    # Each iteration: pick a random selected codeword and a random unselected codeword, swap
    # Compute delta in uncovered count
    
    # For speed, we maintain cov array and update incrementally
    # When we swap out c_old and in c_new:
    #   cov[neighbors[c_old]] -= 1
    #   cov[neighbors[c_new]] += 1
    #   Recompute uncovered count efficiently
    
    # To recompute uncovered count efficiently, we track how many points have cov==0
    # Before swap: u_old = count of u where cov[u]==0
    # After swap: some points may change coverage
    
    # Let's just recompute uncovered count by checking the affected neighborhoods
    # Actually, it's easier to just count zeros in cov each time if space is small
    # space_size = 2187, so counting zeros is O(2187) which is fast
    
    # Let's do it properly with incremental updates
    # When removing c_old:
    #   For each u in N(c_old): old_cov = cov[u], new_cov = old_cov - 1
    #   If old_cov == 1 and new_cov == 0: uncovered increases by 1
    #   If old_cov == 0: impossible (c_old covers u)
    # When adding c_new:
    #   For each u in N(c_new): old_cov = cov[u], new_cov = old_cov + 1
    #   If old_cov == 0 and new_cov == 1: uncovered decreases by 1
    
    # We'll do the swap in order: remove then add, tracking uncovered count
    
    iterations = 0
    while (time.time() - t_start) < t_max:
        iterations += 1
        
        # Temperature based on iterations
        temp = max(0.1, 5.0 * (alpha ** iterations))
        
        # Pick random selected codeword to remove
        selected_list = np.where(x == 1)[0]
        if len(selected_list) == 0:
            break
        c_old = int(selected_list[rng.integers(0, len(selected_list))])
        
        # Pick random unselected codeword to add
        unselected_list = np.where(x == 0)[0]
        if len(unselected_list) == 0:
            break
        c_new = int(unselected_list[rng.integers(0, len(unselected_list))])
        
        if c_old == c_new:
            continue
        
        # Compute delta in uncovered count
        # Remove c_old
        nb_old = neighbors[c_old]
        delta_remove = 0
        for u in nb_old:
            if cov[u] == 1:
                # This point becomes uncovered
                delta_remove += 1
            # If cov[u] > 1, no change in uncovered count
            # If cov[u] == 0, impossible
        
        # Add c_new
        nb_new = neighbors[c_new]
        delta_add = 0
        for u in nb_new:
            # After removing c_old, cov[u] may have changed
            # We need to be careful: if u is in both nb_old and nb_new,
            # the order matters. Let's compute net effect.
            pass
        
        # Actually, let's just do the swap and recompute uncovered count
        # It's O(2187) which is acceptable
        
        # Perform swap
        x[c_old] = 0
        x[c_new] = 1
        cov[nb_old] -= 1
        cov[nb_new] += 1
        
        new_uncovered = int(np.sum(cov == 0))
        
        if new_uncovered < uncovered_count:
            # Accept
            uncovered_count = new_uncovered
            if new_uncovered == 0:
                # We have a valid cover with K codewords
                # Try to reduce K
                K -= 1
                # Remove one random codeword
                selected_list = np.where(x == 1)[0]
                if len(selected_list) > 0:
                    c_remove = int(selected_list[rng.integers(0, len(selected_list))])
                    x[c_remove] = 0
                    cov[neighbors[c_remove]] -= 1
                    uncovered_count = int(np.sum(cov == 0))
                # Update best
                if K < best_K:
                    best_K = K
                    best_x = x.copy()
                    best_uncovered = uncovered_count
            elif K < best_K or (K == best_K and new_uncovered < best_uncovered):
                best_K = K
                best_x = x.copy()
                best_uncovered = new_uncovered
        else:
            # Metropolis acceptance
            delta = new_uncovered - uncovered_count
            if delta <= 0 or rng.random() < np.exp(-delta / max(temp, 1e-6)):
                # Accept
                uncovered_count = new_uncovered
                if K < best_K or (K == best_K and new_uncovered < best_uncovered):
                    best_K = K
                    best_x = x.copy()
                    best_uncovered = new_uncovered
            else:
                # Reject: undo swap
                x[c_old] = 1
                x[c_new] = 0
                cov[nb_old] += 1
                cov[nb_new] -= 1
        
        # Check if we've achieved zero uncovered
        if uncovered_count == 0 and K < best_size:
            # Update best and try to reduce K further
            best_K = K
            best_x = x.copy()
            best_uncovered = 0

    # --- Phase 4: Extract best solution and do final removal pass ---
    if best_uncovered == 0:
        final_indices = list(np.where(best_x == 1)[0])
    else:
        # Fall back to greedy solution
        final_indices = best_indices
    
    # Final removal pass on the best solution
    cov = np.zeros(space_size, dtype=np.int32)
    for c in final_indices:
        cov[neighbors[c]] += 1
    
    active = [True] * len(final_indices)
    changed = True
    while changed and (time.time() - t_start) < deadline - 2.0:
        changed = False
        for i in range(len(final_indices)):
            if not active[i]:
                continue
            c_idx = final_indices[i]
            if np.min(cov[neighbors[c_idx]]) >= 2:
                active[i] = False
                cov[neighbors[c_idx]] -= 1
                changed = True
    
    current_indices = [final_indices[i] for i in range(len(final_indices)) if active[i]]
    
    # Verify coverage
    cov_final = np.zeros(space_size, dtype=np.int32)
    for c in current_indices:
        cov_final[neighbors[c]] += 1
    
    if np.any(cov_final == 0):
        # Fallback to greedy
        cov_init = np.zeros(space_size, dtype=np.int32)
        for i in code_indices:
            cov_init[neighbors[i]] += 1
        if not np.any(cov_init == 0):
            return [idx_to_word(i) for i in code_indices]
        return [idx_to_word(i) for i in range(space_size)]
    
    return [idx_to_word(i) for i in current_indices]
