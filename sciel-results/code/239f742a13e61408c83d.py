import numpy as np
import itertools
import time

D = 11

def generate_weight_4_vectors():
    """Generate all vectors of weight 4 in {-1, 0, 1}^D"""
    candidates = []
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates.append(v)
    return np.array(candidates, dtype=np.int8)

def generate_weight_1_vectors():
    """Generate all weight-1 vectors in {-2, 0, 2}^D (norm^2 = 4)"""
    candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            candidates.append(v)
    return np.array(candidates, dtype=np.int64)

def greedy_mis(conflicts, deadline, rng):
    """
    Greedy Maximum Independent Set with degree-based ordering.
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
            
        # Find min degree among alive
        min_deg = np.min(degrees[alive_indices])
        candidates = alive_indices[degrees[alive_indices] == min_deg]
        
        # Random tie-breaking
        chosen = candidates[rng.integers(len(candidates))]
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

def local_improve_swap(selected, vectors, dots, conflicts, deadline, rng):
    """
    Try to improve the selected set by swapping out one vector for another.
    """
    N = len(vectors)
    selected_mask = np.zeros(N, dtype=bool)
    selected_mask[selected] = True
    current_selected = list(selected)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle selected indices to avoid bias
        rng.shuffle(current_selected)
        
        for idx in range(len(current_selected)):
            if time.time() > deadline:
                return current_selected
            
            s_out = current_selected[idx]
            
            # Create mask of "other selected"
            other_selected_mask = selected_mask.copy()
            other_selected_mask[s_out] = False
            
            # Find unselected vectors that do not conflict with any other selected
            if np.any(other_selected_mask):
                has_conflict = conflicts[:, other_selected_mask].any(axis=1)
            else:
                has_conflict = np.zeros(N, dtype=bool)
                
            # Candidates are those without conflict, not currently selected
            candidates = ~has_conflict & ~selected_mask
            
            # We need to pick a candidate u to replace s_out.
            if np.any(candidates):
                cand_indices = np.where(candidates)[0]
                u_in = cand_indices[rng.integers(len(cand_indices))]
                
                # Perform swap
                selected_mask[s_out] = False
                selected_mask[u_in] = True
                
                # Update list
                current_selected[idx] = u_in
                
                improved = True
                break  # Restart the loop with the new configuration
    
    return current_selected

def local_improve_add(selected, vectors, dots, conflicts, deadline, rng):
    """
    Try to add any non-conflicting vectors to the selected set.
    """
    N = len(vectors)
    selected_mask = np.zeros(N, dtype=bool)
    for idx in selected:
        if idx < N:
            selected_mask[idx] = True
    
    current_selected = [s for s in selected if s < N]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try to add any non-conflicting vector
        if np.any(selected_mask):
            has_conflict = conflicts[:, selected_mask].any(axis=1)
        else:
            has_conflict = np.zeros(N, dtype=bool)
        
        candidates = ~has_conflict & ~selected_mask
        
        if np.any(candidates):
            cand_indices = np.where(candidates)[0]
            # Add all candidates (they're all compatible)
            for u in cand_indices:
                selected_mask[u] = True
                current_selected.append(u)
                improved = True
    
    return current_selected

def simulated_annealing_mis(conflicts, initial_set, deadline, rng):
    """
    Simulated annealing for Maximum Independent Set.
    Uses dynamic vertex weights based on local density.
    """
    N = len(conflicts)
    selected_mask = np.zeros(N, dtype=bool)
    for idx in initial_set:
        selected_mask[idx] = True
    
    current_size = int(selected_mask.sum())
    best_size = current_size
    best_mask = selected_mask.copy()
    
    # Compute initial conflict degrees
    conflict_deg = conflicts.sum(axis=1).astype(np.float64)
    
    # Temperature schedule
    t_start = time.time()
    total_time = deadline - t_start
    T0 = 5.0
    T_end = 0.01
    cooling = (T_end / T0) ** (1.0 / max(1, int(total_time * 100)))
    
    T = T0
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline and T > T_end:
        if no_improve_count > max_no_improve:
            # Reset temperature and restart from best
            selected_mask = best_mask.copy()
            current_size = best_size
            T = T0 * 0.5
            no_improve_count = 0
        
        # Choose a random move type
        move_type = rng.integers(3)  # 0: add, 1: remove, 2: swap
        
        if move_type == 0:
            # Try to add a vector
            if current_size == 0:
                weights = np.ones(N)
            else:
                # Number of conflicts with currently selected
                conflicts_with_selected = conflicts[:, selected_mask].sum(axis=1).astype(np.float64)
                # Only consider non-selected vectors with no conflicts
                eligible = ~selected_mask & (conflicts_with_selected == 0)
                if not np.any(eligible):
                    T *= cooling
                    continue
                # Weight by inverse local density (sparser = higher weight)
                local_density = conflicts[eligible].sum(axis=1).astype(np.float64)
                weights = np.zeros(N)
                weights[eligible] = 1.0 / (local_density + 1.0)
            
            total_w = weights.sum()
            if total_w <= 0:
                T *= cooling
                continue
            
            probs = weights / total_w
            u_in = rng.choice(N, p=probs)
            
            if selected_mask[u_in]:
                T *= cooling
                continue
            
            # Check if u_in conflicts with any selected
            if conflicts[u_in, selected_mask].any():
                T *= cooling
                continue
            
            # Accept the addition
            selected_mask[u_in] = True
            current_size += 1
            
            if current_size > best_size:
                best_size = current_size
                best_mask = selected_mask.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                
        elif move_type == 1:
            # Try to remove a vector (to free up space)
            if current_size == 0:
                T *= cooling
                continue
            
            selected_indices = np.where(selected_mask)[0]
            if len(selected_indices) == 0:
                T *= cooling
                continue
            
            # Prefer removing vectors that are in dense regions
            local_density = conflicts[selected_indices].sum(axis=1).astype(np.float64)
            weights = local_density + 1.0
            total_w = weights.sum()
            if total_w <= 0:
                T *= cooling
                continue
            probs = weights / total_w
            s_out = selected_indices[rng.choice(len(selected_indices), p=probs)]
            
            # Accept removal with some probability (it's a downgrade)
            delta = -1.0
            if rng.random() < np.exp(delta / T):
                selected_mask[s_out] = False
                current_size -= 1
                no_improve_count += 1
            else:
                no_improve_count += 1
                
        else:
            # Try a swap: remove one selected, add one unselected
            if current_size == 0:
                T *= cooling
                continue
            
            selected_indices = np.where(selected_mask)[0]
            if len(selected_indices) == 0:
                T *= cooling
                continue
            
            # Choose which to remove (prefer dense)
            local_density = conflicts[selected_indices].sum(axis=1).astype(np.float64)
            weights = local_density + 1.0
            total_w = weights.sum()
            if total_w <= 0:
                T *= cooling
                continue
            probs = weights / total_w
            s_out = selected_indices[rng.choice(len(selected_indices), p=probs)]
            
            # Create mask of other selected
            other_selected = selected_mask.copy()
            other_selected[s_out] = False
            
            # Find unselected vectors that don't conflict with other_selected
            if np.any(other_selected):
                has_conflict = conflicts[:, other_selected].any(axis=1)
            else:
                has_conflict = np.zeros(N, dtype=bool)
            
            candidates = ~has_conflict & ~selected_mask
            if not np.any(candidates):
                no_improve_count += 1
                T *= cooling
                continue
            
            # Weight candidates by sparseness
            cand_indices = np.where(candidates)[0]
            local_density_cand = conflicts[cand_indices].sum(axis=1).astype(np.float64)
            weights_cand = 1.0 / (local_density_cand + 1.0)
            total_w_cand = weights_cand.sum()
            if total_w_cand <= 0:
                no_improve_count += 1
                T *= cooling
                continue
            probs_cand = weights_cand / total_w_cand
            u_in = cand_indices[rng.choice(len(cand_indices), p=probs_cand)]
            
            # Perform swap
            selected_mask[s_out] = False
            selected_mask[u_in] = True
            # Size unchanged
            
            no_improve_count += 1
        
        T *= cooling
    
    # Return the best solution found
    best_indices = np.where(best_mask)[0]
    return best_indices.tolist()

def construct():
    t0 = time.time()
    deadline = t0 + 115  # Safety margin before 150s kill
    
    # Generate weight-4 vectors
    vectors = generate_weight_4_vectors()
    N = len(vectors)
    
    # Generate weight-1 vectors (all compatible with weight-4 and each other)
    weight1_vectors = generate_weight_1_vectors()
    
    if N == 0:
        return weight1_vectors
        
    # Compute dot products for weight-4 vectors
    dots = vectors @ vectors.T
    
    # Conflict if dot > 2 (since norm^2 = 4, need dist^2 >= 4, so 4+4-2*dot >= 4 => dot <= 2)
    conflicts = (dots > 2).astype(np.int8)
    
    # Self-conflicts don't matter since we pick distinct vectors
    np.fill_diagonal(conflicts, 0)
    
    best_set = []
    rng = np.random.default_rng(42)
    
    # Phase 1: Multiple greedy MIS restarts (30 seconds)
    mis_deadline = t0 + 30
    best_greedy = []
    while time.time() < mis_deadline:
        current_set = greedy_mis(conflicts, mis_deadline, rng)
        if len(current_set) > len(best_greedy):
            best_greedy = current_set
    
    # Use the best greedy solution as initial for SA
    initial_set = best_greedy if len(best_greedy) > 0 else list(range(min(50, N)))
    
    # Phase 2: Simulated annealing (50 seconds)
    sa_deadline = t0 + 80
    if time.time() < sa_deadline:
        sa_set = simulated_annealing_mis(conflicts, initial_set, sa_deadline, rng)
        if len(sa_set) > len(best_set):
            best_set = sa_set
    
    # Phase 3: Local improvement with swaps (remaining time - 5s)
    local_deadline = deadline - 5
    if len(best_set) > 0 and time.time() < local_deadline:
        # First try to add any compatible vectors
        improved_set = local_improve_add(best_set, vectors, dots, conflicts, local_deadline, rng)
        if len(improved_set) > len(best_set):
            best_set = improved_set
        
        # Then try swap-based local improvement
        if time.time() < local_deadline - 2:
            swap_deadline = deadline - 3
            improved_set = local_improve_swap(best_set, vectors, dots, conflicts, swap_deadline, rng)
            if len(improved_set) > len(best_set):
                best_set = improved_set
    
    # Final: add any remaining compatible vectors
    if len(best_set) > 0 and time.time() < deadline - 2:
        final_set = local_improve_add(best_set, vectors, dots, conflicts, deadline - 1, rng)
        if len(final_set) > len(best_set):
            best_set = final_set
    
    if not best_set:
        return weight1_vectors
        
    result_vectors_w4 = vectors[best_set].astype(np.int64)
    
    # Combine weight-4 and weight-1 vectors
    final_vectors = np.vstack([result_vectors_w4, weight1_vectors])
    
    return final_vectors
