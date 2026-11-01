import numpy as np
import itertools
import time
import math

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
    Greedy Maximum Independent Set with degree-based ordering and random tie-breaking.
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
            
        # Find min degree among alive nodes
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

def simulated_annealing(conflicts, init_set, deadline, rng):
    """
    Simulated Annealing for Maximum Independent Set.
    """
    N = len(conflicts)
    selected = np.zeros(N, dtype=bool)
    for idx in init_set:
        selected[idx] = True
    
    current_size = np.sum(selected)
    # Calculate initial edges
    # Use dot product for speed
    current_edges = int(np.dot(conflicts[np.ix_(selected, selected)], np.ones((current_size, current_size), dtype=np.int8)) / 2) 
    # Wait, conflicts is symmetric. 
    # sum of all entries in submatrix / 2.
    # np.sum(conflicts[np.ix_(selected, selected)]) / 2
    current_edges = int(np.sum(conflicts[np.ix_(selected, selected)]) // 2)
    
    best_size = current_size if current_edges == 0 else 0
    best_set = selected.copy() if current_edges == 0 else np.zeros(N, dtype=bool)
    
    T = 1.0
    cooling_rate = 0.999
    min_T = 1e-5
    lambda_penalty = 10.0
    
    while time.time() < deadline:
        move_type = rng.integers(3)
        
        if move_type == 0: # Add
            unselected_indices = np.where(~selected)[0]
            if len(unselected_indices) == 0: 
                T *= cooling_rate
                continue
            v = unselected_indices[rng.integers(len(unselected_indices))]
            
            delta_edges = int(np.sum(conflicts[v][selected]))
            
            delta_E = -1 + lambda_penalty * delta_edges
            
            if delta_E < 0 or rng.random() < math.exp(-delta_E / T):
                selected[v] = True
                current_size += 1
                current_edges += delta_edges
                
                if current_edges == 0 and current_size > best_size:
                    best_size = current_size
                    best_set = selected.copy()
                    
        elif move_type == 1: # Remove
            selected_indices = np.where(selected)[0]
            if len(selected_indices) == 0:
                T *= cooling_rate
                continue
            v = selected_indices[rng.integers(len(selected_indices))]
            
            delta_edges = -int(np.sum(conflicts[v][selected]))
            
            delta_E = 1 + lambda_penalty * delta_edges
            
            if delta_E < 0 or rng.random() < math.exp(-delta_E / T):
                selected[v] = False
                current_size -= 1
                current_edges += delta_edges
                
                if current_edges == 0 and current_size > best_size:
                    best_size = current_size
                    best_set = selected.copy()
                    
        else: # Swap
            selected_indices = np.where(selected)[0]
            unselected_indices = np.where(~selected)[0]
            if len(selected_indices) == 0 or len(unselected_indices) == 0:
                T *= cooling_rate
                continue
            
            v_out = selected_indices[rng.integers(len(selected_indices))]
            v_in = unselected_indices[rng.integers(len(unselected_indices))]
            
            deg_out = int(np.sum(conflicts[v_out][selected]))
            # deg_in is number of neighbors of v_in in S, excluding v_out
            # Since v_out is in S, and we are swapping, v_in is not in S.
            # neighbors of v_in in S = sum(conflicts[v_in][selected])
            # But v_out is in S. Is v_in connected to v_out?
            # If yes, that edge is removed when v_out is removed, and added when v_in is added?
            # No.
            # Current edges involving v_out: (v_out, u) for u in S\{v_out}.
            # New edges involving v_in: (v_in, u) for u in S\{v_out}.
            # Note: (v_in, v_out) is NOT in S after swap (v_out removed).
            # So we remove edges (v_out, u) for u in S\{v_out}.
            # We add edges (v_in, u) for u in S\{v_out}.
            
            # sum(conflicts[v_in][selected]) includes (v_in, v_out) if they are connected.
            # But v_out is being removed. So (v_in, v_out) should NOT be counted in new edges.
            # So deg_in = sum(conflicts[v_in][selected]) - conflicts[v_in][v_out]
            
            deg_in = int(np.sum(conflicts[v_in][selected])) - int(conflicts[v_in][v_out])
            
            delta_edges = -deg_out + deg_in
            
            delta_E = 0 + lambda_penalty * delta_edges
            
            if delta_E < 0 or rng.random() < math.exp(-delta_E / T):
                selected[v_out] = False
                selected[v_in] = True
                # current_size unchanged
                current_edges += delta_edges
                
                if current_edges == 0 and current_size > best_size:
                    best_size = current_size
                    best_set = selected.copy()
        
        T *= cooling_rate
        if T < min_T:
            T = min_T

    return np.where(best_set)[0]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Generate weight-4 vectors
    vectors_w4 = generate_weight_4_vectors()
    N4 = len(vectors_w4)
    
    if N4 == 0:
        return generate_weight_1_vectors()
    
    # Generate weight-1 vectors
    vectors_w1 = generate_weight_1_vectors()
    N1 = len(vectors_w1)
    
    # Combine vectors
    all_vectors = np.vstack([vectors_w4, vectors_w1])
    N = len(all_vectors)
    
    # Compute full conflict matrix
    dots_all = all_vectors @ all_vectors.T
    conflicts_all = (dots_all > 2).astype(np.int8)
    np.fill_diagonal(conflicts_all, 0)
    
    best_set = []
    rng = np.random.default_rng(42)
    
    # Time allocation:
    # 10% for Greedy (fast)
    # 90% for SA
    
    greedy_deadline = start_time + 10
    sa_deadline = deadline - 5
    
    # Initial Greedy
    init_set = greedy_mis(conflicts_all, greedy_deadline, rng)
    
    # Run SA
    if time.time() < sa_deadline:
        current_best = simulated_annealing(conflicts_all, init_set, sa_deadline, rng)
        if len(current_best) > len(best_set):
            best_set = current_best
    
    # If we have time left, try another restart with a different init
    if time.time() < deadline - 2:
        rng2 = np.random.default_rng(123)
        init_set2 = greedy_mis(conflicts_all, start_time + 20, rng2)
        if len(init_set2) > len(best_set):
             best_set = init_set2
        # Run SA again
        current_best2 = simulated_annealing(conflicts_all, init_set2, deadline - 1, rng2)
        if len(current_best2) > len(best_set):
            best_set = current_best2
            
    if not best_set:
        return vectors_w1
    
    return all_vectors[best_set].astype(np.int64)
