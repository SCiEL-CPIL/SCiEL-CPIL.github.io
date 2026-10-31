import numpy as np
import itertools
import time

D = 11

def generate_vectors_by_weight(weights):
    """
    Generate vectors of specified weights in {-1, 0, 1}^D.
    Returns a list of lists.
    """
    vectors = []
    for w in weights:
        for combo in itertools.combinations(range(D), w):
            for signs in itertools.product([1, -1], repeat=w):
                v = [0] * D
                for idx, s in zip(combo, signs):
                    v[idx] = s
                vectors.append(v)
    return np.array(vectors, dtype=np.int8)

def greedy_mis(conflicts, deadline, seed=0):
    """
    Simple greedy MIS with random restarts logic handled outside or inside?
    Let's keep it simple: one pass with scoring.
    """
    N = len(conflicts)
    if N == 0:
        return []
    
    alive = np.ones(N, dtype=bool)
    degrees = conflicts.sum(axis=1).astype(np.float64)
    
    # Scoring: prefer low degree
    # We can just use degree.
    
    selected_indices = []
    rng = np.random.RandomState(seed)
    
    while np.any(alive):
        if time.time() > deadline:
            break
        
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
        
        # Score: lower degree is better
        alive_degrees = degrees[alive_indices]
        
        # Add noise
        noise = rng.random(len(alive_indices)) * 0.1
        score = alive_degrees + noise
        
        best_local_idx = np.argmin(score)
        chosen = alive_indices[best_local_idx]
        selected_indices.append(chosen)
        
        # Remove chosen and neighbors
        neighbors = np.where(conflicts[chosen])[0]
        to_remove = np.unique(np.concatenate(([chosen], neighbors)))
        alive[to_remove] = False
        
        # Update degrees
        remaining_alive = np.where(alive)[0]
        if len(remaining_alive) > 0:
            removed_conflicts = conflicts[remaining_alive][:, to_remove].sum(axis=1)
            degrees[remaining_alive] -= removed_conflicts

    return selected_indices

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    # Weights to include
    # W2: 220 vectors
    # W3: 1320 vectors
    # W4: 5280 vectors
    # Total: ~6820 candidates
    
    # We know W3 and W4 are mutually compatible and compatible with W2.
    # W2 are only compatible with W2 if dot <= 0.
    
    w2_vectors = generate_vectors_by_weight([2])
    w3_vectors = generate_vectors_by_weight([3])
    w4_vectors = generate_vectors_by_weight([4])
    
    N_w2 = len(w2_vectors)
    
    # Compute conflict matrix for W2s only
    # Conflict if dot > 0
    dots_w2 = w2_vectors @ w2_vectors.T
    conflicts_w2 = dots_w2 > 0
    
    # Find MIS for W2s
    best_w2_selected = []
    num_restarts = 50 # Small graph, can afford more restarts
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        selected = greedy_mis(conflicts_w2, deadline - 5, seed=r)
        if len(selected) > len(best_w2_selected):
            best_w2_selected = selected
            
    # Construct final set
    # All W3 and W4 are safe to include
    # Add selected W2s
    
    final_vectors = np.vstack([w3_vectors, w4_vectors, w2_vectors[best_w2_selected]])
    
    return final_vectors.astype(np.int64)
