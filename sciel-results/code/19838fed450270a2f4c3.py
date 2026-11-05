import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    DEADLINE = 145.0  # 5s safety margin before 150s kill

    A = np.asarray(adjacency, dtype=bool)
    N = A.shape[0]

    if N == 0:
        return np.array([])

    # Ensure symmetric
    A = (A | A.T)
    np.fill_diagonal(A, False)

    # Precompute neighbor bitmasks
    neighbor_masks = [0] * N
    for i in range(N):
        mask = 0
        # Using np.where is faster for sparse matrices, but A is dense bool.
        # Iterating is fine for N=200.
        for j in range(N):
            if A[i, j]:
                mask |= (1 << j)
        neighbor_masks[i] = mask

    def evaluate_perm(perm):
        blocked = 0
        count = 0
        for v in perm:
            if (blocked >> v) & 1:
                continue
            count += 1
            blocked |= (1 << v)
            blocked |= neighbor_masks[v]
        return count

    # Initial Permutation: Sort by degree (ascending)
    degrees = A.sum(axis=1)
    perm = list(np.argsort(degrees, kind='stable'))
    
    current_size = evaluate_perm(perm)
    best_perm = perm.copy()
    best_size = current_size
    
    rng = np.random.default_rng(42)
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    while time.time() < DEADLINE:
        i = rng.integers(0, N)
        j = rng.integers(0, N)
        if i == j:
            continue
            
        perm[i], perm[j] = perm[j], perm[i]
        
        new_size = evaluate_perm(perm)
        delta = new_size - current_size
        
        if delta > 0:
            current_size = new_size
            if current_size > best_size:
                best_size = current_size
                best_perm = perm.copy()
        else:
            if T > min_T:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    current_size = new_size
                else:
                    perm[i], perm[j] = perm[j], perm[i]
            
        T *= cooling
        if T < min_T:
            T = min_T

    scores = np.zeros(N)
    for k, v in enumerate(best_perm):
        scores[v] = N - k
        
    return scores
