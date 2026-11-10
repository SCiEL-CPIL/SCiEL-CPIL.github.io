import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Finds a large subset of Z/205Z with no two elements differing by a nonzero square.
    Uses a multi-start greedy approach combined with a local search (hill climbing with kicks).
    """
    m = 205
    
    # 1. Precompute forbidden differences and conflict matrix
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    conflict_matrix = np.zeros((m, m), dtype=bool)
    for d in forbidden:
        for i in range(m):
            j = (i + d) % m
            conflict_matrix[i, j] = True
            conflict_matrix[j, i] = True

    # Helper to get neighbors of a set S (boolean mask)
    def get_neighbors(S):
        neighbors = np.zeros(m, dtype=bool)
        indices = np.where(S)[0]
        for u in indices:
            neighbors |= conflict_matrix[u]
        return neighbors

    # Greedy initialization
    def greedy_init():
        S = np.zeros(m, dtype=bool)
        order = np.random.permutation(m)
        neighbors_of_S = np.zeros(m, dtype=bool)
        for v in order:
            if not neighbors_of_S[v]:
                S[v] = True
                neighbors_of_S |= conflict_matrix[v]
        return S

    # Local Search
    def local_search(S, time_limit=1.0):
        S = S.copy()
        current_size = np.sum(S)
        best_S = S.copy()
        best_size = current_size
        start_time = time.time()
        
        # Parameters for kick
        kick_sizes = [2, 3, 4, 5, 6]
        
        while time.time() - start_time < time_limit:
            neighbors_of_S = get_neighbors(S)
            candidates = ~S & ~neighbors_of_S
            
            if np.any(candidates):
                # Greedy Add
                cand_idx = np.where(candidates)[0]
                # Heuristic: Pick a random candidate to maintain diversity
                v = cand_idx[np.random.randint(len(cand_idx))]
                S[v] = True
                current_size += 1
                
                if current_size > best_size:
                    best_size = current_size
                    best_S = S.copy()
                continue
            
            # Local Optimum: Perform a Kick
            if current_size == 0:
                break
                
            k = random.choice(kick_sizes)
            if current_size < k:
                break
                
            S_old = S.copy()
            size_old = current_size
            
            indices = np.where(S)[0]
            to_remove = np.random.choice(indices, size=k, replace=False)
            S[to_remove] = False
            current_size -= k
            
            # Re-greedy fill
            neighbors_of_S = get_neighbors(S)
            candidates = ~S & ~neighbors_of_S
            cand_idx = np.where(candidates)[0]
            np.random.shuffle(cand_idx)
            
            for v in cand_idx:
                # Check again to be safe, though neighbors_of_S is updated
                if not neighbors_of_S[v]:
                    S[v] = True
                    current_size += 1
                    neighbors_of_S |= conflict_matrix[v]
            
            # Acceptance criteria
            if current_size > size_old:
                # Strict improvement
                if current_size > best_size:
                    best_size = current_size
                    best_S = S.copy()
            elif current_size == size_old:
                # Plateau: Accept with probability 0.5 to escape local minima
                if random.random() < 0.5:
                    if current_size > best_size:
                        best_size = current_size
                        best_S = S.copy()
            else:
                # Degradation: Revert
                S[:] = S_old
                current_size = size_old
                
        return best_S

    # Main Search Loop
    deadline = time.time() + 550  # Safety margin
    best_global = np.zeros(m, dtype=bool)
    best_global_size = 0
    
    # First, run a deterministic greedy to establish a baseline
    # (Though random is usually better, this ensures we have *something*)
    S_det = np.zeros(m, dtype=bool)
    neighbors_det = np.zeros(m, dtype=bool)
    for v in range(m):
        if not neighbors_det[v]:
            S_det[v] = True
            neighbors_det |= conflict_matrix[v]
    best_global = S_det
    best_global_size = np.sum(S_det)
    
    # Multi-start random search
    restart_count = 0
    while time.time() < deadline:
        # Generate random initial solution
        S_init = greedy_init()
        
        # Allocate time for local search
        # If we have lots of time left, give it more.
        # If we are close to deadline, give it less.
        time_left = deadline - time.time()
        if time_left < 5:
            break
            
        # Give each restart a slice of time, e.g., 5 seconds
        # Or run until no improvement? 
        # Let's run local search for a fixed small duration or until stuck.
        # Given m=205, local search steps are fast.
        # Let's try running for 2 seconds per restart.
        ls_time = min(2.0, time_left - 5)
        
        S_best = local_search(S_init, time_limit=ls_time)
        size = np.sum(S_best)
        
        if size > best_global_size:
            best_global = S_best
            best_global_size = size
            
        restart_count += 1
        # Optional: If we hit a very high size, we can increase confidence
        # but for now, keep searching.

    # Return the best set found
    result_indices = np.where(best_global)[0]
    return result_indices.astype(np.int64)

# EVOLVE-BLOCK-END
