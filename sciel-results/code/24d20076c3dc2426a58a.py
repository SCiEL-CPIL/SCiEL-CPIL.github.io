import numpy as np
import itertools
import time
import random
from collections import deque

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin

    # 1. Generate all candidate vectors with norm^2 = 4
    # These are the "minimal" vectors that allow for dense packing.
    candidates = []
    
    # Type A: Weight 1, entry +/-2. (1, 0, ..., 0) * 2
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            candidates.append(tuple(v))
            
    # Type B: Weight 2, entries +/-1. (1, 1, 0, ...)
    for pos1, pos2 in itertools.combinations(range(D), 2):
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                v = [0] * D
                v[pos1] = s1
                v[pos2] = s2
                candidates.append(tuple(v))
                
    # Type C: Weight 4, entries +/-1. (1, 1, 1, 1, 0, ...)
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(tuple(v))
            
    n = len(candidates)
    # Convert to numpy array for fast dot product calculation
    vecs = np.array(candidates, dtype=np.int64)
    
    # 2. Compute conflict matrix
    # Conflict if dot > 2 (since norm^2=4, threshold is 4/2=2)
    dots = vecs @ vecs.T
    conflicts = (dots > 2).astype(np.int32)
    np.fill_diagonal(conflicts, 0)
    
    # Convert conflicts to bitsets for fast set operations
    # conflict_bits[i] is an integer with bit j set if i conflicts with j
    conflict_bits = [0] * n
    for i in range(n):
        bits = 0
        # Use numpy to find indices of conflicts
        nbrs = np.where(conflicts[i] == 1)[0]
        for j in nbrs:
            bits |= (1 << j)
        conflict_bits[i] = bits
        
    # Helper to count bits
    def popcount(x):
        return bin(x).count('1')
        
    # Helper to get a random set bit
    def get_random_set_bit(x, rng):
        if x == 0:
            return -1
        # Convert to list of set bits? Too slow if done often.
        # Better: random index check?
        # For n~5000, we can just pick a random index and check.
        # Or use a list of active vertices.
        pass

    # We will maintain:
    # S: bitset of current independent set
    # in_set: boolean array for quick lookup (maybe not needed if we use bitsets)
    
    # 3. Greedy Initialization
    def greedy_init(rng):
        S = 0
        # Order vertices randomly
        indices = list(range(n))
        rng.shuffle(indices)
        for i in indices:
            # Check if i conflicts with current S
            if (conflict_bits[i] & S) == 0:
                S |= (1 << i)
        return S

    # 4. Simulated Annealing / Local Search
    def local_search(initial_S, max_time):
        S = initial_S
        current_size = popcount(S)
        best_S = S
        best_size = current_size
        
        # Precompute list of all indices for random selection
        all_indices = list(range(n))
        
        T = 10.0
        T_min = 0.01
        cooling = 0.9995
        ls_deadline = time.time() + max_time
        
        while time.time() < ls_deadline and T > T_min:
            # Pick a random vertex
            v = rng.choice(all_indices)
            
            if (S >> v) & 1:
                # v is in S. Try to remove it.
                # Delta = -1
                if rng.random() < np.exp(-1.0 / T):
                    S &= ~(1 << v)
                    current_size -= 1
            else:
                # v is not in S. Try to add it (with swap if necessary)
                conflicts_in_S = conflict_bits[v] & S
                c_count = popcount(conflicts_in_S)
                
                if c_count == 0:
                    # Can add freely
                    S |= (1 << v)
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_S = S
                else:
                    # Swap: remove conflicts, add v
                    # Delta = 1 - c_count
                    delta = 1 - c_count
                    if delta > 0:
                        # Improvement
                        S = (S & ~conflicts_in_S) | (1 << v)
                        current_size += delta
                        if current_size > best_size:
                            best_size = current_size
                            best_S = S
                    else:
                        # Deterioration or neutral
                        # Accept with probability exp(delta / T)
                        if rng.random() < np.exp(delta / T):
                            S = (S & ~conflicts_in_S) | (1 << v)
                            current_size += delta
            
            T *= cooling
            
            # Periodic restart if stuck? 
            # Or just let SA handle it.
            
        return best_S, best_size

    # 5. Main Loop
    rng = np.random.default_rng(42)
    
    best_S = 0
    best_size = 0
    
    # Multiple restarts
    # Each restart: Greedy Init + Local Search
    # We have ~115s. Let's do ~50-100 restarts depending on speed.
    
    # Estimate time per restart.
    # Greedy: O(N^2) bit ops? No, O(N) checks, each check is bit op. Very fast.
    # Local Search: Iterations depend on time.
    
    n_restarts = 0
    max_restarts = 200
    
    while time.time() < deadline - 5:
        n_restarts += 1
        # Greedy init
        S_init = greedy_init(rng)
        init_size = popcount(S_init)
        
        if init_size > best_size:
            best_size = init_size
            best_S = S_init
            
        # Local search
        # Allocate remaining time / expected restarts
        remaining = deadline - 5 - time.time()
        if remaining < 1.0:
            break
            
        # Heuristic for LS time: more time for early restarts? Or equal?
        # Let's try equal time per restart, but cap it.
        # If we do 50 restarts in 100s, that's 2s per restart.
        # Let's start with 1.0s per restart and adjust if we have time.
        ls_time = min(1.5, remaining * 0.2) # Use 20% of remaining time, cap at 1.5s
        if ls_time < 0.2:
            ls_time = 0.2
            
        S_best, size_best = local_search(S_init, ls_time)
        
        if size_best > best_size:
            best_size = size_best
            best_S = S_best

    # Convert best_S to array of indices
    final_indices = []
    mask = best_S
    while mask:
        # Get lowest set bit
        lsb = mask & (-mask)
        idx = lsb.bit_length() - 1
        final_indices.append(idx)
        mask ^= lsb
        
    if not final_indices:
        return np.array([], dtype=np.int64)
        
    result_vectors = vecs[final_indices]
    return result_vectors
