import itertools
import numpy as np
import time

D = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate all weight 4 vectors in R^11
    # There are C(11,4) * 2^4 = 330 * 16 = 5280 such vectors.
    indices = list(itertools.combinations(range(D), 4))
    signs = list(itertools.product([1, -1], repeat=4))
    
    vectors_list = []
    for idx in indices:
        for s in signs:
            v = np.zeros(D, dtype=np.int64)
            for i, val in zip(idx, s):
                v[i] = val
            vectors_list.append(v)
    
    A = np.array(vectors_list)
    N = len(A)
    
    # The condition for two distinct weight 4 vectors u, v to be compatible is:
    # ||u - v||^2 >= 4
    # (u-v).(u-v) = u.u + v.v - 2u.v = 4 + 4 - 2u.v = 8 - 2u.v >= 4
    # 4 >= 2u.v => u.v <= 2
    
    # We want to find a large subset of A such that for all distinct u, v in subset, u.v <= 2.
    
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin for 150s limit
    
    best_size = 0
    best_indices = np.array([], dtype=int)
    
    # We can try to find a set of size 512.
    # If we find one, we can stop.
    
    # Greedy algorithm with random shuffling
    # To make it faster, we can pre-compute the dot product matrix?
    # N=5280. N^2 = 27.8M.
    # Memory for int8 or int16? Dot products range -4 to 4.
    # int8 is enough. 27.8M bytes ~ 28MB.
    # This allows O(1) lookup for compatibility.
    
    # Let's compute the Gram matrix G
    # G[i, j] = A[i] . A[j]
    # We only need to know if G[i, j] <= 2.
    # We can store a boolean matrix Compat where Compat[i, j] = (G[i, j] <= 2)
    
    # However, computing G takes N^2 * D operations.
    # 5280^2 * 11 ~ 300M ops.
    # With NumPy, A @ A.T is fast.
    
    G = A @ A.T
    # G is int64.
    # Create boolean mask for compatibility (excluding self)
    # We want to select indices such that for all i != j in S, G[i, j] <= 2.
    
    # Since G is symmetric and G[i, i] = 4, we can just check G[i, j] <= 2.
    
    # To speed up the greedy step, we can maintain a bitset of compatible candidates?
    # Or just use the matrix.
    
    # Greedy with restarts
    for restart in range(500):
        if time.time() > deadline:
            break
            
        # Shuffle
        order = np.random.permutation(N)
        selected = []
        
        # Optimization: Instead of checking against all selected,
        # we can maintain a list of "forbidden" indices?
        # No, the condition is pairwise.
        
        # Simple loop
        for idx in order:
            if time.time() > deadline:
                break
            
            if len(selected) == 0:
                selected.append(idx)
                continue
            
            # Check if idx is compatible with all in selected
            # G[idx, selected] should all be <= 2
            # Note: G[idx, idx] is 4, but idx is not in selected yet.
            # If idx is in selected, we skip (but it's not, because we pick from order and don't replace).
            # Wait, if we pick idx, it's added to selected.
            # We check against existing selected.
            
            # Vectorized check
            # G[idx, selected]
            # If any is > 2, reject.
            
            # To avoid overhead of indexing G[idx, selected] if selected is large,
            # we can just do it.
            
            # However, Python loop over 'order' is slow.
            # We can try to vectorize the selection?
            # Hard to do greedy vectorized.
            
            # Let's check time.
            # 5280 iterations.
            # Inside: G[idx, selected]
            # If selected size is 500, this is 500 lookups.
            # 5280 * 500 = 2.6M lookups.
            # This is very fast in NumPy if we slice?
            # G[idx, :] is a view. G[idx, selected] is a gather.
            
            # Let's try.
            
            # One optimization: if best_size reaches 512, break.
            
            dots = G[idx, selected]
            if np.all(dots <= 2):
                selected.append(idx)
                
                if len(selected) > best_size:
                    best_size = len(selected)
                    best_indices = np.array(selected)
                    
                    if best_size >= 512:
                        break
        
        if best_size >= 512:
            break

    if best_size == 0:
        # Fallback
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)

    return A[best_indices]
# EVOLVE-BLOCK-END
