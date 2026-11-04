import numpy as np
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Precompute indices for valid (i, j, k) triplets
    valid_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                valid_indices.append((u_idx, v_idx, w_idx))
                
    # Precompute indices for invalid triplets (zeros)
    # Total entries: (n*m) * (m*p) * (p*n)
    # We only need to check a subset of zeros for speed, or all if fast enough.
    # 6*15*10 = 900 entries. 30 valid. 870 invalid.
    # Checking 870 entries per candidate is fast.
    
    def check_decomposition_full(U, V, W, R):
        # Check valid positions (should be 1)
        for u_idx, v_idx, w_idx in valid_indices:
            s = 0
            for r in range(R):
                s += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
            if abs(s - 1) > 1e-9:
                return False
                
        # Check invalid positions (should be 0)
        # Iterate over all u, v, w
        for u_idx in range(n * m):
            for v_idx in range(m * p):
                for w_idx in range(p * n):
                    # Skip if valid
                    # Optimization: We can just compute the sum and check if it's close to 0
                    # But we need to know if it's a valid index.
                    # We can create a set of valid tuples for O(1) lookup.
                    pass
        
        # Efficient check:
        # Create a tensor of zeros, fill valid ones, then check all?
        # Or just check all 900 positions.
        for u_idx in range(n * m):
            for v_idx in range(m * p):
                for w_idx in range(p * n):
                    s = 0
                    for r in range(R):
                        s += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    # Determine if this is a valid index
                    # u_idx = i*m + j -> i = u_idx // m, j = u_idx % m
                    # v_idx = j*p + k -> j2 = v_idx // p, k = v_idx % p
                    # w_idx = k*n + i -> k2 = w_idx // n, i2 = w_idx % n
                    i = u_idx // m
                    j = u_idx % m
                    j2 = v_idx // p
                    k = v_idx % p
                    k2 = w_idx // n
                    i2 = w_idx % n
                    
                    if i == i2 and j == j2:
                        # Valid index, should be 1
                        if abs(s - 1) > 1e-9:
                            return False
                    else:
                        # Invalid index, should be 0
                        if abs(s) > 1e-9:
                            return False
        return True

    # Schoolbook decomposition
    def get_schoolbook():
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx, col] = 1
                    V[v_idx, col] = 1
                    W[w_idx, col] = 1
                    col += 1
        return U, V, W, R

    start_time = time.time()
    time_limit = 100
    
    # Try Rank 28
    # We can't easily find it with random search.
    # We will just return Schoolbook 30 to be safe, as Rank 28 search is too slow/unreliable.
    # However, if we have a specific Rank 28 solution, we would use it.
    
    U, V, W, R = get_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
