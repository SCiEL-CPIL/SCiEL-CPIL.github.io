import numpy as np
import time
import random

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Precompute the target tensor T
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) with matching j
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    T = np.zeros((n * m, m * p, p * n), dtype=np.int8)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def compute_error(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # X[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # Efficient way: 
        # X = np.einsum('ar,br,cr->abc', U, V, W)
        # This might be slow for large R and large tensors.
        # n*m=15, m*p=35, p*n=21. Size is 15*35*21 = 11025.
        # R ~ 50-100.
        # einsum should be fine.
        X = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((X - T) ** 2)

    def snap_to_half_int(arr):
        # Round to nearest 0.5
        return np.round(arr * 2) / 2

    def is_valid(U, V, W):
        # Check if entries are half-integers and decomposition is exact
        if not np.allclose(U * 2, np.round(U * 2)): return False
        if not np.allclose(V * 2, np.round(V * 2)): return False
        if not np.allclose(W * 2, np.round(W * 2)): return False
        
        X = np.einsum('ar,br,cr->abc', U, V, W)
        return np.array_equal(X, T)

    # Schoolbook fallback
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # Try to find a lower rank decomposition
    # We will try ranks from a lower bound up to schoolbook
    # Lower bound for matrix multiplication rank is often around n*m*p/2?
    # Let's try from 60 up to 104.
    # Actually, let's start from 104 and go down, stopping if we find one.
    # Or start from a guess like 50 and go up?
    # Lower rank is better. So we want the MINIMUM rank.
    # We should search from low to high? No, we don't know the min.
    # If we search low to high, we might waste time on impossible low ranks.
    # If we search high to low, we find a valid one quickly but might not be minimal.
    # The score is 1/rank. We want the smallest rank.
    # So we should try to find the *smallest* possible rank.
    # However, we don't know it.
    # Strategy: Try a few low ranks (e.g., 40-60) with limited time.
    # If none found, try higher ranks.
    # If none found before deadline, return schoolbook.

    best_rank = rank_schoolbook
    best_U, best_V, best_W = get_schoolbook()

    # Heuristic: The rank of 3x5x7 is likely between 40 and 80.
    # Let's try ranks in a specific order.
    # We will use a simple local search on the discrete half-integer grid.
    
    def local_search(rank, max_iters=500):
        # Initialize with random half-integers
        # Scale: entries are usually small (0, 1, -1, 0.5, -0.5)
        U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n * m, rank))
        V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m * p, rank))
        W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p * n, rank))
        
        current_err = compute_error(U, V, W)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                return None, None, None, current_err
            
            # Pick a random matrix and a random entry to perturb
            mat_idx = random.randint(0, 2)
            if mat_idx == 0:
                mat = U
            elif mat_idx == 1:
                mat = V
            else:
                mat = W
            
            i, j = random.randint(0, mat.shape[0]-1), random.randint(0, mat.shape[1]-1)
            old_val = mat[i, j]
            
            # Propose a new value from the set
            new_val = random.choice([-1, -0.5, 0, 0.5, 1])
            if new_val == old_val:
                continue
                
            mat[i, j] = new_val
            new_err = compute_error(U, V, W)
            
            if new_err < current_err:
                current_err = new_err
            else:
                # Revert
                mat[i, j] = old_val
                
            # Early termination if perfect
            if current_err == 0:
                return U, V, W, 0

        return U, V, W, current_err

    # Try different ranks
    # We prioritize finding *any* improvement over 105.
    # Let's try ranks 90, 80, 70, 60, 50.
    # If we find a valid one at 90, we record it.
    # Then we try to beat 90 with 80, etc.
    
    ranks_to_try = [95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40]
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        # Try a few restarts for this rank
        found = False
        for restart in range(5):
            if time.time() > deadline - 5:
                break
                
            U, V, W, err = local_search(r, max_iters=200)
            if err == 0:
                if is_valid(U, V, W):
                    best_rank = r
                    best_U, best_V, best_W = U, V, W
                    found = True
                    break
        
        if found:
            # If we found a valid decomposition, we can continue to try to lower the rank
            # or stop if we are confident.
            # Let's continue to try lower ranks in the list.
            pass

    # Convert to list of lists for the required output format
    # The interface expects {'U': ..., 'V': ..., 'W': ...}
    # The values should be integer/half-integer.
    # We use float representation which is exact for .5
    
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}

# EVOLVE-BLOCK-END
