"""
Rewritten program: search for minimal-rank exact tensor decomposition of 3x3 * 3x4 matmul.
Uses greedy/local search with integer/half-integer constraints.
"""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 4]
n, m, p = DIMENSIONS

def _verify(U, V, W):
    """
    Verify if U, V, W exactly decompose the matmul tensor.
    U: (n*m, r)
    V: (m*p, r)
    W: (p*n, r)
    Returns True if valid.
    """
    r = U.shape[1]
    if r == 0:
        return False
        
    U_3d = U.reshape(n, m, r)
    V_3d = V.reshape(m, p, r)
    W_3d = W.reshape(p, n, r)
    
    try:
        T_reconstructed = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    except Exception:
        return False
    
    # Check if T_reconstructed is exactly 1 everywhere
    # Since we are searching for integer/half-integer solutions, 
    # and the target is 1, we check for closeness to 1.
    if not np.allclose(T_reconstructed, 1.0, atol=1e-8, rtol=1e-8):
        return False
    
    # Check if entries are integer or half-integer
    # Multiply by 2 and check if close to integer
    U2 = 2 * U
    V2 = 2 * V
    W2 = 2 * W
    
    if not np.allclose(U2, np.round(U2), atol=1e-6): return False
    if not np.allclose(V2, np.round(V2), atol=1e-6): return False
    if not np.allclose(W2, np.round(W2), atol=1e-6): return False
    
    return True

def _error(U, V, W):
    """
    Calculate the squared error between the reconstructed tensor and the target (all ones).
    """
    r = U.shape[1]
    if r == 0:
        return n * m * p
        
    U_3d = U.reshape(n, m, r)
    V_3d = V.reshape(m, p, r)
    W_3d = W.reshape(p, n, r)
    
    T_rec = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    target = np.ones((n, m, p))
    return np.sum((T_rec - target) ** 2)

def _local_search(rank, max_iters=5000, time_limit=10.0, seed=None):
    """
    Local search for a valid decomposition of the given rank.
    Uses a sparse initialization and moves non-zeros around.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    start_time = time.time()
    
    # Initialize with random sparse structure
    # Each column has a few non-zeros.
    # For rank r, we want to cover all 36 terms (i,j,k) roughly.
    # Let's start with a random permutation of indices for each factor.
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Initialize non-zero positions
    # For each column c, pick random indices i, j, k
    # and set U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] to non-zero values.
    
    # To ensure we can potentially reach the all-ones tensor, 
    # we should ensure that for every (i,j,k), there is at least one column c 
    # where U[i*m+j,c], V[j*p+k,c], W[k*n+i,c] are all non-zero? 
    # Not necessarily, but it helps.
    
    # Let's just randomize non-zero values from {-2, -1, -0.5, 0.5, 1, 2}
    vals = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
    
    for c in range(rank):
        # Pick random indices
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        
        U[i*m+j, c] = random.choice(vals)
        V[j*p+k, c] = random.choice(vals)
        W[k*n+i, c] = random.choice(vals)
        
        # Maybe add a second non-zero to increase connectivity?
        if random.random() < 0.5:
            i2 = random.randint(0, n-1)
            j2 = random.randint(0, m-1)
            k2 = random.randint(0, p-1)
            U[i2*m+j2, c] = random.choice(vals)
            V[j2*p+k2, c] = random.choice(vals)
            W[k2*n+i2, c] = random.choice(vals)
            
    current_err = _error(U, V, W)
    
    # Local search loop
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        if current_err < 1e-12:
            return U, V, W, current_err
            
        # Pick a random column
        c = random.randint(0, rank - 1)
        
        # Pick a random factor
        factor = random.randint(0, 2) # 0: U, 1: V, 2: W
        
        # Pick a random row index to modify
        if factor == 0:
            size = n * m
        elif factor == 1:
            size = m * p
        else:
            size = p * n
            
        idx = random.randint(0, size - 1)
        
        # Try changing the value at this index for column c
        # Original value
        if factor == 0:
            orig_val = U[idx, c]
        elif factor == 1:
            orig_val = V[idx, c]
        else:
            orig_val = W[idx, c]
            
        # Try a new value from a small set
        # Include 0 to allow sparsification
        new_vals = [0.0, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0]
        random.shuffle(new_vals)
        
        best_new_err = current_err
        best_new_val = orig_val
        best_new_idx = idx
        
        # Try moving the non-zero to a different index?
        # Let's stick to value changes first, then try index moves if stuck.
        
        for new_val in new_vals:
            if new_val == orig_val:
                continue
                
            if factor == 0:
                U[idx, c] = new_val
            elif factor == 1:
                V[idx, c] = new_val
            else:
                W[idx, c] = new_val
                
            new_err = _error(U, V, W)
            
            if new_err < best_new_err:
                best_new_err = new_err
                best_new_val = new_val
                best_new_idx = idx
            else:
                # Revert
                if factor == 0:
                    U[idx, c] = orig_val
                elif factor == 1:
                    V[idx, c] = orig_val
                else:
                    W[idx, c] = orig_val
                    
        # Apply best move
        if best_new_err < current_err:
            if factor == 0:
                U[idx, c] = best_new_val
            elif factor == 1:
                V[idx, c] = best_new_val
            else:
                W[idx, c] = best_new_val
            current_err = best_new_err
        else:
            # If no improvement, try a random index move for a non-zero element
            # Find a non-zero element in the chosen factor
            if factor == 0:
                non_zeros = np.nonzero(U[:, c])[0]
            elif factor == 1:
                non_zeros = np.nonzero(V[:, c])[0]
            else:
                non_zeros = np.nonzero(W[:, c])[0]
                
            if len(non_zeros) > 0:
                old_idx = non_zeros[random.randint(0, len(non_zeros)-1)]
                val = (U[old_idx, c] if factor==0 else V[old_idx, c] if factor==1 else W[old_idx, c])
                
                # Set old to 0
                if factor == 0:
                    U[old_idx, c] = 0
                elif factor == 1:
                    V[old_idx, c] = 0
                else:
                    W[old_idx, c] = 0
                    
                # Set new random index to val
                new_idx = random.randint(0, size - 1)
                if factor == 0:
                    U[new_idx, c] = val
                elif factor == 1:
                    V[new_idx, c] = val
                else:
                    W[new_idx, c] = val
                    
                new_err = _error(U, V, W)
                
                if new_err > current_err:
                    # Revert
                    if factor == 0:
                        U[new_idx, c] = 0
                        U[old_idx, c] = val
                    elif factor == 1:
                        V[new_idx, c] = 0
                        V[old_idx, c] = val
                    else:
                        W[new_idx, c] = 0
                        W[old_idx, c] = val
                else:
                    current_err = new_err

    return U, V, W, current_err

def _schoolbook():
    """Returns the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # 1. Schoolbook solution (rank 36)
    U_sb, V_sb, W_sb = _schoolbook()
    best_rank = U_sb.shape[1]
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # 2. Try local search for lower ranks
    # Lower bound for 3x3*3x4 is likely around 24-28.
    # We try ranks from 35 down to 24.
    
    for rank in range(35, 23, -1):
        if time.time() > deadline:
            break
            
        # Try a few restarts for this rank
        # Limit time per rank to ensure we try multiple ranks
        time_per_rank = 5.0
        
        for restart in range(10):
            if time.time() > deadline:
                break
                
            U, V, W, err = _local_search(rank, max_iters=2000, time_limit=time_per_rank, seed=random.randint(0, 10000))
            
            if err < 1e-12:
                if _verify(U, V, W):
                    best_rank = rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W_sb.tolist()} # Wait, W is from local search
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    # If we find a valid one, we can stop searching for this rank
                    # and move to the next lower rank? 
                    # Actually, we want the MINIMAL rank. 
                    # So if we find rank 35, we should try 34 next.
                    # But if we find rank 24, we can stop? 
                    # No, we don't know if 23 is possible. 
                    # But given the time limit, finding 35 is a huge improvement over 36.
                    # Let's continue searching for lower ranks if time permits.
                    pass
            # If we found a valid solution, we update best_result
            # but we keep searching for lower ranks in the outer loop.
            
    return best_result

if __name__ == "__main__":
    res = construct()
    print("Found rank:", len(res['U'][0]))
    print(res['U'][:2])
