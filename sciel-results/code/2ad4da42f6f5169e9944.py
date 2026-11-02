import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor indices
    # U: (n*m, rank) -> index i*m+j
    # V: (m*p, rank) -> index j*p+k
    # W: (p*n, rank) -> index k*n+i  (Note: W[k*n+i, r] in prompt, but standard is often i*p+k for C. 
    # Prompt says: W[k*n+i, r]. Let's stick to prompt convention for W index.
    # C index in prompt: W[k*n+i, r]. This implies C is stored as k*n+i? 
    # Usually C is i*p+k. 
    # "output factor follows the official trace convention W[k*n+i,r]"
    # This means the output index is $k \cdot n + i$.
    # So $C_{k,i}$? Or $C_{i,k}$ stored in $k \cdot n + i$?
    # If $C$ is $n \times p$, standard flatten is $i \cdot p + k$.
    # If $W$ index is $k \cdot n + i$, it's column-major for $C$ if $C$ is $n \times p$?
    # Let's assume the tensor entry $T_{\alpha, \beta, \gamma}$ corresponds to:
    # $\alpha = i \cdot m + j$
    # $\beta = j \cdot p + k$
    # $\gamma = k \cdot n + i$
    
    # Let's verify this convention.
    # If $\gamma = k \cdot n + i$, then for a fixed $k$, $i$ varies from 0 to n-1.
    # This is consistent with column-major storage of an $n \times p$ matrix (cols are $k$).
    
    # Construct Target Tensor
    # Size: (n*m) x (m*p) x (p*n)
    # n=2, m=5, p=6
    # U_dim = 10, V_dim = 30, W_dim = 12
    
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    
    # Target tensor T[U, V, W]
    # T is 1 if the term contributes.
    # Term A[i,j] * B[j,k] contributes to C[i,k].
    # U_idx = i*m + j
    # V_idx = j*p + k
    # W_idx = k*n + i
    
    target = np.zeros((U_dim, V_dim, W_dim), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target[u_idx, v_idx, w_idx] = 1

    # Schoolbook Decomposition
    rank_schoolbook = n * m * p
    U_s = np.zeros((U_dim, rank_schoolbook), dtype=int)
    V_s = np.zeros((V_dim, rank_schoolbook), dtype=int)
    W_s = np.zeros((W_dim, rank_schoolbook), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U_s[u_idx, col] = 1
                V_s[v_idx, col] = 1
                W_s[w_idx, col] = 1
                col += 1
                
    # Verify Schoolbook
    # T_reconstructed = sum_r U[:, r] * V[:, r] * W[:, r] (outer product)
    # This is a tensor sum.
    # We can check by iterating over non-zeros of target.
    
    def verify(U, V, W):
        # Check if sum_r U[:,r] V[:,r] W[:,r] == target
        # Efficient check:
        # For each (u, v, w) in target non-zeros, check if sum_r U[u,r] V[v,r] W[w,r] == 1
        # And for zero entries, check if sum is 0.
        # Since target is sparse (60 ones), we can just check those.
        # But we must ensure no other entries are non-zero.
        # We can compute the full tensor if it's small.
        # 10 x 30 x 12 = 3600 entries. Very small.
        T_rec = np.zeros((U_dim, V_dim, W_dim), dtype=int)
        for r in range(U.shape[1]):
            T_rec += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
        return np.allclose(T_rec, target)

    if verify(U_s, V_s, W_s):
        best_U, best_V, best_W = U_s, V_s, W_s
        best_rank = rank_schoolbook
    else:
        # Should not happen
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    # Try to find lower rank
    # Use time limit
    start_time = time.time()
    time_limit = 110 # Leave 10s margin
    
    # Try ranks from 30 to 59
    # Use random search with integer coefficients -2..2
    import random
    
    found = False
    for rank in range(30, best_rank):
        if time.time() - start_time > time_limit:
            break
            
        attempts = 0
        while time.time() - start_time < time_limit and attempts < 50:
            attempts += 1
            U = np.zeros((U_dim, rank), dtype=int)
            V = np.zeros((V_dim, rank), dtype=int)
            W = np.zeros((W_dim, rank), dtype=int)
            
            # Random initialization
            for r in range(rank):
                # Sparse random vectors
                # Most entries are 0
                for i in range(U_dim):
                    if random.random() < 0.1:
                        U[i, r] = random.randint(-2, 2)
                for j in range(V_dim):
                    if random.random() < 0.1:
                        V[j, r] = random.randint(-2, 2)
                for k in range(W_dim):
                    if random.random() < 0.1:
                        W[k, r] = random.randint(-2, 2)
                        
            if verify(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = rank
                found = True
                break # Try next lower rank? Or stop?
                # We want minimal rank, so break and try lower?
                # The loop goes 30 to 59. So if we find 40, we break and try 39?
                # No, the loop is `for rank in range(30, best_rank)`.
                # If best_rank is 60, it tries 30, 31...
                # If we find 40, we set best_rank=40 and break inner loop.
                # Then outer loop continues? No, `break` exits the `while`.
                # The `for` loop continues to next rank?
                # Wait, `for rank in range(30, best_rank)`.
                # If best_rank changes to 40, the range is fixed at start?
                # No, range is evaluated once.
                # So it will try 30, 31, ..., 59.
                # If we find 40, we should ideally stop or continue to find lower?
                # We want minimal. So we should try lower ranks.
                # But 30 is lower than 40.
                # So if we find 40, we should continue to try 39, 38...
                # But the loop starts at 30.
                # So if we find 40, we have already tried 30-39.
                # So we can break the for loop if we find a rank?
                # No, we want the MINIMAL.
                # So we should iterate from low to high.
                # If we find a solution at rank R, we can stop?
                # No, there might be a solution at R-1.
                # But we started at 30.
                # So if we find 40, we know 30-39 failed.
                # So 40 is the best so far.
                # We can stop the search if we are confident.
                # But to be safe, we can just return the first one found if it's < 60.
                # Or we can try a few more.
                pass

    # If we found a better one, return it.
    if found or best_rank < rank_schoolbook:
         # Convert to lists
         return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
         return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
