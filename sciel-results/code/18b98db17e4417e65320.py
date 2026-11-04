import numpy as np
import random
import time

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank to search for, starting from a plausible lower bound
    # Schoolbook is 48.
    # We try to find a decomposition with rank < 48.
    
    # The tensor T is of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    # But the interface requires U (n*m, R), V (m*p, R), W (p*n, R)
    # such that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 if (i,j,k) valid, 0 else.
    # Wait, the tensor is defined on the indices.
    # The "multiplication tensor" has 1s at (i*m+j, j*p+k, k*n+i) for i in 0..n-1, j in 0..m-1, k in 0..p-1.
    # Note: The third index in the tensor definition in the prompt is W[k*n+i, r].
    # The standard tensor for matrix mult M_{n,m,p} has indices (i, j, k) mapped to (i*m+j, j*p+k, k*n+i).
    
    def get_tensor():
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        return T

    T = get_tensor()
    
    # We search for rank R from 40 to 47.
    # If we find a valid one, we return it.
    # To save time, we only do a limited number of iterations.
    
    best_rank = n * m * p
    best_factors = None
    
    # Pre-compute indices for faster access
    # We will use a local search on the factors U, V, W.
    # Variables are constrained to half-integers.
    
    def error(U, V, W):
        # Calculate tensor reconstruction
        # T_recon[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
        # Here i is index in U (0..n*m-1), j in V (0..m*p-1), k in W (0..p*n-1)
        # Efficiently:
        # U (n*m, R), V (m*p, R), W (p*n, R)
        # T_recon is (n*m, m*p, p*n)
        # This is a 3-way tensor product.
        # For small R, we can do:
        # Temp = U @ V.T  -> (n*m, m*p) for each r? No.
        # T_recon = sum_r np.einsum('ir,jr,kr->ijk', U, V, W)
        # This might be slow in Python loop.
        # Let's use einsum.
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.sum((T - T_recon)**2)

    def random_init(R):
        # Initialize with small random values, then round to half-integers
        # Or start with schoolbook and prune?
        # Random init is better for escaping local minima.
        scale = 1.0
        U = np.random.uniform(-scale, scale, (n*m, R))
        V = np.random.uniform(-scale, scale, (m*p, R))
        W = np.random.uniform(-scale, scale, (p*n, R))
        # Round to nearest half-integer
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        return U, V, W

    def local_search(U, V, W, max_iters=500):
        U = U.copy()
        V = V.copy()
        W = W.copy()
        curr_err = error(U, V, W)
        R = U.shape[1]
        
        # Candidate values for updates
        # We only change one element at a time.
        # To speed up, we can pre-compute contributions.
        # But for R~40, dims ~12-16, it's small.
        
        for it in range(max_iters):
            # Pick a random matrix and a random element
            mat_idx = random.randint(0, 2)
            if mat_idx == 0:
                mat = U
                other1 = V
                other2 = W
                dim = n*m
            elif mat_idx == 1:
                mat = V
                other1 = U
                other2 = W
                dim = m*p
            else:
                mat = W
                other1 = U
                other2 = V
                dim = p*n
            
            row = random.randint(0, dim - 1)
            col = random.randint(0, R - 1)
            
            # Try changing mat[row, col] to nearby half-integers
            curr_val = mat[row, col]
            # Try +/- 0.5, +/- 1.0, +/- 1.5
            deltas = [-1.5, -1.0, -0.5, 0.5, 1.0, 1.5]
            best_delta = 0
            best_local_err = curr_err
            
            for d in deltas:
                new_val = curr_val + d
                # Optional: bound values to avoid explosion
                if new_val < -5 or new_val > 5:
                    continue
                mat[row, col] = new_val
                new_err = error(mat, other1 if mat_idx != 1 else V, other2 if mat_idx != 2 else W)
                # Note: if mat_idx==1, other1 is U, other2 is W. 
                # If mat_idx==0, other1 is V, other2 is W.
                # The error function expects U, V, W order.
                if mat_idx == 0:
                    e = error(mat, V, W)
                elif mat_idx == 1:
                    e = error(U, mat, W)
                else:
                    e = error(U, V, mat)
                
                if e < best_local_err:
                    best_local_err = e
                    best_delta = d
            
            if best_delta != 0:
                curr_err = best_local_err
                # The update was already applied in the loop if it was the best? 
                # No, we need to apply the best_delta found.
                # The loop left mat at the last tried value.
                # Reset to curr_val + best_delta
                mat[row, col] = curr_val + best_delta
            else:
                # Reset to original value if no improvement
                mat[row, col] = curr_val
            
            if curr_err < 1e-6:
                return U, V, W, curr_err
                
        return U, V, W, curr_err

    # Search ranks
    start_rank = 40
    end_rank = 47
    
    for R in range(start_rank, end_rank + 1):
        # Try a few random restarts
        found = False
        for restart in range(10):
            U, V, W = random_init(R)
            U, V, W, err = local_search(U, V, W, max_iters=200)
            if err < 1e-6:
                # Verify exactness with integers/half-integers
                # The local search maintains half-integers.
                # Check if it's truly exact
                T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
                if np.allclose(T, T_recon, atol=1e-9):
                    # Convert to lists for return
                    # Ensure they are exactly representable (0, 0.5, 1.0 etc)
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
                found = True # Found a close one, maybe refine?
        
        if found:
            # If we found a close one but not exact, we might need more iterations
            # But for the sake of the 120s limit, we break and try next rank or return best so far?
            # Actually, if we find a valid one, we return immediately.
            pass

    # If no better rank found, return schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
