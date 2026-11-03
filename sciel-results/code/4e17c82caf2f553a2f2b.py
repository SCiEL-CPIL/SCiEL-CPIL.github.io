import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Target rank is 33
    # We will attempt to find a decomposition of rank 33.
    # If not found within time limit, we return a valid decomposition of higher rank.
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0
    
    # The tensor T is defined by T[i, j, k] = 1 if i,j,k match the multiplication pattern
    # Actually, the tensor for matrix multiplication A(3x5) * B(5x6) = C(3x6)
    # T[i, j, k] = 1 if the term A[i,j] * B[j,k] contributes to C[i,k]
    # In the tensor decomposition U, V, W:
    # C[i,k] = sum_r U[i, r] * V[j, r] * W[k, r] is NOT correct.
    # The standard form is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Decomposition: C[i,k] = sum_r U[i, r] V[r, k] ... No.
    # The task interface:
    # U: (n*m, rank) -> U[i*m+j, r]
    # V: (m*p, rank) -> V[j*p+k, r]
    # W: (p*n, rank) -> W[k*n+i, r]
    # The decomposition is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,i} delta_{j,j} delta_{k,k} ?
    # No, the tensor T_{i,j,k} for matrix multiplication is 1 if the indices are consistent?
    # The multiplication tensor T has entries T_{i,j,k} = 1 if i,j,k are the indices for A[i,j], B[j,k], C[i,k].
    # So T[i,j,k] = 1 for all i,j,k in the range? No.
    # The tensor is defined such that <T, A \otimes B> = C.
    # T_{i,j,k} = 1 if the product A_{i,j} B_{j,k} contributes to C_{i,k}.
    # In the standard basis, T_{i,j,k} = 1 for all i,j,k?
    # No, the indices are (i, j, k) where i is row of A/C, j is col of A/row of B, k is col of B/C.
    # So T[i, j, k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
    # The decomposition is:
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's verify the indices.
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # We want to find U, V, W such that for all i, j, k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of n*m*p = 3*5*6 = 90 equations.
    # Unknowns: rank * (n*m + m*p + p*n) = rank * (15 + 30 + 18) = rank * 63.
    # For rank 33, unknowns = 33 * 63 = 2079.
    # Equations = 90.
    # This is an underdetermined system.
    
    # We can use a local search to find integer solutions.
    
    def check_solution(U, V, W):
        # Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
        # U: 15 x R
        # V: 30 x R
        # W: 18 x R
        # We can compute the tensor contraction.
        # T_calc[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Vectorized check:
        # Reshape U to (n, m, R)
        # Reshape V to (m, p, R)
        # Reshape W to (p, n, R)
        
        U_reshaped = np.array(U).reshape(n, m, -1)
        V_reshaped = np.array(V).reshape(m, p, -1)
        W_reshaped = np.array(W).reshape(p, n, -1)
        
        # Sum over R
        # T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
        # Note: W index is k*n + i, so W_reshaped[k, i, r]
        
        # We can compute this using einsum
        # T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        T_calc = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        # Target T is all ones
        error = np.sum((T_calc - 1.0)**2)
        return error, T_calc

    def random_init(rank):
        U = np.random.randint(-2, 3, size=(n*m, rank))
        V = np.random.randint(-2, 3, size=(m*p, rank))
        W = np.random.randint(-2, 3, size=(p*n, rank))
        return U, V, W

    # Try rank 33
    best_rank = n * m * p
    best_solution = None
    
    # Initialize with schoolbook
    U_sb = np.zeros((n*m, best_rank))
    V_sb = np.zeros((m*p, best_rank))
    W_sb = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_solution = (U_sb, V_sb, W_sb)
    
    # Search for lower rank
    for rank in range(33, best_rank):
        if time.time() > deadline:
            break
            
        # Try a few random restarts for this rank
        for restart in range(10):
            if time.time() > deadline:
                break
                
            U, V, W = random_init(rank)
            error, _ = check_solution(U, V, W)
            
            # Local search
            for step in range(1000):
                if time.time() > deadline:
                    break
                    
                if error == 0:
                    best_rank = rank
                    best_solution = (U, V, W)
                    break
                    
                # Pick a random variable
                mat = np.random.choice(['U', 'V', 'W'])
                if mat == 'U':
                    idx = np.random.randint(n*m)
                    col = np.random.randint(rank)
                    # Perturb
                    old_val = U[idx, col]
                    U[idx, col] += np.random.choice([-1, 1])
                    new_error, _ = check_solution(U, V, W)
                    if new_error < error:
                        error = new_error
                    else:
                        U[idx, col] = old_val
                        
                elif mat == 'V':
                    idx = np.random.randint(m*p)
                    col = np.random.randint(rank)
                    old_val = V[idx, col]
                    V[idx, col] += np.random.choice([-1, 1])
                    new_error, _ = check_solution(U, V, W)
                    if new_error < error:
                        error = new_error
                    else:
                        V[idx, col] = old_val
                        
                else:
                    idx = np.random.randint(p*n)
                    col = np.random.randint(rank)
                    old_val = W[idx, col]
                    W[idx, col] += np.random.choice([-1, 1])
                    new_error, _ = check_solution(U, V, W)
                    if new_error < error:
                        error = new_error
                    else:
                        W[idx, col] = old_val
            
            if error == 0:
                break

    U, V, W = best_solution
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-START
# The code above is inside construct()
# EVOLVE-BLOCK-END
