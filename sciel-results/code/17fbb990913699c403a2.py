import numpy as np
import itertools
import time

DIMENSIONS = [2, 3, 4]

def check_decomposition(U, V, W, n=2, m=3, p=4):
    """
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    rank = U.shape[1]
    # Construct T from U, V, W
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                T[i, j, k] = s
    
    # Check if T matches the matrix multiplication tensor
    # T[i, j, k] should be 1 if it's a valid multiplication tensor?
    # The tensor for matrix multiplication is defined such that C_ik = sum_j A_ij B_jk.
    # The tensor T has entries T[i, j, k] = 1.
    # Wait, the definition of the tensor decomposition of matrix multiplication:
    # The tensor T_{ijk} is 1 for all i, j, k.
    # So we need sum_r U_{ir} V_{jr} W_{kr} = 1 for all i, j, k.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not np.isclose(T[i, j, k], 1.0):
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    # Target rank 19
    target_rank = 19
    
    # Fallback rank 22
    def get_rank_22():
        rank = 22
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Block 1: A[0:2, 0:2] * B[0:2, 0:2] (Strassen)
        # Indices: i in 0..1, j in 0..1, k in 0..1
        # Strassen terms for 2x2x2
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A10)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # This is complex to map to global indices.
        # Let's use the simple schoolbook for the first 2x2x2 block to be safe, 
        # but that's rank 8. 8+8+8=24.
        # The prompt says rank 22 is possible.
        # I will just return the schoolbook rank 24 if I can't find better quickly.
        # But I want to try rank 19.
        
        return None

    # Try to find rank 19
    start_time = time.time()
    max_time = 100.0
    
    for trial in range(1000):
        if time.time() - start_time > max_time:
            break
            
        # Random U, V with small integers
        U = np.random.randint(-1, 2, size=(n*m, target_rank))
        V = np.random.randint(-1, 2, size=(m*p, target_rank))
        
        # Solve for W
        W_valid = True
        W = np.zeros((p*n, target_rank))
        
        for k in range(p):
            # System for W[:, k*n : (k+1)*n]
            # Unknowns: w[i, r] for i in 0..n-1, r in 0..rank-1
            # Equations: sum_r U[i, r] V[j, r] w[i, r] = 1 for all i, j
            # Note: The equation for T[i, j, k] involves w[i, r] (from W[k*n+i, r])
            # So for fixed k, the unknowns are W[k*n + i, r] for i=0..n-1.
            
            # Build A_k (m x n*rank)
            A_k = np.zeros((m, n * target_rank))
            for j in range(m):
                for i in range(n):
                    for r in range(target_rank):
                        A_k[j, i * target_rank + r] = U[i*m + j, r] * V[j*p + k, r]
            
            # b_k is 1 for all j
            b_k = np.ones(m)
            
            # Solve A_k w = b_k
            # Under-determined. Use least squares to get a particular solution?
            # Or find null space.
            # Since we want integer/half-integer, we can try to find a sparse solution.
            # For now, use lstsq and check if it's close to integers.
            w_sol, residuals, rank_k, s = np.linalg.lstsq(A_k, b_k, rcond=None)
            
            # Reshape w_sol to (n, target_rank)
            w_mat = w_sol.reshape(n, target_rank)
            
            # Check if w_mat entries are close to integers or half-integers
            is_valid = True
            for i in range(n):
                for r in range(target_rank):
                    val = w_mat[i, r]
                    # Check distance to nearest 0.5
                    nearest = round(val * 2) / 2
                    if not np.isclose(val, nearest, atol=1e-3):
                        is_valid = False
                        break
                if not is_valid:
                    break
            
            if is_valid:
                # Store the rounded values
                W[k*n : (k+1)*n, :] = np.round(w_mat * 2) / 2
            else:
                W_valid = False
                break
        
        if W_valid:
            # Verify the full decomposition
            if check_decomposition(U, V, W, n, m, p):
                # Convert to lists of floats/ints
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
        
        # If not valid, try again

    # Fallback to Rank 24 (Schoolbook)
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
