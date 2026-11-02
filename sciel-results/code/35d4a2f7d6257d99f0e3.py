"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, rank):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    Returns True if valid, False otherwise.
    """
    try:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # The tensor T[i, j, k] for matrix multiplication C = A * B
        # where A is n x m, B is m x p, C is n x p.
        # The term A[i, k] * B[k, j] contributes to C[i, j].
        # The factorization is:
        # sum_r U[i, k, r] V[k, j, r] W[i, j, r] ? 
        # Let's check the seed's indexing:
        # U[i*m + j] -> A[i, j]? No, seed says U[i*m+j]. 
        # In seed:
        # U[i*m+j] = 1
        # V[j*p+k] = 1
        # W[k*n+i] = 1
        # This corresponds to the term A[i, j] * B[j, k] contributing to C[i, k]?
        # Let's check the product:
        # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
        # If c corresponds to (i, j, k), the product is 1.
        # This sums to 1 for all i, j, k.
        # So the tensor T[i, j, k] = 1.
        # This represents the bilinear map where the output is indexed by (i, k) 
        # and the input by (i, j) and (j, k).
        # Wait, the output W is indexed by k*n+i. 
        # This implies the output matrix is p x n? 
        # But C is n x p.
        # The prompt says: "The output factor follows the official trace convention W[k*n+i,r]."
        # This means the output index is (k, i) in a p x n matrix?
        # Or is it C[i, k] stored as W[k*n+i]?
        # If C is n x p, row i, col k.
        # If we store C as a vector of size n*p, index is i*p + k.
        # But the prompt says W[k*n+i]. This is index k*n + i.
        # This corresponds to a matrix of size p x n, row k, col i.
        # This is C^T? Or C with indices swapped?
        # Let's assume the tensor is defined by the factors.
        # We just need to ensure that for all i, j, k:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        
        # Construct the tensor from factors
        # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # We can compute this efficiently using einsum or loops
        # T is n x m x p
        T_computed = np.zeros((n, m, p), dtype=float)
        
        # To speed up, we can use matrix multiplication tricks
        # But for n=5, m=5, p=5, rank<=125, direct loop is fine.
        
        for c in range(rank):
            # u_c is vector of size n*m
            # v_c is vector of size m*p
            # w_c is vector of size p*n
            
            # We need to add u_c[i, j] * v_c[j, k] * w_c[k, i] to T[i, j, k]
            # This is a rank-1 tensor in the 3-way space.
            
            u_vec = U_arr[:, c]
            v_vec = V_arr[:, c]
            w_vec = W_arr[:, c]
            
            # Reshape
            u_mat = u_vec.reshape(n, m)
            v_mat = v_vec.reshape(m, p)
            w_mat = w_vec.reshape(p, n)
            
            # T[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
            # This is not a simple outer product because of the index linking.
            # We can iterate over i, j, k
            for i in range(n):
                for j in range(m):
                    u_val = u_mat[i, j]
                    if u_val == 0: continue
                    for k in range(p):
                        v_val = v_mat[j, k]
                        if v_val == 0: continue
                        w_val = w_mat[k, i]
                        if w_val == 0: continue
                        T_computed[i, j, k] += u_val * v_val * w_val
                        
        # The target tensor T_target[i, j, k] is 1 for all i, j, k
        T_target = np.ones((n, m, p), dtype=float)
        
        if not np.allclose(T_computed, T_target, atol=1e-9):
            return False
        return True
    except Exception:
        return False

def construct():
    n, m, p = DIMENSIONS
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
                
    # Optional: Verify
    # if not _verify_decomposition(U, V, W, n, m, p, rank):
    #     raise ValueError("Decomposition failed")
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
