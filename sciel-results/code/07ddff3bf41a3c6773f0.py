import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    DEADLINE = 130
    
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 coefficients
    U_coeffs = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (0, -1)],
        [(1, 1), (3, -1)],
    ]
    V_coeffs = [
        [(0, 1), (3, 1)],
        [(0, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (0, -1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (3, 1)],
    ]
    W_coeffs = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, -1)],
        [(1, 1), (3, 1)],
        [(0, 1), (2, 1)],
        [(0, -1), (1, 1)],
        [(3, 1)],
        [(0, 1)],
    ]
    
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for a_block in range(2):
        for b_col_block in range(3):
            for s in range(7):
                for (idx, coeff) in U_coeffs[s]:
                    i_loc, j_loc = divmod(idx, 2)
                    j_global = a_block * 2 + j_loc
                    u_row = i_loc * m + j_global
                    U[u_row, col] += coeff
                
                for (idx, coeff) in V_coeffs[s]:
                    j_loc, k_loc = divmod(idx, 2)
                    j_global = a_block * 2 + j_loc
                    k_global = b_col_block * 2 + k_loc
                    v_row = j_global * p + k_global
                    V[v_row, col] += coeff
                
                for (idx, coeff) in W_coeffs[s]:
                    k_loc, i_loc = divmod(idx, 2)
                    k_global = b_col_block * 2 + k_loc
                    w_row = k_global * n + i_loc
                    W[w_row, col] += coeff
                
                col += 1
    
    # Fast verification using NumPy
    # Tensor T is sum of e_ij \otimes e_jk \otimes e_ki
    # We check if U @ V.T @ W.T reconstructs the tensor?
    # No, the decomposition is T = sum_r U_r \otimes V_r \otimes W_r
    # This is equivalent to checking if the 3-way array matches.
    # We can flatten and check.
    
    # Construct the target tensor T (flattened)
    # T[i, j, k] = 1 if indices match?
    # The tensor is defined by the multiplication structure.
    # T_flat has size n*m*p? No, it's a 3-mode tensor.
    # We can check by evaluating the contraction.
    # For random A, B, check if C = A B matches the contraction of the decomposition.
    
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B
    
    # Contraction: C[k, i] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Note: W index is k*n+i.
    # U index is i*m+j.
    # V index is j*p+k.
    
    # We can do this efficiently.
    # C_decomposed = zeros(p, n)
    # for r in range(rank):
    #    U_r = U[:, r].reshape(n, m)
    #    V_r = V[:, r].reshape(m, p)
    #    W_r = W[:, r].reshape(p, n)
    #    # The term is sum_{i,j,k} U_r[i,j] V_r[j,k] W_r[k,i] e_{ik}
    #    # This is effectively trace(W_r.T (U_r V_r))
    #    # Wait, the tensor decomposition gives C = sum_r (U_r V_r) \otimes W_r?
    #    # No, C_{ik} = sum_r U_r[i,:] \cdot V_r[:,k] * W_r[k,i]?
    #    # Let's just loop r and add to C_decomposed.
    
    C_decomposed = np.zeros((p, n))
    for r in range(rank):
        u_vec = U[:, r]
        v_vec = V[:, r]
        w_vec = W[:, r]
        
        # u_vec is size n*m. Reshape to (n, m).
        u_mat = u_vec.reshape(n, m)
        # v_vec is size m*p. Reshape to (m, p).
        v_mat = v_vec.reshape(m, p)
        # w_vec is size p*n. Reshape to (p, n).
        w_mat = w_vec.reshape(p, n)
        
        # The contribution to C[k, i] is sum_j u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
        # This is sum_j (u_mat[i, j] v_mat[j, k]) w_mat[k, i]
        # Let M = u_mat @ v_mat. (n, p)
        # Then C_decomposed[k, i] += M[i, k] * w_mat[k, i]
        M = u_mat @ v_mat
        # M is (n, p). w_mat is (p, n).
        # We need to sum over k? No, for each (k, i), add M[i, k] * w_mat[k, i].
        # This is element-wise product of M.T (p, n) and w_mat (p, n).
        C_decomposed += M.T * w_mat
        
    if not np.allclose(C_decomposed, C_expected.T):
        # Fallback to schoolbook
        rank_schoolbook = n * m * p
        U = np.zeros((n * m, rank_schoolbook))
        V = np.zeros((m * p, rank_schoolbook))
        W = np.zeros((p * n, rank_schoolbook))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
