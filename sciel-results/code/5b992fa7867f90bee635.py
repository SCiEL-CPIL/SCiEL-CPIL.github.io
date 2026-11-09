import numpy as np
import random

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Target tensor
    T = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
                
    # Greedy Search
    R = T.copy()
    U_list = []
    V_list = []
    W_list = []
    
    max_rank = 63
    for r in range(max_rank):
        # Power Iteration
        u_vec = np.random.rand(dim_u)
        v_vec = np.random.rand(dim_v)
        w_vec = np.random.rand(dim_w)
        
        for _ in range(50):
            u_new = np.tensordot(R, np.outer(v_vec, w_vec), axes=([1, 2], [0, 1]))
            v_new = np.tensordot(R, np.outer(u_vec, w_vec), axes=([0, 2], [0, 1]))
            w_new = np.tensordot(R, np.outer(u_vec, v_vec), axes=([0, 1], [0, 1]))
            
            u_vec = u_new / (np.linalg.norm(u_new) + 1e-10)
            v_vec = v_new / (np.linalg.norm(v_new) + 1e-10)
            w_vec = w_new / (np.linalg.norm(w_new) + 1e-10)
            
        coeff = np.tensordot(R, np.tensordot(u_vec, np.outer(v_vec, w_vec), axes=([0], [0])), axes=([0, 1, 2], [0, 1, 2]))
        
        R = R - coeff * np.tensordot(u_vec, np.outer(v_vec, w_vec), axes=([0], [0]))
        
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)
        
        if np.linalg.norm(R) < 1e-6:
            break
            
    # Round to nearest half-integer
    U = np.round(np.array(U_list).T * 2) / 2
    V = np.round(np.array(V_list).T * 2) / 2
    W = np.round(np.array(W_list).T * 2) / 2
    
    # Verify
    if verify(n, m, p, U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return schoolbook(n, m, p)

def verify(n, m, p, U, V, W):
    # Check if sum_r U[i,j,r] V[j',k',r] W[k,i,r] == delta(i,i') delta(j,j') delta(k,k')
    # We check for all i,j,j',k,k'
    # Actually, we can just check the tensor equality
    # Reconstruct T from U,V,W
    # T_recon[i, j, j', k, k', i'] = sum_r U[i*m+j, r] V[j'*p+k, r] W[k*n+i', r]
    # This is too big.
    # Instead, check C[i,k] = sum_j A[i,j] B[j,k]
    # We can use random A, B to check.
    for _ in range(10):
        A = np.random.randint(-5, 5, (n, m))
        B = np.random.randint(-5, 5, (m, p))
        C_target = A @ B
        
        C_calc = np.zeros((n, p))
        for r in range(U.shape[1]):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            # u is (n*m, 1), v is (m*p, 1), w is (p*n, 1)
            # u reshaped to (n, m)
            u_mat = u.reshape(n, m)
            v_mat = v.reshape(m, p)
            w_mat = w.reshape(p, n).T # w is indexed k*n+i, so reshape to (p, n) then T to (n, p)?
            # w index is k*n + i. So w[k*n+i] is coeff for C[i,k].
            # So w_mat[i, k] = w[k*n+i].
            # w.reshape(p, n) gives w[k, i].
            # So w_mat = w.reshape(p, n).T -> w_mat[i, k] = w[k, i].
            
            term = (A * u_mat).sum() * (B * v_mat).sum()
            C_calc += term * w_mat
            
        if not np.allclose(C_calc, C_target, atol=1e-5):
            return False
    return True

def schoolbook(n, m, p):
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
