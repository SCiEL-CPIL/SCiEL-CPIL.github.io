import numpy as np
import time

DIMENSIONS = [2, 2, 4]

def _construct_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n))
    # T[i, j, k] = 1 if indices match for multiplication
    # i: row of A, j: col of A / row of B, k: col of B
    # Output index: (i, k) in W? 
    # W index is k*n + i.
    # So W index w = k*n + i.
    # A index a = i*m + j? No, U shape is (n*m, rank).
    # Standard: A is n x m. Index a = i*m + j.
    # B is m x p. Index b = j*p + k.
    # C is n x p. Index c = k*n + i.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _peeling_search(T, target_rank, max_iters=1000, tol=1e-9):
    """
    Iteratively subtracts rank-1 components from T.
    Returns (U, V, W) if rank <= target_rank, else None.
    """
    R = T.copy()
    U_list = []
    V_list = []
    W_list = []
    
    for r in range(target_rank):
        # Flatten R to matrix M of shape (n*m, m*p*p*n) ? No.
        # Flatten to (n*m, (m*p)*(p*n)) to find rank-1 approx?
        # Rank-1 tensor u v w corresponds to outer product u (v w^T) in flattening?
        # T = sum u_i v_j w_k.
        # Flatten T_{i, (j,k)} -> M_{i, jk}.
        # M = U * (V * W^T)^T ?
        # If T = u v w, then M_{i, jk} = u_i v_j w_k.
        # M = u (v w^T).
        # So we want to find u, v, w such that M is close to u (v w^T).
        # This is equivalent to finding rank-1 approx of M?
        # No, M is not arbitrary. M_{i, jk} has structure.
        # But for the purpose of peeling, we can just find the best rank-1 tensor approximation.
        
        # Reshape R to (n*m, m*p*p*n)
        # Actually, we can just do SVD on the flattening R_{i, (j,k)}.
        # But the rank-1 tensor u v w flattens to u (v w^T).
        # v w^T is a matrix of size (m*p) x (p*n).
        # So M = u * (v w^T).
        # This is a rank-1 matrix M.
        # So the best rank-1 tensor approximation is given by the top singular vector of the flattening?
        # Let's check.
        # Minimize || R - u v w ||_F.
        # This is not exactly the same as minimizing || Flatten(R) - u (v w^T) ||_F.
        # Actually, it IS the same, because the Frobenius norm is preserved under flattening.
        # And u (v w^T) is a rank-1 matrix.
        # So we find the best rank-1 matrix approximation of Flatten(R).
        # Let Flatten(R) = M.
        # Best rank-1 approx is sigma_1 * x_1 y_1^T.
        # So M_approx = sigma_1 x_1 y_1^T.
        # We need to factor x_1 y_1^T into u (v w^T).
        # x_1 is vector of size n*m. So u = x_1.
        # y_1 is vector of size (m*p)*(p*n).
        # We need y_1 = v w^T.
        # This requires y_1 to be a rank-1 matrix when reshaped to (m*p) x (p*n).
        # But y_1 is just a vector. We can reshape it to Y of size (m*p) x (p*n).
        # Y is rank 1? No, Y is the matrix we got from SVD.
        # Wait. The SVD gives M = sigma x y^T.
        # y is a vector. We reshape y to Y.
        # Is Y rank 1? No.
        # So the best rank-1 matrix approximation of the flattening does NOT necessarily correspond to a rank-1 tensor.
        # A rank-1 tensor flattens to a matrix of rank 1.
        # But not all rank-1 matrices come from rank-1 tensors?
        # Yes they do. If M = u v^T, and we reshape v to Y, then M = u Y^T.
        # But Y must be of the form w v^T? No.
        # The tensor is u (v \otimes w).
        # Flattening is u (v \otimes w)^T = u (w v^T).
        # So the matrix part must be a rank-1 matrix of the form w v^T.
        # Any rank-1 matrix Z can be written as w v^T.
        # So yes, the best rank-1 matrix approximation of the flattening corresponds to a rank-1 tensor.
        
        # 1. Flatten R to M (n*m, m*p*p*n)
        M = R.reshape(n*m, -1)
        # 2. SVD
        U_svd, S, Vt_svd = np.linalg.svd(M, full_matrices=False)
        # 3. Top component
        sigma = S[0]
        u = U_svd[:, 0]
        y = Vt_svd[0, :]
        # 4. Reshape y to Y (m*p, p*n)
        Y = y.reshape(m*p, p*n)
        # 5. Factor Y = w v^T? Or v w^T?
        # We need Y = v w^T? 
        # Tensor term: u_i v_j w_k.
        # Flatten index (j, k).
        # M_{i, jk} = u_i v_j w_k.
        # So Y_{jk} = v_j w_k.
        # So Y = v w^T.
        # We can find v, w by SVD of Y.
        U_y, S_y, Vt_y = np.linalg.svd(Y, full_matrices=False)
        v = U_y[:, 0] * S_y[0]
        w = Vt_y[0, :]
        
        # Construct tensor term
        term = np.einsum('i,j,k->ijk', u, v, w)
        
        # Subtract
        R -= term
        
        # Store
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)
        
        # Check convergence
        if np.linalg.norm(R, 'fro') < tol:
            break
            
    if np.linalg.norm(R, 'fro') < tol:
        # We have a decomposition of rank <= target_rank
        # Pad with zeros if rank < target_rank
        while len(U_list) < target_rank:
            U_list.append(np.zeros(n*m))
            V_list.append(np.zeros(m*p))
            W_list.append(np.zeros(p*n))
        
        U = np.array(U_list).T
        V = np.array(V_list).T
        W = np.array(W_list).T
        
        # Verify
        # Reconstruct
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        if np.allclose(T_recon, T, atol=tol):
            return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # 1. Construct Tensor
    T = _construct_tensor()
    
    # 2. Try Peeling for Rank 12
    best_result = None
    if time.time() < deadline:
        res = _peeling_search(T, 12, max_iters=1000)
        if res:
            best_result = res
            # Convert to lists of lists for return
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # 3. Try Peeling for Rank 13
    if time.time() < deadline:
        res = _peeling_search(T, 13, max_iters=1000)
        if res:
            best_result = res
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # 4. Fallback to Rank 14 (Strassen)
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen linear forms for A (2x2)
    # Indices: 0: a00, 1: a01, 2: a10, 3: a11
    U[0][0] = 1; U[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    U[0][2] = 1
    U[3][3] = 1
    U[0][4] = 1; U[1][4] = 1
    U[2][5] = 1; U[0][5] = -1
    U[1][6] = 1; U[3][6] = -1
    
    U[0][7] = 1; U[3][7] = 1
    U[2][8] = 1; U[3][8] = 1
    U[0][9] = 1
    U[3][10] = 1
    U[0][11] = 1; U[1][11] = 1
    U[2][12] = 1; U[0][12] = -1
    U[1][13] = 1; U[3][13] = -1
    
    V[0][0] = 1; V[5][0] = 1
    V[0][1] = 1
    V[1][2] = 1; V[5][2] = -1
    V[4][3] = 1; V[0][3] = -1
    V[5][4] = 1
    V[0][5] = 1; V[1][5] = 1
    V[4][6] = 1; V[5][6] = 1
    
    V[2][7] = 1; V[7][7] = 1
    V[2][8] = 1
    V[3][9] = 1; V[7][9] = -1
    V[6][10] = 1; V[2][10] = -1
    V[7][11] = 1
    V[2][12] = 1; V[3][12] = 1
    V[6][13] = 1; V[7][13] = 1
    
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    W[1][1] = 1; W[1][3] = 1
    W[2][2] = 1; W[2][4] = 1
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    W[4][7] = 1; W[4][10] = 1; W[4][11] = -1; W[4][13] = 1
    W[5][8] = 1; W[5][10] = 1
    W[6][9] = 1; W[6][11] = 1
    W[7][7] = 1; W[7][8] = -1; W[7][9] = 1; W[7][12] = 1
    
    return {"U": U, "V": V, "W": W}

