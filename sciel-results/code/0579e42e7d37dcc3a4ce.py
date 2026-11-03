import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m   # 8
    mp = m * p   # 28
    pn = p * n   # 14
    
    # The tensor T has entries T[i, j, k] = 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}
    # In the decomposition: T = sum_r U[:, r] \otimes V[:, r] \otimes W[:, r]
    # We want to find U, V, W of rank r such that this holds.
    
    # We will use the linear algebraic approach.
    # Fix V to be a random matrix of size (mp, r).
    # Then the equation becomes:
    # T = sum_r U[:, r] \otimes (V[:, r] \otimes W[:, r])
    # This is a linear system in the unknowns U and W.
    # Specifically, let X_r = V[:, r].
    # We need to find U and W such that the tensor sum matches.
    
    # A more efficient way:
    # Flatten the tensor T into a matrix M of size (nm, mp * pn) or similar?
    # The tensor equation is:
    # sum_r U[i, r] * V[j, r] * W[k, r] = T[i, j, k]
    
    # Let's fix V.
    # For a fixed V, we can solve for U and W.
    # This is still bilinear in U and W.
    # However, we can iterate:
    # 1. Initialize V randomly.
    # 2. Solve for U, W given V. This can be done by flattening.
    #    Let M be a matrix of size (nm, mp * pn) where M[i, j * pn + k] = T[i, j, k].
    #    We want M = sum_r U[:, r] * (V[:, r] \otimes W[:, r])^T ? No.
    #    
    #    Let's reshape.
    #    For each r, the term is U[:, r] \otimes V[:, r] \otimes W[:, r].
    #    This contributes to M[i, idx(j, k)] the value U[i, r] * V[j, r] * W[k, r].
    #    
    #    Let's define a matrix A_r of size (nm, mp * pn) such that:
    #    A_r[i, j * pn + k] = U[i, r] * V[j, r] * W[k, r]
    #    Then M = sum_r A_r.
    #    
    #    This is not linear in U and W simultaneously.
    #    
    #    Alternative: Fix W as well? No.
    #    
    #    Standard Strassen algorithm:
    #    1. Choose V randomly.
    #    2. For each i, the equation is:
    #       sum_r U[i, r] * (V[:, r] \otimes W[:, r]) = T[i, :, :]
    #    3. This is a linear system for U and W? No, W is unknown.
    #    
    #    Let's try a different grouping.
    #    Group by j.
    #    T[:, j, :] = sum_r U[:, r] * V[j, r] * W[:, r]^T
    #    Let B_j be the matrix T[:, j, :] of size (n, p).
    #    B_j = sum_r V[j, r] * (U[:, r] \otimes W[:, r]^T)
    #    
    #    Let Y_r = U[:, r] \otimes W[:, r]^T. Y_r is (n, p).
    #    B_j = sum_r V[j, r] Y_r
    #    
    #    This implies that the matrix B_j lies in the span of the matrices Y_r.
    #    Let Y be a matrix of size (n*p, r) formed by stacking Y_r.
    #    Then B_j (flattened) = Y * V[j, :]^T.
    #    
    #    So, for all j, the flattened B_j must be in the row space of Y? No, column space.
    #    Flattened B_j is a vector of length n*p.
    #    Y is (n*p, r). V[j, :] is (r,).
    #    So Flattened(B_j) = Y * V[j, :].
    #    
    #    This means that the set of vectors {Flattened(B_j)} for j=0..m-1 must lie in the column space of Y.
    #    Let B_mat be the matrix of size (n*p, m) where column j is Flattened(B_j).
    #    Then B_mat = Y * V^T.
    #    
    #    So, if we fix V, we need to find Y such that B_mat = Y * V^T.
    #    This is a least squares problem if V^T is not square/invertible, but V is (m, r).
    #    If r >= m, V^T is (r, m). We want Y (n*p, r) such that Y * V^T = B_mat.
    #    This is solvable if the rows of V^T span the required space.
    #    Actually, Y = B_mat * (V^T)^+ + Z * (I - (V^T)^+ V^T) ...
    #    Since we want an exact solution, we need B_mat to be in the column space of Y.
    #    Wait, Y is unknown.
    #    
    #    Let's reverse it.
    #    B_mat = Y * V^T.
    #    If we fix V, we can compute the minimum rank Y that satisfies this?
    #    No, Y must be of the form sum_r U[:, r] \otimes W[:, r]^T.
    #    This is a constraint on Y.
    #    
    #    This approach is getting complicated. Let's use the standard "fix two, solve one" iteration.
    #    
    #    Algorithm:
    #    1. Initialize U, V, W randomly.
    #    2. Iterate:
    #       a. Fix V, W. Solve for U.
    #          For each i: sum_r U[i, r] * (V[:, r] \otimes W[:, r]) = T[i, :, :]
    #          This is a linear system for U[i, :].
    #          Let M_i be the matrix (mp*pn, r) where M_i[idx(j, k), r] = V[j, r] * W[k, r].
    #          Then M_i * U[i, :] = T[i, :, :] (flattened).
    #          Solve M_i * U[i, :] = T_flat[i].
    #       b. Fix U, W. Solve for V.
    #          For each j: sum_r V[j, r] * (U[:, r] \otimes W[:, r]) = T[:, j, :]
    #          Similar linear system.
    #       c. Fix U, V. Solve for W.
    #          For each k: sum_r W[k, r] * (U[:, r] \otimes V[:, r]) = T[:, :, k]
    #          Similar linear system.
    #    3. Check if the decomposition is exact.
    #    
    #    We need to handle the linear systems carefully.
    #    If the matrix M_i is rank-deficient, we use least squares.
    #    If the residual is non-zero, the decomposition fails for this rank.
    
    def flatten_tensor(T, n, m, p):
        # T is (n, m, p)
        # Return (n, m*p)
        return T.reshape(n, m * p)
        
    def try_rank(r, n, m, p, max_iters=20, seed=0):
        np.random.seed(seed)
        
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        # Precompute T
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = 1.0
        
        T_flat_n = T.reshape(n, m * p) # (n, mp)
        T_flat_m = T.reshape(m, n * p) # (m, np) -- note: T[i,j,k] -> index i*p+k? No.
        # Let's be precise with indices.
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        # System for U:
        # For each i:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i, j, k]
        # Let's group by i.
        # For fixed i, we have variables U[i*m+j, r] for j=0..m-1, r=0..r-1.
        # This is not a simple linear system for U because V and W depend on r.
        
        # Let's use the matrix form derived earlier:
        # B_mat (n*p, m) = Y (n*p, r) * V^T (r, m)
        # Where Y[:, r] = U[:, r] \otimes W[:, r] (flattened to n*p)
        # And B_mat[:, j] = T[:, j, :] (flattened to n*p)
        
        # Let's verify B_mat construction.
        # B_mat[idx(i, k), j] should be T[i, j, k].
        # idx(i, k) = k*n + i? Or i*p + k?
        # W index is k*n + i. So let's use idx_W = k*n + i.
        # Then B_mat[idx_W, j] = T[i, j, k].
        
        B_mat = np.zeros((n * p, m))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_w = k * n + i
                    B_mat[idx_w, j] = T[i, j, k]
                    
        # Now, Y * V^T = B_mat
        # If we fix V, we can solve for Y.
        # Y = B_mat * pinv(V^T)
        # Then we need to check if Y can be decomposed into sum_r U[:, r] \otimes W[:, r].
        # This is a tensor decomposition of Y (which is a matrix) into rank-1 matrices?
        # No, Y is (n*p, r). We want Y[:, r] = U[:, r] \otimes W[:, r].
        # This means the columns of Y must be in the span of Kronecker products.
        # Actually, we just need to find U and W such that for each r,
        # Y[:, r] = U[:, r] \otimes W[:, r].
        # This is not always possible for arbitrary Y.
        
        # Better strategy:
        # Iterate:
        # 1. Fix V, W. Solve for U.
        #    For each i, j:
        #    T[i, j, :] (vector of length p) = sum_r U[i*m+j, r] * (V[j*p+0..p-1, r] * W[0..p-1 * n + i, r])
        #    This is messy.
        
        # Let's stick to the B_mat = Y * V^T idea.
        # If we fix V, we get Y.
        # Then we need to decompose Y into Kronecker products.
        # This is equivalent to finding U and W such that Y = sum_r U[:, r] W[:, r]^T? No.
        # Y[:, r] is a vector. U[:, r] is a vector of length nm. W[:, r] is a vector of length pn.
        # The Kronecker product U[:, r] \otimes W[:, r] has length nm * pn.
        # But Y[:, r] has length n*p.
        # This implies that the Kronecker product structure must be compatible.
        
        # Let's look at the indices again.
        # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        
        # Let's fix V and W.
        # For a fixed i, the equation for all j, k is:
        # T[i, j, k] = sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r])
        
        # Let's define a matrix M_i of size (m*p, r).
        # M_i[idx(j, k), r] = V[j*p+k, r] * W[k*n+i, r]
        # Then T[i, :, :] (flattened to m*p) = M_i * U[i*m: i*m+m, :]
        
        # So, for each i:
        # 1. Construct M_i (m*p, r).
        # 2. Solve M_i * U_i = T_flat[i] (least squares).
        # 3. Set U[i*m: i*m+m, :] = U_i.
        
        # Similarly, fix U and W. Solve for V.
        # For each j:
        # T[:, j, :] (flattened to n*p) = N_j * V_j
        # N_j[idx(i, k), r] = U[i*m+j, r] * W[k*n+i, r]
        # V_j is V[j*p: j*p+p, :]
        
        # Similarly, fix U and V. Solve for W.
        # For each k:
        # T[:, :, k] (flattened to n*m) = P_k * W_k
        # P_k[idx(i, j), r] = U[i*m+j, r] * V[j*p+k, r]
        # W_k is W[k*n: k*n+n, :]
        
        # This is a valid alternating minimization / linear least squares approach.
        
        best_res = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for it in range(max_iters):
            # Solve for U
            for i in range(n):
                M_i = np.zeros((m * p, r))
                for j in range(m):
                    for k in range(p):
                        idx_jk = j * p + k
                        idx_w = k * n + i
                        M_i[idx_jk, :] = V[idx_jk, :] * W[idx_w, :]
                T_flat_i = T[i, :, :].reshape(m * p)
                U_i, _, _, _ = np.linalg.lstsq(M_i, T_flat_i, rcond=None)
                U[i * m: i * m + m, :] = U_i
            
            # Solve for V
            for j in range(m):
                N_j = np.zeros((n * p, r))
                for i in range(n):
                    for k in range(p):
                        idx_ik = k * n + i # Wait, N_j index should correspond to T[:, j, :] flattening
                        # Let's flatten T[:, j, :] as idx(i, k) = k*n + i?
                        # T[i, j, k]. If we flatten to (n*p), let's use idx = k*n + i.
                        idx_ik = k * n + i
                        idx_u = i * m + j
                        idx_w = k * n + i
                        N_j[idx_ik, :] = U[idx_u, :] * W[idx_w, :]
                T_flat_j = T[:, j, :].reshape(n * p) # Note: reshape order
                # T[:, j, :] is (n, p). reshape(n*p) is row-major: T[0,0], T[0,1]...
                # My idx_ik = k*n + i is column-major?
                # T[i, k]. If I do T[:, j, :].reshape(n*p), element (i, k) is at i*p + k.
                # So I must match the indexing.
                # Let's use idx_ik = i * p + k.
                
                # Redo N_j construction with idx_ik = i * p + k
                N_j = np.zeros((n * p, r))
                for i in range(n):
                    for k in range(p):
                        idx_ik = i * p + k
                        idx_u = i * m + j
                        idx_w = k * n + i
                        N_j[idx_ik, :] = U[idx_u, :] * W[idx_w, :]
                T_flat_j = T[:, j, :].reshape(n * p)
                V_j, _, _, _ = np.linalg.lstsq(N_j, T_flat_j, rcond=None)
                V[j * p: j * p + p, :] = V_j
            
            # Solve for W
            for k in range(p):
                P_k = np.zeros((n * m, r))
                for i in range(n):
                    for j in range(m):
                        idx_ij = i * m + j
                        idx_u = i * m + j
                        idx_v = j * p + k
                        P_k[idx_ij, :] = U[idx_u, :] * V[idx_v, :]
                T_flat_k = T[:, :, k].reshape(n * m)
                W_k, _, _, _ = np.linalg.lstsq(P_k, T_flat_k, rcond=None)
                W[k * n: k * n + n, :] = W_k
            
            # Calculate residual
            res = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                        res += (val - 1.0) ** 2
            
            if res < best_res:
                best_res = res
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
            if best_res < 1e-10:
                break
        
        if best_res < 1e-10:
            return best_U, best_V, best_W, best_res
        return None, None, None, best_res

    # Try ranks from 40 upwards
    # Schoolbook is 56. Current is 46.
    # Lower bound: max(n*m, m*p, p*n) = max(8, 28, 14) = 28.
    # Also, rank >= (n*m*p) / max(n, m, p)? No.
    # Let's try 40, 41, 42...
    
    deadline = time.time() + 140 # Safety margin
    
    for r in range(40, 47):
        if time.time() > deadline:
            break
        for seed in range(5):
            if time.time() > deadline:
                break
            U, V, W, res = try_rank(r, n, m, p, max_iters=30, seed=seed)
            if U is not None:
                # Round to half-integers? The problem allows integer/half-integer.
                # The search finds floating point. We can try to round.
                # But if it's exact, it should be close to integers.
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                
                # Verify rounded version
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_round[i * m + j, :], V_round[j * p + k, :] * W_round[k * n + i, :])
                            if abs(val - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if valid:
                    return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
                
                # If rounded fails, return the float version? 
                # "integer/half-integer entries" is required.
                # If we can't round, we might need to adjust.
                # But for now, if it's exact float, it might be exact rational.
                
    # Fallback to the known rank 46 construction from the current program
    # (Re-implementing the correct block construction)
    
    U_2x2x2 = np.zeros((4, 7), dtype=int)
    V_2x2x2 = np.zeros((4, 7), dtype=int)
    W_2x2x2 = np.zeros((4, 7), dtype=int)
    
    # Strassen 2x2x2
    U_2x2x2[0, 0] = 1; U_2x2x2[3, 0] = 1
    V_2x2x2[1, 0] = 1; V_2x2x2[2, 0] = 1
    W_2x2x2[0, 0] = 1; W_2x2x2[3, 0] = 1
    
    U_2x2x2[2, 1] = 1; U_2x2x2[3, 1] = 1
    V_2x2x2[0, 1] = 1
    W_2x2x2[2, 1] = 1; W_2x2x2[3, 1] = -1
    
    U_2x2x2[0, 2] = 1
    V_2x2x2[0, 2] = 1; V_2x2x2[2, 2] = -1
    W_2x2x2[1, 2] = 1; W_2x2x2[3, 2] = 1
    
    U_2x2x2[3, 3] = 1
    V_2x2x2[3, 3] = 1; V_2x2x2[1, 3] = -1
    W_2x2x2[0, 3] = 1; W_2x2x2[2, 3] = 1
    
    U_2x2x2[0, 4] = 1; U_2x2x2[1, 4] = 1
    V_2x2x2[3, 4] = 1
    W_2x2x2[0, 4] = -1; W_2x2x2[1, 4] = 1
    
    U_2x2x2[2, 5] = 1; U_2x2x2[0, 5] = -1
    V_2x2x2[0, 5] = 1; V_2x2x2[1, 5] = 1
    W_2x2x2[3, 5] = 1
    
    U_2x2x2[1, 6] = 1; U_2x2x2[3, 6] = -1
    V_2x2x2[2, 6] = 1; V_2x2x2[3, 6] = 1
    W_2x2x2[0, 6] = 1
    
    U_2x2x1 = np.zeros((4, 2), dtype=int)
    V_2x2x1 = np.zeros((2, 2), dtype=int)
    W_2x2x1 = np.zeros((2, 2), dtype=int)
    for j in range(2):
        for i in range(2):
            U_2x2x1[i * 2 + j, j] = 1
        V_2x2x1[j, j] = 1
        for i in range(2):
            W_2x2x1[i, j] = 1
            
    total_rank = 46
    U = np.zeros((nm, total_rank), dtype=int)
    V = np.zeros((mp, total_rank), dtype=int)
    W = np.zeros((pn, total_rank), dtype=int)
    
    col = 0
    for block in range(2):
        for sub in range(3):
            for r in range(7):
                for i in range(2):
                    for j in range(2):
                        U[i * 4 + (2 * block + j), col] += U_2x2x2[i * 2 + j, r]
                for j in range(2):
                    for k in range(2):
                        V[(2 * block + j) * 7 + (sub * 2 + k), col] += V_2x2x2[j * 2 + k, r]
                for k in range(2):
                    for i in range(2):
                        W[(sub * 2 + k) * 2 + i, col] += W_2x2x2[k * 2 + i, r]
                col += 1
        for r in range(2):
            for i in range(2):
                for j in range(2):
                    U[i * 4 + (2 * block + j), col] += U_2x2x1[i * 2 + j, r]
            for j in range(2):
                V[(2 * block + j) * 7 + 6, col] += V_2x2x1[j, r]
            for i in range(2):
                W[6 * 2 + i, col] += W_2x2x1[i, r]
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
