import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the true multiplication tensor in 3D form
    # T[i, j, k] = 1 if output[i,k] depends on A[i,j]*B[j,k]
    # Using trace convention: W[k*n + i, r], so W index = k*n + i
    # U index = i*m + j, V index = j*p + k
    # Tensor T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i for some i,j,k
    
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    # Try ranks from 12 up to 17 (trivial is 18)
    for rank in range(12, 18):
        best_err = float('inf')
        best_decomp = None
        
        # Multiple random restarts
        for restart in range(50):
            np.random.seed(42 + restart * 1000)
            
            # Initialize random decomposition
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Normalize to help convergence
            for r in range(rank):
                U[:, r] /= (np.linalg.norm(U[:, r]) + 1e-10)
                V[:, r] /= (np.linalg.norm(V[:, r]) + 1e-10)
                W[:, r] /= (np.linalg.norm(W[:, r]) + 1e-10)
            
            # Iterative refinement: alternate between fixing two factors and solving for the third
            converged = False
            for it in range(2000):
                # Solve for U given V, W: minimize ||T - U(V^T ⊗ W^T)||_F^2
                # T as matrix: reshape to (nm, mp*pn)
                # T_mat[i, v*pn + w] = T[i, v, w]
                # We want: T_mat ≈ U * M where M[r, v*pn+w] = V[v,r]*W[w,r]
                # This is: T_mat ≈ U @ M.T ? No...
                # T[i, v, w] = sum_r U[i,r] * V[v,r] * W[w,r]
                # For fixed V,W: T_mat[i, (v,w)] = sum_r U[i,r] * (V[v,r]*W[w,r])
                # So M[r, (v,w)] = V[v,r] * W[w,r], shape (rank, mp*pn)
                # T_mat = U @ M  =>  solve for U: U = T_mat @ M.T @ (M @ M.T)^-1
                
                M_vw = np.zeros((mp * pn, rank))
                for v in range(mp):
                    for w in range(pn):
                        for r in range(rank):
                            M_vw[v * pn + w, r] = V[v, r] * W[w, r]
                
                T_mat = T.reshape(nm, mp * pn)
                
                # Solve U: T_mat ≈ U @ M_vw.T
                # U = T_mat @ M_vw @ (M_vw.T @ M_vw)^-1
                try:
                    MvT_Mv = M_vw.T @ M_vw
                    U_new = T_mat @ M_vw @ np.linalg.pinv(MvT_Mv)
                except np.linalg.LinAlgError:
                    continue
                
                # Solve for V
                M_uw = np.zeros((nm * pn, rank))
                for u in range(nm):
                    for w in range(pn):
                        for r in range(rank):
                            M_uw[u * pn + w, r] = U_new[u, r] * W[w, r]
                T_mat2 = T.reshape(nm * pn, 1).T  # not quite right, let me redo
                
                # T[i, v, w] = sum_r U[i,r] * V[v,r] * W[w,r]
                # For fixed U,W: T_mat3[v, (i,w)] = sum_r V[v,r] * (U[i,r]*W[w,r])
                M3 = np.zeros((nm * pn, rank))
                for i in range(nm):
                    for w in range(pn):
                        for r in range(rank):
                            M3[i * pn + w, r] = U_new[i, r] * W[w, r]
                T_mat3 = T.reshape(nm, mp, pn).transpose(1, 0, 2).reshape(mp, nm * pn)
                try:
                    M3T_M3 = M3.T @ M3
                    V_new = T_mat3 @ M3 @ np.linalg.pinv(M3T_M3)
                except np.linalg.LinAlgError:
                    continue
                
                # Solve for W
                M4 = np.zeros((nm * mp, rank))
                for i in range(nm):
                    for v in range(mp):
                        for r in range(rank):
                            M4[i * mp + v, r] = U_new[i, r] * V_new[v, r]
                T_mat4 = T.reshape(nm, mp, pn).transpose(2, 0, 1).reshape(pn, nm * mp)
                try:
                    M4T_M4 = M4.T @ M4
                    W_new = T_mat4 @ M4 @ np.linalg.pinv(M4T_M4)
                except np.linalg.LinAlgError:
                    continue
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence
                err = 0.0
                for r in range(rank):
                    outer = np.einsum('ir,vr,wr->ivw', U, V, W)
                    err += np.sum((outer - T) ** 2)
                err = np.sqrt(err / (nm * mp * pn))
                
                if err < 1e-10:
                    converged = True
                    break
            
            if converged:
                best_decomp = (U.copy(), V.copy(), W.copy())
                break
        
        if best_decomp is not None:
            U, V, W = best_decomp
            # Convert to list of lists with integer/half-integer check
            U_list = [[float(U[i, r]) for r in range(rank)] for i in range(nm)]
            V_list = [[float(V[i, r]) for r in range(rank)] for i in range(mp)]
            W_list = [[float(W[i, r]) for r in range(rank)] for i in range(pn)]
            
            # Verify exactness
            max_err = 0.0
            for i in range(nm):
                for v in range(mp):
                    for w in range(pn):
                        val = sum(U_list[i][r] * V_list[v][r] * W_list[w][r] for r in range(rank))
                        target = T[i, v, w]
                        max_err = max(max_err, abs(val - target))
            
            if max_err < 1e-8:
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to trivial rank 18
    rank = nm * mp
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U[u][col] = 1
                V[v][col] = 1
                W[w][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
