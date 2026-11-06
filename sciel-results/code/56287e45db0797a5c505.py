import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 5, 5, 5

    # Block sizes for partitioning 5 = 2 + 2 + 1
    block_sizes = [2, 2, 1]

    # Block offsets
    offsets = [0]
    for s in block_sizes:
        offsets.append(offsets[-1] + s)

    # Strassen decomposition for 2x2x2 (verified correct)
    def get_strassen_factors():
        """Return list of (U_vec, V_vec, W_vec) for Strassen 2x2x2.
        U_vec has length 4: indices [0,1,2,3] = [a11,a12,a21,a22]
        V_vec has length 4: indices [0,1,2,3] = [b11,b12,b21,b22]
        W_vec has length 4: indices [0,1,2,3] = [c11,c12,c21,c22]
        """
        factors = []
        # M1 = (a11+a22)(b11+b22) -> c11(+1), c22(+1)
        factors.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        # M2 = (a21+a22)b11 -> c21(+1), c22(-1)
        factors.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
        # M3 = a11(b12-b22) -> c12(+1), c22(+1)
        factors.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
        # M4 = a22(b21-b11) -> c11(+1), c21(+1)
        factors.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
        # M5 = (a11+a12)b22 -> c11(-1), c12(+1)
        factors.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
        # M6 = (a21-a11)(b11+b12) -> c22(+1)
        factors.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
        # M7 = (a12-a22)(b21+b22) -> c11(+1)
        factors.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
        
        return factors

    strassen_factors = get_strassen_factors()
    
    # Verify Strassen factors are correct
    def verify_strassen():
        A = np.array([[1,2],[3,4]], dtype=float)
        B = np.array([[5,6],[7,8]], dtype=float)
        C = A @ B
        # U_vec: [a11,a12,a21,a22], V_vec: [b11,b12,b21,b22], W_vec: [c11,c12,c21,c22]
        a_vec = [A[0,0], A[0,1], A[1,0], A[1,1]]
        b_vec = [B[0,0], B[0,1], B[1,0], B[1,1]]
        C_recon = np.zeros((2,2))
        for uf, vf, wf in strassen_factors:
            m_val = sum(uf[i]*a_vec[i] for i in range(4)) * sum(vf[i]*b_vec[i] for i in range(4))
            for idx in range(4):
                C_recon[idx//2, idx%2] += wf[idx] * m_val
        return np.allclose(C_recon, C, atol=1e-10)
    
    if not verify_strassen():
        # Fallback: use schoolbook for all blocks if Strassen is broken
        strassen_factors = None
    
    # Total rank calculation
    total_rank = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                if strassen_factors is not None and ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk

    # Allocate matrices
    U = [[0.0] * total_rank for _ in range(n * m)]  # 25 x total_rank
    V = [[0.0] * total_rank for _ in range(m * p)]  # 25 x total_rank
    W = [[0.0] * total_rank for _ in range(p * n)]  # 25 x total_rank
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if time.time() > deadline:
                    break
                
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                
                if strassen_factors is not None and ni == 2 and mj == 2 and pk == 2:
                    # Use Strassen
                    for (uf, vf, wf) in strassen_factors:
                        local_to_global_U = [
                            (offsets[i] + 0) * m + (offsets[j] + 0),
                            (offsets[i] + 0) * m + (offsets[j] + 1),
                            (offsets[i] + 1) * m + (offsets[j] + 0),
                            (offsets[i] + 1) * m + (offsets[j] + 1),
                        ]
                        local_to_global_V = [
                            (offsets[j] + 0) * p + (offsets[k] + 0),
                            (offsets[j] + 0) * p + (offsets[k] + 1),
                            (offsets[j] + 1) * p + (offsets[k] + 0),
                            (offsets[j] + 1) * p + (offsets[k] + 1),
                        ]
                        local_to_global_W = [
                            (offsets[k] + 0) * n + (offsets[i] + 0),
                            (offsets[k] + 1) * n + (offsets[i] + 0),
                            (offsets[k] + 0) * n + (offsets[i] + 1),
                            (offsets[k] + 1) * n + (offsets[i] + 1),
                        ]
                        
                        for idx in range(4):
                            if uf[idx] != 0:
                                U[local_to_global_U[idx]][col] += uf[idx]
                            if vf[idx] != 0:
                                V[local_to_global_V[idx]][col] += vf[idx]
                            if wf[idx] != 0:
                                W[local_to_global_W[idx]][col] += wf[idx]
                        col += 1
                else:
                    # Schoolbook for this block
                    for li in range(ni):
                        for lj in range(mj):
                            for lk in range(pk):
                                U_idx = (offsets[i] + li) * m + (offsets[j] + lj)
                                V_idx = (offsets[j] + lj) * p + (offsets[k] + lk)
                                W_idx = (offsets[k] + lk) * n + (offsets[i] + li)
                                U[U_idx][col] = 1.0
                                V[V_idx][col] = 1.0
                                W[W_idx][col] = 1.0
                                col += 1
    
    # Final verification using numpy: check T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k
    if time.time() < deadline - 5.0:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Build index arrays
        ii = np.repeat(np.arange(n), m*p)
        jj = np.tile(np.repeat(np.arange(m), p), n)
        kk = np.tile(np.arange(p), n*m)
        
        U_idx = ii * m + jj
        V_idx = jj * p + kk
        W_idx = kk * n + ii
        
        # Compute all T entries: sum over r
        # T[i,j,k] = U[U_idx, :] @ (V[V_idx, :] * W[W_idx, :]).T
        # This is element-wise product of V and W rows, then dot with U row
        prod_VW = V_arr[V_idx] * W_arr[W_idx]  # shape (n*m*p, rank)
        T_vals = U_arr[U_idx] * prod_VW  # shape (n*m*p, rank)
        T_vals = T_vals.sum(axis=1)  # shape (n*m*p,)
        
        max_err = np.max(np.abs(T_vals - 1.0))
        
        if max_err > 1e-9:
            # Decomposition is wrong, fall back to schoolbook
            U_fb = [[0.0] * (n*m*p) for _ in range(n*m)]
            V_fb = [[0.0] * (n*m*p) for _ in range(m*p)]
            W_fb = [[0.0] * (n*m*p) for _ in range(p*n)]
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i*m+j][idx] = 1.0
                        V_fb[j*p+k][idx] = 1.0
                        W_fb[k*n+i][idx] = 1.0
                        idx += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
