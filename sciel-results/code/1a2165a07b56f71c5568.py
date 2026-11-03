import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    n, m, p = DIMENSIONS
    
    # Build the matmul tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    def try_als(rank, max_restarts=3, max_iters=30):
        """Try to find exact rank-rank decomposition via ALS"""
        if time.time() > deadline:
            return None
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(42 + restart * 1000)
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
            
            prev_err = np.inf
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # Update U: for each i, solve A_i @ U[i,:] = T[i,:,:].flatten()
                # A_i columns are kron(V[:,r], W[:,r])
                A_U = np.zeros((m*p*p*n, rank))
                for r in range(rank):
                    A_U[:, r] = np.kron(V[:, r], W[:, r])
                AU_pinv = np.linalg.pinv(A_U)
                U_new = np.zeros((n*m, rank))
                for i in range(n*m):
                    U_new[i, :] = AU_pinv @ T[i, :, :].flatten()
                U = U_new
                
                # Update V
                A_V = np.zeros((n*m*p*n, rank))
                for r in range(rank):
                    A_V[:, r] = np.kron(U[:, r], W[:, r])
                AV_pinv = np.linalg.pinv(A_V)
                V_new = np.zeros((m*p, rank))
                for j in range(m*p):
                    V_new[j, :] = AV_pinv @ T[:, j, :].flatten()
                V = V_new
                
                # Update W
                A_W = np.zeros((n*m*m*p, rank))
                for r in range(rank):
                    A_W[:, r] = np.kron(U[:, r], V[:, r])
                AW_pinv = np.linalg.pinv(A_W)
                W_new = np.zeros((p*n, rank))
                for k in range(p*n):
                    W_new[k, :] = AW_pinv @ T[:, :, k].flatten()
                W = W_new
                
                T_rec = np.einsum('ir,jr,kr->ijk', U, V, W)
                err = np.linalg.norm(T_rec - T)
                if err < 1e-8:
                    break
                if err > prev_err * (1 + 1e-6):
                    break
                prev_err = err
            
            # Check if we found a valid solution
            if prev_err < 1e-6:
                # Round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                T_rec_r = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
                if np.linalg.norm(T_rec_r - T) < 1e-6:
                    return (U_r, V_r, W_r)
        return None
    
    # Try decreasing ranks from 41 down
    for rank in range(41, 11, -1):
        if time.time() > deadline - 10:
            break
        result = try_als(rank, max_restarts=2, max_iters=25)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: rank-42 Strassen-based construction
    if time.time() > deadline - 2:
        # Return trivial rank-48
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Rank 42: Block Strassen
    U = np.zeros((n * m, 42), dtype=float)
    V = np.zeros((m * p, 42), dtype=float)
    W = np.zeros((p * n, 42), dtype=float)
    
    # Strassen 2x2x2 terms: (U_terms, V_terms, W_terms)
    # Standard Strassen: M1=(A00+A11)(B00+B11), M2=(A10+A11)B00,
    # M3=A00(B10-B11), M4=A11(B01-B00), M5=(A00+A01)B11,
    # M6=(A10-A00)(B00+B01), M7=(A01-A10)(B10+B11)
    # C00=M1+M4-M5+M7, C01=M2+M5, C10=M1-M3+M6+M7, C11=M3+M2-M4+M6
    strassen = [
        # (u_terms, v_terms, w_terms) each term is (local_idx, coeff)
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),   # M1
        ([(2,1),(3,1)], [(0,1)],           [(2,1),(3,-1)]),  # M2
        ([(0,1)],           [(1,1),(3,-1)], [(1,1),(3,1)]),   # M3
        ([(3,1)],           [(2,1),(0,-1)], [(0,1),(2,1)]),   # M4
        ([(0,1),(1,1)],     [(3,1)],           [(0,-1),(1,1)]),  # M5
        ([(2,1),(0,-1)],    [(0,1),(1,1)],     [(3,1)]),           # M6
        ([(1,1),(3,-1)],    [(2,1),(3,1)],     [(0,1)]),           # M7
    ]
    
    # Four 2x2 blocks: (a_row_start, a_col_start, b_row_start, b_col_start)
    # A is 2x4, B is 4x6
    # Block 0: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # Block 1: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    # Block 2: A[0:2,0:2] * B[0:2,4:6] -> C[0:2,4:6]
    # Block 3: A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]  (same output cols as block 0)
    # Block 4: A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    # Block 5: A[0:2,2:4] * B[2:4,4:6] -> C[0:2,4:6]
    
    blocks = [
        (0, 0, 0, 0),  # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        (0, 0, 0, 2),  # A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
        (0, 0, 0, 4),  # A[0:2,0:2] * B[0:2,4:6] -> C[0:2,4:6]
        (0, 2, 2, 0),  # A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
        (0, 2, 2, 2),  # A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
        (0, 2, 2, 4),  # A[0:2,2:4] * B[2:4,4:6] -> C[0:2,4:6]
    ]
    
    col = 0
    for (ar, ac, br, bc) in blocks:
        # Local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        a_map = {0: (ar+0)*m+(ac+0), 1: (ar+0)*m+(ac+1), 2: (ar+1)*m+(ac+0), 3: (ar+1)*m+(ac+1)}
        b_map = {0: (br+0)*p+(bc+0), 1: (br+0)*p+(bc+1), 2: (br+1)*p+(bc+0), 3: (br+1)*p+(bc+1)}
        w_map = {0: (bc+0)*n+(ar+0), 1: (bc+1)*n+(ar+0), 2: (bc+0)*n+(ar+1), 3: (bc+1)*n+(ar+1)}
        
        for u_terms, v_terms, w_terms in strassen:
            for li, c in u_terms:
                U[a_map[li], col] += c
            for li, c in v_terms:
                V[b_map[li], col] += c
            for li, c in w_terms:
                W[w_map[li], col] += c
            col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
