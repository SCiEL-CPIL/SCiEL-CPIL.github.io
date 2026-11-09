import numpy as np
import time

def construct():
    n, m, p = 2, 4, 5
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    def verify(U, V, W, atol=1e-9):
        """Verify tensor decomposition exactly."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        T = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            u = U_arr[:, r].reshape(n, m)
            v = V_arr[:, r].reshape(m, p)
            w = W_arr[:, r].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i*m+j, j*p+k, k*n+i] += u[i,j] * v[j,k] * w[k,i]
        
        Texp = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Texp[i*m+j, j*p+k, k*n+i] = 1.0
        
        return np.allclose(T, Texp, atol=atol)

    def build_rank_36():
        """4 Strassen 2x2x2 blocks + 8 schoolbook for col 4"""
        strassen_U = [[1,0,0,1],[0,0,1,1],[1,0,0,0],[0,0,0,1],[1,1,0,0],[-1,0,1,0],[0,1,0,-1]]
        strassen_V = [[1,0,0,1],[1,0,0,0],[0,1,0,-1],[-1,0,1,0],[0,0,0,1],[1,1,0,0],[0,0,1,1]]
        strassen_W = [[1,0,0,1],[0,1,0,-1],[0,0,1,1],[1,1,0,0],[-1,0,1,0],[0,0,0,1],[1,0,0,0]]
        
        rank = 4*7 + 8
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        col = 0
        for j0 in (0, 2):
            for k0 in (0, 2):
                for r in range(7):
                    for li in range(2):
                        for lj in range(2):
                            U[li*m + (j0+lj)][col] = strassen_U[r][li*2+lj]
                    for lj in range(2):
                        for lk in range(2):
                            V[(j0+lj)*p + (k0+lk)][col] = strassen_V[r][lj*2+lk]
                    for lk in range(2):
                        for li in range(2):
                            W[(k0+lk)*n + li][col] = strassen_W[r][lk*2+li]
                    col += 1
        
        for i in range(n):
            for j in range(m):
                U[i*m+j][col] = 1
                V[j*p+4][col] = 1
                W[4*n+i][col] = 1
                col += 1
        
        assert col == rank
        return U, V, W

    def try_reduce_rank(U, V, W, target_rank, budget_s=20.0):
        """
        Try to find a decomposition of target_rank by optimization.
        Uses the structure of the matrix multiplication tensor.
        """
        if time.time() - start_time > deadline - 10:
            return None
        
        U0 = np.array(U, dtype=float)
        V0 = np.array(V, dtype=float)
        W0 = np.array(W, dtype=float)
        
        # Initialize with first target_rank columns + perturbation
        np.random.seed(42)
        U_opt = U0[:, :target_rank].copy()
        V_opt = V0[:, :target_rank].copy()
        W_opt = W0[:, :target_rank].copy()
        
        # Add small random perturbation to escape local minima
        U_opt += 0.01 * np.random.randn(n*m, target_rank)
        V_opt += 0.01 * np.random.randn(m*p, target_rank)
        W_opt += 0.01 * np.random.randn(p*n, target_rank)
        
        # Build target tensor
        Texp = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Texp[i*m+j, j*p+k, k*n+i] = 1.0
        
        def compute_error(U_o, V_o, W_o):
            T = np.zeros((n*m, m*p, p*n))
            for r in range(target_rank):
                u = U_o[:, r].reshape(n, m)
                v = V_o[:, r].reshape(m, p)
                w = W_o[:, r].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T[i*m+j, j*p+k, k*n+i] += u[i,j] * v[j,k] * w[k,i]
            return np.max(np.abs(T - Texp))
        
        # Simple iterative refinement: try to zero out error terms
        # Use a Gauss-Seidel-like approach
        best_err = compute_error(U_opt, V_opt, W_opt)
        
        for iteration in range(200):
            if time.time() - start_time > deadline - 5:
                break
            
            err = compute_error(U_opt, V_opt, W_opt)
            if err < 1e-8:
                return (U_opt.tolist(), V_opt.tolist(), W_opt.tolist())
            
            # Try small adjustments to each factor
            # For each term r, adjust U, V, W to reduce error
            for r in range(target_rank):
                if time.time() - start_time > deadline - 5:
                    break
                
                # Compute contribution of term r
                u = U_opt[:, r].reshape(n, m)
                v = V_opt[:, r].reshape(m, p)
                w = W_opt[:, r].reshape(p, n)
                
                # Try small perturbations
                for delta in [0.01, -0.01, 0.005, -0.005]:
                    U_trial = U_opt.copy()
                    U_trial[:, r] += delta * np.random.randn(n*m)
                    err_trial = compute_error(U_trial, V_opt, W_opt)
                    if err_trial < best_err:
                        best_err = err_trial
                        U_opt = U_trial
                    del U_trial
                    
                    V_trial = V_opt.copy()
                    V_trial[:, r] += delta * np.random.randn(m*p)
                    err_trial = compute_error(U_opt, V_trial, W_opt)
                    if err_trial < best_err:
                        best_err = err_trial
                        V_opt = V_trial
                    del V_trial
                    
                    W_trial = W_opt.copy()
                    W_trial[:, r] += delta * np.random.randn(p*n)
                    err_trial = compute_error(U_opt, V_opt, W_trial)
                    if err_trial < best_err:
                        best_err = err_trial
                        W_opt = W_trial
                    del W_trial
        
        final_err = compute_error(U_opt, V_opt, W_opt)
        if final_err < 1e-8:
            return (U_opt.tolist(), V_opt.tolist(), W_opt.tolist())
        return None

    def build_rank_32_attempt():
        """
        Try to construct rank-32 decomposition using 2 blocks of M_{2,2,5} with rank 16 each.
        M_{2,2,5} rank 16 construction:
        Use 2 Strassen blocks (rank 7 each) for the first 4 columns,
        and 2 additional terms to handle the 5th column with sharing.
        
        Actually, the known rank-16 construction for M_{2,2,5} uses:
        - The 7 Strassen terms for M_{2,2,2}
        - 9 additional terms for the remaining 3 columns, but with sharing
        Total: 7 + 9 = 16
        
        Let me try a specific construction.
        """
        if time.time() - start_time > deadline - 15:
            return None
        
        # For M_{2,2,5}, use the following approach:
        # Split B into B1 (cols 0-1) and B2 (cols 2-4)
        # Use Strassen for B1 (rank 7)
        # For B2, use a rank-9 decomposition
        # Total: 7 + 9 = 16 per block, 32 total
        
        # Actually, let's try a different split:
        # Use 4 Strassen terms (for 2x2x2) and handle the rest differently
        
        # Known rank-16 for M_{2,2,5}:
        # This is non-trivial. Let me use a numerical approach instead.
        
        # Try to find rank-16 for M_{2,2,5} by optimization
        n2, m2, p2 = 2, 2, 5
        target_rank = 16
        
        # Build M_{2,2,5} tensor
        Texp2 = np.zeros((n2*m2, m2*p2, p2*n2))
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    Texp2[i*m2+j, j*p2+k, k*n2+i] = 1.0
        
        # Initialize with schoolbook (rank 20) and try to reduce
        rank_school = n2 * m2 * p2
        U_s = np.zeros((n2*m2, rank_school))
        V_s = np.zeros((m2*p2, rank_school))
        W_s = np.zeros((p2*n2, rank_school))
        
        col = 0
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    U_s[i*m2+j, col] = 1
                    V_s[j*p2+k, col] = 1
                    W_s[k*n2+i, col] = 1
                    col += 1
        
        # Try to find rank-16 by optimization
        np.random.seed(123)
        U_opt = U_s[:, :target_rank].copy()
        V_opt = V_s[:, :target_rank].copy()
        W_opt = W_s[:, :target_rank].copy()
        
        U_opt += 0.01 * np.random.randn(n2*m2, target_rank)
        V_opt += 0.01 * np.random.randn(m2*p2, target_rank)
        W_opt += 0.01 * np.random.randn(p2*n2, target_rank)
        
        def compute_error_2x2x5(U_o, V_o, W_o):
            T = np.zeros((n2*m2, m2*p2, p2*n2))
            for r in range(target_rank):
                u = U_o[:, r].reshape(n2, m2)
                v = V_o[:, r].reshape(m2, p2)
                w = W_o[:, r].reshape(p2, n2)
                for i in range(n2):
                    for j in range(m2):
                        for k in range(p2):
                            T[i*m2+j, j*p2+k, k*n2+i] += u[i,j] * v[j,k] * w[k,i]
            return np.max(np.abs(T - Texp2))
        
        best_err = compute_error_2x2x5(U_opt, V_opt, W_opt)
        
        for iteration in range(100):
            if time.time() - start_time > deadline - 10:
                break
            
            err = compute_error_2x2x5(U_opt, V_opt, W_opt)
            if err < 1e-8:
                break
            
            # Try small adjustments
            for r in range(target_rank):
                if time.time() - start_time > deadline - 10:
                    break
                
                for delta in [0.01, -0.01]:
                    U_trial = U_opt.copy()
                    U_trial[:, r] += delta * np.random.randn(n2*m2)
                    err_trial = compute_error_2x2x5(U_trial, V_opt, W_opt)
                    if err_trial < best_err:
                        best_err = err_trial
                        U_opt = U_trial
                    
                    V_trial = V_opt.copy()
                    V_trial[:, r] += delta * np.random.randn(m2*p2)
                    err_trial = compute_error_2x2x5(U_opt, V_trial, W_opt)
                    if err_trial < best_err:
                        best_err = err_trial
                        V_opt = V_trial
                    
                    W_trial = W_opt.copy()
                    W_trial[:, r] += delta * np.random.randn(p2*n2)
                    err_trial = compute_error_2x2x5(U_opt, V_opt, W_trial)
                    if err_trial < best_err:
                        best_err = err_trial
                        W_opt = W_trial
        
        if compute_error_2x2x5(U_opt, V_opt, W_opt) < 1e-8:
            # Found rank-16 for M_{2,2,5}! Now build the full M_{2,4,5}
            # Use two copies for the two j-blocks
            U_full = np.zeros((n*m, 2*target_rank))
            V_full = np.zeros((m*p, 2*target_rank))
            W_full = np.zeros((p*n, 2*target_rank))
            
            for block in range(2):
                j0 = block * 2
                for r in range(target_rank):
                    # U: 2x2 block at rows 0-1, cols j0-j0+1
                    u_block = U_opt[:, r].reshape(n2, m2)
                    for li in range(n2):
                        for lj in range(m2):
                            U_full[li*m + (j0+lj), block*target_rank + r] = u_block[li, lj]
                    
                    # V: 2x5 block at rows j0-j0+1, cols 0-4
                    v_block = V_opt[:, r].reshape(m2, p2)
                    for lj in range(m2):
                        for lk in range(p2):
                            V_full[(j0+lj)*p + lk, block*target_rank + r] = v_block[lj, lk]
                    
                    # W: 5x2 block at rows 0-4, cols 0-1
                    w_block = W_opt[:, r].reshape(p2, n2)
                    for lk in range(p2):
                        for li in range(n2):
                            W_full[lk*n + li, block*target_rank + r] = w_block[lk, li]
            
            U_list = U_full.tolist()
            V_list = V_full.tolist()
            W_list = W_full.tolist()
            
            if verify(U_list, V_list, W_list):
                return U_list, V_list, W_list
        
        return None

    # Build the rank-36 baseline
    U, V, W = build_rank_36()
    
    if not verify(U, V, W):
        # Fallback to schoolbook
        rank = n*m*p
        U_s = [[0]*rank for _ in range(n*m)]
        V_s = [[0]*rank for _ in range(m*p)]
        W_s = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j][col] = 1
                    V_s[j*p+k][col] = 1
                    W_s[k*n+i][col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    best_U, best_V, best_W = U, V, W
    best_rank = len(U[0])
    
    # Try to reduce to rank 35
    if time.time() - start_time < deadline - 20:
        result_35 = try_reduce_rank(U, V, W, 35, budget_s=20.0)
        if result_35 is not None:
            U35, V35, W35 = result_35
            if verify(U35, V35, W35):
                best_U, best_V, best_W = U35, V35, W35
                best_rank = 35
    
    # Try to reduce to rank 32
    if best_rank > 32 and time.time() - start_time < deadline - 20:
        result_32 = build_rank_32_attempt()
        if result_32 is not None:
            U32, V32, W32 = result_32
            if verify(U32, V32, W32):
                best_U, best_V, best_W = U32, V32, W32
                best_rank = 32
    
    return {"U": best_U, "V": best_V, "W": best_W}
