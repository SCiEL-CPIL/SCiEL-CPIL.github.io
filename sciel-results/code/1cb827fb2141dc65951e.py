"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 140  # 140s budget, leave 10s margin
    
    n, m, p = DIMENSIONS
    import numpy as np
    
    # Build the target tensor T[a,b,c] where a in [0,4), b in [0,16), c in [0,16)
    # T[2i+j, 8j+k, 2k+i] = 1 for i in {0,1}, j in {0,1}, k in {0..7}
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Helper: verify a decomposition (U, V, W) against T
    def verify(U, V, W, tol=1e-9):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        # Compute reconstructed tensor
        # T_recon[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Use einsum for efficiency
        T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        max_err = np.max(np.abs(T_recon - T))
        return max_err < tol, max_err
    
    # Helper: ALS step - fix V,W and solve for U
    def als_step_U(V, W, T, max_iter=100, tol=1e-12):
        """Given V (16xr) and W (16xr), solve for U (4xr) such that
        sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
        For fixed a: sum_r U[a,r] * (V*W)[b,c] = T[a,b,c]
        where (V*W)[b,c] = sum_r V[b,r]*W[c,r]
        """
        r = V.shape[1]
        # Compute VW = V @ W^T (16x16)
        VW = V @ W.T
        U = np.zeros((4, r))
        for a in range(4):
            # We need U[a,:] such that for all b,c: sum_r U[a,r]*VW[b,r]*W[c,r]... 
            # Actually, T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed a, this is a linear system in U[a,:]:
            # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
            # Let's flatten: for each (b,c), we have an equation
            # This is a linear system: M_a * U[a,:] = T[a,:,:].flatten()
            # where M_a is (256 x r) with M_a[(b,c), r] = V[b,r]*W[c,r]
            M = np.zeros((16*16, r))
            for b in range(16):
                for c in range(16):
                    M[b*16+c, :] = V[:, r] * W[c, r]  # This is wrong, let me fix
            # Actually M[(b,c), r] = V[b,r] * W[c,r]
            M = np.zeros((16*16, r))
            for b in range(16):
                for c in range(16):
                    M[b*16+c, :] = V[b, :] * W[c, :]
            rhs = T[a, :, :].flatten()
            # Solve least squares
            U[a, :], residuals, rank_M, sv = np.linalg.lstsq(M, rhs, rcond=None)
        return U
    
    # Simpler ALS: use the fact that for fixed V,W, the problem for each a is linear
    def als_step_U_simple(V, W, T):
        """Solve for U given V, W"""
        r = V.shape[1]
        U = np.zeros((4, r))
        for a in range(4):
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed a: let X = V @ W^T (16x16), then... no that's not right
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # = <U[a,:], V[b,:] * W[c,:]> elementwise dot
            # So for each (b,c): sum_r U[a,r] * (V[b,r]*W[c,r]) = T[a,b,c]
            # This is: (V * W^T)[b,c] dotted with U[a,:] = T[a,b,c]
            # Where (V*W^T)[b,c] = V[b,:] * W[c,:] (elementwise)
            # So we have a linear system: M @ U[a,:] = T[a,:,:].flatten()
            # M is 256 x r, M[idx, r'] = V[b, r'] * W[c, r'] where idx = b*16+c
            M = np.zeros((256, r))
            for b in range(16):
                for c in range(16):
                    M[b*16+c, :] = V[b, :] * W[c, :]
            rhs = T[a, :, :].flatten()
            U[a, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
        return U
    
    def als_step_V(U, W, T):
        """Solve for V given U, W"""
        r = W.shape[1]
        V = np.zeros((16, r))
        for b in range(16):
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed b: sum_r V[b,r] * (U[a,r]*W[c,r]) = T[a,b,c]
            # M[idx, r'] = U[a, r'] * W[c, r'] where idx = a*16+c
            M = np.zeros((64, r))
            for a in range(4):
                for c in range(16):
                    M[a*16+c, :] = U[a, :] * W[c, :]
            rhs = T[:, b, :].flatten()
            V[b, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
        return V
    
    def als_step_W(U, V, T):
        """Solve for W given U, V"""
        r = V.shape[1]
        W = np.zeros((16, r))
        for c in range(16):
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed c: sum_r W[c,r] * (U[a,r]*V[b,r]) = T[a,b,c]
            # M[idx, r'] = U[a, r'] * V[b, r'] where idx = a*16+b
            M = np.zeros((64, r))
            for a in range(4):
                for b in range(16):
                    M[a*16+b, :] = U[a, :] * V[b, :]
            rhs = T[:, :, c].flatten()
            W[c, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
        return W
    
    def als_decompose(T, rank, max_iter=200, tol=1e-12, seed=0):
        """ALS decomposition to rank r"""
        rng = np.random.RandomState(seed)
        U = rng.randn(4, rank) * 0.1
        V = rng.randn(16, rank) * 0.1
        W = rng.randn(16, rank) * 0.1
        
        for it in range(max_iter):
            U_old = U.copy()
            U = als_step_U_simple(V, W, T)
            V = als_step_V(U, W, T)
            W = als_step_W(U, V, T)
            
            # Check convergence
            U_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            err = np.max(np.abs(U_recon - T))
            if err < tol:
                return U, V, W, err
            if np.max(np.abs(U - U_old)) < tol:
                break
        
        U_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        err = np.max(np.abs(U_recon - T))
        return U, V, W, err
    
    # First, try to find a solution with rank < 28
    best_result = None
    best_rank = 29  # Start above 28
    
    # Try ranks from 27 down to 20
    for target_rank in range(27, 19, -1):
        if time.time() > deadline - 30:
            break
        
        found = False
        # Try multiple random seeds
        for seed in range(20):
            if time.time() > deadline - 30:
                break
            
            U, V, W, err = als_decompose(T, target_rank, max_iter=300, tol=1e-10, seed=seed)
            
            if err < 1e-9:
                # Found a solution! Verify more carefully
                ok, max_err = verify(U, V, W, tol=1e-9)
                if ok:
                    best_result = (U, V, W)
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            # Try to go lower
            continue
        else:
            # No solution found for this rank, stop
            break
    
    # If we didn't find anything better than 28, use the Strassen solution
    if best_result is None or best_rank >= 28:
        # Use Strassen for each 2x2 block of B, giving rank 4*7=28
        num_blocks = p // 2  # 4 blocks of 2x2
        rank = num_blocks * 7  # 28
        
        U = [[0] * rank for _ in range(n * m)]  # 4 x 28
        V = [[0] * rank for _ in range(m * p)]  # 16 x 28
        W = [[0] * rank for _ in range(p * n)]  # 16 x 28
        
        for b in range(num_blocks):
            base = b * 7
            # M1 = (A[0,0]+A[1,1]) * (B[0,2b]+B[1,2b+1])
            U[0][base + 0] += 1
            U[3][base + 0] += 1
            V[0 * p + 2 * b][base + 0] += 1
            V[1 * p + 2 * b + 1][base + 0] += 1
            W[2 * b * n + 0][base + 0] += 1
            W[(2 * b + 1) * n + 1][base + 0] += 1
            
            # M2 = (A[1,0]+A[1,1]) * B[0,2b]
            U[2][base + 1] += 1
            U[3][base + 1] += 1
            V[0 * p + 2 * b][base + 1] += 1
            W[2 * b * n + 1][base + 1] += 1
            W[(2 * b + 1) * n + 1][base + 1] -= 1
            
            # M3 = A[0,0] * (B[0,2b+1]-B[1,2b+1])
            U[0][base + 2] += 1
            V[0 * p + 2 * b + 1][base + 2] += 1
            V[1 * p + 2 * b + 1][base + 2] -= 1
            W[(2 * b + 1) * n + 0][base + 2] += 1
            W[(2 * b + 1) * n + 1][base + 2] += 1
            
            # M4 = A[1,1] * (B[1,2b]-B[0,2b])
            U[3][base + 3] += 1
            V[1 * p + 2 * b][base + 3] += 1
            V[0 * p + 2 * b][base + 3] -= 1
            W[2 * b * n + 0][base + 3] += 1
            W[2 * b * n + 1][base + 3] += 1
            
            # M5 = (A[0,0]+A[0,1]) * B[1,2b+1]
            U[0][base + 4] += 1
            U[1][base + 4] += 1
            V[1 * p + 2 * b + 1][base + 4] += 1
            W[2 * b * n + 0][base + 4] -= 1
            W[(2 * b + 1) * n + 0][base + 4] += 1
            
            # M6 = (A[1,0]-A[0,0]) * (B[0,2b]+B[0,2b+1])
            U[2][base + 5] += 1
            U[0][base + 5] -= 1
            V[0 * p + 2 * b][base + 5] += 1
            V[0 * p + 2 * b + 1][base + 5] += 1
            W[(2 * b + 1) * n + 1][base + 5] += 1
            
            # M7 = (A[0,1]-A[1,1]) * (B[1,2b]+B[1,2b+1])
            U[1][base + 6] += 1
            U[3][base + 6] -= 1
            V[1 * p + 2 * b][base + 6] += 1
            V[1 * p + 2 * b + 1][base + 6] += 1
            W[2 * b * n + 0][base + 6] += 1
        
        # Verify
        ok, max_err = verify(U, V, W, tol=1e-10)
        if not ok:
            # Fallback: schoolbook rank 32
            rank_s = n * m * p
            U_s = [[0] * rank_s for _ in range(n * m)]
            V_s = [[0] * rank_s for _ in range(m * p)]
            W_s = [[0] * rank_s for _ in range(p * n)]
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_s[i * m + j][idx] = 1
                        V_s[j * p + k][idx] = 1
                        W_s[k * n + i][idx] = 1
                        idx += 1
            return {"U": U_s, "V": V_s, "W": W_s}
        
        return {"U": U, "V": V, "W": W}
    else:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
