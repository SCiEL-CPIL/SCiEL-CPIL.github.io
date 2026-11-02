import numpy as np
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Precompute the target tensor T
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    T_flat = T.flatten()
    
    start_time = time.time()
    deadline = start_time + 140
    
    # Try ranks from 14 up to 17
    # Theoretical lower bound is 9, known optimal is 14.
    # Schoolbook is 18.
    
    def run_als(U, V, W, max_iters=50, tol=1e-9):
        """
        Alternating Least Squares for tensor decomposition.
        U: (nm, rank), V: (mp, rank), W: (pn, rank)
        """
        rank = U.shape[1]
        # Precompute T reshaped for efficient linear solves
        # Y_a is T[a, :, :] flattened to (mp*pn,)
        # M is (mp*pn, rank) where M[(b,c), s] = V[b,s] * W[c,s]
        # U[a, :] = T[a, :, :].reshape(-1) @ pinv(M)
        
        for _ in range(max_iters):
            # Update U
            # M is (mp*pn, rank)
            M = np.einsum('bs,cs->bc,s', V, W).reshape(mp * pn, rank)
            # Solve U @ M = T_flat_reshaped
            # T_reshaped is (nm, mp*pn)
            T_reshaped = T.reshape(nm, mp * pn)
            # U = T_reshaped @ pinv(M)
            # Using lstsq for better numerical stability
            try:
                U_new, _, _, _ = np.linalg.lstsq(M.T, T_reshaped.T, rcond=None)
                U = U_new.T
            except:
                break
                
            # Update V
            # M is (nm*pn, rank) where M[(a,c), s] = U[a,s] * W[c,s]
            M = np.einsum('as,cs->ac,s', U, W).reshape(nm * pn, rank)
            # T_reshaped is (mp, nm*pn)
            T_reshaped_v = T.transpose(1, 0, 2).reshape(mp, nm * pn)
            try:
                V_new, _, _, _ = np.linalg.lstsq(M.T, T_reshaped_v.T, rcond=None)
                V = V_new.T
            except:
                break
                
            # Update W
            # M is (nm*mp, rank) where M[(a,b), s] = U[a,s] * V[b,s]
            M = np.einsum('as,bs->ab,s', U, V).reshape(nm * mp, rank)
            # T_reshaped is (pn, nm*mp)
            T_reshaped_w = T.transpose(2, 0, 1).reshape(pn, nm * mp)
            try:
                W_new, _, _, _ = np.linalg.lstsq(M.T, T_reshaped_w.T, rcond=None)
                W = W_new.T
            except:
                break
                
            # Check convergence
            R = np.einsum('as,bs,cs->abc', U, V, W)
            err = np.linalg.norm(R - T)
            if err < tol:
                break
                
        return U, V, W, np.linalg.norm(np.einsum('as,bs,cs->abc', U, V, W) - T)

    best_sol = None
    best_rank = nm * mp * pn # Schoolbook
    
    # Try Rank 14 first
    for rank in [14, 15, 16, 17]:
        if time.time() > deadline - 10:
            break
            
        max_restarts = 50 if rank == 14 else 10
        
        for trial in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(42 + rank * 100 + trial)
            # Scale initialization
            U0 = np.random.randn(nm, rank)
            V0 = np.random.randn(mp, rank)
            W0 = np.random.randn(pn, rank)
            
            U, V, W, err = run_als(U0, V0, W0, max_iters=100)
            
            if err < 1e-8:
                # Round to nearest 0.5
                U_out = [[round(x*2)/2 for x in row] for row in U]
                V_out = [[round(x*2)/2 for x in row] for row in V]
                W_out = [[round(x*2)/2 for x in row] for row in W]
                
                # Verify exactness
                U_np = np.array(U_out, dtype=float)
                V_np = np.array(V_out, dtype=float)
                W_np = np.array(W_out, dtype=float)
                R = np.einsum('as,bs,cs->abc', U_np, V_np, W_np)
                
                if np.allclose(R, T, atol=1e-9):
                    return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fallback: Schoolbook decomposition of rank n*m*p
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
