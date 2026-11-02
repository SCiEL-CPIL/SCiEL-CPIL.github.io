"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    deadline = time.time() + 135

    # Build the matrix multiplication tensor T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U, V, W, tol=1e-8):
        recon = np.einsum('at,bt,ct->abc', U, V, W)
        return np.max(np.abs(recon - T)) < tol

    def to_lists(U, V, W, r):
        U_list = [[float(U[i, t]) for t in range(r)] for i in range(n * m)]
        V_list = [[float(V[i, t]) for t in range(r)] for i in range(m * p)]
        W_list = [[float(W[i, t]) for t in range(r)] for i in range(p * n)]
        return {"U": U_list, "V": V_list, "W": W_list}

    # Strassen 2x2x2 decomposition (rank 7)
    def strassen_2x2_decomp():
        U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
        U[0,0]=1.0; U[3,0]=1.0; V[0,0]=1.0; V[3,0]=1.0; W[0,0]=1.0; W[3,0]=1.0
        U[2,1]=1.0; U[3,1]=1.0; V[0,1]=1.0; W[1,1]=1.0; W[3,1]=-1.0
        U[0,2]=1.0; V[1,2]=1.0; V[3,2]=-1.0; W[2,2]=1.0; W[3,2]=1.0
        U[3,3]=1.0; V[2,3]=1.0; V[0,3]=-1.0; W[0,3]=1.0; W[1,3]=1.0
        U[0,4]=1.0; U[1,4]=1.0; V[3,4]=1.0; W[0,4]=-1.0; W[2,4]=1.0
        U[2,5]=1.0; U[0,5]=-1.0; V[0,5]=1.0; V[1,5]=1.0; W[3,5]=1.0
        U[1,6]=1.0; U[3,6]=-1.0; V[2,6]=1.0; V[3,6]=1.0; W[0,6]=1.0
        return U, V, W

    # Build rank-28 Strassen block decomposition
    def build_strassen28():
        U_s, V_s, W_s = strassen_2x2_decomp()
        full_rank = 28
        U_full = np.zeros((n * m, full_rank))
        V_full = np.zeros((m * p, full_rank))
        W_full = np.zeros((p * n, full_rank))
        blocks = [
            (lambda li,lj: li*4+lj,          lambda lj,lk: lj*4+lk,          lambda lk,li: lk*2+li),
            (lambda li,lj: li*4+(lj+2),      lambda lj,lk: (lj+2)*4+lk,      lambda lk,li: lk*2+li),
            (lambda li,lj: li*4+lj,          lambda lj,lk: lj*4+(lk+2),      lambda lk,li: (lk+2)*2+li),
            (lambda li,lj: li*4+(lj+2),      lambda lj,lk: (lj+2)*4+(lk+2),  lambda lk,li: (lk+2)*2+li),
        ]
        for bi, (ga, gb, gc) in enumerate(blocks):
            for t in range(7):
                col = bi * 7 + t
                for li in range(2):
                    for lj in range(2):
                        U_full[ga(li, lj), col] = U_s[li * 2 + lj, t]
                for lj in range(2):
                    for lk in range(2):
                        V_full[gb(lj, lk), col] = V_s[lj * 2 + lk, t]
                for lk in range(2):
                    for li in range(2):
                        W_full[gc(lk, li), col] = W_s[lk * 2 + li, t]
        return U_full, V_full, W_full

    U28, V28, W28 = build_strassen28()

    # Fallback: schoolbook rank 32
    def build_schoolbook():
        r = n * m * p
        U = np.zeros((n * m, r)); V = np.zeros((m * p, r)); W = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, r

    if not verify(U28, V28, W28, tol=1e-8):
        U, V, W, r = build_schoolbook()
        return to_lists(U, V, W, r)

    # === Strategy: try to reduce rank by freeing columns and using alternating minimization ===
    # The idea: take the rank-28 decomposition, freeze most columns, and optimize a few free columns
    # to see if we can achieve rank 27 or lower exactly.

    def try_rank_reduction(free_cols, n_iters, seed, tol=1e-10):
        """
        Try to reduce rank by optimizing `free_cols` columns while keeping others fixed.
        Returns (U, V, W, rank) if successful, else None.
        """
        rng = np.random.RandomState(seed)
        U = U28.copy(); V = V28.copy(); W = W28.copy()
        
        # Pick which columns to free
        if free_cols >= 28:
            return None
        
        # Initialize free columns with small random values
        U[:, free_cols:] = rng.randn(n * m, 28 - free_cols) * 0.3
        V[:, free_cols:] = rng.randn(m * p, 28 - free_cols) * 0.3
        W[:, free_cols:] = rng.randn(p * n, 28 - free_cols) * 0.3
        
        # Fixed columns
        fixed = slice(0, free_cols)
        base_U = U[:, fixed]; base_V = V[:, fixed]; base_W = W[:, fixed]
        base = np.einsum('at,bt,ct->abc', base_U, base_V, base_W)
        
        # Alternating minimization
        for it in range(n_iters):
            if time.time() > deadline:
                return None
            
            # Update U
            # minimize ||base + Uf ⊗ Vf ⊗ Wf - T||^2 over Uf
            # grad_U = einsum('abc,bt,ct->at', F, Vf, Wf)
            Uf = U[:, free_cols:]
            Vf = V[:, free_cols:]
            Wf = W[:, free_cols:]
            recon = base + np.einsum('at,bt,ct->abc', Uf, Vf, Wf)
            F = recon - T
            gU = np.einsum('abc,bt,ct->at', F, Vf, Wf)
            gV = np.einsum('abc,at,ct->bt', F, Uf, Wf)
            gW = np.einsum('abc,at,bt->ct', F, Uf, Vf)
            
            # Line search with backtracking
            err = float(np.sum(F * F))
            lr = 0.1 / (1.0 + err)
            
            for _ in range(20):
                U_new = U.copy()
                V_new = V.copy()
                W_new = W.copy()
                U_new[:, free_cols:] -= lr * gU
                V_new[:, free_cols:] -= lr * gV
                W_new[:, free_cols:] -= lr * gW
                
                recon_new = base + np.einsum('at,bt,ct->abc', U_new[:, free_cols:], V_new[:, free_cols:], W_new[:, free_cols:])
                err_new = float(np.sum((recon_new - T) ** 2))
                
                if err_new < err:
                    U, V, W = U_new, V_new, W_new
                    err = err_new
                    break
                lr *= 0.5
            
            if err < tol:
                break
        
        # Check if we can snap to half-integers
        Us = np.round(U * 2) / 2
        Vs = np.round(V * 2) / 2
        Ws = np.round(W * 2) / 2
        if verify(Us, Vs, Ws, tol=1e-8):
            return Us, Vs, Ws, 28
        
        # Try integer snap
        Ui = np.round(U)
        Vi = np.round(V)
        Wi = np.round(W)
        if verify(Ui, Vi, Wi, tol=1e-8):
            return Ui, Vi, Wi, 28
        
        return None

    # Try to find a rank-27 decomposition by removing one column and adjusting others
    # This is harder; let's try a different approach: use the structure of the tensor
    # to find a lower-rank decomposition directly.

    # Alternative: try to find a rank-27 decomposition by solving a system of equations
    # The tensor has 2*4*4 = 32 entries. A rank-r decomposition has 3*r*(something) parameters.
    # For r=27, we have 3*27*8 = 648 parameters, so it should be feasible if the tensor
    # has the right structure.

    # Let's try a more direct approach: use the fact that the matrix multiplication tensor
    # for 2x4x4 can be decomposed using 4 Strassen blocks of rank 7, giving rank 28.
    # To get rank 27, we need to find a linear dependency among the 28 terms.

    # Check if the 28 columns are linearly independent in some sense
    # If not, we can reduce the rank.

    # For now, let's just return the rank-28 decomposition if we can't find a better one.
    # The current best is rank 28 with score 1.1429.

    # Try a few random restarts to see if we can find a rank-27 decomposition
    for seed in [0, 1, 2, 3, 5, 7, 11, 13]:
        if time.time() > deadline:
            break
        res = try_rank_reduction(26, 500, seed)  # free 2 columns
        if res is not None:
            Uf, Vf, Wf, r = res
            if r < 28:
                return to_lists(Uf, Vf, Wf, r)

    # Return the verified rank-28 Strassen decomposition
    return to_lists(U28, V28, W28, 28)
# EVOLVE-BLOCK-END
