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

    # === Rank reduction via alternating least squares with structured initialization ===
    # Start from rank-28, try to reduce to rank 24 by finding a low-rank approximation
    # that still exactly represents the tensor.

    def try_rank_reduction(target_rank, n_restarts=5, n_iters=300, seed=0):
        """
        Try to find a rank-`target_rank` decomposition by starting from rank-28
        and using ALS (alternating least squares) to fit the tensor.
        """
        if time.time() > deadline:
            return None
        
        rng = np.random.RandomState(seed)
        
        best_err = float('inf')
        best_solution = None
        
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            # Initialize: take first target_rank columns of rank-28, add small perturbation
            if target_rank <= 28:
                U = U28[:, :target_rank].copy()
                V = V28[:, :target_rank].copy()
                W = W28[:, :target_rank].copy()
                # Add small random perturbation to escape local minima
                U += rng.randn(n * m, target_rank) * 0.01
                V += rng.randn(m * p, target_rank) * 0.01
                W += rng.randn(p * n, target_rank) * 0.01
            else:
                U = rng.randn(n * m, target_rank) * 0.5
                V = rng.randn(m * p, target_rank) * 0.5
                W = rng.randn(p * n, target_rank) * 0.5
            
            err = float(np.sum((np.einsum('at,bt,ct->abc', U, V, W) - T) ** 2))
            
            for it in range(n_iters):
                if time.time() > deadline - 5:
                    break
                
                # Alternating least squares
                # Fix V and W, solve for U
                # T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]
                # For each a: T[a,:,:] = U[a,:] @ (V^T * W^T) ... 
                # Actually: T[a,b,c] = sum_t U[a,t] * (V[b,t] * W[c,t])
                # Let M_a[b,c] = T[a,b,c], and let Z[b,c] = sum_t V[b,t]*W[c,t]
                # Then M_a = U[a,:] outer product with... no, this is a 3-way tensor.
                
                # Better: vectorize. T.reshape(n*m, m*p*p*n) ... no.
                # Use: for fixed V,W, the equation for U is:
                # sum_t U[a,t] * V[b,t] * W[c,t] = T[a,b,c] for all a,b,c
                # This is linear in U. Let's form the system.
                
                # Alternative: use gradient-based ALS
                recon = np.einsum('at,bt,ct->abc', U, V, W)
                F = recon - T
                
                # Gradient descent with adaptive step
                gU = np.einsum('abc,bt,ct->at', F, V, W)
                gV = np.einsum('abc,at,ct->bt', F, U, W)
                gW = np.einsum('abc,at,bt->ct', F, U, V)
                
                norm_gU = float(np.sum(gU * gU)) ** 0.5 + 1e-12
                norm_gV = float(np.sum(gV * gV)) ** 0.5 + 1e-12
                norm_gW = float(np.sum(gW * gW)) ** 0.5 + 1e-12
                max_norm = max(norm_gU, norm_gV, norm_gW)
                lr = 0.5 / (max_norm + 1e-6)
                
                improved = False
                for scale in [1.0, 0.5, 0.25, 0.1, 0.05]:
                    U_new = U - lr * scale * gU
                    V_new = V - lr * scale * gV
                    W_new = W - lr * scale * gW
                    recon_new = np.einsum('at,bt,ct->abc', U_new, V_new, W_new)
                    err_new = float(np.sum((recon_new - T) ** 2))
                    if err_new < err:
                        U, V, W = U_new, V_new, W_new
                        err = err_new
                        improved = True
                        break
                
                if not improved:
                    break
                
                if err < 1e-10:
                    break
            
            if err < best_err:
                best_err = err
                best_solution = (U.copy(), V.copy(), W.copy())
            
            if best_err < 1e-8:
                break
        
        if best_err > 1e-6:
            return None
        
        Uf, Vf, Wf = best_solution
        
        # Snap to half-integers
        Us = np.round(Uf * 2) / 2
        Vs = np.round(Vf * 2) / 2
        Ws = np.round(Wf * 2) / 2
        if verify(Us, Vs, Ws, tol=1e-8):
            return Us, Vs, Ws, target_rank
        
        # Try integer snap
        Ui = np.round(Uf)
        Vi = np.round(Vf)
        Wi = np.round(Wf)
        if verify(Ui, Vi, Wi, tol=1e-8):
            return Ui, Vi, Wi, target_rank
        
        # Try quarter-integer snap
        Uq = np.round(Uf * 4) / 4
        Vq = np.round(Vf * 4) / 4
        Wq = np.round(Wf * 4) / 4
        if verify(Uq, Vq, Wq, tol=1e-8):
            return Uq, Vq, Wq, target_rank
        
        return None

    # Try to find rank-24 decomposition
    best_result = None
    best_rank = 28

    # Try rank 24 with multiple seeds
    for seed in range(10):
        if time.time() > deadline - 15:
            break
        res = try_rank_reduction(24, n_restarts=3, n_iters=200, seed=seed)
        if res is not None:
            Uf, Vf, Wf, r = res
            if r < best_rank:
                best_rank = r
                best_result = (Uf, Vf, Wf, r)
                break

    # If rank 24 found, try rank 23
    if best_rank == 24 and time.time() < deadline - 15:
        for seed in range(5):
            if time.time() > deadline - 5:
                break
            res = try_rank_reduction(23, n_restarts=2, n_iters=150, seed=seed)
            if res is not None:
                Uf, Vf, Wf, r = res
                if r < best_rank:
                    best_rank = r
                    best_result = (Uf, Vf, Wf, r)
                    break

    # If we found something better, use it; otherwise fall back to rank 28
    if best_result is not None and best_rank < 28:
        Uf, Vf, Wf, r = best_result
        return to_lists(Uf, Vf, Wf, r)

    # Return the verified rank-28 Strassen decomposition
    return to_lists(U28, V28, W28, 28)
# EVOLVE-BLOCK-END
