"""Matrix multiplication 4x4x6 via 4x4x4 Strassen + 4x4x2 Strassen-block."""
DIMENSIONS = [4, 4, 6]

import time
import numpy as np

def _strassen_2x2():
    """2x2x2 Strassen factors. U: (4,7), V: (4,7), W: (4,7)."""
    n = m = p = 2
    r = 7
    U = [[0.0]*r for _ in range(4)]
    V = [[0.0]*r for _ in range(4)]
    W = [[0.0]*r for _ in range(4)]
    prods = [
        ([(0,0,1.0),(1,1,1.0)], [(0,0,1.0),(1,1,1.0)], [(0,0,1.0),(1,1,1.0)]),
        ([(1,0,1.0),(1,1,1.0)], [(0,0,1.0)], [(1,0,1.0),(1,1,-1.0)]),
        ([(0,0,1.0)], [(0,1,1.0),(1,1,-1.0)], [(0,1,1.0),(1,1,1.0)]),
        ([(1,1,1.0)], [(1,0,1.0),(0,0,-1.0)], [(0,0,1.0),(1,0,1.0)]),
        ([(0,0,1.0),(0,1,1.0)], [(1,1,1.0)], [(0,0,-1.0),(0,1,1.0)]),
        ([(1,0,1.0),(0,0,-1.0)], [(0,0,1.0),(0,1,1.0)], [(1,1,1.0)]),
        ([(0,1,1.0),(1,1,-1.0)], [(1,0,1.0),(1,1,1.0)], [(0,0,1.0)]),
    ]
    for c,(ub,vb,wb) in enumerate(prods):
        for (i,j,cv) in ub: U[i*m+j][c] += cv
        for (j,k,cv) in vb: V[j*p+k][c] += cv
        for (i,k,cv) in wb: W[k*n+i][c] += cv
    return U, V, W

def _build_444():
    """4x4x4 Strassen via Kronecker of 2x2 Strassen. Returns U(16,49), V(16,49), W(16,49)."""
    Us, Vs, Ws = _strassen_2x2()
    r = 49
    U = [[0.0]*r for _ in range(16)]
    V = [[0.0]*r for _ in range(16)]
    W = [[0.0]*r for _ in range(16)]
    for i1 in range(2):
        for i2 in range(2):
            i = 2*i1+i2
            for j1 in range(2):
                for j2 in range(2):
                    j = 2*j1+j2
                    for r1 in range(7):
                        for r2 in range(7):
                            col = 7*r1+r2
                            U[i*4+j][col] = Us[i1*2+j1][r1]*Us[i2*2+j2][r2]
    for j1 in range(2):
        for j2 in range(2):
            j = 2*j1+j2
            for k1 in range(2):
                for k2 in range(2):
                    k = 2*k1+k2
                    for r1 in range(7):
                        for r2 in range(7):
                            col = 7*r1+r2
                            V[j*4+k][col] = Vs[j1*2+k1][r1]*Vs[j2*2+k2][r2]
    for k1 in range(2):
        for k2 in range(2):
            k = 2*k1+k2
            for i1 in range(2):
                for i2 in range(2):
                    i = 2*i1+i2
                    for r1 in range(7):
                        for r2 in range(7):
                            col = 7*r1+r2
                            W[k*4+i][col] = Ws[k1*2+i1][r1]*Ws[k2*2+i2][r2]
    return U, V, W

def _verify(U, V, W, n, m, p, rank):
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    return float(np.max(np.abs(T - 1.0)))

def _try_reduce_rank(U, V, W, n, m, p, rank, deadline):
    """
    Try to reduce the rank by dropping columns and adjusting the rest.
    Returns (best_U, best_V, best_W, best_rank) as lists.
    """
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    # Target tensor
    T = np.zeros((n, m, p), dtype=np.float64)
    T[:] = 1.0  # Actually the matmul tensor is delta_{i,j,k} where A[i,j]*B[j,k]->C[i,k]
    # Wait, the tensor T[i,j,k] = 1 if the decomposition produces the identity map.
    # The decomposition is sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = delta_{i,j,k} ? No.
    # The tensor for matrix multiplication is T[i,j,k] = 1 if we are computing C[i,k] += A[i,j]*B[j,k].
    # So T[i,j,k] = 1 for all i,j,k.
    # The reconstruction is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    
    best_U, best_V, best_W = Ua.copy(), Va.copy(), Wa.copy()
    best_rank = rank
    
    # Try to reduce by 1
    for target_rank in [rank - 1]:
        if time.time() > deadline - 10:
            break
            
        # Initialize with random perturbation of the last column
        # Or just drop the last column and try to fix
        U_try = np.zeros((n*m, target_rank), dtype=np.float64)
        V_try = np.zeros((m*p, target_rank), dtype=np.float64)
        W_try = np.zeros((p*n, target_rank), dtype=np.float64)
        
        # Copy first target_rank-1 columns
        U_try[:, :target_rank-1] = Ua[:, :target_rank-1]
        V_try[:, :target_rank-1] = Va[:, :target_rank-1]
        W_try[:, :target_rank-1] = Wa[:, :target_rank-1]
        
        # Last column is the one we're trying to optimize to absorb the missing part
        # Initialize with small random values
        U_try[:, target_rank-1] = np.random.randn(n*m) * 0.1
        V_try[:, target_rank-1] = np.random.randn(m*p) * 0.1
        W_try[:, target_rank-1] = np.random.randn(p*n) * 0.1
        
        # Gradient descent on the last column only? Or all?
        # Let's try optimizing all columns but with a strong prior on the first rank-1
        # Actually, simpler: optimize the last column to minimize the residual.
        # Residual R[i,j,k] = 1 - sum_{r < target_rank-1} U[i,j,r]V[j,k,r]W[k,i,r]
        # We want U[i,j,target_rank-1] * V[j,k,target_rank-1] * W[k,i,target_rank-1] = R[i,j,k]
        
        # This is a rank-1 approximation problem.
        # Let R_tensor = R reshaped to (n*m, m*p, p*n)
        # We want to find u, v, w such that u ⊗ v ⊗ w ≈ R_tensor
        
        # Compute residual from first target_rank-1 columns
        R_res = np.ones((n*m, m*p, p*n), dtype=np.float64)
        for r in range(target_rank-1):
            term = np.einsum('a,b,c->abc', U_try[:, r], V_try[:, r], W_try[:, r])
            R_res -= term
            
        # Now find best rank-1 approximation of R_res
        # Use SVD-like method for tensors? Or just optimize u, v, w directly.
        # Let's do a few iterations of alternating minimization.
        
        u = U_try[:, target_rank-1]
        v = V_try[:, target_rank-1]
        w = W_try[:, target_rank-1]
        
        for it in range(50):
            if time.time() > deadline - 5:
                break
            # Update u: u = argmin_u ||R_res - u ⊗ v ⊗ w||
            # This is equivalent to u = (R_res * (v ⊗ w)) / ||v ⊗ w||^2
            # R_res * (v ⊗ w) means contracting R_res with v and w
            # u_new[i] = sum_{j,k} R_res[i,j,k] * v[j] * w[k]
            u_new = np.einsum('ijk,j,k->i', R_res, v, w)
            norm_vw = np.dot(v, v) * np.dot(w, w)
            if norm_vw > 1e-12:
                u = u_new / norm_vw
            
            # Update v
            v_new = np.einsum('ijk,i,k->j', R_res, u, w)
            norm_uw = np.dot(u, u) * np.dot(w, w)
            if norm_uw > 1e-12:
                v = v_new / norm_uw
            
            # Update w
            w_new = np.einsum('ijk,i,j->k', R_res, u, v)
            norm_uv = np.dot(u, u) * np.dot(v, v)
            if norm_uv > 1e-12:
                w = w_new / norm_uv
        
        U_try[:, target_rank-1] = u
        V_try[:, target_rank-1] = v
        W_try[:, target_rank-1] = w
        
        # Check error
        recon = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for r in range(target_rank):
            recon += np.einsum('a,b,c->abc', U_try[:, r], V_try[:, r], W_try[:, r])
            
        # The target is all 1s in the (i,j,k) space where i,j,k are the indices.
        # Wait, the target tensor for matmul is T[i,j,k] = 1 for all i,j,k.
        # So we want recon[i,j,k] = 1 for all i,j,k.
        err = np.max(np.abs(recon - 1.0))
        
        if err < 1e-6:
            best_U = U_try
            best_V = V_try
            best_W = W_try
            best_rank = target_rank
            break  # Can't go lower than rank-1 in this simple loop
        
    return best_U.tolist(), best_V.tolist(), best_W.tolist(), best_rank

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    t0 = time.time()
    deadline = t0 + 110.0

    Us, Vs, Ws = _strassen_2x2()
    U444, V444, W444 = _build_444()
    
    rank_444 = 49
    rank_442 = 28
    total_rank = rank_444 + rank_442  # 77

    U = [[0.0]*total_rank for _ in range(n*m)]  # 16 x 77
    V = [[0.0]*total_rank for _ in range(m*p)]  # 24 x 77
    W = [[0.0]*total_rank for _ in range(p*n)]  # 24 x 77

    # Part 1: 4x4x4 Strassen for columns 0-3 of B -> columns 0-3 of C
    for idx in range(16):
        for c in range(rank_444):
            if U444[idx][c] != 0:
                U[idx][c] = U444[idx][c]
    for j in range(4):
        for k in range(4):
            for c in range(rank_444):
                if V444[j*4+k][c] != 0:
                    V[j*6+k][c] = V444[j*4+k][c]
    for k in range(4):
        for i in range(4):
            for c in range(rank_444):
                if W444[k*4+i][c] != 0:
                    W[k*4+i][c] = W444[k*4+i][c]

    # Part 2: 4x4x2 Strassen-block for columns 4-5 of B -> columns 4-5 of C
    products = [
        (0, 0),  # A[0:2,0:2] * B[0:2,4:6]
        (0, 1),  # A[0:2,2:4] * B[2:4,4:6]
        (1, 0),  # A[2:4,0:2] * B[0:2,4:6]
        (1, 1),  # A[2:4,2:4] * B[2:4,4:6]
    ]
    
    for p_idx, (I, J) in enumerate(products):
        col_off = rank_444 + p_idx*7
        bot = J
        
        for i0 in range(2):
            for j0 in range(2):
                i = 2*I + i0
                j = 2*J + j0
                for s in range(7):
                    v = Us[2*i0+j0][s]
                    if v: U[i*4+j][col_off+s] += v
        for j0 in range(2):
            for k0 in range(2):
                jg = 2*bot + j0
                kg = 4 + k0
                for s in range(7):
                    v = Vs[2*j0+k0][s]
                    if v: V[jg*6+kg][col_off+s] += v
        for i0 in range(2):
            for k0 in range(2):
                ig = 2*I + i0
                kg = 4 + k0
                for s in range(7):
                    v = Ws[2*k0+i0][s]
                    if v: W[kg*4+ig][col_off+s] += v

    err = _verify(U, V, W, n, m, p, total_rank)
    
    if err > 1e-6:
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Try to reduce rank
    if time.time() < deadline - 20:
        U_red, V_red, W_red, new_rank = _try_reduce_rank(U, V, W, n, m, p, total_rank, deadline)
        if new_rank < total_rank:
            err_red = _verify(U_red, V_red, W_red, n, m, p, new_rank)
            if err_red < 1e-6:
                return {"U": U_red, "V": V_red, "W": W_red}

    return {"U": U, "V": V, "W": W}
