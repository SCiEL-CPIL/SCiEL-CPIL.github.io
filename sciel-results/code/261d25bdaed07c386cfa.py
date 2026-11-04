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

def _build_target_tensor(n, m, p):
    """Build the correct target tensor in (n*m, m*p, p*n) space.
    T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k; 0 elsewhere."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify(U, V, W, n, m, p, rank):
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    return float(np.max(np.abs(T - 1.0)))

def _build_target_tensor(n, m, p):
    """Build the correct target tensor in (n*m, m*p, p*n) space.
    T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k; 0 elsewhere."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _try_reduce_rank(U, V, W, n, m, p, rank, deadline):
    """
    Try to reduce the rank by dropping columns and adjusting the rest.
    Returns (best_U, best_V, best_W, best_rank) as lists.
    """
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    # Build the correct target tensor
    target = _build_target_tensor(n, m, p)
    
    best_U, best_V, best_W = Ua.copy(), Va.copy(), Wa.copy()
    best_rank = rank
    
    # Try to reduce by 1, 2, 3 ranks
    for drop in [1, 2, 3]:
        target_rank = rank - drop
        if target_rank < 1:
            break
        if time.time() > deadline - 15:
            break
        
        # Copy first target_rank columns
        U_try = np.zeros((n*m, target_rank), dtype=np.float64)
        V_try = np.zeros((m*p, target_rank), dtype=np.float64)
        W_try = np.zeros((p*n, target_rank), dtype=np.float64)
        
        U_try[:, :] = Ua[:, :target_rank]
        V_try[:, :] = Va[:, :target_rank]
        W_try[:, :] = Wa[:, :target_rank]
        
        # Compute residual
        R_res = target.copy()
        for r in range(target_rank):
            term = np.einsum('a,b,c->abc', U_try[:, r], V_try[:, r], W_try[:, r])
            R_res -= term
        
        # Try to absorb the residual with one additional rank-1 term
        # Use alternating minimization
        u = np.random.randn(n*m) * 0.1
        v = np.random.randn(m*p) * 0.1
        w = np.random.randn(p*n) * 0.1
        
        for it in range(100):
            if time.time() > deadline - 5:
                break
            u_new = np.einsum('ijk,j,k->i', R_res, v, w)
            norm_vw = np.dot(v, v) * np.dot(w, w)
            if norm_vw > 1e-12:
                u = u_new / norm_vw
            v_new = np.einsum('ijk,i,k->j', R_res, u, w)
            norm_uw = np.dot(u, u) * np.dot(w, w)
            if norm_uw > 1e-12:
                v = v_new / norm_uw
            w_new = np.einsum('ijk,i,j->k', R_res, u, v)
            norm_uv = np.dot(u, u) * np.dot(v, v)
            if norm_uv > 1e-12:
                w = w_new / norm_uv
        
        # Add the new column
        U_try2 = np.zeros((n*m, target_rank + 1), dtype=np.float64)
        V_try2 = np.zeros((m*p, target_rank + 1), dtype=np.float64)
        W_try2 = np.zeros((p*n, target_rank + 1), dtype=np.float64)
        U_try2[:, :target_rank] = U_try
        V_try2[:, :target_rank] = V_try
        W_try2[:, :target_rank] = W_try
        U_try2[:, target_rank] = u
        V_try2[:, target_rank] = v
        W_try2[:, target_rank] = w
        
        # Check error
        recon = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for r in range(target_rank + 1):
            recon += np.einsum('a,b,c->abc', U_try2[:, r], V_try2[:, r], W_try2[:, r])
        
        err = np.max(np.abs(recon - target))
        
        if err < 1e-6 and (target_rank + 1) < best_rank:
            best_U = U_try2
            best_V = V_try2
            best_W = W_try2
            best_rank = target_rank + 1
    
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
