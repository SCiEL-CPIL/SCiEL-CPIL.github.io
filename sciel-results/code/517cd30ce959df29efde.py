import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _verify_decomposition_fast(U, V, W, n, m, p):
    """Verify decomposition using vectorized numpy operations."""
    if len(U) == 0 or len(U[0]) == 0:
        return False
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Compute reconstructed tensor
    T_recon = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
    
    # Construct target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T_target[u, v, w] = 1.0
                
    err = np.max(np.abs(T_recon - T_target))
    return err < 1e-10

def _build_base_decomposition():
    """Build the rank-44 Strassen+trivial decomposition."""
    n, m, p = 3, 4, 4
    rank = 44
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    
    col = 0
    for b_r in range(2):
        for b_c in range(2):
            # 7 Strassen terms for A[0:2, b_r*2:b_r*2+2] * B[b_r*2:b_r*2+2, b_c*2:b_c*2+2] -> C[0:2, b_c*2:b_c*2+2]
            a0 = 0*m + (b_r*2+0)  # A[0, b_r*2]
            a1 = 1*m + (b_r*2+1)  # A[1, b_r*2+1]
            a2 = 0*m + (b_r*2+1)  # A[0, b_r*2+1]
            a3 = 1*m + (b_r*2+0)  # A[1, b_r*2]
            e0 = (b_r*2+0)*p + (b_c*2+0)  # B[b_r*2, b_c*2]
            e1 = (b_r*2+1)*p + (b_c*2+1)  # B[b_r*2+1, b_c*2+1]
            e2 = (b_r*2+0)*p + (b_c*2+1)  # B[b_r*2, b_c*2+1]
            e3 = (b_r*2+1)*p + (b_c*2+0)  # B[b_r*2+1, b_c*2]
            c0 = (b_c*2+0)*n + 0  # C[0, b_c*2]
            c1 = (b_c*2+1)*n + 1  # C[1, b_c*2+1]
            c2 = (b_c*2+1)*n + 0  # C[0, b_c*2+1]
            c3 = (b_c*2+0)*n + 1  # C[1, b_c*2]
            
            # m1 = (a+d)(e+h) -> C[0,0] += , C[1,1] +=
            U[a0][col] = 1; U[a1][col] = 1
            V[e0][col] = 1; V[e1][col] = 1
            W[c0][col] = 1; W[c1][col] = 1
            col += 1
            # m2 = (c+d)e -> C[0,1] += , C[1,1] -=
            U[a3][col] = 1; U[a1][col] = 1
            V[e0][col] = 1
            W[c3][col] = 1; W[c1][col] = -1
            col += 1
            # m3 = a(f-h) -> C[0,1] += , C[1,1] +=
            U[a0][col] = 1
            V[e2][col] = 1; V[e1][col] = -1
            W[c2][col] = 1; W[c1][col] = 1
            col += 1
            # m4 = d(g-e) -> C[0,0] += , C[1,0] +=
            U[a1][col] = 1
            V[e3][col] = 1; V[e0][col] = -1
            W[c0][col] = 1; W[c3][col] = 1
            col += 1
            # m5 = (a+b)h -> C[0,0] -= , C[0,1] +=
            U[a0][col] = 1; U[a2][col] = 1
            V[e1][col] = 1
            W[c0][col] = -1; W[c2][col] = 1
            col += 1
            # m6 = (c-a)(e+f) -> C[1,1] +=
            U[a0][col] = -1; U[a3][col] = 1
            V[e0][col] = 1; V[e2][col] = 1
            W[c1][col] = 1
            col += 1
            # m7 = (b-d)(g+h) -> C[0,0] +=
            U[a2][col] = 1; U[a1][col] = -1
            V[e3][col] = 1; V[e1][col] = 1
            W[c0][col] = 1
            col += 1
            
            # 4 trivial terms for row 2: A[2, b_r*2:b_r*2+2] * B[b_r*2:b_r*2+2, b_c*2:b_c*2+2] -> C[2, b_c*2:b_c*2+2]
            for j_local in range(2):
                for k_local in range(2):
                    U[2*m + (b_r*2+j_local)][col] = 1
                    V[(b_r*2+j_local)*p + (b_c*2+k_local)][col] = 1
                    W[(b_c*2+k_local)*n + 2][col] = 1
                    col += 1
    
    return U, V, W

def _check_rank_reduction_greedy(U, V, W, n, m, p):
    """Try to reduce rank by greedily removing columns."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    active = list(range(rank))
    
    changed = True
    while changed and _time_left() > 3.0:
        changed = False
        i = 0
        while i < len(active):
            idx = active[i]
            new_active = active[:i] + active[i+1:]
            if len(new_active) == 0:
                i += 1
                continue
            U_sub = U_arr[:, new_active]
            V_sub = V_arr[:, new_active]
            W_sub = W_arr[:, new_active]
            if _verify_decomposition_fast(U_sub, V_sub, W_sub, n, m, p):
                active = new_active
                changed = True
                continue
            else:
                i += 1
                
    if len(active) < rank:
        U_new = U_arr[:, active].tolist()
        V_new = V_arr[:, active].tolist()
        W_new = W_arr[:, active].tolist()
        return U_new, V_new, W_new, len(active)
        
    return U, V, W, rank

def construct():
    n, m, p = 3, 4, 4
    
    U, V, W = _build_base_decomposition()
    
    if not _verify_decomposition_fast(U, V, W, n, m, p):
        # Fallback: trivial rank-48
        rank = 48
        U = [[0.0]*rank for _ in range(n*m)]
        V = [[0.0]*rank for _ in range(m*p)]
        W = [[0.0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
    
    if _time_left() > 5.0:
        U, V, W, new_rank = _check_rank_reduction_greedy(U, V, W, n, m, p)
    
    return {"U": U, "V": V, "W": W}
