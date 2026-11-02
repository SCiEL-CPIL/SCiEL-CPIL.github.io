"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    deadline = 120.0
    n, m, p = DIMENSIONS
    u_dim = n * m
    v_dim = m * p
    w_dim = p * n

    def _get_strassen_2x2_2x2():
        # Returns U, V, W for 2x2 * 2x2
        # U is 4x7, V is 4x7, W is 4x7
        # A: 00, 01, 10, 11
        # B: 00, 01, 10, 11
        # C: 00, 10, 01, 11 (Trace convention: W[k*n+i, r] -> C[c,a] index is c*n+a? 
        # Wait, W dimension is p*n. For 2x2, p=2, n=2.
        # W index k. k = c*n + a? 
        # The task says W[k*n+i, r]. This is for the general case.
        # For the sub-block, we just need the local factors.
        
        U_loc = [[0]*7 for _ in range(4)]
        V_loc = [[0]*7 for _ in range(4)]
        W_loc = [[0]*7 for _ in range(4)]
        
        # Indices in local 2x2 space:
        # A: 0,1,2,3
        # B: 0,1,2,3
        # C: 0,1,2,3 (C00, C10, C01, C11) -> indices 0,1,2,3
        
        # M1
        U_loc[0][0] = 1; U_loc[3][0] = 1
        V_loc[0][0] = 1; V_loc[3][0] = 1
        W_loc[0][0] = 1; W_loc[3][0] = 1
        
        # M2
        U_loc[2][1] = 1; U_loc[3][1] = 1
        V_loc[0][1] = 1
        W_loc[1][1] = 1; W_loc[3][1] = -1
        
        # M3
        U_loc[0][2] = 1
        V_loc[1][2] = 1; V_loc[3][2] = -1
        W_loc[2][2] = 1; W_loc[3][2] = 1
        
        # M4
        U_loc[3][3] = 1
        V_loc[2][3] = 1; V_loc[0][3] = -1
        W_loc[0][3] = 1; W_loc[1][3] = 1
        
        # M5
        U_loc[0][4] = 1; U_loc[1][4] = 1
        V_loc[3][4] = 1
        W_loc[0][4] = -1; W_loc[2][4] = 1
        
        # M6
        U_loc[2][5] = 1; U_loc[0][5] = -1
        V_loc[0][5] = 1; V_loc[1][5] = 1
        W_loc[3][5] = 1
        
        # M7
        U_loc[1][6] = 1; U_loc[3][6] = -1
        V_loc[2][6] = 1; V_loc[3][6] = 1
        W_loc[0][6] = 1
        
        return U_loc, V_loc, W_loc

    def _build_rank14():
        rank = 14
        U = [[0] * rank for _ in range(u_dim)]
        V = [[0] * rank for _ in range(v_dim)]
        W = [[0] * rank for _ in range(w_dim)]
        
        U_loc, V_loc, W_loc = _get_strassen_2x2_2x2()
        
        # Block 1: B cols 0,1
        # V indices: B00->0, B01->1, B10->4, B11->5
        # W indices: C00->0, C10->1, C01->2, C11->3
        for r in range(7):
            for i in range(4):
                if U_loc[i][r] != 0:
                    U[i][r] = U_loc[i][r]
            # Map local V index to global
            # Local 0 (B00) -> Global 0
            # Local 1 (B01) -> Global 1
            # Local 2 (B10) -> Global 4
            # Local 3 (B11) -> Global 5
            v_map = [0, 1, 4, 5]
            for j_loc in range(4):
                if V_loc[j_loc][r] != 0:
                    V[v_map[j_loc]][r] = V_loc[j_loc][r]
                    
            # Map local W index to global
            # Local 0 (C00) -> Global 0
            # Local 1 (C10) -> Global 1
            # Local 2 (C01) -> Global 2
            # Local 3 (C11) -> Global 3
            w_map = [0, 1, 2, 3]
            for k_loc in range(4):
                if W_loc[k_loc][r] != 0:
                    W[w_map[k_loc]][r] = W_loc[k_loc][r]
                    
        # Block 2: B cols 2,3
        # V indices: B02->2, B03->3, B12->6, B13->7
        # W indices: C02->4, C12->5, C03->6, C13->7
        offset = 7
        for r in range(7):
            r_global = r + offset
            for i in range(4):
                if U_loc[i][r] != 0:
                    U[i][r_global] = U_loc[i][r]
            
            v_map = [2, 3, 6, 7]
            for j_loc in range(4):
                if V_loc[j_loc][r] != 0:
                    V[v_map[j_loc]][r_global] = V_loc[j_loc][r]
                    
            w_map = [4, 5, 6, 7]
            for k_loc in range(4):
                if W_loc[k_loc][r] != 0:
                    W[w_map[k_loc]][r_global] = W_loc[k_loc][r]
                    
        return U, V, W

    def _verify(U, V, W):
        rank = len(U[0])
        for i in range(u_dim):
            a = i // m
            b = i % m
            for j in range(v_dim):
                b2 = j // p
                c = j % p
                for k in range(w_dim):
                    c2 = k // n
                    a2 = k % n
                    expected = 1.0 if (a == a2 and b == b2 and c == c2) else 0.0
                    s = 0.0
                    for r in range(rank):
                        s += U[i][r] * V[j][r] * W[k][r]
                    if abs(s - expected) > 1e-9:
                        return False
        return True

    def _build_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(u_dim)]
        V = [[0] * rank for _ in range(v_dim)]
        W = [[0] * rank for _ in range(w_dim)]
        col = 0
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    i = a * m + b
                    j = b * p + c
                    k = c * n + a
                    U[i][col] = 1
                    V[j][col] = 1
                    W[k][col] = 1
                    col += 1
        return U, V, W

    try:
        U, V, W = _build_rank14()
        if time.time() - t_start < deadline - 5.0:
            if _verify(U, V, W):
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass

    U, V, W = _build_schoolbook()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
