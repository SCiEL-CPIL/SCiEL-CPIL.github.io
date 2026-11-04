"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_222():
    """Return (U, V, W) for 2x2x2 matrix multiplication, rank 7 (Strassen)."""
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    U[0][0] = 1; V[0][0] = 1; V[3][0] = 1; W[0][0] = 1
    U[0][1] = 1; U[1][1] = 1; V[3][1] = 1; W[2][1] = 1
    U[2][2] = 1; U[3][2] = 1; V[0][2] = 1; W[1][2] = 1
    U[3][3] = 1; V[2][3] = 1; V[3][3] = 1; W[3][3] = 1
    U[0][4] = 1; U[3][4] = 1; V[0][4] = 1; W[0][4] = 1
    U[1][5] = 1; U[3][5] = -1; V[1][5] = 1; V[3][5] = 1; W[2][5] = 1
    U[0][6] = 1; U[2][6] = -1; V[0][6] = 1; V[1][6] = 1; W[2][6] = 1
    
    return U, V, W

def _trivial_1mp(m, p):
    """Return (U, V, W) for 1xm*p matrix multiplication, rank m*p (schoolbook)."""
    n = 1
    rank = m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for j in range(m):
        for k in range(p):
            U[j][col] = 1
            V[j * p + k][col] = 1
            W[k][col] = 1
            col += 1
    return U, V, W

def _tensor_product(U1, V1, W1, U2, V2, W2):
    """Tensor product of two tensor decompositions."""
    r1 = len(U1[0]) if U1 else 0
    r2 = len(U2[0]) if U2 else 0
    rank = r1 * r2
    nm1 = len(U1)
    mp1 = len(V1)
    pn1 = len(W1)
    nm2 = len(U2)
    mp2 = len(V2)
    pn2 = len(W2)
    
    U = [[0.0] * rank for _ in range(nm1 * nm2)]
    V = [[0.0] * rank for _ in range(mp1 * mp2)]
    W = [[0.0] * rank for _ in range(pn1 * pn2)]
    
    for r1_idx in range(r1):
        for r2_idx in range(r2):
            c = r1_idx * r2 + r2_idx
            for i1 in range(nm1):
                if U1[i1][r1_idx] != 0:
                    for i2 in range(nm2):
                        if U2[i2][r2_idx] != 0:
                            U[i1 * nm2 + i2][c] = U1[i1][r1_idx] * U2[i2][r2_idx]
            for i1 in range(mp1):
                if V1[i1][r1_idx] != 0:
                    for i2 in range(mp2):
                        if V2[i2][r2_idx] != 0:
                            V[i1 * mp2 + i2][c] = V1[i1][r1_idx] * V2[i2][r2_idx]
            for i1 in range(pn1):
                if W1[i1][r1_idx] != 0:
                    for i2 in range(pn2):
                        if W2[i2][r2_idx] != 0:
                            W[i1 * pn2 + i2][c] = W1[i1][r1_idx] * W2[i2][r2_idx]
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                if abs(val - 1) > 1e-10:
                    return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, deadline, max_iterations=200):
    """Try to reduce rank by greedily removing columns."""
    import numpy as np
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    
    for iteration in range(max_iterations):
        if time.time() > deadline - 5:
            break
        found = False
        for r in range(rank):
            if time.time() > deadline - 5:
                break
            # Try removing column r
            mask = [i for i in range(rank) if i != r]
            U_test = U_np[:, mask]
            V_test = V_np[:, mask]
            W_test = W_np[:, mask]
            
            # Verify
            ok = True
            for i in range(n):
                if not ok:
                    break
                for j in range(m):
                    if not ok:
                        break
                    for k in range(p):
                        val = np.sum(U_test[i*m+j] * V_test[j*p+k] * W_test[k*n+i])
                        if abs(val - 1) > 1e-10:
                            ok = False
                            break
            
            if ok:
                U_np, V_np, W_np = U_test, V_test, W_test
                rank -= 1
                found = True
                break
        if not found:
            break
    
    return U_np.tolist(), V_np.tolist(), W_np.tolist()

def construct():
    start_time = time.time()
    deadline = 145.0
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Build T_{3,4,8} using tensor product: T_{2,4,8} ⊕ T_{1,4,8}
    U_s, V_s, W_s = _strassen_222()
    U_t, V_t, W_t = _trivial_1mp(2, 4)
    U_248, V_248, W_248 = _tensor_product(U_s, V_s, W_s, U_t, V_t, W_t)
    U_148, V_148, W_148 = _trivial_1mp(4, 8)
    
    rank_248 = len(U_248[0])
    rank_148 = len(U_148[0])
    total_rank = rank_248 + rank_148
    
    U_full = [[0.0] * total_rank for _ in range(12)]
    V_full = [[0.0] * total_rank for _ in range(32)]
    W_full = [[0.0] * total_rank for _ in range(24)]
    
    for i in range(8):
        for r in range(rank_248):
            U_full[i][r] = U_248[i][r]
    for i in range(32):
        for r in range(rank_248):
            V_full[i][r] = V_248[i][r]
    for i in range(16):
        for r in range(rank_248):
            W_full[i][r] = W_248[i][r]
    
    for i in range(4):
        for r in range(rank_148):
            U_full[8 + i][rank_248 + r] = U_148[i][r]
    for i in range(32):
        for r in range(rank_148):
            V_full[i][rank_248 + r] = V_148[i][r]
    for i in range(8):
        for r in range(rank_148):
            W_full[16 + i][rank_248 + r] = W_148[i][r]
    
    if not _verify_decomposition(U_full, V_full, W_full, n, m, p):
        # Fall back to schoolbook
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
    
    # Try to reduce rank
    if time.time() < deadline - 10:
        U_red, V_red, W_red = _try_reduce_rank(U_full, V_full, W_full, n, m, p, deadline)
        U_final, V_final, W_final = U_red, V_red, W_red
    else:
        U_final, V_final, W_final = U_full, V_full, W_full
    
    U_int = [[int(round(x)) if abs(x - round(x)) < 1e-10 else x for x in row] for row in U_final]
    V_int = [[int(round(x)) if abs(x - round(x)) < 1e-10 else x for x in row] for row in V_final]
    W_int = [[int(round(x)) if abs(x - round(x)) < 1e-10 else x for x in row] for row in W_final]
    return {"U": U_int, "V": V_int, "W": W_int}
# EVOLVE-BLOCK-END
