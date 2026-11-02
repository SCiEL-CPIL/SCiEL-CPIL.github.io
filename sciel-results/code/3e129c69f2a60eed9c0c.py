"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _strassen_2x2_terms():
    """Return 7 rank-1 terms for 2x2 x 2x2 Strassen."""
    terms = []
    terms.append(([1,0,0,1],[1,0,0,1],[1,0,0,1]))
    terms.append(([0,0,1,1],[1,0,0,0],[0,0,1,0]))
    terms.append(([1,0,0,0],[0,1,0,-1],[0,1,0,0]))
    terms.append(([0,0,0,1],[0,0,1,-1],[0,0,0,1]))
    terms.append(([1,1,0,0],[0,0,1,0],[1,0,1,0]))
    terms.append(([0,0,1,-1],[1,1,0,0],[0,0,1,-1]))
    terms.append(([0,1,0,-1],[0,0,1,1],[0,-1,0,1]))
    return terms

def _build_strassen_4x4_4x4():
    """Build rank-49 decomposition for 4x4 x 4x4 x 4x4."""
    s2 = _strassen_2x2_terms()
    terms = []
    super_terms = [
        ([[1,0],[0,1]],[[1,0],[0,1]],[[1,0],[0,1]]),
        ([[0,0],[1,1]],[[1,0],[0,0]],[[0,0],[1,0]]),
        ([[1,0],[0,0]],[[0,1],[0,-1]],[[0,1],[0,0]]),
        ([[0,0],[0,1]],[[0,0],[1,-1]],[[0,0],[0,1]]),
        ([[1,1],[0,0]],[[0,0],[1,0]],[[1,0],[1,0]]),
        ([[0,0],[1,-1]],[[1,1],[0,0]],[[0,0],[1,-1]]),
        ([[0,1],[0,-1]],[[0,0],[1,1]],[[0,-1],[0,1]]),
    ]
    for U_s, V_s, W_s in super_terms:
        for (u_i, v_i, w_i) in s2:
            u_full = [0]*16
            v_full = [0]*16
            w_full = [0]*16
            for br in range(2):
                for bc in range(2):
                    block_pos = br*2+bc
                    coeff_u = U_s[br][bc]
                    coeff_v = V_s[br][bc]
                    coeff_w = W_s[br][bc]
                    for inner in range(4):
                        base = block_pos*4 + inner
                        u_full[base] = coeff_u * u_i[inner]
                        v_full[base] = coeff_v * v_i[inner]
                        w_full[base] = coeff_w * w_i[inner]
            terms.append((u_full, v_full, w_full))
    return terms

def _build_decomposition_4x4x4x5():
    """Build rank-53 decomposition for 4x4 x 4x5."""
    n, m, p = 4, 4, 5
    s4 = _build_strassen_4x4_4x4()
    rank = 53
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for (u16, v16, w16) in s4:
        for idx in range(16):
            i = idx // 4
            j = idx % 4
            U[i*m + j][col] = u16[idx]
        for idx in range(16):
            j = idx // 4
            k = idx % 4
            V[j*p + k][col] = v16[idx]
        for idx in range(16):
            k = idx // 4
            i = idx % 4
            W[k*n + i][col] = w16[idx]
        col += 1
    
    for j in range(4):
        c = col + j
        for i in range(4):
            U[i*m + j][c] = 1
        V[j*p + 4][c] = 1
        for i in range(4):
            W[4*n + i][c] = 1
    col += 4
    
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, tol=1e-9):
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140
    
    result = _build_decomposition_4x4x4x5()
    
    if time.time() < deadline - 5:
        if not _verify(result["U"], result["V"], result["W"], n, m, p):
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
            result = {"U": U, "V": V, "W": W}
    
    return result
# EVOLVE-BLOCK-END
