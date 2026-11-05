import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2_coeffs():
    """
    Returns list of 7 tuples (U_coeffs, V_coeffs, W_coeffs)
    U_coeffs: dict {(u, v): val} for A_block
    V_coeffs: dict {(v, w): val} for B_block
    W_coeffs: dict {(w, u): val} for C_block contribution
    Indices u,v,w in {0,1}
    """
    # P1 = (A00+A11)(B00+B11)
    p1 = ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}) 
    # Correction: W coeffs must match C = ...
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Let's define W contribution for each P to C
    # P1: C00 += 1, C11 += 1
    # P2: C10 += 1, C11 += -1
    # P3: C01 += 1, C11 += 1
    # P4: C00 += 1, C10 += 1
    # P5: C00 += -1, C01 += 1
    # P6: C11 += 1
    # P7: C00 += 1
    
    coeffs = []
    
    # P1
    coeffs.append((
        {(0,0):1, (1,1):1}, 
        {(0,0):1, (1,1):1}, 
        {(0,0):1, (1,1):1}
    ))
    
    # P2 = (A10+A11)B00
    coeffs.append((
        {(1,0):1, (1,1):1}, 
        {(0,0):1}, 
        {(0,1):1, (1,1):-1} # C10 += 1, C11 -= 1. W index (w,u). C10 is u=1,w=0? No.
                             # C_{u,w}. C10 is u=1, w=0. So W(0,1) += 1.
                             # C11 is u=1, w=1. So W(1,1) -= 1.
    ))
    # Fix W index mapping: W[k*n+i] corresponds to C[i,k].
    # Local C[u,w]. Global C[i,k]. i=2I+u, k=2J+w.
    # W index is k*6+i.
    # So W contribution at (w, u) goes to C[u, w].
    # So for C10 (u=1, w=0), we need W(0, 1).
    # For C11 (u=1, w=1), we need W(1, 1).
    
    # Let's re-verify P2 W coeffs:
    # C10 += 1 => W(0,1) += 1
    # C11 -= 1 => W(1,1) -= 1
    coeffs[1] = ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1})

    # P3 = A00(B01-B11)
    coeffs.append((
        {(0,0):1}, 
        {(0,1):1, (1,1):-1}, 
        {(1,0):1, (1,1):1} # C01 += 1 => W(1,0) += 1. C11 += 1 => W(1,1) += 1.
    ))

    # P4 = A11(B10-B00)
    coeffs.append((
        {(1,1):1}, 
        {(1,0):1, (0,0):-1}, 
        {(0,0):1, (0,1):1} # C00 += 1 => W(0,0) += 1. C10 += 1 => W(0,1) += 1.
    ))

    # P5 = (A00+A01)B11
    coeffs.append((
        {(0,0):1, (0,1):1}, 
        {(1,1):1}, 
        {(0,0):-1, (0,1):1} # C00 -= 1 => W(0,0) -= 1. C01 += 1 => W(1,0) += 1.
    ))
    # Fix P5 W: C01 is u=0, w=1. W(1,0).
    coeffs[4] = ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1})

    # P6 = (A10-A00)(B00+B01)
    coeffs.append((
        {(1,0):1, (0,0):-1}, 
        {(0,0):1, (0,1):1}, 
        {(1,1):1} # C11 += 1 => W(1,1) += 1.
    ))

    # P7 = (A01-A11)(B10+B11)
    coeffs.append((
        {(0,1):1, (1,1):-1}, 
        {(1,0):1, (1,1):1}, 
        {(0,0):1} # C00 += 1 => W(0,0) += 1.
    ))
    
    return coeffs

def _build_block_strassen():
    n, m, p = 6, 6, 6
    rank = 189
    U = np.zeros((n*m, rank), dtype=object)
    V = np.zeros((m*p, rank), dtype=object)
    W = np.zeros((p*n, rank), dtype=object)
    
    strassen_coeffs = _strassen_2x2_coeffs()
    
    col = 0
    for I in range(3):
        for J in range(3):
            for K in range(3):
                for s in range(7):
                    u_c, v_c, w_c = strassen_coeffs[s]
                    
                    # U: A block (I, K). Global i=2I+u, j=2K+v
                    for (u, v), val in u_c.items():
                        i_global = 2*I + u
                        j_global = 2*K + v
                        U[i_global*m + j_global, col] += val
                        
                    # V: B block (K, J). Global j=2K+v, k=2J+w
                    for (v, w), val in v_c.items():
                        j_global = 2*K + v
                        k_global = 2*J + w
                        V[j_global*p + k_global, col] += val
                        
                    # W: C block (I, J). Global k=2J+w, i=2I+u
                    for (w, u), val in w_c.items():
                        k_global = 2*J + w
                        i_global = 2*I + u
                        W[k_global*n + i_global, col] += val
                        
                    col += 1
                    
    return U, V, W

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
                    s += Ua[i * m + j, c] * Va[j * p + k, c] * Wa[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS
    try:
        U, V, W = _build_block_strassen()
        # Quick verification on a subset or full if fast enough
        # 189 rank, 216 entries. 216 * 189 ops ~ 40k. Very fast.
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Fallback to schoolbook if verification fails
            pass
    except Exception:
        pass
    
    # Schoolbook fallback
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
