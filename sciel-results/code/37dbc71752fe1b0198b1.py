"""Program for matrix multiplication 4x4 x 4x6 tensor decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2.
    
    Returns (U, V, W) where:
      U: (4, 7) - for A[i,j], row index = i*2+j
      V: (4, 7) - for B[j,k], row index = j*2+k
      W: (4, 7) - for C[i,k], row index = k*2+i  (trace convention: W[k*n+i] with n=2)
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    
    # M2 = (A10 + A11) * B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    
    # M3 = A00 * (B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    
    # M4 = A11 * (B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    
    # M5 = (A00 + A01) * B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    
    # W: C[i,k] contributions, W row = k*2 + i
    # C[0,0]: i=0,k=0 -> W[0]; C[0,1]: i=0,k=1 -> W[2]
    # C[1,0]: i=1,k=0 -> W[1]; C[1,1]: i=1,k=1 -> W[3]
    
    # C[0,0] = M1 + M4 - M5 + M7
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    
    # C[0,1] = M3 + M5
    W[2, 2] = 1; W[2, 4] = 1
    
    # C[1,0] = M2 + M4
    W[1, 1] = 1; W[1, 3] = 1
    
    # C[1,1] = M1 - M2 + M3 + M6
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    
    return U, V, W


def _build_outer_terms():
    """
    Build the outer terms for 2x2 * 2x3 -> 2x3 block multiplication.
    7 terms from Strassen (cols 0,1) + 4 terms schoolbook (col 2).
    Each term: dict with 'U', 'V', 'W' mapping (block_row, block_col) -> coeff.
    """
    sU, sV, sW = _strassen_2x2()
    terms = []
    
    # 7 Strassen terms for the 2x2 * 2x2 part (block cols 0,1)
    for t in range(7):
        U_d, V_d, W_d = {}, {}, {}
        for a in range(2):
            for b in range(2):
                c = sU[a*2+b, t]
                if c: U_d[(a,b)] = U_d.get((a,b), 0) + c
        for b in range(2):
            for c in range(2):
                cv = sV[b*2+c, t]
                if cv: V_d[(b,c)] = V_d.get((b,c), 0) + cv
        for a in range(2):
            for c in range(2):
                cw = sW[c*2+a, t]
                if cw: W_d[(a,c)] = W_d.get((a,c), 0) + cw
        terms.append({'U': U_d, 'V': V_d, 'W': W_d})
    
    # 4 schoolbook terms for block col 2
    for a in range(2):
        for b in range(2):
            terms.append({'U': {(a,b): 1}, 'V': {(b,2): 1}, 'W': {(a,2): 1}})
    
    return terms


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    outer_terms = _build_outer_terms()
    sU, sV, sW = _strassen_2x2()
    R = len(outer_terms) * 7  # 77
    
    U = np.zeros((n*m, R), dtype=int)
    V = np.zeros((m*p, R), dtype=int)
    W = np.zeros((p*n, R), dtype=int)
    
    for oi, ot in enumerate(outer_terms):
        for t in range(7):
            col = oi * 7 + t
            for (ai, bj), cu in ot['U'].items():
                if cu == 0: continue
                for ai2 in range(2):
                    for bj2 in range(2):
                        ci = sU[ai2*2 + bj2, t]
                        if ci:
                            U[(2*ai+ai2)*4 + (2*bj+bj2), col] += cu * ci
            for (bj, ck), cv in ot['V'].items():
                if cv == 0: continue
                for bj2 in range(2):
                    for ck2 in range(2):
                        ci = sV[bj2*2 + ck2, t]
                        if ci:
                            V[(2*bj+bj2)*6 + (2*ck+ck2), col] += cv * ci
            for (ai, ck), cw in ot['W'].items():
                if cw == 0: continue
                for ai2 in range(2):
                    for ck2 in range(2):
                        ci = sW[ck2*2 + ai2, t]
                        if ci:
                            W[(2*ck+ck2)*4 + (2*ai+ai2), col] += cw * ci
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
