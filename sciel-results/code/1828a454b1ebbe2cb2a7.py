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


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Strategy: Use 2x2 block structure with Kronecker product decomposition.
    # Total rank: 11 outer terms * 7 inner terms = 77.
    
    sU, sV, sW = _strassen_2x2()
    
    # Define the 11 outer terms explicitly.
    # Each term is (U_indices, V_indices, W_indices)
    # Indices are (block_row, block_col)
    
    outer_terms = []
    
    # 7 Strassen terms for cols 0,1
    # We extract the non-zero entries from sU, sV, sW
    for t in range(7):
        U_idx = [(a, b) for a in range(2) for b in range(2) if sU[a*2+b, t] != 0]
        V_idx = [(b, c) for b in range(2) for c in range(2) if sV[b*2+c, t] != 0]
        W_idx = [(a, c) for a in range(2) for c in range(2) if sW[c*2+a, t] != 0]
        
        U_coeffs = { (a,b): sU[a*2+b, t] for a,b in U_idx }
        V_coeffs = { (b,c): sV[b*2+c, t] for b,c in V_idx }
        W_coeffs = { (a,c): sW[c*2+a, t] for a,c in W_idx }
        
        outer_terms.append((U_coeffs, V_coeffs, W_coeffs))
        
    # 4 schoolbook terms for col 2
    for a in range(2):
        for b in range(2):
            U_coeffs = { (a,b): 1 }
            V_coeffs = { (b,2): 1 }
            W_coeffs = { (a,2): 1 }
            outer_terms.append((U_coeffs, V_coeffs, W_coeffs))
            
    total_rank = len(outer_terms) * 7
    U = np.zeros((n*m, total_rank), dtype=int)
    V = np.zeros((m*p, total_rank), dtype=int)
    W = np.zeros((p*n, total_rank), dtype=int)
    
    for ot_idx, (U_coeffs, V_coeffs, W_coeffs) in enumerate(outer_terms):
        for t in range(7):
            col = ot_idx * 7 + t
            
            # U part
            for (ai, bj), coeff_outer in U_coeffs.items():
                for ai2 in range(2):
                    for bj2 in range(2):
                        coeff_inner = sU[ai2*2 + bj2, t]
                        if coeff_outer != 0 and coeff_inner != 0:
                            u_row = (2*ai + ai2) * 4 + (2*bj + bj2)
                            U[u_row, col] += coeff_outer * coeff_inner
            
            # V part
            for (bj, ck), coeff_outer in V_coeffs.items():
                for bj2 in range(2):
                    for ck2 in range(2):
                        coeff_inner = sV[bj2*2 + ck2, t]
                        if coeff_outer != 0 and coeff_inner != 0:
                            v_row = (2*bj + bj2) * 6 + (2*ck + ck2)
                            V[v_row, col] += coeff_outer * coeff_inner
            
            # W part
            for (ai, ck), coeff_outer in W_coeffs.items():
                for ai2 in range(2):
                    for ck2 in range(2):
                        coeff_inner = sW[ck2*2 + ai2, t]
                        if coeff_outer != 0 and coeff_inner != 0:
                            w_row = (2*ck + ck2) * 4 + (2*ai + ai2)
                            W[w_row, col] += coeff_outer * coeff_inner
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
