"""Program for matrix multiplication 4x4 x 4x6 tensor decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2.
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
    
    # W: C[i,k] contributions, W row = k*2 + i (trace convention)
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
    
    # We decompose the 4x4x6 multiplication using a 2x2 block structure.
    # A (4x4) -> 2x2 blocks of 2x2
    # B (4x6) -> 2x3 blocks of 2x2
    # C (4x6) -> 2x3 blocks of 2x2
    
    # The outer multiplication is 2x2 * 2x3 -> 2x3.
    # This can be split into:
    # 1. Multiplication of the first 2 columns of B (2x2 block matrix mult): Rank 7 (Strassen)
    # 2. Multiplication of the last 1 column of B (2x1 block matrix mult): Rank 4 (Schoolbook)
    # Total outer rank: 11.
    # Each outer term corresponds to a 2x2 * 2x2 -> 2x2 multiplication, which has rank 7 (Strassen).
    # Total rank: 11 * 7 = 77.
    
    sU, sV, sW = _strassen_2x2()
    
    # Define the outer terms.
    # Each term is a tuple (U_outer, V_outer, W_outer) where each is a dict mapping block indices to coefficients.
    # Block indices:
    # A: (ai, aj) with ai, aj in {0, 1}
    # B: (bj, bk) with bj in {0, 1}, bk in {0, 1, 2}
    # C: (ci, ck) with ci in {0, 1}, ck in {0, 1, 2}
    
    outer_terms = []
    
    # 1. Strassen terms for columns 0, 1
    # These compute C[:, 0:2] = A[:, 0:2] * B[:, 0:2]
    # The 7 Strassen terms for 2x2 matrix multiplication.
    # We need to map the Strassen U,V,W (which are for 2x2*2x2->2x2) to the block structure.
    # sU[ai*2+aj, t] is the coefficient for A[ai, aj] in term t.
    # sV[bj*2+bk, t] is the coefficient for B[bj, bk] in term t.
    # sW[ck*2+ci, t] is the coefficient for C[ci, ck] in term t.
    
    for t in range(7):
        U_dict = {}
        V_dict = {}
        W_dict = {}
        
        for ai in range(2):
            for aj in range(2):
                coeff = sU[ai*2+aj, t]
                if coeff != 0:
                    U_dict[(ai, aj)] = U_dict.get((ai, aj), 0) + coeff
        
        for bj in range(2):
            for bk in range(2):
                coeff = sV[bj*2+bk, t]
                if coeff != 0:
                    V_dict[(bj, bk)] = V_dict.get((bj, bk), 0) + coeff
        
        for ci in range(2):
            for ck in range(2):
                coeff = sW[ck*2+ci, t]
                if coeff != 0:
                    W_dict[(ci, ck)] = W_dict.get((ci, ck), 0) + coeff
        
        outer_terms.append((U_dict, V_dict, W_dict))
        
    # 2. Schoolbook terms for column 2
    # C[:, 2] = A[:, 0] * B[:, 2] + A[:, 1] * B[:, 1] ? No.
    # C[ci, 2] = sum_bj A[ci, bj] B[bj, 2]
    # So for each ci in {0,1} and bj in {0,1}, we have a product A[ci, bj] * B[bj, 2].
    # This results in 4 terms.
    
    for ci in range(2):
        for bj in range(2):
            U_dict = {(ci, bj): 1}
            V_dict = {(bj, 2): 1}
            W_dict = {(ci, 2): 1}
            outer_terms.append((U_dict, V_dict, W_dict))
            
    total_rank = len(outer_terms) * 7
    U = np.zeros((n*m, total_rank), dtype=int)
    V = np.zeros((m*p, total_rank), dtype=int)
    W = np.zeros((p*n, total_rank), dtype=int)
    
    for ot_idx, (U_outer, V_outer, W_outer) in enumerate(outer_terms):
        for t in range(7):
            col = ot_idx * 7 + t
            
            # U part
            for (ai, aj), coeff_outer in U_outer.items():
                for ai2 in range(2):
                    for aj2 in range(2):
                        coeff_inner = sU[ai2*2 + aj2, t]
                        if coeff_outer != 0 and coeff_inner != 0:
                            u_row = (2*ai + ai2) * 4 + (2*aj + aj2)
                            U[u_row, col] += coeff_outer * coeff_inner
            
            # V part
            for (bj, bk), coeff_outer in V_outer.items():
                for bj2 in range(2):
                    for bk2 in range(2):
                        coeff_inner = sV[bj2*2 + bk2, t]
                        if coeff_outer != 0 and coeff_inner != 0:
                            v_row = (2*bj + bj2) * 6 + (2*bk + bk2)
                            V[v_row, col] += coeff_outer * coeff_inner
            
            # W part
            for (ci, ck), coeff_outer in W_outer.items():
                for ci2 in range(2):
                    for ck2 in range(2):
                        coeff_inner = sW[ck2*2 + ci2, t]
                        if coeff_outer != 0 and coeff_inner != 0:
                            # Trace convention: W[k*n+i] -> W[ (2*ck+ck2)*4 + (2*ci+ci2) ]
                            w_row = (2*ck + ck2) * 4 + (2*ci + ci2)
                            W[w_row, col] += coeff_outer * coeff_inner
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
