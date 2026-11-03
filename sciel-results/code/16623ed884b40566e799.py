"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    deadline = time.time() + 110
    
    # Strassen-based rank-112 decomposition for 4x4 * 4x8
    # Block structure: 2x2 blocks of 2x2 (A) and 2x4 (B)
    # For each A-block (2x2) * B-block (2x4), split B into two 2x2 halves
    # Use 7 Strassen terms per 2x2*2x2, so 14 per 2x2*2x4
    # 4 A-blocks * 2 B-half-blocks * 2 sub-halves * 7 = 112
    
    rank = 112
    U = [[0]*rank for _ in range(n*m)]  # 16 x 112
    V = [[0]*rank for _ in range(m*p)]  # 32 x 112
    W = [[0]*rank for _ in range(p*n)]  # 32 x 112
    
    # Strassen 2x2*2x2 terms: (A_coeffs, X_coeffs, C_coeffs)
    # Local 2x2 indices: [0,0],[0,1],[1,0],[1,1] -> flat [0,1,2,3]
    # C00 = P1+P4-P5+P7, C01 = P2+P5, C10 = P3+P6, C11 = P1-P2+P3+P4
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # P1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),   # P2
        ([1, 0, 0, 0], [1, 0, 0, -1], [0, 0, 1, 1]),   # P3
        ([0, 0, 0, 1], [0, 0, 0, 1], [1, 0, 0, 1]),    # P4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # P5
        ([0, -1, 1, 0], [1, 1, 0, 0], [0, 0, 1, 0]),   # P6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),   # P7
    ]
    
    col = 0
    for ai in range(2):      # A row block
        for aj in range(2):  # A col block (= B row block)
            for half in range(2):  # B col half (0=cols 0-3, 1=cols 4-7)
                for sub in range(2):  # sub-half within B (0=first 2 cols, 1=next 2)
                    for t in range(7):
                        a_c = strassen_terms[t][0]
                        x_c = strassen_terms[t][1]
                        c_c = strassen_terms[t][2]
                        
                        # Fill U: A[ai*2+di, aj*2+dj]
                        for di in range(2):
                            for dj in range(2):
                                coeff = a_c[di*2+dj]
                                if coeff != 0:
                                    u_idx = (ai*2+di)*4 + (aj*2+dj)
                                    U[u_idx][col] = coeff
                        
                        # Fill V: B[aj*2+dj, half*4+sub*2+dk]
                        for dj in range(2):
                            for dk in range(2):
                                coeff = x_c[dj*2+dk]
                                if coeff != 0:
                                    v_idx = (aj*2+dj)*8 + (half*4+sub*2+dk)
                                    V[v_idx][col] = coeff
                        
                        # Fill W: C[ai*2+di, half*4+sub*2+dk], W[k*n+i]
                        for di in range(2):
                            for dk in range(2):
                                coeff = c_c[di*2+dk]
                                if coeff != 0:
                                    k = half*4+sub*2+dk
                                    i = ai*2+di
                                    w_idx = k*4 + i
                                    W[w_idx][col] = coeff
                        
                        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
