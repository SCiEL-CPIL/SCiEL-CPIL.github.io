"""Matrix multiplication 2x4x6 - low rank decomposition via Strassen blocks."""
import numpy as np
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    deadline = time.time() + 140
    
    # Use 6 Strassen blocks (2x2x2) to decompose M_{2,4,6}.
    # j = 2*a + b, k = 2*c + d, with a in {0,1}, b in {0,1}, c in {0,1,2}, d in {0,1}
    # Each block (a,c) is a 2x2x2 tensor, decomposed with rank 7 (Strassen).
    # Total rank = 6 * 7 = 42.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2x2:
    # Uf[i, b, f], Vf[b, d, f], Wf[d, i, f]
    Uf = np.zeros((2, 2, 7))
    Vf = np.zeros((2, 2, 7))
    Wf = np.zeros((2, 2, 7))
    
    # M1: (A00+A11)(B00+B11) -> C00+=1, C11+=1
    Uf[0,0,0] = 1; Uf[1,1,0] = 1
    Vf[0,0,0] = 1; Vf[1,1,0] = 1
    Wf[0,0,0] = 1; Wf[1,1,0] = 1
    
    # M2: (A10+A11)B00 -> C10+=1, C11-=1
    Uf[1,0,1] = 1; Uf[1,1,1] = 1
    Vf[0,0,1] = 1
    Wf[0,1,1] = 1; Wf[1,1,1] = -1
    
    # M3: A00(B01-B11) -> C01+=1, C11+=1
    Uf[0,0,2] = 1
    Vf[0,1,2] = 1; Vf[1,1,2] = -1
    Wf[1,0,2] = 1; Wf[1,1,2] = 1
    
    # M4: A11(B10-B00) -> C00+=1, C10+=1
    Uf[1,1,3] = 1
    Vf[1,0,3] = 1; Vf[0,0,3] = -1
    Wf[0,0,3] = 1; Wf[0,1,3] = 1
    
    # M5: (A00+A01)B11 -> C00-=1, C01+=1
    Uf[0,0,4] = 1; Uf[0,1,4] = 1
    Vf[1,1,4] = 1
    Wf[0,0,4] = -1; Wf[1,0,4] = 1
    
    # M6: (A10-A00)(B00+B01) -> C11+=1
    Uf[1,0,5] = 1; Uf[0,0,5] = -1
    Vf[0,0,5] = 1; Vf[0,1,5] = 1
    Wf[1,1,5] = 1
    
    # M7: (A01-A11)(B10+B11) -> C00+=1
    Uf[0,1,6] = 1; Uf[1,1,6] = -1
    Vf[1,0,6] = 1; Vf[1,1,6] = 1
    Wf[0,0,6] = 1
    
    # Assemble
    col = 0
    for a in range(2):
        for c in range(3):
            for f in range(7):
                for i in range(2):
                    for b in range(2):
                        if Uf[i, b, f] != 0:
                            U[i * m + (2*a + b)][col] = Uf[i, b, f]
                for b in range(2):
                    for d in range(2):
                        if Vf[b, d, f] != 0:
                            V[(2*a + b) * p + (2*c + d)][col] = Vf[b, d, f]
                for d in range(2):
                    for i in range(2):
                        if Wf[d, i, f] != 0:
                            W[(2*c + d) * n + i][col] = Wf[d, i, f]
                col += 1
    
    # Verify
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1
                actual = 0
                for c in range(rank):
                    actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                max_err = max(max_err, abs(actual - expected))
    
    if max_err > 1e-10:
        # Fallback to schoolbook
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
