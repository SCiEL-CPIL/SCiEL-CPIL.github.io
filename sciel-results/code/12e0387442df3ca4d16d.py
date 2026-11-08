import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    R = 88

    # Initialize factor matrices as lists of lists to match interface
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]

    c = 0

    # Part A: i=0 slice (1x4x8)
    # Trivial decomposition: rank 32
    # U[j, c] = 1, V[j*p+k, c] = 1, W[k*n+0, c] = 1
    for j in range(m):
        for k in range(p):
            U[j][c] = 1.0
            V[j * p + k][c] = 1.0
            W[k * n][c] = 1.0
            c += 1
            if time.time() - t0 > 145:
                return {"U": U, "V": V, "W": W}

    # Part B: i=1,2 slice (2x4x8)
    # Decomposed into 8 blocks of 2x2x2 using Strassen
    # Strassen factors for 2x2x2
    # Local indices for U (2x2): (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # Local indices for V (2x2): (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # Local indices for W (2x2): (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    
    # Strassen 7 multiplications
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Factors U_t[l], V_t[l], W_t[l] for t in 0..6, l in 0..3
    # l corresponds to (i_local, j_local) for U, (j_local, k_local) for V, (k_local, i_local) for W
    # i_local, j_local, k_local in {0,1}
    
    def idx2(l):
        return (l // 2, l % 2)

    # Pre-calculate non-zero entries for Strassen factors
    # U factors: A components
    # V factors: B components
    # W factors: C components
    
    # t=0: (A00+A11)(B00+B11)
    # U: A00, A11 -> l=0, l=3
    # V: B00, B11 -> l=0, l=3
    # W: C00+=, C01+=, C10+=, C11+= -> l=0,1,2,3 all get +1
    # Actually, W contribution is to all 4 blocks with coefficient 1.
    
    # Let's just use the explicit matrices from the previous code but generated differently.
    
    # SU[t][l]
    # t=0: A00(1), A11(1) -> l=0:1, l=3:1
    # t=1: A10(1), A11(1) -> l=2:1, l=3:1
    # t=2: A00(1) -> l=0:1
    # t=3: A11(1) -> l=3:1
    # t=4: A00(1), A01(1) -> l=0:1, l=1:1
    # t=5: A10(1), A00(-1) -> l=2:1, l=0:-1
    # t=6: A01(1), A11(-1) -> l=1:1, l=3:-1
    
    SU = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    # SV[t][l]
    # t=0: B00(1), B11(1) -> l=0:1, l=3:1
    # t=1: B00(1) -> l=0:1
    # t=2: B01(1), B11(-1) -> l=1:1, l=3:-1
    # t=3: B10(1), B00(-1) -> l=2:1, l=0:-1
    # t=4: B11(1) -> l=3:1
    # t=5: B00(1), B01(1) -> l=0:1, l=1:1
    # t=6: B10(1), B11(1) -> l=2:1, l=3:1
    
    SV = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    # SW[t][l]
    # t=0: C00(1), C01(1), C10(1), C11(1) -> l=0:1, l=1:1, l=2:1, l=3:1
    # t=1: C00(1), C01(1) -> l=0:1, l=1:1 (Wait, M2=(A10+A11)B00. B00 is top-left. C = M2 * Identity? No.
    # Standard Strassen:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 + M7
    
    # Let's re-verify SW based on this formula.
    # M1 contributes to C00(+), C01(+), C10(+), C11(+) -> [1,1,1,1]
    # M2 contributes to C10(+) -> [0,0,1,0]
    # M3 contributes to C01(+), C11(-) -> [0,1,0,-1]
    # M4 contributes to C00(-), C10(+) -> [-1,0,1,0]
    # M5 contributes to C00(+), C01(+) -> [1,1,0,0]
    # M6 contributes to C11(+) -> [0,0,0,1]
    # M7 contributes to C00(-), C11(+) -> [-1,0,0,1]
    
    # Wait, the previous code had:
    # SW = [[1,0,0,1],[0,0,1,-1],[0,1,0,1],[1,0,1,0],[-1,1,0,0],[0,0,0,1],[1,0,0,0]]
    # This does not match my derived SW.
    # Let's check the previous code's Strassen implementation.
    # It might be a different permutation or variant.
    # Given the previous code scored 1.0909 (valid), I should stick to its specific factor definitions to ensure correctness, even if they look different from standard textbook Strassen.
    # The factors in the previous code:
    # SU = [[1,0,0,1],[0,0,1,1],[1,0,0,0],[0,0,0,1],[1,1,0,0],[-1,0,1,0],[0,1,0,-1]]
    # SV = [[1,0,0,1],[1,0,0,0],[0,1,0,-1],[-1,0,1,0],[0,0,0,1],[1,1,0,0],[0,0,1,1]]
    # SW = [[1,0,0,1],[0,0,1,-1],[0,1,0,1],[1,0,1,0],[-1,1,0,0],[0,0,0,1],[1,0,0,0]]
    
    # I will use these exact factors.
    
    # Block loop for Part B
    # i_offset = 1
    # j_block 0: j=0,1. j_block 1: j=2,3
    # k_block 0: k=0,1. k_block 1: k=2,3. k_block 2: k=4,5. k_block 3: k=6,7
    
    for jb in range(2):
        jo = jb * 2
        for kb in range(4):
            ko = kb * 2
            for t in range(7):
                # Update U
                for l in range(4):
                    val = SU[t][l]
                    if val != 0:
                        li, lj = idx2(l)
                        gi = 1 + li
                        gj = jo + lj
                        U[gi * m + gj][c] += float(val)
                
                # Update V
                for l in range(4):
                    val = SV[t][l]
                    if val != 0:
                        lj, lk = idx2(l)
                        gj = jo + lj
                        gk = ko + lk
                        V[gj * p + gk][c] += float(val)
                
                # Update W
                for l in range(4):
                    val = SW[t][l]
                    if val != 0:
                        lk, li = idx2(l)
                        gk = ko + lk
                        gi = 1 + li
                        W[gk * n + gi][c] += float(val)
                
                c += 1
                if time.time() - t0 > 145:
                    return {"U": U, "V": V, "W": W}
            
            if time.time() - t0 > 145:
                return {"U": U, "V": V, "W": W}
        
        if time.time() - t0 > 145:
            return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
