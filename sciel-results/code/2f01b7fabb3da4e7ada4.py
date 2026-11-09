"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # We construct a rank 18 decomposition.
    # Strategy: Split m=3 into 1 and 2.
    # Part 1: 2x1x5 (Rank 2)
    # Part 2: 2x2x5 (Rank 16) -> Split p=5 into 2 and 3.
    #   Part 2a: 2x2x2 (Rank 7, Strassen)
    #   Part 2b: 2x2x3 (Rank 9, Strassen + Rank 2)
    
    rank = 18
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # --- Part 1: 2x1x5 ---
    # Indices: i in {0,1}, j=0, k in {0..4}
    # U index: i*m + j = i*3 + 0
    # V index: j*p + k = 0*5 + k = k
    # W index: k*n + i = k*2 + i
    
    # Term 0: i=0
    U[0*3 + 0, 0] = 1.0
    V[0*5 + 0, 0] = 1.0 # V row 0 (j=0)
    # W for k=0..4, i=0
    for k in range(5):
        W[k*2 + 0, 0] = 1.0
        
    # Term 1: i=1
    U[1*3 + 0, 1] = 1.0
    V[0*5 + 0, 1] = 1.0
    for k in range(5):
        W[k*2 + 1, 1] = 1.0
        
    # --- Part 2: 2x2x5 ---
    # This part handles j in {1,2} (since j=0 was Part 1? No, m=3, indices 0,1,2).
    # My split was m=1 (j=0) and m=2 (j=1,2).
    # So Part 2 uses j=1,2.
    # U index: i*3 + j. j in {1,2}.
    # V index: j*5 + k. j in {1,2}.
    # W index: k*2 + i.
    
    # Part 2a: 2x2x2 (k in {0,1} for the first 2 columns of B part 2)
    # Wait, B is 3x5.
    # Part 1 used B rows j=0.
    # Part 2 uses B rows j=1,2.
    # The columns of B are k=0..4.
    # We split the 5 columns into k=0,1 (2 cols) and k=2,3,4 (3 cols).
    # So Part 2a handles j in {1,2} and k in {0,1}.
    # Part 2b handles j in {1,2} and k in {2,3,4}.
    
    # Part 2a: 2x2x2 Strassen
    # Local indices: i in {0,1}, j_local in {0,1} (global j=1,2), k_local in {0,1} (global k=0,1)
    # U global index: i*3 + (1 + j_local)
    # V global index: (1 + j_local)*5 + (0 + k_local)
    # W global index: (0 + k_local)*2 + i
    
    # Strassen terms for 2x2x2
    # U vectors (size 4, local j 0,1 -> global 1,2)
    # V vectors (size 4, local k 0,1 -> global 0,1)
    # W vectors (size 4, local i 0,1, k 0,1)
    
    s = 2 # offset for rank
    # Term 2 (r=2)
    U[0*3+1, 2] = 1; U[1*3+2, 2] = 1 # A00, A11
    V[1*5+0, 2] = 1; V[2*5+1, 2] = 1 # B00, B11
    W[0*2+0, 2] = 1; W[1*2+1, 2] = 1 # C00, C11
    
    # Term 3 (r=3)
    U[1*3+1, 3] = 1; U[1*3+2, 3] = 1 # A10, A11
    V[1*5+0, 3] = 1 # B00
    W[1*2+0, 3] = 1; W[1*2+1, 3] = -1 # C10, C11
    
    # Term 4 (r=4)
    U[0*3+1, 4] = 1 # A00
    V[1*5+1, 4] = 1; V[2*5+1, 4] = -1 # B01, B11
    W[0*2+1, 4] = 1; W[1*2+1, 4] = 1 # C01, C11
    
    # Term 5 (r=5)
    U[1*3+2, 5] = 1 # A11
    V[2*5+0, 5] = 1; V[1*5+0, 5] = -1 # B10, B00
    W[0*2+0, 5] = 1; W[1*2+0, 5] = 1 # C00, C10
    
    # Term 6 (r=6)
    U[0*3+1, 6] = 1; U[0*3+2, 6] = 1 # A00, A01
    V[2*5+1, 6] = 1 # B11
    W[0*2+0, 6] = -1; W[0*2+1, 6] = 1 # C00, C01
    
    # Term 7 (r=7)
    U[1*3+1, 7] = 1; U[0*3+1, 7] = -1 # A10, A00
    V[1*5+0, 7] = 1; V[1*5+1, 7] = 1 # B00, B01
    W[1*2+1, 7] = 1 # C11
    
    # Term 8 (r=8)
    U[0*3+2, 8] = 1; U[1*3+2, 8] = -1 # A01, A11
    V[2*5+0, 8] = 1; V[2*5+1, 8] = 1 # B10, B11
    W[0*2+0, 8] = 1 # C00
    
    # Part 2b: 2x2x3 (k in {2,3,4})
    # Rank 9 decomposition: Strassen (7) for first 2 cols (k=2,3) + Rank 2 for last col (k=4).
    # Local k indices: 0,1,2 corresponding to global 2,3,4.
    
    s = 9
    # Strassen for k_local 0,1 (global 2,3)
    # Term 9 (r=9)
    U[0*3+1, 9] = 1; U[1*3+2, 9] = 1
    V[1*5+2, 9] = 1; V[2*5+3, 9] = 1
    W[2*2+0, 9] = 1; W[3*2+1, 9] = 1
    
    # Term 10 (r=10)
    U[1*3+1, 10] = 1; U[1*3+2, 10] = 1
    V[1*5+2, 10] = 1
    W[3*2+0, 10] = 1; W[3*2+1, 10] = -1
    
    # Term 11 (r=11)
    U[0*3+1, 11] = 1
    V[1*5+3, 11] = 1; V[2*5+3, 11] = -1
    W[2*2+1, 11] = 1; W[3*2+1, 11] = 1
    
    # Term 12 (r=12)
    U[1*3+2, 12] = 1
    V[2*5+2, 12] = 1; V[1*5+2, 12] = -1
    W[2*2+0, 12] = 1; W[3*2+0, 12] = 1
    
    # Term 13 (r=13)
    U[0*3+1, 13] = 1; U[0*3+2, 13] = 1
    V[2*5+3, 13] = 1
    W[2*2+0, 13] = -1; W[2*2+1, 13] = 1
    
    # Term 14 (r=14)
    U[1*3+1, 14] = 1; U[0*3+1, 14] = -1
    V[1*5+2, 14] = 1; V[1*5+3, 14] = 1
    W[3*2+1, 14] = 1
    
    # Term 15 (r=15)
    U[0*3+2, 15] = 1; U[1*3+2, 15] = -1
    V[2*5+2, 15] = 1; V[2*5+3, 15] = 1
    W[2*2+0, 15] = 1
    
    # Rank 2 for k_local 2 (global 4)
    # Term 16 (r=16) i=0
    U[0*3+1, 16] = 1; U[0*3+2, 16] = 1 # A00, A01
    V[1*5+4, 16] = 1; V[2*5+4, 16] = 1 # B0, B1 (rows 1,2)
    W[4*2+0, 16] = 1 # C0
    
    # Term 17 (r=17) i=1
    U[1*3+1, 17] = 1; U[1*3+2, 17] = 1 # A10, A11
    V[1*5+4, 17] = 1; V[2*5+4, 17] = 1
    W[4*2+1, 17] = 1 # C1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
