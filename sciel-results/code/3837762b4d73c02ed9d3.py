import time
import numpy as np

def construct():
    n, m, p = 3, 4, 4
    rank = 44
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    
    # Helper to add a term
    def add_term(a_coeffs, b_coeffs, c_coeffs):
        nonlocal col
        # a_coeffs: dict (i, j) -> coeff
        # b_coeffs: dict (j, k) -> coeff
        # c_coeffs: dict (k, i) -> coeff  # Note: W index is k*n+i
        
        for (i, j), coeff in a_coeffs.items():
            if coeff != 0:
                U[i*m + j, col] += coeff
        for (j, k), coeff in b_coeffs.items():
            if coeff != 0:
                V[j*p + k, col] += coeff
        for (k, i), coeff in c_coeffs.items():
            if coeff != 0:
                W[k*n + i, col] += coeff
        col += 1
        
    # Strassen 2x2x2x2
    def strassen_2x2(a_base, b_base, c_base):
        # a_base: (i, j) offset in A
        # b_base: (j, k) offset in B
        # c_base: (k, i) offset in C
        # Local indices 0,1 for rows/cols
        
        # M1 = (A00+A11)(B00+B11)
        add_term({(a_base[0], a_base[1]): 1, (a_base[0]+1, a_base[1]+1): 1}, 
                 {(b_base[0], b_base[1]): 1, (b_base[0]+1, b_base[1]+1): 1},
                 {(c_base[0], c_base[1]): 1, (c_base[0]+1, c_base[1]): 1, 
                  (c_base[0], c_base[1]+1): 1, (c_base[0]+1, c_base[1]+1): 1})
        # Wait, C indices in W are (k, i).
        # C00 is (0,0) in local C. Global C index (i=0, k=0) -> W[0*n+0]
        # C01 is (0,1) in local C. Global C index (i=0, k=1) -> W[1*n+0]
        # C10 is (1,0) in local C. Global C index (i=1, k=0) -> W[0*n+1]
        # C11 is (1,1) in local C. Global C index (i=1, k=1) -> W[1*n+1]
        
        # Let's re-verify W index mapping.
        # Task says: W[k*n+i, r]
        # So for C[i, k], the W index is k*n + i.
        
        # M1 contributes to C00, C11.
        # C00: i=0, k=0. W[0*n+0]
        # C11: i=1, k=1. W[1*n+1]
        # My add_term above for c_coeffs was wrong for M1.
        # M1 = (A00+A11)(B00+B11)
        # C00 += M1
        # C11 += M1
        
        # Let's rewrite the Strassen terms carefully.
        
        pass

    # I will implement the 44 terms explicitly to avoid bugs in the helper logic.
    
    # Block 1: C_TL = A_TL B_11 + A_TR B_21
    # A_TL is at (0,0) in A. B_11 is at (0,0) in B.
    # A_TR is at (0,2) in A. B_21 is at (2,0) in B.
    # C_TL is at (0,0) in C.
    
    # Strassen for A_TL * B_11
    # A indices: (0,0), (0,1), (1,0), (1,1)
    # B indices: (0,0), (0,1), (1,0), (1,1)
    # C indices: (0,0), (0,1), (1,0), (1,1)
    
    # M1
    add_term({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1})
    # M2
    add_term({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,0): 1, (1,0): 1})
    # M3
    add_term({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1})
    # M4
    add_term({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1})
    # M5
    add_term({(0,0): 1, (1,0): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1})
    # M6
    add_term({(0,1): 1, (1,0): -1}, {(0,0): 1, (1,0): 1}, {(1,1): 1})
    # M7
    add_term({(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}, {(0,0): 1, (1,1): -1})
    
    # Strassen for A_TR * B_21
    # A indices: (0,2), (0,3), (1,2), (1,3)
    # B indices: (2,0), (2,1), (3,0), (3,1)
    # C indices: (0,0), (0,1), (1,0), (1,1) (Same C block)
    
    # M1
    add_term({(0,2): 1, (1,3): 1}, {(2,0): 1, (3,1): 1}, {(0,0): 1, (1,1): 1})
    # M2
    add_term({(1,2): 1, (1,3): 1}, {(2,0): 1}, {(0,0): 1, (1,0): 1})
    # M3
    add_term({(0,2): 1}, {(2,1): 1, (3,1): -1}, {(0,1): 1, (1,1): 1})
    # M4
    add_term({(1,3): 1}, {(3,0): 1, (2,0): -1}, {(0,0): 1, (1,0): 1})
    # M5
    add_term({(0,2): 1, (1,2): 1}, {(3,1): 1}, {(0,0): -1, (0,1): 1})
    # M6
    add_term({(0,3): 1, (1,2): -1}, {(2,0): 1, (3,0): 1}, {(1,1): 1})
    # M7
    add_term({(0,3): 1, (1,3): -1}, {(2,1): 1, (3,1): 1}, {(0,0): 1, (1,1): -1})
    
    # Block 2: C_TR = A_TL B_12 + A_TR B_22
    # A_TL: (0,0)..(1,1)
    # B_12: (0,2)..(1,3)
    # C_TR: (0,2)..(1,3) -> W indices (2,0), (3,0), (2,1), (3,1)
    # C[0,2] -> W[2*3+0] = W[6]
    # C[0,3] -> W[3*3+0] = W[9]
    # C[1,2] -> W[2*3+1] = W[7]
    # C[1,3] -> W[3*3+1] = W[10]
    
    # Strassen for A_TL * B_12
    # M1
    add_term({(0,0): 1, (1,1): 1}, {(0,2): 1, (1,3): 1}, {(2,0): 1, (3,1): 1})
    # M2
    add_term({(1,0): 1, (1,1): 1}, {(0,2): 1}, {(2,0): 1, (2,1): 1})
    # M3
    add_term({(0,0): 1}, {(0,3): 1, (1,3): -1}, {(2,1): 1, (3,1): 1})
    # M4
    add_term({(1,1): 1}, {(1,2): 1, (0,2): -1}, {(2,0): 1, (2,1): 1})
    # M5
    add_term({(0,0): 1, (1,0): 1}, {(1,3): 1}, {(2,0): -1, (2,1): 1})
    # M6
    add_term({(0,1): 1, (1,0): -1}, {(0,2): 1, (1,2): 1}, {(3,1): 1})
    # M7
    add_term({(0,1): 1, (1,1): -1}, {(0,3): 1, (1,3): 1}, {(2,0): 1, (3,1): -1})
    
    # Strassen for A_TR * B_22
    # A_TR: (0,2)..(1,3)
    # B_22: (2,2)..(3,3)
    # C_TR: (0,2)..(1,3)
    
    # M1
    add_term({(0,2): 1, (1,3): 1}, {(2,2): 1, (3,3): 1}, {(2,0): 1, (3,1): 1})
    # M2
    add_term({(1,2): 1, (1,3): 1}, {(2,2): 1}, {(2,0): 1, (2,1): 1})
    # M3
    add_term({(0,2): 1}, {(2,3): 1, (3,3): -1}, {(2,1): 1, (3,1): 1})
    # M4
    add_term({(1,3): 1}, {(3,2): 1, (2,2): -1}, {(2,0): 1, (2,1): 1})
    # M5
    add_term({(0,2): 1, (1,2): 1}, {(3,3): 1}, {(2,0): -1, (2,1): 1})
    # M6
    add_term({(0,3): 1, (1,2): -1}, {(2,2): 1, (3,2): 1}, {(3,1): 1})
    # M7
    add_term({(0,3): 1, (1,3): -1}, {(2,3): 1, (3,3): 1}, {(2,0): 1, (3,1): -1})
    
    # Block 3: C_BL = A_BL B_11 + A_BR B_21
    # A_BL: (2,0), (2,1)
    # B_11: (0,0)..(1,1)
    # C_BL: (2,0), (2,1) -> W indices (0,2), (1,2) -> W[0*3+2]=W[2], W[1*3+2]=W[5]
    
    # Schoolbook A_BL * B_11
    # C[2,0] = A[2,0]B[0,0] + A[2,1]B[1,0]
    # C[2,1] = A[2,0]B[0,1] + A[2,1]B[1,1]
    
    # T1: A[2,0]B[0,0] -> C[2,0]
    add_term({(2,0): 1}, {(0,0): 1}, {(0,2): 1})
    # T2: A[2,1]B[1,0] -> C[2,0]
    add_term({(2,1): 1}, {(1,0): 1}, {(0,2): 1})
    # T3: A[2,0]B[0,1] -> C[2,1]
    add_term({(2,0): 1}, {(0,1): 1}, {(1,2): 1})
    # T4: A[2,1]B[1,1] -> C[2,1]
    add_term({(2,1): 1}, {(1,1): 1}, {(1,2): 1})
    
    # Schoolbook A_BR * B_21
    # A_BR: (2,2), (2,3)
    # B_21: (2,0)..(3,1)
    # C_BL: (2,0), (2,1)
    
    # C[2,0] += A[2,2]B[2,0] + A[2,3]B[3,0]
    # C[2,1] += A[2,2]B[2,1] + A[2,3]B[3,1]
    
    # T5
    add_term({(2,2): 1}, {(2,0): 1}, {(0,2): 1})
    # T6
    add_term({(2,3): 1}, {(3,0): 1}, {(0,2): 1})
    # T7
    add_term({(2,2): 1}, {(2,1): 1}, {(1,2): 1})
    # T8
    add_term({(2,3): 1}, {(3,1): 1}, {(1,2): 1})
    
    # Block 4: C_BR = A_BL B_12 + A_BR B_22
    # A_BL: (2,0), (2,1)
    # B_12: (0,2)..(1,3)
    # C_BR: (2,2), (2,3) -> W indices (2,2), (3,2) -> W[2*3+2]=W[8], W[3*3+2]=W[11]
    
    # Schoolbook A_BL * B_12
    # C[2,2] = A[2,0]B[0,2] + A[2,1]B[1,2]
    # C[2,3] = A[2,0]B[0,3] + A[2,1]B[1,3]
    
    # T9
    add_term({(2,0): 1}, {(0,2): 1}, {(2,2): 1})
    # T10
    add_term({(2,1): 1}, {(1,2): 1}, {(2,2): 1})
    # T11
    add_term({(2,0): 1}, {(0,3): 1}, {(3,2): 1})
    # T12
    add_term({(2,1): 1}, {(1,3): 1}, {(3,2): 1})
    
    # Schoolbook A_BR * B_22
    # A_BR: (2,2), (2,3)
    # B_22: (2,2)..(3,3)
    # C_BR: (2,2), (2,3)
    
    # C[2,2] += A[2,2]B[2,2] + A[2,3]B[3,2]
    # C[2,3] += A[2,2]B[2,3] + A[2,3]B[3,3]
    
    # T13
    add_term({(2,2): 1}, {(2,2): 1}, {(2,2): 1})
    # T14
    add_term({(2,3): 1}, {(3,2): 1}, {(2,2): 1})
    # T15
    add_term({(2,2): 1}, {(2,3): 1}, {(3,2): 1})
    # T16
    add_term({(2,3): 1}, {(3,3): 1}, {(3,2): 1})
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    # Verify
    n, m, p = 3, 4, 4
    recon = np.einsum('ijc,jkc,kic->ijk', U.reshape(n,m,-1), V.reshape(m,p,-1), W.reshape(p,n,-1))
    expected = np.ones((n,m,p))
    print("Max Error:", np.max(np.abs(recon - expected)))
    print("Rank:", U.shape[1])
