import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Strassen's algorithm for 3x3x3: rank 7 decomposition
    # The 7 terms for multiplying two 3x3 matrices C = A * B
    # Using the standard Strassen decomposition adapted for 3x3:
    #
    # M1 = (A00 + A22)(B00 + B22)
    # M2 = (A20 + A22)B00
    # M3 = A00(B01 - B21)
    # M4 = A22(B21 - B01)
    # M5 = (A01 + A11)B11
    # M6 = (A10 + A11)(B01 + B11)
    # M7 = (A00 + A10)(B10 - B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C02 = M2 - M4 + M6
    # C10 = M1 - M3 + M5 - M7
    # C11 = M2 - M1 + M4 + M7
    # C12 = -M2 + M6
    # C20 = M3 - M6 + M7
    # C21 = -M4 - M7
    # C22 = M1 - M2 + M4 - M5
    
    # Define the 7 rank-1 terms for 3x3x3
    # Each term is (u, v, w) where u is 9-dim (A indices), v is 9-dim (B indices), w is 9-dim (C indices)
    # A index: i*m+j = i*3+j, B index: j*p+k = j*3+k, C index: k*n+i = k*3+i
    
    # For each Strassen term, we need to express the linear combinations
    # Term 1: (A00+A22)(B00+B22)
    # A part: A00=(0,0), A22=(2,2) -> indices 0 and 8
    # B part: B00=(0,0), B22=(2,2) -> indices 0 and 8
    # C part: contributes to C00(+1), C10(-1), C11(-1), C22(+1)
    
    strassen_terms = []
    
    # Helper to build a term from linear combinations
    def make_term(a_combos, b_combos, c_combos):
        """a_combos: list of (a_index, coeff), b_combos: list of (b_index, coeff), c_combos: list of (c_index, coeff)"""
        u = np.zeros(9)
        v = np.zeros(9)
        w = np.zeros(9)
        for idx, coeff in a_combos:
            u[idx] = coeff
        for idx, coeff in b_combos:
            v[idx] = coeff
        for idx, coeff in c_combos:
            w[idx] = coeff
        return u, v, w
    
    # M1 = (A00 + A22)(B00 + B22)
    # A00 = index 0, A22 = index 8
    # B00 = index 0, B22 = index 8
    # C: C00=+1, C10=-1, C11=-1, C22=+1
    # C indices: C00=(0,0)=0, C10=(1,0)=3, C11=(1,1)=4, C22=(2,2)=8
    strassen_terms.append(make_term(
        [(0, 1), (8, 1)],
        [(0, 1), (8, 1)],
        [(0, 1), (3, -1), (4, -1), (8, 1)]
    ))
    
    # M2 = (A20 + A22)B00
    # A20 = index 6, A22 = index 8
    # B00 = index 0
    # C: C02=+1, C11=+1, C12=-1, C22=-1
    # C02=(0,2)=2, C11=(1,1)=4, C12=(1,2)=5, C22=(2,2)=8
    strassen_terms.append(make_term(
        [(6, 1), (8, 1)],
        [(0, 1)],
        [(2, 1), (4, 1), (5, -1), (8, -1)]
    ))
    
    # M3 = A00(B01 - B21)
    # A00 = index 0
    # B01 = index 1, B21 = index 7
    # C: C01=+1, C10=-1, C20=+1
    # C01=(0,1)=1, C10=(1,0)=3, C20=(2,0)=6
    strassen_terms.append(make_term(
        [(0, 1)],
        [(1, 1), (7, -1)],
        [(1, 1), (3, -1), (6, 1)]
    ))
    
    # M4 = A22(B21 - B01)
    # A22 = index 8
    # B21 = index 7, B01 = index 1
    # C: C00=+1, C11=+1, C21=-1, C22=-1
    # C00=(0,0)=0, C11=(1,1)=4, C21=(2,1)=7, C22=(2,2)=8
    strassen_terms.append(make_term(
        [(8, 1)],
        [(7, 1), (1, -1)],
        [(0, 1), (4, 1), (7, -1), (8, -1)]
    ))
    
    # M5 = (A01 + A11)B11
    # A01 = index 1, A11 = index 4
    # B11 = index 4
    # C: C00=-1, C01=+1, C10=+1, C22=-1
    # C00=(0,0)=0, C01=(0,1)=1, C10=(1,0)=3, C22=(2,2)=8
    strassen_terms.append(make_term(
        [(1, 1), (4, 1)],
        [(4, 1)],
        [(0, -1), (1, 1), (3, 1), (8, -1)]
    ))
    
    # M6 = (A10 + A11)(B01 + B11)
    # A10 = index 3, A11 = index 4
    # B01 = index 1, B11 = index 4
    # C: C02=+1, C12=+1, C20=-1
    # C02=(0,2)=2, C12=(1,2)=5, C20=(2,0)=6
    strassen_terms.append(make_term(
        [(3, 1), (4, 1)],
        [(1, 1), (4, 1)],
        [(2, 1), (5, 1), (6, -1)]
    ))
    
    # M7 = (A00 + A10)(B10 - B11)
    # A00 = index 0, A10 = index 3
    # B10 = index 3, B11 = index 4
    # C: C00=+1, C10=-1, C11=+1, C20=+1, C21=-1
    # C00=(0,0)=0, C10=(1,0)=3, C11=(1,1)=4, C20=(2,0)=6, C21=(2,1)=7
    strassen_terms.append(make_term(
        [(0, 1), (3, 1)],
        [(3, 1), (4, -1)],
        [(0, 1), (3, -1), (4, 1), (6, 1), (7, -1)]
    ))
    
    # Now build the full 3x3x7 decomposition
    # First 7 terms handle the 3x3x3 part (columns 0,1,2 of B)
    # Remaining 4 columns (3,4,5,6) use schoolbook: 3*3*4 = 36 terms
    # Total rank = 7 + 36 = 43
    
    # But wait - let me verify the Strassen decomposition is correct first
    # Build the tensor for 3x3x3
    T3 = np.zeros((9, 9, 9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                T3[i*3+j, j*3+k, k*3+i] = 1
    
    # Build the decomposition from Strassen terms
    U3 = np.zeros((9, 7))
    V3 = np.zeros((9, 7))
    W3 = np.zeros((9, 7))
    for r in range(7):
        u, v, w = strassen_terms[r]
        U3[:, r] = u
        V3[:, r] = v
        W3[:, r] = w
    
    T3_recon = np.einsum('ar,br,cr->abc', U3, V3, W3)
    error3 = np.max(np.abs(T3_recon - T3))
    
    if error3 > 1e-10:
        # Strassen decomposition is wrong, fall back to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Strassen is correct, now build full 3x3x7
    # Rank = 7 (for cols 0-2) + 36 (schoolbook for cols 3-6) = 43
    rank = 7 + 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Copy Strassen terms for the 3x3x3 part
    U[:, :7] = U3
    V[:, :7] = V3
    W[:, :7] = W3
    
    # Schoolbook for columns 3,4,5,6
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(3, p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Verify the full decomposition
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    error = np.max(np.abs(T_recon - T))
    
    if error > 1e-10:
        # Something went wrong, fall back to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
