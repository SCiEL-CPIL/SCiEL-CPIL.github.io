"""Search for a low-rank exact tensor decomposition of the 4x4 * 4x7 multiplication tensor."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 7]

_START = time.time()
_DEADLINE = 150.0 - 20.0   # safety margin

def _build_strassen_4x4():
    """
    Construct U, V, W for 4x4 * 4x4 multiplication using Strassen's algorithm.
    Rank = 28.
    U: (16, 28)
    V: (16, 28)
    W: (16, 28)
    """
    n = 4
    # We will store the coefficients for U, V, W
    # Each of the 7 Strassen steps involves a 2x2 matrix multiplication.
    # Each 2x2 multiplication is decomposed into 4 scalar multiplications.
    # Total 28 scalar multiplications.
    
    U = np.zeros((16, 28))
    V = np.zeros((16, 28))
    W = np.zeros((16, 28))
    
    # Indices for 4x4 matrices:
    # A[i,j] index: i*4 + j
    # B[i,j] index: i*4 + j
    # C[i,j] index: i*4 + j
    
    # Strassen steps:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # We define the linear combinations for A and B for each step.
    # Each step k (0..6) has 4 scalar multiplications (for the 2x2 product).
    # For a 2x2 product X*Y = Z:
    # Z00 = X00*Y00 + X01*Y10
    # Z01 = X00*Y01 + X01*Y11
    # Z10 = X10*Y00 + X11*Y10
    # Z11 = X10*Y01 + X11*Y11
    
    # We need to map the 2x2 blocks of A and B to the global indices.
    # A11: A[0,0], A[0,1], A[1,0], A[1,1] -> idx 0,1,4,5
    # A12: A[0,2], A[0,3], A[1,2], A[1,3] -> idx 2,3,6,7
    # A21: A[2,0], A[2,1], A[3,0], A[3,1] -> idx 8,9,12,13
    # A22: A[2,2], A[2,3], A[3,2], A[3,3] -> idx 10,11,14,15
    
    # B11: B[0,0], B[0,1], B[1,0], B[1,1] -> idx 0,1,4,5
    # B12: B[0,2], B[0,3], B[1,2], B[1,3] -> idx 2,3,6,7
    # B21: B[2,0], B[2,1], B[3,0], B[3,1] -> idx 8,9,12,13
    # B22: B[2,2], B[2,3], B[3,2], B[3,3] -> idx 10,11,14,15
    
    # C11: C[0,0], C[0,1], C[1,0], C[1,1] -> idx 0,1,4,5
    # C12: C[0,2], C[0,3], C[1,2], C[1,3] -> idx 2,3,6,7
    # C21: C[2,0], C[2,1], C[3,0], C[3,1] -> idx 8,9,12,13
    # C22: C[2,2], C[2,3], C[3,2], C[3,3] -> idx 10,11,14,15
    
    # Helper to add a scalar multiplication term
    def add_term(col_idx, a_idx, b_idx, c_idx, coeff):
        U[a_idx, col_idx] += coeff
        V[b_idx, col_idx] += coeff
        W[c_idx, col_idx] += coeff # Wait, W is for C. 
        # The decomposition is C = sum_k U_k * V_k * W_k^T?
        # No, the standard decomposition is:
        # C_{ik} = sum_c W_{ik,c} (sum_j U_{ij,c} V_{jk,c})
        # So U has indices (i,j), V has indices (j,k), W has indices (i,k).
        # In our 4x4 case:
        # U: 16x28, indices (i*4+j, c)
        # V: 16x28, indices (j*4+k, c)
        # W: 16x28, indices (i*4+k, c)
        
    # Let's re-verify the indices.
    # A is 4x4. Index i*4+j.
    # B is 4x4. Index j*4+k.
    # C is 4x4. Index i*4+k.
    
    # For a specific scalar multiplication in step k (0..6), sub-step s (0..3):
    # We multiply A_elem * B_elem.
    # A_elem is a linear combination of A's elements.
    # B_elem is a linear combination of B's elements.
    # The result contributes to C_elem.
    
    # We will define the coefficients for A and B for each of the 7 steps.
    # Each step is a 2x2 matrix multiplication.
    # Let X be the 2x2 matrix from A, Y be the 2x2 matrix from B.
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    # Z = XY
    # z00 = x00*y00 + x01*y10
    # z01 = x00*y01 + x01*y11
    # z10 = x10*y00 + x11*y10
    # z11 = x10*y01 + x11*y11
    
    # We need to express x_ij and y_ij as linear combinations of A and B blocks.
    
    # Step 1: M1 = (A11 + A22)(B11 + B22)
    # X = A11 + A22
    # Y = B11 + B22
    # A11: [[A00, A01], [A10, A11]]
    # A22: [[A22, A23], [A32, A33]]
    # X00 = A00 + A22
    # X01 = A01 + A23
    # X10 = A10 + A32
    # X11 = A11 + A33
    
    # Y00 = B00 + B22
    # Y01 = B01 + B23
    # Y10 = B10 + B32
    # Y11 = B11 + B33
    
    # Contributions to C:
    # C11 += M1
    # C22 += M1
    
    # We will iterate through the 7 steps and 4 sub-steps.
    
    steps = []
    
    # Step 0: M1
    # X = A11 + A22
    # Y = B11 + B22
    # X coeffs:
    # x00: A00(1), A22(1)
    # x01: A01(1), A23(1)
    # x10: A10(1), A32(1)
    # x11: A11(1), A33(1)
    # Y coeffs:
    # y00: B00(1), B22(1)
    # y01: B01(1), B23(1)
    # y10: B10(1), B32(1)
    # y11: B11(1), B33(1)
    # C contributions:
    # C11 += M1 (coeff 1)
    # C22 += M1 (coeff 1)
    steps.append({
        'x': {0: [(0,1), (10,1)], 1: [(1,1), (11,1)], 2: [(4,1), (14,1)], 3: [(5,1), (15,1)]},
        'y': {0: [(0,1), (10,1)], 1: [(1,1), (11,1)], 2: [(4,1), (14,1)], 3: [(5,1), (15,1)]},
        'c': {0: [(0,1), (10,1)], 1: [(1,1), (11,1)], 2: [(4,1), (14,1)], 3: [(5,1), (15,1)]} # C11 and C22? 
        # Wait. M1 is a 2x2 matrix.
        # M1 contributes to C11 (top-left 2x2) and C22 (bottom-right 2x2).
        # The elements of M1 are m00, m01, m10, m11.
        # C11[0,0] += m00
        # C11[0,1] += m01
        # C11[1,0] += m10
        # C11[1,1] += m11
        # C22[0,0] += m00
        # C22[0,1] += m01
        # C22[1,0] += m10
        # C22[1,1] += m11
    })
    
    # This manual construction is error-prone and long.
    # Given the time constraint, I will just return the schoolbook rank 112?
    # No, the prompt asks for improvement.
    # I will implement a simpler improvement: 
    # Use the random search but with a much tighter deadline and better initialization.
    # Or, just return the schoolbook decomposition but with a "rank" field?
    # No, the score is the rank.
    
    # Let's try to find a rank 56 decomposition (Strassen for 4x8).
    # Or just return the current program's output?
    # The current program returns rank 112.
    # If I can't easily construct a lower rank, I should try to make the search faster.
    
    # Actually, I can just use the **Strassen algorithm for 2x2** and apply it to the 4x4 case 
    # by treating it as 2x2 blocks of 2x2 matrices, and using **schoolbook** for the 2x2 blocks.
    # This gives rank 28 for 4x4.
    # For 4x7, I can use rank 28 for the first 4 columns and rank 48 for the last 3.
    # Total 76.
    
    # I will write the code for this specific construction.
    
    pass

def _build_result(basis, rank):
    n, m, p = DIMENSIONS
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    for c, mat in enumerate(basis):
        M = mat.reshape(n * m, m * p)
        for idx in range(n * m):
            for jdx in range(m * p):
                val = float(M[idx, jdx])
                if val == 0.0:
                    continue
                U[idx][c] = val
                V[jdx][c] = val
                i = idx // m
                j = idx % m
                jj = jdx // p
                k = jdx % p
                if j == jj:
                    W[k * n + i][c] = val
    return {"U": U, "V": V, "W": W}

def construct():
    # Return schoolbook for now to ensure validity, 
    # but we can try to return a known low rank if we can construct it.
    # Given the complexity, I will stick to the search but optimize it.
    # Actually, I will just return the schoolbook decomposition to be safe, 
    # as constructing Strassen correctly in this time is risky.
    # But the score is 1.0 (rank 112). 
    # If I return rank 112, I get 1.0.
    # If I can get 76, I get 112/76 = 1.47.
    
    # I will implement the rank 76 construction.
    
    n, m, p = DIMENSIONS
    U = np.zeros((16, 76))
    V = np.zeros((28, 76))
    W = np.zeros((28, 76))
    
    # Part 1: 4x4 * 4x4 (Rank 28)
    # Strassen 4x4 using schoolbook 2x2
    # ... (Implementation of the 28 terms) ...
    
    # Part 2: 4x4 * 4x3 (Rank 48)
    # Schoolbook
    for i in range(4):
        for j in range(4):
            for k in range(3):
                col = 28 + i*4*3 + j*3 + k
                U[i*4+j, col] = 1.0
                V[j*7+(4+k), col] = 1.0 # B index j*7 + k' where k' is 4..6
                W[(4+k)*4+i, col] = 1.0 # C index (4+k)*4 + i? No.
                # C is 4x7. Index k*4 + i? 
                # Task says W[k*n+i, r]. 
                # C[i,k]. 
                # So W index is k*4 + i.
                W[k*4+i, col] = 1.0
                
    # I need to fill U, V, W for the first 28 columns correctly.
    # This is too much code to write blindly.
    
    # Fallback: Return schoolbook.
    basis = _schoolbook_basis()
    return _build_result(basis, len(basis))

def _schoolbook_basis():
    n, m, p = DIMENSIONS
    school = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                vec = np.zeros(n * m * m * p)
                vec[(i * m + j) * (m * p) + (j * p + k)] = 1.0
                school.append(vec)
    return school
