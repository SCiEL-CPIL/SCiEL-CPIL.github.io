"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0  # Leave margin before 120s limit
    
    n, m, p = DIMENSIONS
    # Rank 24 decomposition
    rank = 24
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def add_term(i, j, k, u_val, v_val, w_val, col):
        # i, j, k are 0-indexed
        U[i * m + j, col] = u_val
        V[j * p + k, col] = v_val
        W[k * n + i, col] = w_val

    col = 0
    
    # 1. A11 B11
    add_term(0, 0, 0, 1, 1, 1, col); col += 1
    
    # 2. A11 B12 (2 terms)
    add_term(0, 0, 1, 1, 1, 1, col); col += 1
    add_term(0, 0, 2, 1, 1, 1, col); col += 1
    
    # 3. A12 B22 (4 terms)
    # A12 is (0,1), (0,2)
    # B22 is (1,1), (1,2), (2,1), (2,2)
    # C12 = A11 B12 + A12 B22
    # C13 = A11 B13 + A12 B23
    # Terms: A01 B11, A01 B12, A02 B21, A02 B22
    add_term(0, 1, 1, 1, 1, 1, col); col += 1
    add_term(0, 1, 2, 1, 1, 1, col); col += 1
    add_term(0, 2, 1, 1, 1, 1, col); col += 1
    add_term(0, 2, 2, 1, 1, 1, col); col += 1
    
    # 4. A21 B11 (2 terms)
    # A21 is (1,0), (2,0)
    add_term(1, 0, 0, 1, 1, 1, col); col += 1
    add_term(2, 0, 0, 1, 1, 1, col); col += 1
    
    # 5. A22 B21 (4 terms)
    # A22 is (1,1)..(2,2)
    # B21 is (1,0), (2,0)
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    # Terms for C21 part from A22 B21:
    # A11 B10, A12 B20, A21 B10, A22 B20
    add_term(1, 1, 0, 1, 1, 1, col); col += 1
    add_term(1, 2, 0, 1, 1, 1, col); col += 1
    add_term(2, 1, 0, 1, 1, 1, col); col += 1
    add_term(2, 2, 0, 1, 1, 1, col); col += 1
    
    # 6. A21 B12 (4 terms)
    # Outer product
    add_term(1, 0, 1, 1, 1, 1, col); col += 1
    add_term(1, 0, 2, 1, 1, 1, col); col += 1
    add_term(2, 0, 1, 1, 1, 1, col); col += 1
    add_term(2, 0, 2, 1, 1, 1, col); col += 1
    
    # 7. A22 B22 (7 terms - Strassen)
    # Indices:
    # A: 11(4), 12(5), 21(7), 22(8)
    # B: 11(4), 12(5), 21(7), 22(8)
    # C: 11(4), 12(7), 21(5), 22(8) (W index 3k+i)
    
    # P1 = (A11+A22)(B11+B22)
    # U: 4:1, 8:1
    # V: 4:1, 8:1
    # W: 4:1, 7:1, 5:1, 8:1 (Coeffs: C11+=1, C12+=0, C21+=0, C22+=-1? No, check Strassen)
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # We need to add terms for each Pm with appropriate U, V, W coefficients.
    # The W coefficient is the coefficient of Pm in the final sum for C_ik.
    
    # P1
    U[4, col] = 1; U[8, col] = 1
    V[4, col] = 1; V[8, col] = 1
    W[4, col] = 1  # C11
    W[7, col] = 0  # C12
    W[5, col] = 0  # C21
    W[8, col] = 1  # C22
    col += 1
    
    # P2 = (A21+A22)B11
    U[7, col] = 1; U[8, col] = 1
    V[4, col] = 1
    W[5, col] = 1  # C21
    W[8, col] = -1 # C22
    col += 1
    
    # P3 = A11(B12-B22)
    U[4, col] = 1
    V[5, col] = 1; V[8, col] = -1
    W[7, col] = 1  # C12
    W[8, col] = 1  # C22
    col += 1
    
    # P4 = A22(B21-B11)
    U[8, col] = 1
    V[7, col] = 1; V[4, col] = -1
    W[4, col] = 1  # C11
    W[5, col] = 1  # C21
    col += 1
    
    # P5 = (A11+A12)B22
    U[4, col] = 1; U[5, col] = 1
    V[8, col] = 1
    W[4, col] = -1 # C11
    W[7, col] = 1  # C12
    col += 1
    
    # P6 = (A21-A11)(B11+B12)
    U[7, col] = 1; U[4, col] = -1
    V[4, col] = 1; V[5, col] = 1
    W[8, col] = 1  # C22
    col += 1
    
    # P7 = (A12-A22)(B21+B22)
    U[5, col] = 1; U[8, col] = -1
    V[7, col] = 1; V[8, col] = 1
    W[4, col] = 1  # C11
    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
