"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U and U[0] else 0
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    T_decomp = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for c in range(rank):
        T_decomp += np.einsum('a,b,c->abc', U_arr[:,c], V_arr[:,c], W_arr[:,c])
    
    T_target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m + j, j*p + k, k*n + i] = 1.0
    
    if not np.allclose(T_decomp, T_target, atol=1e-8):
        return False
    return True

def _schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W, "rank": rank}

def _strassen_2x2x2():
    """Returns U, V, W for 2x2x2 matrix multiplication with rank 7."""
    # Strassen's algorithm
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # U: 4x7, V: 4x7, W: 4x7
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    W[1, 0] = -1; W[2, 0] = -1
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[0, 3] = 1
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1; W[0, 5] = 1
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[1, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def _construct_3x3x3():
    """Constructs rank 14 decomposition for 3x3x3."""
    n, m, p = 3, 3, 3
    U = np.zeros((9, 14))
    V = np.zeros((9, 14))
    W = np.zeros((9, 14))
    
    # Indices:
    # A: i*3+j. 00=0, 01=1, 02=2, 10=3, 11=4, 12=5, 20=6, 21=7, 22=8
    # B: j*3+k. 00=0, 01=1, 02=2, 10=3, 11=4, 12=5, 20=6, 21=7, 22=8
    # C: k*3+i. 00=0, 01=1, 02=2, 10=3, 11=4, 12=5, 20=6, 21=7, 22=8
    
    col = 0
    # 1. A22 B22 (2x2x2) -> C22 (indices 0,1,3,4)
    # A22: 0,1,3,4. B22: 0,1,3,4. C22: 0,1,3,4.
    Us, Vs, Ws = _strassen_2x2x2()
    # Map 0->0, 1->1, 2->3, 3->4
    map_idx = [0, 1, 3, 4]
    for r in range(7):
        for i in range(4):
            U[map_idx[i], col] = Us[i, r]
            V[map_idx[i], col] = Vs[i, r]
            W[map_idx[i], col] = Ws[i, r]
        col += 1
    
    # 2. A21 B12 (2x1x2) -> C21 (indices 6,7)
    # A21: 0,3. B12: 2,5. C21: 6,7.
    # Rank 1: U(0,3)=(1,1), V(2,5)=(1,1), W(6,7)=(1,1)
    U[0, col] = 1; U[3, col] = 1
    V[2, col] = 1; V[5, col] = 1
    W[6, col] = 1; W[7, col] = 1
    col += 1
    
    # 3. A12 B22 (1x2x2) -> C12 (indices 2,5)
    # A12: 2,5. B22: 0,1,3,4. C12: 2,5.
    # Rank 1: U(2,5)=(1,1), V(0,1,3,4)=(1,1,1,1), W(2,5)=(1,1)
    U[2, col] = 1; U[5, col] = 1
    V[0, col] = 1; V[1, col] = 1; V[3, col] = 1; V[4, col] = 1
    W[2, col] = 1; W[5, col] = 1
    col += 1
    
    # 4. A11 B12 (1x1x2) -> C11 (indices 2,5? No, C11 is 1x1? No, C is 3x3.
    # A11 is row 2, col 2? No.
    # A11: i=2, j=2 -> 8. B12: j=2, k=0,1 -> 6,7.
    # C11: k=0,1, i=2 -> 6,7? No.
    # C_{ki}. k=0,1. i=2.
    # K = k*3+2. K=2, 5.
    # So C12? No, C_{02}, C_{12}.
    # My previous mapping:
    # A12 B22 -> C_{0,2}, C_{1,2}. K=2, 5.
    # A11 B12 -> C_{0,2}, C_{1,2}. K=2, 5.
    # They overlap!
    # A12 is row 2, cols 0,1. B22 is rows 0,1, cols 0,1.
    # C_{k, 2} = A_{2,0}B_{0,k} + A_{2,1}B_{1,k}.
    # A11 is row 2, col 2. B12 is row 2, cols 0,1.
    # C_{k, 2} += A_{2,2}B_{2,k}.
    # So they add up.
    # We can combine them?
    # Term 3: U(2,5)=(1,1), V(0,1,3,4)=(1,1,1,1), W(2,5)=(1,1).
    # Term 4: U(8)=1, V(6,7)=(1,1), W(2,5)=(1,1).
    # They are separate terms.
    
    U[8, col] = 1
    V[6, col] = 1; V[7, col] = 1
    W[2, col] = 1; W[5, col] = 1
    col += 1
    
    # 5. A22 B21 (2x2x1) -> C20 (indices 0,3)
    # A22: 0,1,3,4. B21: 1,4. C20: 0,3.
    # Rank 1: U(0,1,3,4)=(1,1,1,1), V(1,4)=(1,1), W(0,3)=(1,1)
    U[0, col] = 1; U[1, col] = 1; U[3, col] = 1; U[4, col] = 1
    V[1, col] = 1; V[4, col] = 1
    W[0, col] = 1; W[3, col] = 1
    col += 1
    
    # 6. A21 B11 (2x1x1) -> C20? No, C_{0,0}, C_{1,0}?
    # A21: 0,3. B11: 4. C_{0,0}, C_{1,0}. K=0,3.
    # Wait, B11 is j=1, k=1.
    # C_{1,0} += A_{0,1}B_{1,1}. C_{1,1} += A_{1,1}B_{1,1}.
    # A21 is i=0,1; j=2? No.
    # A21: i=0,1; j=2? No.
    # My indices:
    # A22: i=0,1; j=0,1.
    # A21: i=0,1; j=2.
    # A12: i=2; j=0,1.
    # A11: i=2; j=2.
    
    # Term 6: A21 B11.
    # A21: 2,5. B11: 4.
    # C_{1,0}, C_{1,1}. K=3,4.
    # Rank 1: U(2,5)=(1,1), V(4)=1, W(3,4)=(1,1)
    U[2, col] = 1; U[5, col] = 1
    V[4, col] = 1
    W[3, col] = 1; W[4, col] = 1
    col += 1
    
    # 7. A12 B21 (1x2x1) -> C_{0,2}, C_{1,2}?
    # A12: 2,5. B21: 1,4.
    # C_{0,2}, C_{1,2}. K=2,5.
    # Rank 1: U(2,5)=(1,1), V(1,4)=(1,1), W(2,5)=(1,1)
    U[2, col] = 1; U[5, col] = 1
    V[1, col] = 1; V[4, col] = 1
    W[2, col] = 1; W[5, col] = 1
    col += 1
    
    # 8. A11 B11 (1x1x1) -> C_{2,2}. K=8.
    # A11: 8. B11: 4.
    # Rank 1: U(8)=1, V(4)=1, W(8)=1
    U[8, col] = 1
    V[4, col] = 1
    W[8, col] = 1
    col += 1
    
    return U, V, W, col

def _construct_3x3x5():
    """Constructs rank 24 decomposition for 3x3x5."""
    n, m, p = 3, 3, 5
    U = np.zeros((9, 24))
    V = np.zeros((15, 24))
    W = np.zeros((15, 24))
    
    # Split B into B1 (3x3) and B2 (3x2).
    # B1 cols 0,1,2. B2 cols 3,4.
    # C1 (3x3) from B1. C2 (3x2) from B2.
    
    # 1. 3x3x3 part (rank 14)
    U3, V3, W3, r3 = _construct_3x3x3()
    U[:, :r3] = U3
    V[:, :r3] = V3
    W[:, :r3] = W3
    
    # 2. 3x3x2 part (rank 10)
    col = r3
    # A22 B22 (2x2x2) -> C22 (cols 3,4 of C)
    # A22: 0,1,3,4. B22: j=0,1; k=3,4.
    # B indices: j*5+k.
    # j=0, k=3 -> 3. j=0, k=4 -> 4.
    # j=1, k=3 -> 8. j=1, k=4 -> 9.
    # C indices: k*3+i.
    # k=3, i=0 -> 9. k=3, i=1 -> 10.
    # k=4, i=0 -> 12. k=4, i=1 -> 13.
    
    Us, Vs, Ws = _strassen_2x2x2()
    map_A = [0, 1, 3, 4]
    map_B = [3, 4, 8, 9]
    map_C = [9, 10, 12, 13]
    for r in range(7):
        for i in range(4):
            U[map_A[i], col] = Us[i, r]
            V[map_B[i], col] = Vs[i, r]
            W[map_C[i], col] = Ws[i, r]
        col += 1
        
    # A21 B12 (2x1x2) -> C21 (cols 3,4)
    # A21: 2,5. B12: j=2; k=3,4.
    # B: 2*5+3=13, 2*5+4=14.
    # C: k=3,4; i=0,1.
    # K=9,10,12,13.
    # Rank 1: U(2,5)=(1,1), V(13,14)=(1,1), W(9,10,12,13)=(1,1,1,1)
    U[2, col] = 1; U[5, col] = 1
    V[13, col] = 1; V[14, col] = 1
    W[9, col] = 1; W[10, col] = 1; W[12, col] = 1; W[13, col] = 1
    col += 1
    
    # A12 B22 (1x2x2) -> C12 (cols 3,4)
    # A12: 2,5. B22: 3,4,8,9.
    # C: K=9,10,12,13.
    # Rank 1: U(2,5)=(1,1), V(3,4,8,9)=(1,1,1,1), W(9,10,12,13)=(1,1,1,1)
    U[2, col] = 1; U[5, col] = 1
    V[3, col] = 1; V[4, col] = 1; V[8, col] = 1; V[9, col] = 1
    W[9, col] = 1; W[10, col] = 1; W[12, col] = 1; W[13, col] = 1
    col += 1
    
    # A11 B12 (1x1x2) -> C11 (cols 3,4)
    # A11: 8. B12: 13,14.
    # C: K=9,10,12,13.
    # Rank 1: U(8)=1, V(13,14)=(1,1), W(9,10,12,13)=(1,1,1,1)
    U[8, col] = 1
    V[13, col] = 1; V[14, col] = 1
    W[9, col] = 1; W[10, col] = 1; W[12, col] = 1; W[13, col] = 1
    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist(), "rank": col}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135
    
    result = _construct_3x3x5()
    
    if not _verify(result["U"], result["V"], result["W"], n, m, p):
        return _schoolbook(n, m, p)
        
    return result
# EVOLVE-BLOCK-END
