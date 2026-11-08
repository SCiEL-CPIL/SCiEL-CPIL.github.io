"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    n, m, p = DIMENSIONS
    # We use a decomposition of rank 26 by partitioning the 3x3 matrices
    # into 2x2 and 1x1 blocks.
    # Rank 26 < 27 (schoolbook).
    
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term: u_idx, v_idx, w_idx, coeff
    # U[u_idx][col] = coeff
    # V[v_idx][col] = 1
    # W[w_idx][col] = 1
    # But we need to handle sums.
    # It is easier to just add to the matrices directly.
    
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal col
        U[u_idx][col] += coeff
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col += 1

    # 1. A2 B2 (Top-left 2x2 * 2x2) -> Rank 7 (Strassen)
    # Indices:
    # A2: (0,0), (0,1), (1,0), (1,1) -> U indices 0, 1, 3, 4
    # B2: (0,0), (0,1), (1,0), (1,1) -> V indices 0, 1, 3, 4
    # C22: (0,0), (0,1), (1,0), (1,1) -> W indices 0, 3, 1, 4 (k*n+i)
    # W index for C_{ik} is k*3 + i.
    # C_{00} -> 0*3+0 = 0
    # C_{01} -> 0*3+1 = 1? No, C_{ik}. i is row, k is col.
    # W[k*n+i].
    # C_{00}: k=0, i=0 -> 0
    # C_{01}: k=1, i=0 -> 3
    # C_{10}: k=0, i=1 -> 1
    # C_{11}: k=1, i=1 -> 4
    
    # Strassen M1 = (A00+A11)(B00+B11)
    # M1 contributes to C00, C11
    # C00 += M1, C11 += M1
    # We need to add M1 to C00 and C11.
    # But M1 is a single multiplication.
    # U: A00=1, A11=1. V: B00=1, B11=1.
    # W: C00=1, C11=1.
    
    # Let's define the Strassen terms explicitly.
    # M1
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    # M2 = (A10+A11)B00
    # C10 += M2, C11 -= M2
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    
    # M3 = A00(B01-B11)
    # C01 += M3, C11 += M3
    U[0][col] = 1
    V[1][col] = 1; V[4][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    # M4 = A11(B10-B00)
    # C00 += M4, C10 += M4
    U[4][col] = 1
    V[3][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # M5 = (A00+A01)B11
    # C00 -= M5, C01 += M5
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    # C11 += M6
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    # C00 += M7
    U[1][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = 1
    col += 1
    
    # 2. a b^T (Top-right 2x1 * Bottom-left 1x2) -> Rank 4
    # a = A[0:2, 2] -> U indices 2, 5
    # b^T = B[2, 0:2] -> V indices 6, 7?
    # V index for B_{jk} is j*3 + k.
    # B_{20} -> 2*3+0 = 6.
    # B_{21} -> 2*3+1 = 7.
    # C_{ij} for i in 0,1 and j in 0,1.
    # C_{00} (W0) += A02 B20
    # C_{01} (W3) += A02 B21
    # C_{10} (W1) += A12 B20
    # C_{11} (W4) += A12 B21
    
    add_term(2, 6, 0, 1)
    add_term(2, 7, 3, 1)
    add_term(5, 6, 1, 1)
    add_term(5, 7, 4, 1)
    
    # 3. A2 b (Top-left 2x2 * Top-right 2x1) -> Rank 4
    # A2: U 0,1,3,4
    # b = B[0:2, 2] -> V indices 2, 5
    # C_{i2} for i in 0,1.
    # C_{02} (W[2*3+0]=6) += A00 B02 + A01 B12
    # C_{12} (W[2*3+1]=7) += A10 B02 + A11 B12
    
    add_term(0, 2, 6, 1)
    add_term(1, 5, 6, 1)
    add_term(3, 2, 7, 1)
    add_term(4, 5, 7, 1)
    
    # 4. a b3 (Top-right 2x1 * Bottom-right 1x1) -> Rank 2
    # a: U 2,5
    # b3 = B[2,2] -> V index 8
    # C_{02} (W6) += A02 B22
    # C_{12} (W7) += A12 B22
    
    add_term(2, 8, 6, 1)
    add_term(5, 8, 7, 1)
    
    # 5. a^T B2 (Bottom-left 1x2 * Top-left 2x2) -> Rank 4
    # a^T = A[2, 0:2] -> U indices 6, 7
    # B2: V 0,1,3,4
    # C_{2j} for j in 0,1.
    # C_{20} (W[0*3+2]=2) += A20 B00 + A21 B10
    # C_{21} (W[1*3+2]=5) += A20 B01 + A21 B11
    
    add_term(6, 0, 2, 1)
    add_term(7, 3, 2, 1)
    add_term(6, 1, 5, 1)
    add_term(7, 4, 5, 1)
    
    # 6. a3 b^T (Bottom-right 1x1 * Bottom-left 1x2) -> Rank 2
    # a3 = A[2,2] -> U index 8
    # b^T = B[2, 0:2] -> V indices 6, 7
    # C_{20} (W2) += A22 B20
    # C_{21} (W5) += A22 B21
    
    add_term(8, 6, 2, 1)
    add_term(8, 7, 5, 1)
    
    # 7. a^T b (Bottom-left 1x2 * Top-right 2x1) -> Rank 2
    # a^T: U 6,7
    # b: V 2,5
    # C_{22} (W[2*3+2]=8) += A20 B02 + A21 B12
    
    add_term(6, 2, 8, 1)
    add_term(7, 5, 8, 1)
    
    # 8. a3 b3 (Bottom-right 1x1 * Bottom-right 1x1) -> Rank 1
    # a3: U 8
    # b3: V 8
    # C_{22} (W8) += A22 B22
    
    add_term(8, 8, 8, 1)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
