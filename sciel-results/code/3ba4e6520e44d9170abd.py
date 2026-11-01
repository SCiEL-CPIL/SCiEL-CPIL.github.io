"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for 2x4 * 4x6 matrix multiplication.
    Uses block-Strassen: split the 4x4 middle into 2x2 blocks, apply Strassen to each
    pair of 2x2 blocks, and handle cross-block terms with schoolbook.
    
    The multiplication (A 2x4) * (B 4x6) = C 2x6.
    A = [A1 A2] where A1, A2 are 2x2.
    B = [B1; B2] where B1, B2 are 2x4 (top and bottom halves of B).
    C = A1*B1 + A2*B2.
    
    We can apply Strassen to each 2x2 * 2x4 multiplication, but Strassen is for square
    matrices. Let's think differently.
    
    Actually, let's use the known result: M(2,4,6) can be done with rank less than 48.
    
    One approach: Use the fact that 4 = 2*2 and 6 = 3*2.
    
    Let A (2x4) = [A1 A2], each 2x2.
    Let B (4x6) = [[B11 B12]; [B21 B22]] where Bij are 2x3.
    
    C = A1*B1 + A2*B2 where B1 = [B11 B12] (2x6) and B2 = [B21 B22] (2x6).
    
    So C = A1*B1 + A2*B2, which is a sum of two 2x2 * 2x6 multiplications.
    
    M(2,2,6) <= ? We can write 6 = 2*3, so 2x6 = [2x2 2x2 2x2].
    M(2,2,6) = M(2,2,2)*3? No, that's not right.
    
    Actually, M(2,2,6): multiply 2x2 by 2x6. This is equivalent to 3 separate 2x2*2x2 multiplications
    (since 2x6 = [2x2 | 2x2 | 2x2]), so M(2,2,6) <= 3*M(2,2,2) = 3*7 = 21.
    
    So M(2,4,6) <= M(2,2,6) + M(2,2,6) = 21 + 21 = 42.
    
    This matches the inspiration! Let's implement this.
    
    Strassen for 2x2 * 2x2:
    Given X (2x2) and Y (2x2):
    M1 = (X11+X22)*(Y11+Y22)
    M2 = (X21+X22)*Y11
    M3 = X11*(Y12-Y22)
    M4 = X22*(Y21-Y11)
    M5 = (X11+X12)*Y22
    M6 = (X21-X11)*(Y11+Y12)
    M7 = (X12-X22)*(Y21+Y22)
    
    Result:
    Z11 = M1 + M4 - M5 + M7
    Z12 = M3 + M5
    Z21 = M2 + M4
    Z22 = M1 - M2 + M3 + M6
    
    For 2x2 * 2x6, we split the 2x6 into three 2x2 blocks and apply Strassen to each.
    That gives 3*7 = 21 terms.
    
    So total rank = 2 * 21 = 42.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    
    # We'll build U, V, W matrices.
    # U has shape (n*m, rank) = (8, rank)
    # V has shape (m*p, rank) = (24, rank)
    # W has shape (p*n, rank) = (12, rank)
    
    # The tensor T[i,j,k] = delta_{i,k} ... no, for matrix multiplication:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k].
    # The decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Indexing:
    # U row index: i*m + j, where i in [0,n), j in [0,m)
    # V row index: j*p + k, where j in [0,m), k in [0,p)
    # W row index: k*n + i, where k in [0,p), i in [0,n)
    
    # Strategy:
    # A (2x4) = [A1 A2], A1 = A[0:2, 0:2], A2 = A[0:2, 2:4]
    # B (4x6) = [B1; B2], B1 = B[0:2, :], B2 = B[2:4, :]
    # C = A1*B1 + A2*B2
    #
    # B1 is 2x6, split into B1a = B[0:2,0:2], B1b = B[0:2,2:4], B1c = B[0:2,4:6]
    # B2 is 2x6, split into B2a = B[2:4,0:2], B2b = B[2:4,2:4], B2c = B[2:4,4:6]
    #
    # A1*B1 = A1*B1a + A1*B1b + A1*B1c (each is 2x2*2x2)
    # A2*B2 = A2*B2a + A2*B2b + A2*B2c (each is 2x2*2x2)
    #
    # Total: 6 Strassen multiplications, each of rank 7, total rank 42.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Strassen coefficients for 2x2*2x2
    # M1 = (X11+X22)*(Y11+Y22)
    # M2 = (X21+X22)*Y11
    # M3 = X11*(Y12-Y22)
    # M4 = X22*(Y21-Y11)
    # M5 = (X11+X12)*Y22
    # M6 = (X21-X11)*(Y11+Y12)
    # M7 = (X12-X22)*(Y21+Y22)
    #
    # Z11 = M1 + M4 - M5 + M7
    # Z12 = M3 + M5
    # Z21 = M2 + M4
    # Z22 = M1 - M2 + M3 + M6
    
    # For each of the 6 Strassen blocks, we need to set up U, V, W contributions.
    # The 6 blocks are:
    # (A1, B1a), (A1, B1b), (A1, B1c), (A2, B2a), (A2, B2b), (A2, B2c)
    
    # For a 2x2*2x2 block where A-block is at rows i1..i1+1, cols j1..j1+1 of A,
    # and B-block is at rows j1..j1+1, cols k1..k1+1 of B:
    #
    # A indices: i in {i1, i1+1}, j in {j1, j1+1}
    # B indices: j in {j1, j1+1}, k in {k1, k1+1}
    # C indices: i in {i1, i1+1}, k in {k1, k1+1}
    
    def add_strassen(i1, j1, k1):
        """Add Strassen decomposition for A[i1:i1+2, j1:j1+2] * B[j1:j1+2, k1:k1+2] -> C[i1:i1+2, k1:k1+2]"""
        nonlocal col
        # A11 = A[i1, j1], A12 = A[i1, j1+1], A21 = A[i1+1, j1], A22 = A[i1+1, j1+1]
        # B11 = B[j1, k1], B12 = B[j1, k1+1], B21 = B[j1+1, k1], B22 = B[j1+1, k1+1]
        # C11 = C[i1, k1], C12 = C[i1, k1+1], C21 = C[i1+1, k1], C22 = C[i1+1, k1+1]
        
        # M1 = (A11+A22)*(B11+B22)
        # U: A11 and A22 get +1; V: B11 and B22 get +1; W: C11 gets +1, C22 gets +1
        U[i1 * m + j1][col] += 1       # A11
        U[(i1+1) * m + (j1+1)][col] += 1  # A22
        V[j1 * p + k1][col] += 1       # B11
        V[(j1+1) * p + (k1+1)][col] += 1  # B22
        W[k1 * n + i1][col] += 1       # C11
        W[(k1+1) * n + (i1+1)][col] += 1  # C22
        col += 1
        
        # M2 = (A21+A22)*B11
        U[(i1+1) * m + j1][col] += 1       # A21
        U[(i1+1) * m + (j1+1)][col] += 1  # A22
        V[j1 * p + k1][col] += 1           # B11
        W[k1 * n + (i1+1)][col] += 1       # C21
        col += 1
        
        # M3 = A11*(B12-B22)
        U[i1 * m + j1][col] += 1           # A11
        V[j1 * p + (k1+1)][col] += 1       # B12
        V[(j1+1) * p + (k1+1)][col] -= 1  # -B22
        W[(k1+1) * n + i1][col] += 1       # C12
        W[(k1+1) * n + (i1+1)][col] += 1  # C22
        col += 1
        
        # M4 = A22*(B21-B11)
        U[(i1+1) * m + (j1+1)][col] += 1  # A22
        V[(j1+1) * p + k1][col] += 1       # B21
        V[j1 * p + k1][col] -= 1           # -B11
        W[k1 * n + i1][col] += 1           # C11
        W[k1 * n + (i1+1)][col] += 1       # C21
        col += 1
        
        # M5 = (A11+A12)*B22
        U[i1 * m + j1][col] += 1           # A11
        U[i1 * m + (j1+1)][col] += 1       # A12
        V[(j1+1) * p + (k1+1)][col] += 1  # B22
        W[k1 * n + i1][col] -= 1           # -C11
        W[(k1+1) * n + i1][col] += 1       # C12
        col += 1
        
        # M6 = (A21-A11)*(B11+B12)
        U[(i1+1) * m + j1][col] += 1       # A21
        U[i1 * m + j1][col] -= 1           # -A11
        V[j1 * p + k1][col] += 1           # B11
        V[j1 * p + (k1+1)][col] += 1       # B12
        W[(k1+1) * n + (i1+1)][col] += 1  # C22
        col += 1
        
        # M7 = (A12-A22)*(B21+B22)
        U[i1 * m + (j1+1)][col] += 1       # A12
        U[(i1+1) * m + (j1+1)][col] -= 1  # -A22
        V[(j1+1) * p + k1][col] += 1       # B21
        V[(j1+1) * p + (k1+1)][col] += 1  # B22
        W[k1 * n + i1][col] += 1           # C11
        col += 1
    
    # Block 1: A1 * B1a -> C[0:2, 0:2]
    add_strassen(0, 0, 0)
    # Block 2: A1 * B1b -> C[0:2, 2:4]
    add_strassen(0, 0, 2)
    # Block 3: A1 * B1c -> C[0:2, 4:6]
    add_strassen(0, 0, 4)
    # Block 4: A2 * B2a -> C[0:2, 0:2]
    add_strassen(0, 2, 0)
    # Block 5: A2 * B2b -> C[0:2, 2:4]
    add_strassen(0, 2, 2)
    # Block 6: A2 * B2c -> C[0:2, 4:6]
    add_strassen(0, 2, 4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
