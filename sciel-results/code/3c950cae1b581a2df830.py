import time
import numpy as np

DIMENSIONS = [3, 4, 4]
DEADLINE = 150.0  # wall-clock limit; use ~120s for safety margin as per contract
START_TIME = time.time()

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank = n*m*p = 48. We will try to beat this with block decomposition.
    # The matrix multiplication tensor T has entries T[i,j,k] = A[i,j]*B[j,k] -> C[i,k].
    # We partition the index sets to exploit structure.
    # n=3, m=4, p=4.
    # Strategy: Partition i into {0,1} and {2}.
    # Partition j into {0,1} and {2,3}.
    # Partition k into {0,1} and {2,3}.
    #
    # Block C[0:2, 0:2] depends on A[0:2, 0:4] * B[0:4, 0:2].
    #   A[0:2, 0:4] can be split into A[0:2, 0:2] and A[0:2, 2:4].
    #   So C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2].
    #   The first term is M_{2,2,2} (rank 7 via Strassen).
    #   The second term is M_{2,2,2} (rank 7 via Strassen).
    #   Total rank for this block: 14.
    #
    # Block C[0:2, 2:4] depends on A[0:2, 0:4] * B[0:4, 2:4].
    #   Similarly: M_{2,2,2} + M_{2,2,2} -> rank 14.
    #
    # Block C[2, 0:2] depends on A[2, 0:4] * B[0:4, 0:2].
    #   A[2, 0:4] is a row vector (1x4). B[0:4, 0:2] is 4x2.
    #   This is M_{1,4,2}. Minimal rank?
    #   For M_{1,4,2}, we compute y_k = sum_j a_j b_{j,k}.
    #   This is equivalent to multiplying a 1x4 matrix by a 4x2 matrix.
    #   Schoolbook rank is 1*4*2 = 8.
    #   Can we do better?
    #   The tensor is a slice. Generally, rank of M_{1,m,p} is m*p for m,p >= 2?
    #   Actually, for M_{1,4,2}, it is equivalent to the outer product of a vector and a matrix?
    #   No, it's linear in A and linear in B.
    #   Let A = (a1, a2, a3, a4). B is 4x2.
    #   C1 = a1 B11 + a2 B21 + a3 B31 + a4 B41
    #   C2 = a1 B12 + a2 B22 + a3 B32 + a4 B42
    #   This is a bilinear map from (1x4) x (4x2) to (1x2).
    #   This is isomorphic to M_{2,1,4} or similar.
    #   Actually, rank of M_{1,m,p} is m*p if m,p >= 1?
    #   Wait, the multiplication of a 1xm matrix and mxp matrix is just scaling rows of B by A.
    #   C_k = sum_j A_j B_{j,k}.
    #   This is exactly M_{1,m,p}. The rank is known to be m*p for m,p >= 2?
    #   Actually, no. For M_{1,4,2}, the tensor is sum_j e_j (in A) \otimes e_j (in B) \otimes ...?
    #   Let's check small cases. M_{1,2,2}: C1 = a1 b11 + a2 b21, C2 = a1 b12 + a2 b22.
    #   Rank 4? Or less?
    #   The tensor T in basis a_i, b_{jk}, c_k.
    #   T = sum_{i,k} a_i \otimes (sum_j b_{jk} e_j) \otimes c_k ? No.
    #   T = sum_{j,k} a_j \otimes b_{jk} \otimes c_k? No, c_k is the output index.
    #   T = sum_{j=1}^4 sum_{k=1}^2 (a_j \otimes b_{jk}) \otimes c_k.
    #   This is a sum of 8 terms. Are they independent?
    #   Likely rank 8.
    #   So C[2, 0:2] uses rank 8.
    #
    # Block C[2, 2:4] similarly uses rank 8.
    #
    # Total rank = 14 + 14 + 8 + 8 = 44.
    #
    # We need to construct the explicit U, V, W matrices for rank 44.
    #
    # Strassen decomposition for M_{2,2,2}:
    # Inputs: A (2x2), B (2x2).
    # Outputs: C (2x2).
    # 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B01 - B11)
    # M4 = A11 (B10 - B00)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # We will map these to global indices.
    #
    # Block 1: C[0:2, 0:2] from A[0:2, 0:2]*B[0:2, 0:2] (Strassen)
    #   i in {0,1}, j in {0,1}, k in {0,1}
    #   Local A indices: i, j. Local B indices: j, k.
    #   Global A[i,j], B[j,k].
    #   Global C[i,k].
    #
    # Block 2: C[0:2, 0:2] from A[0:2, 2:4]*B[2:4, 0:2] (Strassen)
    #   i in {0,1}, j in {2,3}, k in {0,1}
    #   Local A indices: i, j-2. Local B indices: j-2, k.
    #
    # Block 3: C[0:2, 2:4] from A[0:2, 0:2]*B[0:2, 2:4] (Strassen)
    #   i in {0,1}, j in {0,1}, k in {2,3}
    #
    # Block 4: C[0:2, 2:4] from A[0:2, 2:4]*B[2:4, 2:4] (Strassen)
    #   i in {0,1}, j in {2,3}, k in {2,3}
    #
    # Block 5: C[2, 0:2] from A[2, 0:4]*B[0:4, 0:2] (Schoolbook M_{1,4,2})
    #   i=2, j in {0,1,2,3}, k in {0,1}
    #   Products: A[2,j] * B[j,k]
    #
    # Block 6: C[2, 2:4] from A[2, 0:4]*B[0:4, 2:4] (Schoolbook M_{1,4,2})
    #   i=2, j in {0,1,2,3}, k in {2,3}
    #   Products: A[2,j] * B[j,k]
    #
    # Total rank = 7*4 + 8*2 = 28 + 16 = 44.
    
    U = [[0] * 44 for _ in range(n * m)]
    V = [[0] * 44 for _ in range(m * p)]
    W = [[0] * 44 for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block
    def add_strassen(i_off, j_off, k_off, col_start):
        # i_off: offset for i index (0 or 1 for the 2x2 block, but we pass 0 for top, 1 for bottom? No, we pass start indices)
        # Actually, let's pass the global indices for the 2x2 block.
        # i0, j0, k0 are the top-left indices of the A, B, C blocks.
        # The block is A[i0:i0+2, j0:j0+2], B[j0:j0+2, k0:k0+2], C[i0:i0+2, k0:k0+2].
        
        # M1 = (A[i0,j0]+A[i0+1,j0+1]) * (B[j0,k0]+B[j0+1,k0+1])
        # U entries:
        U[i0*m + j0][col_start] = 1
        U[(i0+1)*m + (j0+1)][col_start] = 1
        # V entries:
        V[j0*p + k0][col_start] = 1
        V[(j0+1)*p + (k0+1)][col_start] = 1
        # W entries:
        # C[i0,k0] += M1
        # C[i0+1,k0+1] += M1
        W[k0*n + i0][col_start] = 1
        W[(k0+1)*n + (i0+1)][col_start] = 1
        
        # M2 = (A[i0+1,j0]+A[i0+1,j0+1]) * B[j0,k0]
        U[(i0+1)*m + j0][col_start+1] = 1
        U[(i0+1)*m + (j0+1)][col_start+1] = 1
        V[j0*p + k0][col_start+1] = 1
        # C[i0+1,k0] += M2
        W[k0*n + (i0+1)][col_start+1] = 1
        
        # M3 = A[i0,j0] * (B[j0,k0+1]-B[j0+1,k0+1])
        U[i0*m + j0][col_start+2] = 1
        V[j0*p + (k0+1)][col_start+2] = 1
        V[(j0+1)*p + (k0+1)][col_start+2] = -1
        # C[i0,k0+1] += M3
        W[(k0+1)*n + i0][col_start+2] = 1
        
        # M4 = A[i0+1,j0+1] * (B[j0+1,k0]-B[j0,k0])
        U[(i0+1)*m + (j0+1)][col_start+3] = 1
        V[(j0+1)*p + k0][col_start+3] = 1
        V[j0*p + k0][col_start+3] = -1
        # C[i0,k0] += M4
        # C[i0+1,k0] += M4
        W[k0*n + i0][col_start+3] = 1
        W[k0*n + (i0+1)][col_start+3] = 1
        
        # M5 = (A[i0,j0]+A[i0,j0+1]) * B[j0+1,k0+1]
        U[i0*m + j0][col_start+4] = 1
        U[i0*m + (j0+1)][col_start+4] = 1
        V[(j0+1)*p + (k0+1)][col_start+4] = 1
        # C[i0,k0] -= M5
        # C[i0,k0+1] += M5
        W[k0*n + i0][col_start+4] = -1
        W[(k0+1)*n + i0][col_start+4] = 1
        
        # M6 = (A[i0+1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k0+1])
        U[(i0+1)*m + j0][col_start+5] = 1
        U[i0*m + j0][col_start+5] = -1
        V[j0*p + k0][col_start+5] = 1
        V[j0*p + (k0+1)][col_start+5] = 1
        # C[i0+1,k0+1] += M6
        W[(k0+1)*n + (i0+1)][col_start+5] = 1
        
        # M7 = (A[i0,j0+1]-A[i0+1,j0+1]) * (B[j0+1,k0]+B[j0+1,k0+1])
        U[i0*m + (j0+1)][col_start+6] = 1
        U[(i0+1)*m + (j0+1)][col_start+6] = -1
        V[(j0+1)*p + k0][col_start+6] = 1
        V[(j0+1)*p + (k0+1)][col_start+6] = 1
        # C[i0,k0] += M7
        # C[i0+1,k0+1] += M7
        W[k0*n + i0][col_start+6] = 1
        W[(k0+1)*n + (i0+1)][col_start+6] = 1

    # Helper to add a schoolbook M_{1,4,2} block
    def add_schoolbook_142(i_idx, j_start, k_start, col_start):
        # C[i_idx, k_start:k_start+2] from A[i_idx, j_start:j_start+4] * B[j_start:j_start+4, k_start:k_start+2]
        for j in range(4):
            for k in range(2):
                U[i_idx*m + (j_start + j)][col_start + j*2 + k] = 1
                V[(j_start + j)*p + (k_start + k)][col_start + j*2 + k] = 1
                W[(k_start + k)*n + i_idx][col_start + j*2 + k] = 1

    # Block 1: C[0:2, 0:2] from A[0:2, 0:2]*B[0:2, 0:2]
    add_strassen(0, 0, 0, col)
    col += 7
    
    # Block 2: C[0:2, 0:2] from A[0:2, 2:4]*B[2:4, 0:2]
    add_strassen(0, 2, 0, col)
    col += 7
    
    # Block 3: C[0:2, 2:4] from A[0:2, 0:2]*B[0:2, 2:4]
    add_strassen(0, 0, 2, col)
    col += 7
    
    # Block 4: C[0:2, 2:4] from A[0:2, 2:4]*B[2:4, 2:4]
    add_strassen(0, 2, 2, col)
    col += 7
    
    # Block 5: C[2, 0:2] from A[2, 0:4]*B[0:4, 0:2]
    add_schoolbook_142(2, 0, 0, col)
    col += 8
    
    # Block 6: C[2, 2:4] from A[2, 0:4]*B[0:4, 2:4]
    add_schoolbook_142(2, 0, 2, col)
    col += 8
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
