import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS

    # We will build a decomposition using Strassen for 2x2x2 blocks
    # and simple rank-1 terms for remaining dimensions.
    
    # The matrix multiplication tensor T_{i,j,k} = delta_{i,j,k} (if i,j,k are valid indices for the product)
    # Specifically for C = AB where A is n x m and B is m x p:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor is T_{i,j,k} = 1 if (i,j) is a valid A index and (j,k) is a valid B index, else 0.
    # Actually, the tensor decomposition represents the identity of the multiplication operation.
    # T_{i,j,k} = 1 for all valid i in [0,n), j in [0,m), k in [0,p).
    
    # We want to find U, V, W such that T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r}
    
    # Strategy:
    # 1. Split the multiplication into blocks.
    #    Rows of A: 0..2 (n=3)
    #    Cols of A / Rows of B: 0..3 (m=4)
    #    Cols of B: 0..6 (p=7)
    
    # We can use Strassen for the 2x2x2 sub-multiplications.
    # Strassen decomposes the 2x2 matrix multiplication tensor.
    # For a 2x2x2 block (i in {0,1}, j in {0,1}, k in {0,1}), the rank is 7.
    
    # Let's partition the indices:
    # i: {0, 1, 2}
    # j: {0, 1, 2, 3}
    # k: {0, 1, 2, 3, 4, 5, 6}
    
    # We can pair i={0,1} and handle i=2 separately.
    # For i={0,1}, we have a 2x4x7 multiplication.
    # For i=2, we have a 1x4x7 multiplication.
    
    # Rank of M_1,4,7:
    # T_{2,j,k} = 1.
    # This is a rank-1 tensor: u_2=1, v_j=1, w_k=1.
    # So 1 rank unit for row 2.
    
    # Rank of M_2,4,7:
    # We can split j and k.
    # j={0,1,2,3}, k={0,..,6}.
    # Split k into {0,1}, {2,3}, {4,5}, {6}.
    # Split j into {0,1}, {2,3}.
    
    # For k in {0,1}:
    #   j in {0,1}: 2x2x2 block -> Strassen (rank 7)
    #   j in {2,3}: 2x2x2 block -> Strassen (rank 7)
    # For k in {2,3}:
    #   j in {0,1}: 2x2x2 block -> Strassen (rank 7)
    #   j in {2,3}: 2x2x2 block -> Strassen (rank 7)
    # For k in {4,5}:
    #   j in {0,1}: 2x2x2 block -> Strassen (rank 7)
    #   j in {2,3}: 2x2x2 block -> Strassen (rank 7)
    # For k in {6}:
    #   j in {0,1}: 2x2x1 block.
    #   j in {2,3}: 2x2x1 block.
    #   A 2x2x1 multiplication can be done with rank 2?
    #   C_{i,0} = A_{i,0}B_{0,0} + A_{i,1}B_{1,0}
    #   This is sum of two rank-1 terms.
    #   So rank 2 for each 2x2x1 block.
    #   Total for k=6: 2 + 2 = 4.
    
    # Total rank for M_2,4,7: 7*6 + 4 = 42 + 4 = 46?
    # Wait, 6 blocks of 2x2x2?
    # k pairs: (0,1), (2,3), (4,5). That's 3 pairs.
    # j pairs: (0,1), (2,3). That's 2 pairs.
    # So 3 * 2 = 6 blocks of 2x2x2.
    # Rank = 6 * 7 = 42.
    # Plus k=6 part:
    # j={0,1}: 2x2x1. Rank 2.
    # j={2,3}: 2x2x1. Rank 2.
    # Total 4.
    # Total M_2,4,7 rank = 46.
    # Plus M_1,4,7 rank = 1.
    # Total rank = 47.
    
    # Is this minimal?
    # Known bounds:
    # M_2,4,4 rank is 14 (2 * 7).
    # M_2,4,7 = M_2,4,4 + M_2,4,3.
    # M_2,4,3 = M_2,4,2 + M_2,4,1.
    # M_2,4,2 rank 14? No, M_2,4,2 = 2 * M_2,2,2? No.
    # M_2,4,2 can be done with 2 * 7 = 14?
    # Actually, M_2,4,2 is sum of two 2x2x2 multiplications?
    # A (2x4) B (4x2).
    # A = [A1 A2], B = [B1; B2].
    # C = A1 B1 + A2 B2.
    # This is sum of two 2x2x2 multiplications.
    # Rank = 7 + 7 = 14.
    # So M_2,4,2 rank <= 14.
    # M_2,4,4 = M_2,4,2 + M_2,4,2 (shifted k). Rank <= 28.
    # M_2,4,7 = M_2,4,4 + M_2,4,2 + M_2,4,1.
    # M_2,4,1 rank?
    # C_{i,0} = sum_j A_{i,j} B_{j,0}.
    # This is a bilinear form.
    # Rank of M_2,4,1 is 2?
    # T_{i,j,0} = 1.
    # sum_r U_{i,r} V_{j,r} W_{0,r}.
    # If we set W_{0,r}=1, then sum_r U_{i,r} V_{j,r} = 1.
    # This is the identity matrix of size 4 (for j) ? No.
    # It's a 2x4 matrix of 1s.
    # Rank of the matrix of 1s (2x4) is 1.
    # So we can do it with 1 rank term?
    # U_{0,0}=1, U_{1,0}=1.
    # V_{j,0}=1 for all j.
    # W_{0,0}=1.
    # Then T_{i,j,0} = U_{i,0} V_{j,0} W_{0,0} = 1 * 1 * 1 = 1.
    # Yes, rank 1 for M_2,4,1.
    # So M_2,4,7 rank <= 28 + 14 + 1 = 43.
    # Plus M_1,4,7 rank 1.
    # Total rank <= 44.
    
    # Let's implement rank 44.
    
    rank = 44
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    r_idx = 0
    
    # Helper to add a Strassen 2x2x2 block
    # i0, i1: row indices
    # j0, j1: col indices of A / row indices of B
    # k0, k1: col indices of B
    
    def add_strassen(i0, j0, k0):
        nonlocal r_idx
        # Indices:
        # A: (i0,j0), (i0,j1), (i1,j0), (i1,j1) where j1=j0+1, i1=i0+1
        # B: (j0,k0), (j0,k1), (j1,k0), (j1,k1) where k1=k0+1
        # C: (i0,k0), (i0,k1), (i1,k0), (i1,k1)
        
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1
        
        # P1 = (A00+A11)(B00+B11)
        # U: A00, A11
        # V: B00, B11
        # W: C00, C11
        U[i0*m + j0, r_idx] += 1
        U[i1*m + j1, r_idx] += 1
        V[j0*p + k0, r_idx] += 1
        V[j1*p + k1, r_idx] += 1
        W[k0*n + i0, r_idx] += 1
        W[k1*n + i1, r_idx] += 1
        r_idx += 1
        
        # P2 = (A10+A11)B00
        # U: A10, A11
        # V: B00
        # W: C10
        U[i1*m + j0, r_idx] += 1
        U[i1*m + j1, r_idx] += 1
        V[j0*p + k0, r_idx] += 1
        W[k0*n + i1, r_idx] += 1
        r_idx += 1
        
        # P3 = A00(B01-B11)
        # U: A00
        # V: B01, -B11
        # W: C01
        U[i0*m + j0, r_idx] += 1
        V[j0*p + k1, r_idx] += 1
        V[j1*p + k1, r_idx] -= 1
        W[k1*n + i0, r_idx] += 1
        r_idx += 1
        
        # P4 = A11(B10-B00)
        # U: A11
        # V: B10, -B00
        # W: C00, C10
        U[i1*m + j1, r_idx] += 1
        V[j1*p + k0, r_idx] += 1
        V[j0*p + k0, r_idx] -= 1
        W[k0*n + i0, r_idx] += 1
        W[k0*n + i1, r_idx] += 1
        r_idx += 1
        
        # P5 = (A00+A01)B11
        # U: A00, A01
        # V: B11
        # W: -C00, C01
        U[i0*m + j0, r_idx] += 1
        U[i0*m + j1, r_idx] += 1
        V[j1*p + k1, r_idx] += 1
        W[k0*n + i0, r_idx] -= 1
        W[k1*n + i0, r_idx] += 1
        r_idx += 1
        
        # P6 = (A10-A00)(B00+B01)
        # U: A10, -A00
        # V: B00, B01
        # W: C11
        U[i1*m + j0, r_idx] += 1
        U[i0*m + j0, r_idx] -= 1
        V[j0*p + k0, r_idx] += 1
        V[j0*p + k1, r_idx] += 1
        W[k1*n + i1, r_idx] += 1
        r_idx += 1
        
        # P7 = (A01-A11)(B10+B11)
        # U: A01, -A11
        # V: B10, B11
        # W: C00, C11
        U[i0*m + j1, r_idx] += 1
        U[i1*m + j1, r_idx] -= 1
        V[j1*p + k0, r_idx] += 1
        V[j1*p + k1, r_idx] += 1
        W[k0*n + i0, r_idx] += 1
        W[k1*n + i1, r_idx] += 1
        r_idx += 1

    # Part 1: Rows 0-1 (2x4) * Cols 0-6 (4x7)
    # Split Cols 0-6 into three 2-col blocks and one 1-col block
    
    # Block 1: Cols 0-1
    # A_L = A[0:2, 0:2], A_R = A[0:2, 2:4]
    # B_T = B[0:2, 0:2], B_B = B[2:4, 0:2]
    add_strassen(0, 0, 0) # A_L B_T
    add_strassen(0, 2, 0) # A_R B_B
    
    # Block 2: Cols 2-3
    # A_L = A[0:2, 0:2], A_R = A[0:2, 2:4]
    # B_T = B[0:2, 2:4], B_B = B[2:4, 2:4]
    add_strassen(0, 0, 2) # A_L B_T
    add_strassen(0, 2, 2) # A_R B_B
    
    # Block 3: Cols 4-5
    # A_L = A[0:2, 0:2], A_R = A[0:2, 2:4]
    # B_T = B[0:2, 4:6], B_B = B[2:4, 4:6]
    add_strassen(0, 0, 4) # A_L B_T
    add_strassen(0, 2, 4) # A_R B_B
    
    # Block 4: Col 6
    # A is 2x4, B is 4x1
    # Rank 1 term
    for i in range(2):
        for j in range(4):
            U[i*m + j, r_idx] += 1
    for j in range(4):
        V[j*p + 6, r_idx] += 1
    for i in range(2):
        W[6*n + i, r_idx] += 1
    r_idx += 1
    
    # Part 2: Row 2 (1x4) * Cols 0-6 (4x7)
    # A is 1x4, B is 4x7.
    # Rank 1 term
    for j in range(4):
        U[2*m + j, r_idx] += 1
    for j in range(4):
        for k in range(7):
            V[j*p + k, r_idx] += 1
    for k in range(7):
        W[k*n + 2, r_idx] += 1
    r_idx += 1
    
    if r_idx != rank:
        # This shouldn't happen if our counting is correct
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
