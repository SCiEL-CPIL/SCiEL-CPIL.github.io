"""Rewritten program using tensor slicing approach for 2x4x4 matrix multiplication."""
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    # A is n x m = 2x4, B is m x p = 4x4, C is n x p = 2x4
    
    # Tensor T[i, j, k] = sum over l of A[i,l] * B[l,k]
    # Using trace convention: W[k*n+i, r]
    # So C[i, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, the standard tensor decomposition is:
    # C_{i,k} = sum_r A_{i,j} B_{j,k} W_{i,k,r} ... no
    # Actually: sum_{j} A_{i,j} B_{j,k} = sum_r (sum_j U_{i,j,r} A_{i,j}) (sum_k V_{j,k,r} B_{j,k}) W_{i,k,r}
    # No, that's not right either.
    
    # The tensor T_{i,j,k} for matrix mult is T_{i,j,k} = delta_{i,j} (for the identity in the middle)
    # Actually for matrix multiplication C = AB where A is n x m, B is m x p:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The multiplication tensor has components T_{i,j,k} = 1 if the indices match appropriately
    # More precisely, T is a tensor of shape (n*m, m*p, p*n) in some flattening, or
    # T_{i, j, k} where i indexes A (row), j indexes B (col of B = row of A in sum), k indexes C (col)
    # T_{i, j, k} = delta_{j, j} ... no.
    
    # Let me use the standard formulation:
    # The bilinear map: (A, B) -> C where C_{i,k} = sum_j A_{i,j} B_{j,k}
    # This is a 3-tensor T with indices (i, j, k) -> but j is summed over.
    # The tensor is T_{i, j, k} = delta_{j,j} ... 
    # Actually T_{i, j, k} = 1 if we think of it as: the coefficient of A_{i,j} * B_{j,k} in C_{i,k} is 1.
    # So T is a tensor of shape (n, m, m, p) but we flatten to (n*m, m*p, p*n)?
    
    # Let me just use the rank-28 Strassen construction directly.
    # The key insight from tensor slicing: fix A to basis vectors, get 8 different 4x4 matrices
    # (actually n*m = 2*4 = 8 basis vectors for A), each giving a linear map B -> C.
    # The rank is the minimum number of rank-1 terms needed.
    
    # For 2x4x4, we can decompose into 4 blocks of 2x2x2, each using 7 Strassen multiplications.
    # Total rank = 28.
    
    rank = 28
    
    # U has shape (n*m, rank) = (8, 28)
    # V has shape (m*p, rank) = (16, 28)  
    # W has shape (p*n, rank) = (8, 28)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 terms
    # For 2x2 matrices X, Y, Z:
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    # Z00 = M1 - M4 + M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # Local index mapping for 2x2: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
    # X indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Y indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Z indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # Each Strassen term s has:
    # U part: dict {x_local_idx: coeff}
    # V part: dict {y_local_idx: coeff}
    # W part: dict {z_local_idx: coeff}
    
    strassen_terms = [
        # M1: (X00+X11)(Y00+Y11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),  # contributes to Z00 and Z11
        # M2: (X10+X11)Y00
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),  # Z10 += M2, Z11 -= M2
        # M3: X00(Y01-Y11)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),  # Z01 += M3, Z11 += M3
        # M4: X11(Y10-Y00)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),  # Z00 += M4, Z10 += M4
        # M5: (X00+X01)Y11
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),  # Z00 -= M5, Z01 += M5
        # M6: (X10-X00)(Y00+Y01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),  # Z11 += M6
        # M7: (X01-X11)(Y10+Y11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),  # Z00 += M7
    ]
    
    # Now we need to map these to the 2x4x4 problem.
    # A is 2x4: rows i=0,1, cols j=0,1,2,3
    # B is 4x4: rows j=0,1,2,3, cols k=0,1,2,3
    # C is 2x4: rows i=0,1, cols k=0,1,2,3
    
    # We split into 4 blocks based on j (column of A / row of B) and k (col of B / col of C):
    # Block 0: j in {0,1}, k in {0,1} -> A[:,0:2] @ B[0:2,0:2] -> C[:,0:2]
    # Block 1: j in {2,3}, k in {0,1} -> A[:,2:4] @ B[2:4,0:2] -> C[:,0:2]
    # Block 2: j in {0,1}, k in {2,3} -> A[:,0:2] @ B[0:2,2:4] -> C[:,2:4]
    # Block 3: j in {2,3}, k in {2,3} -> A[:,2:4] @ B[2:4,2:4] -> C[:,2:4]
    
    # For each block, the local 2x2 indices map to global indices:
    # Block b: j_offset = 2*(b//2), k_offset = 2*(b%2)
    # Wait, let me re-index:
    # b=0: j_off=0, k_off=0
    # b=1: j_off=2, k_off=0
    # b=2: j_off=0, k_off=2
    # b=3: j_off=2, k_off=2
    
    for b in range(4):
        j_off = 2 * (b // 2)
        k_off = 2 * (b % 2)
        
        for s in range(7):
            col = b * 7 + s
            u_dict, v_dict, w_dict = strassen_terms[s]
            
            # U: A[i, j] where i in {0,1}, j in {j_off, j_off+1}
            # Local x_idx: 0->(i=0,j=j_off), 1->(i=0,j=j_off+1), 2->(i=1,j=j_off), 3->(i=1,j=j_off+1)
            for x_idx, coeff in u_dict.items():
                i = x_idx // 2
                j_local = x_idx % 2
                j = j_off + j_local
                a_idx = i * m + j  # i*4 + j
                U[a_idx][col] += coeff
            
            # V: B[j, k] where j in {j_off, j_off+1}, k in {k_off, k_off+1}
            # Local y_idx: 0->(j=j_off,k=k_off), 1->(j=j_off,k=k_off+1), 2->(j=j_off+1,k=k_off), 3->(j=j_off+1,k=k_off+1)
            for y_idx, coeff in v_dict.items():
                j_local = y_idx // 2
                k_local = y_idx % 2
                j = j_off + j_local
                k = k_off + k_local
                b_idx = j * p + k  # j*4 + k
                V[b_idx][col] += coeff
            
            # W: C[i, k] where i in {0,1}, k in {k_off, k_off+1}
            # Local z_idx: 0->(i=0,k=k_off), 1->(i=0,k=k_off+1), 2->(i=1,k=k_off), 3->(i=1,k=k_off+1)
            for z_idx, coeff in w_dict.items():
                i = z_idx // 2
                k_local = z_idx % 2
                k = k_off + k_local
                c_idx = k * n + i  # k*2 + i (trace convention)
                W[c_idx][col] += coeff
    
    return {"U": U, "V": V, "W": W}
