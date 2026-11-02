import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS
    
    # Strassen 4x4 decomposition for 3x3 blocks
    # We embed 3x3 into 4x4 with a 1 in the (4,4) corner to maintain multiplicative structure?
    # Actually, we can just use the 7 products and the linear combinations.
    # The rank is 7.
    
    # Define the 7 linear forms for A and B in 4x4 Strassen
    # A is 4x4, B is 4x4
    # M1 = (A11+A22)(B11+B22)
    # ...
    
    # For 3x3, we pad with 0s except for the (4,4) entry which we set to 0?
    # If we pad with 0s everywhere, A44 = [[A33, 0], [0, 0]].
    # Then M1 = (A11+A22)(B11+B22). A22 is the (2,2) block of A44, which is the 3x3 A?
    # No, Strassen uses 2x2 blocks.
    # A44 is split into four 2x2 blocks: A11, A12, A21, A22.
    # If A is 3x3, we cannot fit it neatly into 2x2 blocks of a 4x4 matrix without overlapping or padding.
    # A 3x3 matrix padded to 4x4:
    # [ a11 a12 a13 0 ]
    # [ a21 a22 a23 0 ]
    # [ a31 a32 a33 0 ]
    # [  0   0   0  0 ]
    # This 4x4 matrix is not a block matrix of 2x2 blocks in the standard Strassen sense?
    # It is. A11 = [[a11, a12], [a21, a22]], A12 = [[a13, 0], [a23, 0]], etc.
    # So we can apply Strassen.
    # The 7 products are computed.
    # The result C44 is 4x4. The top-left 3x3 is the desired product.
    # So yes, rank is 7.
    
    # We will construct the U, V, W matrices for the whole 3x5 x 5x6 problem.
    # Split: 3x3 + 3x3 + 3x2x6
    # Rank = 7 + 7 + 36 = 50.
    
    R = 50
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # Helper to add Strassen 4x4 (for 3x3) decomposition
    # offset_u, offset_v, offset_w: indices in U, V, W
    # i_offset, k_offset, j_offset: indices in A, B, C
    def add_strassen_3x3(offset, i_off, k_off, j_off):
        # Strassen 4x4 terms
        # We need to map the 4x4 indices to the global indices.
        # A is 3x3, padded to 4x4.
        # A[i, k] for i,k in 0..2. A[3, *] = 0, A[*, 3] = 0.
        # B is 3x3, padded to 4x4.
        # B[k, j] for k,j in 0..2. B[3, *] = 0, B[*, 3] = 0.
        
        # Strassen 2x2 blocks (indices in 4x4 matrix)
        # A11: i=0,1; k=0,1
        # A12: i=0,1; k=2,3
        # A21: i=2,3; k=0,1
        # A22: i=2,3; k=2,3
        
        # Terms:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # We define the linear combinations for U, V, W.
        # U[i*m+k, r] = coeff
        # V[k*p+j, r] = coeff
        # W[j*n+i, r] = 1 (or coeff if output is linear combo)
        # Actually, W is just the output index.
        # The linear combinations are in U and V.
        
        # Let's list the 7 terms.
        # Each term is a product of a linear combo of A and a linear combo of B.
        # U contains the A linear combos.
        # V contains the B linear combos.
        # W contains the C linear combos (which are sums of M's).
        # Wait, the decomposition is C = sum_r U_r V_r W_r.
        # W_r is a matrix? No, W is a vector/matrix in the tensor decomposition.
        # In the tensor T = sum u_r v_r w_r, u, v, w are vectors.
        # Here U, V, W are matrices.
        # U[i, r] is the coefficient of A[i] in u_r.
        # V[k, r] is the coefficient of B[k] in v_r.
        # W[j, r] is the coefficient of C[j] in w_r?
        # No, the standard form is C[i,j] = sum_k A[i,k] B[k,j].
        # Tensor T[i,k,j] = delta.
        # Decomposition: T = sum_r u_r[i] v_r[k] w_r[j].
        # So U[i, r] = u_r[i]. V[k, r] = v_r[k]. W[j, r] = w_r[j].
        # And C[i,j] = sum_r (sum_k U[i,r] A[i,k]) (sum_l V[l,r] B[l,j]) W[j,r]?
        # No.
        # C[i,j] = sum_r U[i,r] V[r, ?] ...
        # The standard tensor decomposition for matrix multiplication is:
        # C_{ij} = \sum_{k} A_{ik} B_{kj} = \sum_{r=1}^R U_{ir} V_{kr} W_{jr} ?
        # No, that's not right.
        # The rank-R decomposition is:
        # C_{ij} = \sum_{r=1}^R (\sum_{k} U_{ik}^{(r)} A_{ik}) (\sum_{l} V_{kl}^{(r)} B_{kl}) W_{jr}^{(r)} ?
        # No.
        # The bilinear form is:
        # C_{ij} = \sum_{r=1}^R U_{i,r} V_{k,r} W_{j,r} ?
        # This sums over r. But k is summed over in the original product.
        # The tensor T_{ikj} has indices i, k, j.
        # T_{ikj} = \delta_{ik} \delta_{kj}? No.
        # T_{ikj} is 1 if k=k and j=j?
        # The multiplication tensor T has entries T_{i,k,j} = 1 if the product A_{ik} B_{kj} contributes to C_{ij}.
        # Actually, T_{i,k,j} is the coefficient of A_{ik} B_{kj} in C_{ij}.
        # It is 1 for all i,k,j.
        # So T_{ikj} = 1.
        # We want T = \sum_{r=1}^R u_r \otimes v_r \otimes w_r.
        # T_{ikj} = \sum_{r} u_{r,i} v_{r,k} w_{r,j}.
        # So C_{ij} = \sum_{k} A_{ik} B_{kj} = \sum_{k} \sum_{r} u_{r,i} v_{r,k} w_{r,j} A_{ik} B_{kj}.
        # This is not quite right.
        # The correct identity is:
        # C_{ij} = \sum_{r} (\sum_{k} u_{r,i} A_{ik}) (\sum_{l} v_{r,l} B_{lj}) w_{r,j} ?
        # No.
        # The standard form is:
        # C_{ij} = \sum_{r} U_{i,r} V_{r,k} W_{j,r} ?
        
        # Let's use the explicit U, V, W matrices from the task description.
        # U (n*m, R), V (m*p, R), W (p*n, R).
        # C_{i,j} = \sum_{k} A_{i,k} B_{k,j}
        # = \sum_{r} (\sum_{k} U_{i*m+k, r} A_{i,k}) (\sum_{l} V_{l*p+j, r} B_{l,j}) ?
        # No, the indices don't match.
        
        # The correct reconstruction is:
        # C_{i,j} = \sum_{r} W_{j*n+i, r} * ( \sum_{k} U_{i*m+k, r} A_{i,k} ) * ( \sum_{l} V_{l*p+j, r} B_{l,j} ) ?
        # No, U and V are multiplied element-wise?
        # The tensor T_{i,k,j} = \sum_{r} U_{i*m+k, r} V_{k*p+j, r} W_{j*n+i, r}.
        # And we require T_{i,k,j} = 1 for all i,k,j.
        # So \sum_{r} U_{i*m+k, r} V_{k*p+j, r} W_{j*n+i, r} = 1.
        
        # So for Strassen, we need to find U, V, W such that this holds.
        # This is complex to hardcode.
        
        # Given the time, I will stick to the **schoolbook** fallback but ensure it is robust.
        # The current program already does this.
        # The only improvement is if I can **verify** a lower rank.
        # Since I cannot reliably hardcode rank 23 or 7 in this format, I will return the schoolbook result.
        
        pass

    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k][col] = 1
                V[k * p + j][col] = 1
                W[j * n + i][col] = 1
                col += 1
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return {"U": U, "V": V, "W": W}

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k][col] = 1
                V[k * p + j][col] = 1
                W[j * n + i][col] = 1
                col += 1
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return {"U": U, "V": V, "W": W}
