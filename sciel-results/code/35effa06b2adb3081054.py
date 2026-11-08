"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill

    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    
    # Strategy: Use Strassen's algorithm for 2x2 block multiplication.
    # A is 2x4, B is 4x8.
    # We can view A as a single row of two 2x2 blocks: [A_L, A_R] where A_L, A_R are 2x2.
    # We can view B as a column of two 2x4 blocks: [B_T; B_B] where B_T, B_B are 2x4.
    # Wait, standard Strassen applies to square matrices. 
    # Let's decompose the problem into 2x2 matrix multiplications.
    # C = A @ B
    # A (2x4) = [A_0 A_1] where A_0, A_1 are 2x2.
    # B (4x8) = [B_0 B_1 B_2 B_3] where B_i are 2x2.
    # C (2x8) = [C_0 C_1 C_2 C_3] where C_i are 2x2.
    # C_i = A_0 @ B_{2i} + A_1 @ B_{2i+1}
    
    # Each term A_0 @ B_{2i} and A_1 @ B_{2i+1} is a 2x2 matrix multiplication.
    # There are 4 columns in C, so 4 such sums.
    # Total 8 instances of 2x2 matrix multiplication.
    # Standard rank for 2x2 is 4. Total rank 32.
    # Strassen rank for 2x2 is 7. Total rank 8 * 7 = 56.
    
    # Can we do better?
    # The tensor is T_{2,4,8}.
    # Lower bounds: max(n*m, m*p, p*n) = max(8, 32, 16) = 32.
    # So rank >= 32.
    # Schoolbook is 64.
    # Strassen-based is 56.
    # Is 32 achievable? Unlikely for these dimensions without specific structure.
    # Is 48 achievable? Maybe by sharing some computations?
    # Let's stick to 56 as a safe improvement over 64.
    # If we can find a rank 48 or lower decomposition, that would be better.
    # However, finding a non-trivial decomposition programmatically in 150s is hard.
    # Constructing the known Strassen decomposition is robust and fast.
    
    # Let's implement the rank 56 Strassen decomposition.
    
    # Strassen formulas for 2x2 matrices X, Y:
    # M1 = (X00 - X11)(Y11 - Y00)
    # M2 = (X00 + X11)Y00
    # M3 = X00(Y01 - Y11)
    # M4 = X11(Y10 - Y00)
    # M5 = (X00 + X01)Y11
    # M6 = (X10 - X00)(Y00 + Y01)
    # M7 = (X01 + X11)(Y10 - Y11)
    
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # We have 8 such multiplications:
    # For i in 0..3:
    #   C_{2i}   = A_0 @ B_{2i}
    #   C_{2i+1} = A_1 @ B_{2i+1}
    
    # Let's define the 7 rank-1 tensors for a generic 2x2 multiplication.
    # Indices for X (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for Y (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for Z (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # u, v, w vectors of length 4.
    
    # M1: (X00 - X11)(Y11 - Y00)
    u1 = [1, 0, 0, -1]
    v1 = [0, 0, -1, 1]  # Y11 - Y00 -> index 3 is 1, index 0 is -1? No.
    # Y indices: 0:00, 1:01, 2:10, 3:11
    # Y11 is index 3. Y00 is index 0.
    # So v1 should be [ -1, 0, 0, 1 ] ?
    # Let's check: sum(v * Y) = -Y00 + Y11. Correct.
    v1 = [-1, 0, 0, 1]
    # Z contribution: Z00 += M1, Z11 += M1
    w1 = [1, 0, 0, 1]
    
    # M2: (X00 + X11)Y00
    u2 = [1, 0, 0, 1]
    v2 = [1, 0, 0, 0]
    # Z10 += M2, Z11 -= M2
    w2 = [0, 0, 1, -1]
    
    # M3: X00(Y01 - Y11)
    u3 = [1, 0, 0, 0]
    v3 = [0, 1, 0, -1] # Y01 - Y11
    # Z01 += M3, Z11 += M3
    w3 = [0, 1, 0, 1]
    
    # M4: X11(Y10 - Y00)
    u4 = [0, 0, 0, 1]
    v4 = [0, 0, 1, -1] # Y10 - Y00
    # Z00 += M4, Z10 += M4
    w4 = [1, 0, 1, 0]
    
    # M5: (X00 + X01)Y11
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    # Z00 -= M5, Z01 += M5
    w5 = [-1, 1, 0, 0]
    
    # M6: (X10 - X00)(Y00 + Y01)
    u6 = [-1, 0, 1, 0] # -X00 + X10
    v6 = [1, 1, 0, 0] # Y00 + Y01
    # Z11 += M6
    w6 = [0, 0, 0, 1]
    
    # M7: (X01 + X11)(Y10 - Y11)
    u7 = [0, 1, 0, 1]
    v7 = [0, 0, 1, -1] # Y10 - Y11
    # Z00 += M7, Z11 -= M7
    w7 = [1, 0, 0, -1]
    
    strassen_terms = [
        (u1, v1, w1),
        (u2, v2, w2),
        (u3, v3, w3),
        (u4, v4, w4),
        (u5, v5, w5),
        (u6, v6, w6),
        (u7, v7, w7)
    ]
    
    # Verify Strassen locally
    def verify_strassen():
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        C_ref = A @ B
        C_calc = np.zeros((2, 2))
        for u, v, w in strassen_terms:
            u_a = np.array(u)
            v_b = np.array(v)
            w_c = np.array(w)
            # Flatten A, B to match index order: 00, 01, 10, 11
            A_flat = A.flatten()
            B_flat = B.flatten()
            scalar_u = np.dot(u_a, A_flat)
            scalar_v = np.dot(v_b, B_flat)
            C_calc_flat = C_calc.flatten()
            C_calc_flat += scalar_u * scalar_v * w_c
            C_calc = C_calc_flat.reshape(2, 2)
        return np.allclose(C_calc, C_ref)
    
    if not verify_strassen():
        # Fallback to schoolbook if Strassen implementation is wrong
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
        
    # Build the full decomposition
    rank = 56
    U = [[0] * rank for _ in range(n * m)] # 8 x 56
    V = [[0] * rank for _ in range(m * p)] # 32 x 56
    W = [[0] * rank for _ in range(p * n)] # 16 x 56
    
    current_col = 0
    
    # A is 2x4. Split into A_L (cols 0,1) and A_R (cols 2,3).
    # B is 4x8. Split into 4 blocks of 2x2?
    # B rows 0,1 are top, rows 2,3 are bottom.
    # B_{2i} is rows 0,1, cols 2i, 2i+1.
    # B_{2i+1} is rows 2,3, cols 2i, 2i+1.
    
    for i in range(4):
        # Block i of C (2x2) is at C[:, 2i:2i+2]
        # C_{2i} = A_L @ B_{2i}
        # A_L is A[:, 0:2] -> indices in U: i_row in 0..1, j_col in 0..1
        # B_{2i} is B[0:2, 2i:2i+2] -> indices in V: j_row in 0..1, k_col in 2i..2i+1
        
        for u, v, w in strassen_terms:
            # u corresponds to A_L (2x2)
            # Global U index: i_row * m + j_col = i_row * 4 + j_col
            # u indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            for li in range(2):
                for lj in range(2):
                    val = u[li * 2 + lj]
                    if val != 0:
                        U[li * m + lj][current_col] += val
            
            # v corresponds to B_{2i} (2x2)
            # Global V index: j_row * p + k_col
            # j_row is 0..1. k_col is 2i .. 2i+1
            for lj in range(2):
                for lk in range(2):
                    val = v[lj * 2 + lk]
                    if val != 0:
                        V[lj * p + (2 * i + lk)][current_col] += val
            
            # w corresponds to C_{2i} (2x2)
            # Global W index: k_row * n + i_col
            # k_row is 2i .. 2i+1. i_col is 0..1
            for lk in range(2):
                for li in range(2):
                    val = w[lk * 2 + li]
                    if val != 0:
                        W[(2 * i + lk) * n + li][current_col] += val
            
            current_col += 1
            
        # C_{2i+1} = A_R @ B_{2i+1}
        # A_R is A[:, 2:4] -> indices in U: i_row in 0..1, j_col in 2..3
        # B_{2i+1} is B[2:4, 2i:2i+2] -> indices in V: j_row in 2..3, k_col in 2i..2i+1
        
        for u, v, w in strassen_terms:
            # u corresponds to A_R (2x2)
            # Global U index: i_row * m + (2 + j_col)
            for li in range(2):
                for lj in range(2):
                    val = u[li * 2 + lj]
                    if val != 0:
                        U[li * m + (2 + lj)][current_col] += val
            
            # v corresponds to B_{2i+1} (2x2)
            # Global V index: (2 + j_row) * p + k_col
            for lj in range(2):
                for lk in range(2):
                    val = v[lj * 2 + lk]
                    if val != 0:
                        V[(2 + lj) * p + (2 * i + lk)][current_col] += val
            
            # w corresponds to C_{2i+1} (2x2)
            # Global W index: k_row * n + i_col
            # k_row is 2i .. 2i+1. i_col is 0..1
            for lk in range(2):
                for li in range(2):
                    val = w[lk * 2 + li]
                    if val != 0:
                        W[(2 * i + lk) * n + li][current_col] += val
            
            current_col += 1

    # Final Verification
    def verify_full():
        A = np.random.randint(-10, 10, (n, m))
        B = np.random.randint(-10, 10, (m, p))
        C_ref = A @ B
        
        U_mat = np.array(U).T # rank x n*m
        V_mat = np.array(V).T # rank x m*p
        W_mat = np.array(W).T # rank x p*n
        
        # C[i, k] = sum_{j, r} A[i, j] V_mat[r, j*p+k] W_mat[r, k*n+i]
        # Efficiently:
        # Temp[r, j] = sum_i A[i, j] U_mat[r, i*m+j]
        # C[i, k] = sum_{j, r} Temp[r, j] V_mat[r, j*p+k] W_mat[r, k*n+i]
        
        # Let's just iterate for simplicity in verification (small sizes)
        C_calc = np.zeros((n, p))
        for r in range(rank):
            # u_r is U_mat[r, :]
            # v_r is V_mat[r, :]
            # w_r is W_mat[r, :]
            # Contribution to C is (A * u_r) outer (v_r) ... no.
            # The tensor is sum_r U_r \otimes V_r \otimes W_r
            # C[i,k] = sum_j A[i,j] B[j,k]
            # = sum_r (sum_j A[i,j] U_r[i*m+j]) (sum_l B[j,l] V_r[j*p+l]) W_r[k*n+i] ? No.
            
            # Correct formula:
            # C = sum_r (A * U_r) (V_r * B^T) W_r^T ?
            # Let's stick to the definition:
            # C[i,k] = sum_{j} A[i,j] B[j,k]
            # Decomposition:
            # A[i,j] B[j,k] = sum_r U_r[i*m+j] V_r[j*p+k] W_r[k*n+i]
            
            # So C[i,k] = sum_r W_r[k*n+i] * (sum_j A[i,j] U_r[i*m+j] ... wait.
            # The middle term is not a simple product of A and B parts unless we sum over j.
            # Let's compute the contribution of rank-r term:
            # T_r[i, k] = sum_j U_r[i*m+j] * A[i,j] * ... no.
            
            # Let's just use the explicit sum.
            # C[i,k] = sum_{j, r} A[i,j] * (U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] / U_r[i*m+j]?) No.
            
            # The identity is:
            # sum_{i,j,k} A[i,j] B[j,k] W[i,j,k] = sum_r (sum_{i,j} A[i,j] U_r[i,j]) (sum_{j,k} B[j,k] V_r[j,k]) (sum_{i,k} C[i,k] W_r[i,k])
            # This is for the tensor T_{ijk} = delta_{jk}? No.
            
            # The matrix multiplication tensor T has T[i,j,k] = 1 if it contributes to C[i,k] from A[i,j]B[j,k].
            # Actually, C[i,k] = sum_j A[i,j] B[j,k].
            # The tensor is T_{i,j,k} such that C = A * B corresponds to contraction.
            # The rank-1 decomposition is:
            # T = sum_r u_r \otimes v_r \otimes w_r
            # where u_r is vector over (i,j), v_r over (j,k), w_r over (i,k).
            # Then C[i,k] = sum_r w_r[i,k] * (sum_j A[i,j] u_r[i,j]) * (sum_j B[j,k] v_r[j,k])?
            # No, that's not right. The contraction is:
            # C[i,k] = sum_{j, r} A[i,j] B[j,k] (coefficient)
            
            # Let's check the seed program's logic.
            # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
            # Sum over c:
            # C[i,k] = sum_c W[k*n+i, c] * (sum_j A[i,j] U[i*m+j, c]) * (sum_j B[j,k] V[j*p+k, c])?
            # No, the standard bilinear form is:
            # C[i,k] = sum_c (sum_j A[i,j] U_c[i,j]) (sum_j B[j,k] V_c[j,k]) W_c[i,k]
            
            # Let's test this formula with the schoolbook case.
            # For fixed c corresponding to (i0, j0, k0):
            # U_c[i,j] is 1 if i=i0, j=j0, else 0.
            # V_c[j,k] is 1 if j=j0, k=k0, else 0.
            # W_c[i,k] is 1 if i=i0, k=k0, else 0.
            
            # Term c contributes to C[i,k]:
            # If i != i0 or k != k0, W_c[i,k] = 0, so contribution is 0.
            # If i=i0, k=k0:
            #   sum_j A[i0,j] U_c[i0,j] = A[i0,j0]
            #   sum_j B[j,k0] V_c[j,k0] = B[j0,k0]
            #   Contribution = A[i0,j0] * B[j0,k0] * 1.
            # Summing over all c (all i0,j0,k0) gives sum_{j0} A[i0,j0] B[j0,k0]. Correct.
            
            # So the verification formula is:
            # C_calc[i,k] = sum_r W_r[k*n+i] * (sum_j A[i,j] U_r[i*m+j]) * (sum_j B[j,k] V_r[j*p+k])
            
            # Let's implement this.
            
            for i in range(n):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        w_val = W_mat[r, k * n + i]
                        if w_val == 0: continue
                        u_sum = 0
                        for j in range(m):
                            u_sum += A[i, j] * U_mat[r, i * m + j]
                        v_sum = 0
                        for j in range(m):
                            v_sum += B[j, k] * V_mat[r, j * p + k]
                        val += w_val * u_sum * v_sum
                    C_calc[i, k] = val
        
        return np.allclose(C_calc, C_ref)

    if not verify_full():
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
