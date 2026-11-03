import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    n, m, p = 3, 4, 6
    
    R = 66
    U = [[0] * R for _ in range(n * m)]  # (12, 66)
    V = [[0] * R for _ in range(m * p)]  # (24, 66)
    W = [[0] * R for _ in range(p * n)]  # (18, 66)
    
    col = 0
    
    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        """Add Strassen 2x2*2x2 block at given offsets."""
        nonlocal col
        # A indices
        A00 = a_r * m + a_c
        A01 = a_r * m + a_c + 1
        A10 = (a_r + 1) * m + a_c
        A11 = (a_r + 1) * m + a_c + 1
        
        # B indices
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        
        # C indices (Trace convention: W[k*n + i, r])
        # c_r is row i, c_c is col k
        C00 = c_c * n + c_r
        C01 = (c_c + 1) * n + c_r
        C10 = c_c * n + (c_r + 1)
        C11 = (c_c + 1) * n + (c_r + 1)
        
        # M1 = (A00+A11)(B00+B11)
        U[A00][col] += 1; U[A11][col] += 1
        V[B00][col] += 1; V[B11][col] += 1
        W[C00][col] += 1; W[C11][col] += 1
        col += 1
        # M2 = (A10+A11)B00
        U[A10][col] += 1; U[A11][col] += 1
        V[B00][col] += 1
        W[C10][col] += 1; W[C11][col] -= 1
        col += 1
        # M3 = A00(B01-B11)
        U[A00][col] += 1
        V[B01][col] += 1; V[B11][col] -= 1
        W[C01][col] += 1; W[C11][col] += 1
        col += 1
        # M4 = A11(B10-B00)
        U[A11][col] += 1
        V[B10][col] += 1; V[B00][col] -= 1
        W[C00][col] += 1; W[C10][col] += 1
        col += 1
        # M5 = (A00+A01)B11
        U[A00][col] += 1; U[A01][col] += 1
        V[B11][col] += 1
        W[C00][col] -= 1; W[C01][col] += 1
        col += 1
        # M6 = (A10-A00)(B00+B01)
        U[A10][col] += 1; U[A00][col] -= 1
        V[B00][col] += 1; V[B01][col] += 1
        W[C11][col] += 1
        col += 1
        # M7 = (A01-A11)(B10+B11)
        U[A01][col] += 1; U[A11][col] -= 1
        V[B10][col] += 1; V[B11][col] += 1
        W[C00][col] += 1
        col += 1
    
    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        """Add 1x2*2x2 block (rank 4) at given offsets."""
        nonlocal col
        # A indices (1x2)
        A0 = a_r * m + a_c
        A1 = a_r * m + a_c + 1
        
        # B indices (2x2)
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        
        # C indices (1x2)
        C0 = c_c * n + c_r
        C1 = (c_c + 1) * n + c_r
        
        # M1: A0*B00
        U[A0][col] += 1; V[B00][col] += 1; W[C0][col] += 1
        col += 1
        # M2: A1*B10
        U[A1][col] += 1; V[B10][col] += 1; W[C0][col] += 1
        col += 1
        # M3: A0*B01
        U[A0][col] += 1; V[B01][col] += 1; W[C1][col] += 1
        col += 1
        # M4: A1*B11
        U[A1][col] += 1; V[B11][col] += 1; W[C1][col] += 1
        col += 1
    
    # Build decomposition:
    # C[0,0:2] = A[0,0:2]*B[0:2,0:2] + A[0,2:4]*B[2:4,0:2]
    add_1x2_2x2(0, 0, 0, 0, 0, 0)   # 1x2*2x2, rank 4
    add_1x2_2x2(0, 2, 2, 0, 0, 0)   # 1x2*2x2, rank 4
    
    # C[0,2:4] = A[0,0:2]*B[0:2,2:4] + A[0,2:4]*B[2:4,2:4]
    add_1x2_2x2(0, 0, 0, 2, 0, 2)   # 1x2*2x2, rank 4
    add_1x2_2x2(0, 2, 2, 2, 0, 2)   # 1x2*2x2, rank 4
    
    # C[0,4:6] = A[0,0:2]*B[0:2,4:6] + A[0,2:4]*B[2:4,4:6]
    add_1x2_2x2(0, 0, 0, 4, 0, 4)   # 1x2*2x2, rank 4
    add_1x2_2x2(0, 2, 2, 4, 0, 4)   # 1x2*2x2, rank 4
    
    # C[1:3,0:2] = A[1:3,0:2]*B[0:2,0:2] + A[1:3,2:4]*B[2:4,0:2]
    add_strassen(1, 0, 0, 0, 1, 0)   # 2x2*2x2, rank 7
    add_strassen(1, 2, 2, 0, 1, 0)   # 2x2*2x2, rank 7
    
    # C[1:3,2:4] = A[1:3,0:2]*B[0:2,2:4] + A[1:3,2:4]*B[2:4,2:4]
    add_strassen(1, 0, 0, 2, 1, 2)   # 2x2*2x2, rank 7
    add_strassen(1, 2, 2, 2, 1, 2)   # 2x2*2x2, rank 7
    
    # C[1:3,4:6] = A[1:3,0:2]*B[0:2,4:6] + A[1:3,2:4]*B[2:4,4:6]
    add_strassen(1, 0, 0, 4, 1, 4)   # 2x2*2x2, rank 7
    add_strassen(1, 2, 2, 4, 1, 4)   # 2x2*2x2, rank 7
    
    assert col == 66, f"Expected 66 columns, got {col}"
    
    # Convert to numpy arrays for nullspace analysis
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check for linear dependencies in the tensor decomposition
    # The tensor T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # We can flatten this to check if the rank-1 components are linearly independent
    
    # Create the flattening matrix where each column is a rank-1 component
    # T_flattened has shape (n*m*p, R) where T_flattened[idx, r] = U[i,r]*V[j,r]*W[k,r]
    # for the appropriate i,j,k corresponding to idx
    
    nmp = n * m * p  # 72
    R_current = R  # 66
    
    # Build the flattening matrix
    T_flat = np.zeros((nmp, R_current))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = (i * m + j) * p + k  # Flatten A[i,j]*B[j,k] -> C[i,k]
                # Actually, let's use the trace convention: W[k*n+i, r]
                # So C[i,k] corresponds to W index k*n + i
                w_idx = k * n + i
                for r in range(R_current):
                    T_flat[idx, r] = U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[w_idx, r]
    
    # Now check the rank of T_flat to see if there are linear dependencies
    # If rank < R_current, we can reduce the number of terms
    try:
        rank_T = np.linalg.matrix_rank(T_flat, tol=1e-10)
    except:
        rank_T = R_current
    
    if rank_T < R_current:
        # There are linear dependencies. Find the nullspace and remove dependent columns
        try:
            # Find columns to remove
            # Use SVD to find the nullspace
            U_svd, s, Vt = np.linalg.svd(T_flat, full_matrices=False)
            
            # Find which columns are in the nullspace (small singular values)
            # We want to keep R_current - (R_current - rank_T) = rank_T columns
            # The columns corresponding to small singular values can be removed
            
            # Get the right singular vectors
            # Vt has shape (min(nmp, R_current), R_current)
            # We want to identify which original columns are linearly dependent
            
            # Alternative approach: find a basis for the column space
            # and express each column in terms of the basis
            
            # Use QR decomposition to find independent columns
            Q, R_qr = np.linalg.qr(T_flat, mode='reduced')
            
            # Find pivots in R_qr
            pivots = []
            for col_idx in range(R_current):
                if abs(R_qr[col_idx, col_idx]) > 1e-10 if col_idx < min(nmp, R_current) else True:
                    pivots.append(col_idx)
            
            # If we have fewer pivots than R_current, we can remove some columns
            if len(pivots) < R_current:
                # Remove non-pivot columns
                keep_cols = pivots
                new_R = len(keep_cols)
                
                # Create new matrices with only the kept columns
                U_new = U_np[:, keep_cols]
                V_new = V_np[:, keep_cols]
                W_new = W_np[:, keep_cols]
                
                # Convert back to lists
                U_result = U_new.tolist()
                V_result = V_new.tolist()
                W_result = W_new.tolist()
                
                return {"U": U_result, "V": V_result, "W": W_result}
        except:
            pass
    
    # Return the original 66-rank decomposition
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
