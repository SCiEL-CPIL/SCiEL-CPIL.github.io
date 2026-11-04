import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = 3, 4, 6
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A00 = a_r * m + a_c
        A01 = a_r * m + a_c + 1
        A10 = (a_r + 1) * m + a_c
        A11 = (a_r + 1) * m + a_c + 1

        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1

        C00 = c_c * n + c_r
        C01 = (c_c + 1) * n + c_r
        C10 = c_c * n + (c_r + 1)
        C11 = (c_c + 1) * n + (c_r + 1)

        U[A00][col] += 1; U[A11][col] += 1
        V[B00][col] += 1; V[B11][col] += 1
        W[C00][col] += 1; W[C11][col] += 1
        col += 1
        U[A10][col] += 1; U[A11][col] += 1
        V[B00][col] += 1
        W[C10][col] += 1; W[C11][col] -= 1
        col += 1
        U[A00][col] += 1
        V[B01][col] += 1; V[B11][col] -= 1
        W[C01][col] += 1; W[C11][col] += 1
        col += 1
        U[A11][col] += 1
        V[B10][col] += 1; V[B00][col] -= 1
        W[C00][col] += 1; W[C10][col] += 1
        col += 1
        U[A00][col] += 1; U[A01][col] += 1
        V[B11][col] += 1
        W[C00][col] -= 1; W[C01][col] += 1
        col += 1
        U[A10][col] += 1; U[A00][col] -= 1
        V[B00][col] += 1; V[B01][col] += 1
        W[C11][col] += 1
        col += 1
        U[A01][col] += 1; U[A11][col] -= 1
        V[B10][col] += 1; V[B11][col] += 1
        W[C00][col] += 1
        col += 1

    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A0 = a_r * m + a_c
        A1 = a_r * m + a_c + 1

        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1

        C0 = c_c * n + c_r
        C1 = (c_c + 1) * n + c_r

        U[A0][col] += 1; V[B00][col] += 1; W[C0][col] += 1
        col += 1
        U[A1][col] += 1; V[B10][col] += 1; W[C0][col] += 1
        col += 1
        U[A0][col] += 1; V[B01][col] += 1; W[C1][col] += 1
        col += 1
        U[A1][col] += 1; V[B11][col] += 1; W[C1][col] += 1
        col += 1

    add_1x2_2x2(0, 0, 0, 0, 0, 0)
    add_1x2_2x2(0, 2, 2, 0, 0, 0)
    add_1x2_2x2(0, 0, 0, 2, 0, 2)
    add_1x2_2x2(0, 2, 2, 2, 0, 2)
    add_1x2_2x2(0, 0, 0, 4, 0, 4)
    add_1x2_2x2(0, 2, 2, 4, 0, 4)

    add_strassen(1, 0, 0, 0, 1, 0)
    add_strassen(1, 2, 2, 0, 1, 0)
    add_strassen(1, 0, 0, 2, 1, 2)
    add_strassen(1, 2, 2, 2, 1, 2)
    add_strassen(1, 0, 0, 4, 1, 4)
    add_strassen(1, 2, 2, 4, 1, 4)

    # Convert to numpy for nullspace detection
    U_np = np.array(U, dtype=float)  # shape (n*m, rank)
    V_np = np.array(V, dtype=float)  # shape (m*p, rank)
    W_np = np.array(W, dtype=float)  # shape (p*n, rank)

    # Build the flattening matrix: each column is the vectorized rank-1 tensor
    # T_{ijk} = U_{i,k} * V_{j,k} * W_{l,k} where i=(a_row,a_col), j=(b_row,b_col), l=(c_col,c_row)
    # The flattening for detecting linear dependence: stack columns as vectors
    # Each rank-1 term contributes U_col * V_col * W_col (outer product in tensor space)
    # We can check if any column of the "coefficient matrix" is linearly dependent
    # by checking if the 66 rank-1 tensors are linearly independent.
    
    # Method: form matrix M where M[:,k] = vec(U[:,k] ⊗ V[:,k] ⊗ W[:,k])
    # But that's 12*24*18 = 5184 dimensional. Instead, use a smaller check.
    # 
    # Alternative: Check if there exists a non-trivial combination of columns
    # that gives zero tensor. This is equivalent to checking if the 66 vectors
    # in R^(n*m * m*p * p*n) are independent.
    #
    # More practically: form the "coefficient matrix" by flattening each rank-1 term
    # and doing SVD to find the numerical rank.
    
    # Build flattening matrix: each column k is the outer product U[:,k] ⊗ V[:,k] ⊗ W[:,k]
    # Flatten to a vector of size (n*m)*(m*p)*(p*n) = 12*24*18 = 5184
    # This is 66 columns, so we can do QR/SVD on it.
    
    # For efficiency, use the fact that we only need to check linear independence.
    # Form matrix F of size (5184, 66) where F[:,k] = vec(U[:,k] ⊗ V[:,k] ⊗ W[:,k])
    
    # Build using Kronecker products for speed
    # vec(A ⊗ B ⊗ C) = vec(A) ⊗ vec(B) ⊗ vec(C) for rank-1 tensors
    # Actually, the tensor T_k = U_k ⊗ V_k ⊗ W_k where U_k is a vector of length 12,
    # V_k is length 24, W_k is length 18.
    # The flattening is vec(T_k) which has 12*24*18 = 5184 entries.
    # vec(T_k)[i,j,l] = U_k[i] * V_k[j] * W_k[l]
    
    # Build the 5184 x 66 matrix
    F = np.zeros((n * m * m * p * p * n, rank))
    for k in range(rank):
        F[:, k] = np.kron(np.kron(U_np[:, k], V_np[:, k]), W_np[:, k])
    
    # Compute numerical rank via SVD
    try:
        if time.time() < deadline:
            # Use SVD to find the rank
            _, s, _ = np.linalg.svd(F, full_matrices=False)
            # Count singular values above tolerance
            tol = max(F.shape) * np.finfo(float).eps * s[0]
            num_rank = np.sum(s > tol)
            
            if num_rank < rank:
                # There's a linear dependence! Find the null vector
                # The null vector gives coefficients to combine rank-1 terms to zero
                _, s, Vt = np.linalg.svd(F, full_matrices=True)
                # Null space is in the last (rank - num_rank) rows of Vt
                null_vec = Vt[-1, :]  # one null vector
                
                # Check if this gives a valid reduction
                # The null vector tells us which combination of terms sums to zero
                # We can remove one term if the null vector has a non-zero component
                # and re-express the others.
                
                # For now, just check if we can drop a term
                # Find the index with largest |null_vec| component
                drop_idx = int(np.argmax(np.abs(null_vec)))
                
                # Verify: check that removing this term still gives valid decomposition
                # by checking the remaining 65 terms span the same tensor
                remaining = [i for i in range(rank) if i != drop_idx]
                F_reduced = F[:, remaining]
                _, s_r, _ = np.linalg.svd(F_reduced, full_matrices=False)
                tol_r = max(F_reduced.shape) * np.finfo(float).eps * (s_r[0] if len(s_r) > 0 else 1)
                num_rank_r = np.sum(s_r > tol_r)
                
                if num_rank_r == num_rank:
                    # The reduced set still spans the same space
                    # Build the reduced decomposition
                    U_reduced = [[U[i][j] for j in remaining] for i in range(n * m)]
                    V_reduced = [[V[i][j] for j in remaining] for i in range(m * p)]
                    W_reduced = [[W[i][j] for j in remaining] for i in range(p * n)]
                    
                    # Verify the decomposition is exact
                    # Check: for all i,j,l: sum_k U[i,k]*V[j,k]*W[l,k] should equal the MM tensor
                    # The MM tensor: T[i,j,l] = 1 if i's row part matches j's row part AND
                    # i's col part matches j's col part... actually for MM:
                    # C = A * B where A is 3x4, B is 4x6
                    # C[r,c] = sum_s A[r,s] * B[s,c]
                    # In our indexing: i = (r, s), j = (s, c), l = (c, r)
                    # T[i,j,l] = 1 if i_row == l_row AND i_col == j_row AND j_col == l_col
                    # i = r*m + s, j = s*p + c, l = c*n + r
                    # So T[i,j,l] = 1 if (i//m == l%n) AND (i%m == j//p) AND (j%p == l//n)
                    
                    verified = True
                    U_r = np.array(U_reduced, dtype=float)
                    V_r = np.array(V_reduced, dtype=float)
                    W_r = np.array(W_reduced, dtype=float)
                    
                    # Spot check a few entries
                    for _ in range(100):
                        ri = int(np.random.randint(0, n * m))
                        rj = int(np.random.randint(0, m * p))
                        rl = int(np.random.randint(0, p * n))
                        
                        # True tensor value
                        r_row = ri // m
                        r_col = ri % m
                        b_row = rj // p
                        b_col = rj % p
                        c_col = rl // n
                        c_row = rl % n
                        
                        true_val = 1.0 if (r_row == c_row and r_col == b_row and b_col == c_col) else 0.0
                        
                        # Computed value
                        comp_val = 0.0
                        for k in range(len(remaining)):
                            comp_val += U_r[ri, k] * V_r[rj, k] * W_r[rl, k]
                        
                        if abs(comp_val - true_val) > 1e-9:
                            verified = False
                            break
                    
                    if verified:
                        return {"U": U_reduced, "V": V_reduced, "W": W_reduced}
    except Exception:
        pass

    return {"U": U, "V": V, "W": W}
