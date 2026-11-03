import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 118
    
    n, m, p = 3, 4, 6
    
    # The tensor equation: for all i,j,k:
    # sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] = 1 if j==k else 0
    
    # Build the rank-66 decomposition
    U66 = [[0] * 66 for _ in range(n * m)]
    V66 = [[0] * 66 for _ in range(m * p)]
    W66 = [[0] * 66 for _ in range(p * n)]
    
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
        
        U66[A00][col] += 1; U66[A11][col] += 1
        V66[B00][col] += 1; V66[B11][col] += 1
        W66[C00][col] += 1; W66[C11][col] += 1
        col += 1
        U66[A10][col] += 1; U66[A11][col] += 1
        V66[B00][col] += 1
        W66[C10][col] += 1; W66[C11][col] -= 1
        col += 1
        U66[A00][col] += 1
        V66[B01][col] += 1; V66[B11][col] -= 1
        W66[C01][col] += 1; W66[C11][col] += 1
        col += 1
        U66[A11][col] += 1
        V66[B10][col] += 1; V66[B00][col] -= 1
        W66[C00][col] += 1; W66[C10][col] += 1
        col += 1
        U66[A00][col] += 1; U66[A01][col] += 1
        V66[B11][col] += 1
        W66[C00][col] -= 1; W66[C01][col] += 1
        col += 1
        U66[A10][col] += 1; U66[A00][col] -= 1
        V66[B00][col] += 1; V66[B01][col] += 1
        W66[C11][col] += 1
        col += 1
        U66[A01][col] += 1; U66[A11][col] -= 1
        V66[B10][col] += 1; V66[B11][col] += 1
        W66[C00][col] += 1
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
        
        U66[A0][col] += 1; V66[B00][col] += 1; W66[C0][col] += 1
        col += 1
        U66[A1][col] += 1; V66[B10][col] += 1; W66[C0][col] += 1
        col += 1
        U66[A0][col] += 1; V66[B01][col] += 1; W66[C1][col] += 1
        col += 1
        U66[A1][col] += 1; V66[B11][col] += 1; W66[C1][col] += 1
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
    
    # Try to reduce rank by finding linear dependence
    # Build the 66 rank-1 tensors as vectors in R^(12*24*18)
    # Check if they are linearly dependent
    U_np = np.array(U66, dtype=float)
    V_np = np.array(V66, dtype=float)
    W_np = np.array(W66, dtype=float)
    
    # Each column l defines a tensor T_l with T_l[i,j,k] = U[i,j,l]*V[j,k,l]*W[k,i,l]
    # We need: sum_l T_l[i,j,k] = 1 if j==k else 0
    # Check if the 66 tensors are linearly dependent
    # Build matrix where each column is a flattened tensor
    tensor_matrix = np.zeros((n*m*p, 66))
    for l in range(66):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor_matrix[i*m*p + j*p + k, l] = U_np[i*m+j, l] * V_np[j*p+k, l] * W_np[k*n+i, l]
    
    # Check rank of tensor_matrix
    # If rank < 66, there's a linear dependence
    try:
        rank_tensor = np.linalg.matrix_rank(tensor_matrix)
        if rank_tensor < 66:
            # Find null space vector
            # Use SVD to find the null space
            U_svd, s, Vt = np.linalg.svd(tensor_matrix, full_matrices=False)
            # The null space is spanned by the last (66 - rank) rows of Vt
            # Find the first null vector
            null_vec = Vt[-1, :]  # last row of Vt
            # Check which component is largest to eliminate
            max_idx = np.argmax(np.abs(null_vec))
            # Express column max_idx as combination of others
            # null_vec[max_idx] * col_max_idx = sum_{l != max_idx} null_vec[l] * col_l
            # So col_max_idx = sum_{l != max_idx} (null_vec[l] / null_vec[max_idx]) * col_l
            
            # Create new matrices with column max_idx eliminated
            U_new = np.zeros((n*m, 65))
            V_new = np.zeros((m*p, 65))
            W_new = np.zeros((p*n, 65))
            
            new_col = 0
            for l in range(66):
                if l == max_idx:
                    continue
                U_new[:, new_col] = U_np[:, l]
                V_new[:, new_col] = V_np[:, l]
                W_new[:, new_col] = W_np[:, l]
                new_col += 1
            
            # Verify the decomposition still works
            # sum_l' U_new[i,j,l']*V_new[j,k,l']*W_new[k,i,l'] should equal original
            # = sum_{l != max_idx} U[i,j,l]*V[j,k,l]*W[k,i,l]
            # = original - U[i,j,max_idx]*V[j,k,max_idx]*W[k,i,max_idx]
            # We need this to still equal 1 if j==k else 0
            # This only works if the null vector has the right sign
            
            # Actually, we need: sum_{l != max_idx} null_vec[l]/null_vec[max_idx] * T_l = T_max_idx
            # So the new decomposition uses coefficients
            # Let's just check if removing one column and adjusting works
            
            # For simplicity, let's try: can we just drop one column if it's in the span?
            # The answer is yes if the tensors are linearly dependent
            
            # Convert to lists with exact fractions if possible
            # For now, use the reduced rank
            U_out = U_new.tolist()
            V_out = V_new.tolist()
            W_out = W_new.tolist()
            
            # Verify
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for l in range(65):
                            s += U_out[i*m+j][l] * V_out[j*p+k][l] * W_out[k*n+i][l]
                        expected = 1.0 if j == k else 0.0
                        err = max(err, abs(s - expected))
            
            if err < 1e-10:
                return {"U": U_out, "V": V_out, "W": W_out}
    except Exception:
        pass
    
    # Return the rank-66 decomposition
    return {"U": U66, "V": V66, "W": W66}
