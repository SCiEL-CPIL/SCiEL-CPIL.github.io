import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 12, 32, 24
    
    # Construct the matrix multiplication tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=np.int8)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    def validate(U, V, W):
        """Check if U, V, W correctly decompose T."""
        rank = U.shape[1]
        T_recon = np.zeros((nm, mp, pn), dtype=np.int64)
        for r in range(rank):
            T_recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        return np.array_equal(T_recon, T.astype(np.int64))
    
    # Schoolbook baseline
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school), dtype=np.int8)
    V_school = np.zeros((mp, rank_school), dtype=np.int8)
    W_school = np.zeros((pn, rank_school), dtype=np.int8)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Block decomposition
    # Split m=4 into two blocks of 2: s=0 (j=0,1), s=1 (j=2,3)
    # Split p=8 into four blocks of 2: t=0 (k=0,1), t=1 (k=2,3), t=2 (k=4,5), t=3 (k=6,7)
    
    # For each (s, t), we have a 3x2x2 sub-problem.
    # We use Strassen (rank 7) for i=0,1 and schoolbook (rank 4) for i=2.
    # Total rank per sub-problem: 11.
    # Total rank: 8 * 11 = 88.
    
    U_block = np.zeros((nm, 88), dtype=np.int8)
    V_block = np.zeros((mp, 88), dtype=np.int8)
    W_block = np.zeros((pn, 88), dtype=np.int8)
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    
    # The tensor terms for Strassen are:
    # U, V, W vectors for each of the 7 terms.
    # For i in {0,1}, j0 in {0,1}, k0 in {0,1}.
    # Local indices:
    # u_idx = 4*i + 2*s + j0
    # v_idx = 16*s + 8*j0 + 2*t + k0
    # w_idx = 6*t + 3*k0 + i
    
    for s in range(2):
        for t in range(4):
            # Part 1: i in {0,1} (2x2x2 sub-problem)
            # We need to map Strassen's 7 terms to the global indices.
            # Strassen operates on 2x2 matrices.
            # Here, the "matrix" is effectively 2x2 in (j0, k0) space for each i?
            # No, the 2x2x2 tensor is for i in {0,1}, j0 in {0,1}, k0 in {0,1}.
            # This is exactly the 2x2x2 matrix multiplication tensor.
            
            # Strassen terms:
            # 1. (A11 + A22)(B11 + B22)
            # A11 corresponds to i=0, j0=0. A22 corresponds to i=1, j0=1.
            # B11 corresponds to j0=0, k0=0. B22 corresponds to j0=1, k0=1.
            
            # Term 1: U = e_{0,0} + e_{1,1}, V = e_{0,0} + e_{1,1}, W = e_{0,0} + e_{1,1}?
            # No, W is the output.
            # The output C is 2x2.
            # C11 = M1 + M4 - M5 - M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M7
            
            # We need to express the tensor T as sum of rank-1 terms.
            # T = sum_{i,j,k} e_i e_j e_k.
            # Using Strassen, we replace the 8 terms with 7 linear combinations.
            
            # Let's define the 7 Strassen terms in terms of U, V, W vectors.
            # U vectors are in R^2 (for i=0,1).
            # V vectors are in R^2 (for j0=0,1). Note: V index depends on j0 and k0.
            # W vectors are in R^2 (for k0=0,1). Note: W index depends on k0 and i.
            
            # This is tricky because V and W indices are coupled with j0 and k0.
            # But for a fixed s, t, the sub-tensor is isolated.
            
            # Let's list the 7 terms for the 2x2x2 part.
            # Term 1: M1 = (A11 + A22)(B11 + B22)
            # U1 = e_0 + e_1 (i=0,1)
            # V1 = e_0 + e_1 (j0=0,1) -- but V index is 16s + 8j0 + 2t + k0.
            # Wait, V is the middle matrix.
            # In the sub-problem, V is 2x2.
            # V1 = e_{j0=0, k0=0} + e_{j0=1, k0=1}.
            # W1 = e_{k0=0, i=0} + e_{k0=1, i=1}? No.
            # W is the output.
            # M1 contributes to C11, C12, C21, C22.
            # Actually, Strassen computes the product C = AB.
            # The tensor is T_{i,j,k} = 1.
            # The Strassen decomposition gives coefficients for the linear combinations.
            
            # Let's just use the known Strassen rank-7 decomposition for 2x2x2.
            # The 7 terms are:
            # 1. U=[1,1], V=[1,0,0,1], W=[1,0,0,1] ? No.
            
            # Let's construct it explicitly.
            # i in {0,1}, j0 in {0,1}, k0 in {0,1}.
            # u_idx = 4*i + 2*s + j0
            # v_idx = 16*s + 8*j0 + 2*t + k0
            # w_idx = 6*t + 3*k0 + i
            
            # Term 1: (A11 + A22)(B11 + B22)
            # U: i=0, j0=0 and i=1, j0=1.
            # V: j0=0, k0=0 and j0=1, k0=1.
            # W: This term contributes to all C_ik.
            # C11 gets +1, C12 gets +1, C21 gets +1, C22 gets +1.
            # So W vector has 1s at (k0=0, i=0), (k0=1, i=0), (k0=0, i=1), (k0=1, i=1).
            # W1 = e_{0,0} + e_{1,0} + e_{0,1} + e_{1,1} in (k0, i) space.
            
            # Let's add this term.
            U_block[u_idx(0, 0), col] += 1
            U_block[u_idx(1, 1), col] += 1
            V_block[v_idx(0, 0), col] += 1
            V_block[v_idx(1, 1), col] += 1
            W_block[w_idx(0, 0), col] += 1
            W_block[w_idx(1, 0), col] += 1
            W_block[w_idx(0, 1), col] += 1
            W_block[w_idx(1, 1), col] += 1
            col += 1
            
            # Term 2: (A21 + A22)B11
            # U: i=1, j0=0 and i=1, j0=1.
            # V: j0=0, k0=0.
            # W: Contributes to C21, C22.
            # C21 gets +1, C22 gets +1.
            # W2 = e_{0,1} + e_{1,1} in (k0, i) space.
            
            U_block[u_idx(1, 0), col] += 1
            U_block[u_idx(1, 1), col] += 1
            V_block[v_idx(0, 0), col] += 1
            W_block[w_idx(0, 1), col] += 1
            W_block[w_idx(1, 1), col] += 1
            col += 1
            
            # Term 3: A11(B21 - B22)
            # U: i=0, j0=0.
            # V: j0=1, k0=0 and j0=1, k0=1.
            # W: Contributes to C12, C22.
            # C12 gets +1, C22 gets -1.
            # W3 = e_{1,0} - e_{1,1} in (k0, i) space.
            
            U_block[u_idx(0, 0), col] += 1
            V_block[v_idx(1, 0), col] += 1
            V_block[v_idx(1, 1), col] -= 1
            W_block[w_idx(1, 0), col] += 1
            W_block[w_idx(1, 1), col] -= 1
            col += 1
            
            # Term 4: A22(B12 - B11)
            # U: i=1, j0=1.
            # V: j0=0, k0=0 and j0=0, k0=1.
            # W: Contributes to C11, C21.
            # C11 gets -1, C21 gets +1.
            # W4 = -e_{0,0} + e_{0,1} in (k0, i) space.
            
            U_block[u_idx(1, 1), col] += 1
            V_block[v_idx(0, 0), col] -= 1
            V_block[v_idx(0, 1), col] += 1
            W_block[w_idx(0, 0), col] -= 1
            W_block[w_idx(0, 1), col] += 1
            col += 1
            
            # Term 5: (A11 + A12)B22
            # U: i=0, j0=0 and i=0, j0=1.
            # V: j0=1, k0=1.
            # W: Contributes to C11, C12.
            # C11 gets -1, C12 gets +1.
            # W5 = -e_{0,0} + e_{1,0} in (k0, i) space.
            
            U_block[u_idx(0, 0), col] += 1
            U_block[u_idx(0, 1), col] += 1
            V_block[v_idx(1, 1), col] += 1
            W_block[w_idx(0, 0), col] -= 1
            W_block[w_idx(1, 0), col] += 1
            col += 1
            
            # Term 6: (A21 - A11)(B11 + B12)
            # U: i=1, j0=0 and i=0, j0=0.
            # V: j0=0, k0=0 and j0=0, k0=1.
            # W: Contributes to C21, C22.
            # C21 gets -1, C22 gets -1.
            # W6 = -e_{0,1} - e_{1,1} in (k0, i) space.
            
            U_block[u_idx(1, 0), col] += 1
            U_block[u_idx(0, 0), col] -= 1
            V_block[v_idx(0, 0), col] += 1
            V_block[v_idx(0, 1), col] += 1
            W_block[w_idx(0, 1), col] -= 1
            W_block[w_idx(1, 1), col] -= 1
            col += 1
            
            # Term 7: (A12 - A22)(B21 + B22)
            # U: i=0, j0=1 and i=1, j0=1.
            # V: j0=1, k0=0 and j0=1, k0=1.
            # W: Contributes to C11, C12.
            # C11 gets +1, C12 gets +1.
            # W7 = e_{0,0} + e_{1,0} in (k0, i) space.
            
            U_block[u_idx(0, 1), col] += 1
            U_block[u_idx(1, 1), col] -= 1
            V_block[v_idx(1, 0), col] += 1
            V_block[v_idx(1, 1), col] += 1
            W_block[w_idx(0, 0), col] += 1
            W_block[w_idx(1, 0), col] += 1
            col += 1
            
            # Part 2: i=2 (1x2x2 sub-problem)
            # Schoolbook: 4 terms.
            # j0 in {0,1}, k0 in {0,1}.
            for j0 in range(2):
                for k0 in range(2):
                    U_block[u_idx(2, j0), col] = 1
                    V_block[v_idx(j0, k0), col] = 1
                    W_block[w_idx(k0, 2), col] = 1
                    col += 1
    
    # Check if col == 88
    if col != 88:
        print(f"Warning: col is {col}, expected 88")
        
    # Validate
    if validate(U_block, V_block, W_block):
        best_U, best_V, best_W = U_block, V_block, W_block
        best_rank = 88
    else:
        print("Block decomposition failed validation, using schoolbook.")
        
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
