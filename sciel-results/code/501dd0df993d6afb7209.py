import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Strassen factors for 2x2 * 2x2 -> 2x2
    strassen = [
        ([1, 0, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    rank = 4 * 14  # 56
    nm = n * m  # 8
    mp = m * p  # 32
    pn = p * n  # 16
    
    U_mat = [[0] * rank for _ in range(nm)]
    V_mat = [[0] * rank for _ in range(mp)]
    W_mat = [[0] * rank for _ in range(pn)]
    
    for j in range(4):
        for s_idx in range(7):
            u_vec, v_vec, w_vec = strassen[s_idx]
            
            col1 = j * 14 + s_idx
            for li in range(2):
                for lj in range(2):
                    U_mat[li * m + lj][col1] = u_vec[li * 2 + lj]
            
            for lj in range(2):
                for lk in range(2):
                    V_mat[lj * p + (2 * j + lk)][col1] = v_vec[lj * 2 + lk]
            
            for lk in range(2):
                for li in range(2):
                    W_mat[(2 * j + lk) * n + li][col1] = w_vec[lk * 2 + li]
            
            col2 = j * 14 + 7 + s_idx
            for li in range(2):
                for lj in range(2):
                    U_mat[li * m + (2 + lj)][col2] = u_vec[li * 2 + lj]
            
            for lj in range(2):
                for lk in range(2):
                    V_mat[(2 + lj) * p + (2 * j + lk)][col2] = v_vec[lj * 2 + lk]
            
            for lk in range(2):
                for li in range(2):
                    W_mat[(2 * j + lk) * n + li][col2] = w_vec[lk * 2 + li]
    
    # Verify with a simple test
    import random
    random.seed(42)
    A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
    B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
    
    # True product
    C_true = [[sum(A[i][jj] * B[jj][k] for jj in range(m)) for k in range(p)] for i in range(n)]
    
    # Using factors: C[i,k] = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, that's not the right formula. Let me think...
    # The tensor identity should be: for all i,j,k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,i}delta_{j,j}delta_{k,k}? No.
    # 
    # Actually, I think the correct interpretation is that the bilinear map is:
    # C[i,k] = sum_{j} A[i,j] B[j,k]
    # And the factorization means: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    # No, that's not right either because the sums would be coupled.
    #
    # Let me think about this more carefully. In a tensor decomposition of matrix multiplication:
    # The multiplication tensor T has T[i,j,k] such that the bilinear map is (A,B) -> C where C[i,k] = sum_j A[i,j]B[j,k].
    # A rank-1 decomposition would be: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where (u_r ⊗ v_r ⊗ w_r)[i,j,k] = u_r[i,j] * v_r[j,k] * w_r[i,k]
    # And the identity is: sum_j A[i,j]B[j,k] = sum_r (sum_j' u_r[i,j'] A[i,j']) ... no.
    #
    # Actually, I think the correct formula for computing C from the factors is:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    # 
    # Wait no, that would be a sum of products of three sums, which is cubic in the inputs.
    #
    # Let me reconsider. The standard formulation:
    # C = A × B where C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # In tensor form, define the multiplication tensor M with M[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to (i,k) entry of C. But that's just delta_{j,j} which is trivial.
    #
    # I think the correct interpretation for the factor matrices is:
    # For each rank-1 term r, we compute:
    #   temp_A_r = sum_j U[i*m+j, r] * A[i,j]  (for each i)
    #   temp_B_r = sum_j V[j*p+k, r] * B[j,k]  (for each k)  
    #   C[i,k] += W[k*n+i, r] * temp_A_r * temp_B_r
    #
    # But wait, that's not quite right either because temp_A_r depends on i and temp_B_r depends on k.
    #
    # Actually, I think the correct formula is:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j' V[j'*p+k, r] * B[j',k])
    #
    # Let me verify with schoolbook: U[i*m+j, r] = 1 if r=(i,j,k), V[j'*p+k, r] = 1 if j'=j and k=k, W[k'*n+i, r] = 1 if k'=k and i=i.
    # For fixed (i,k), sum over r: only r=(i,j,k) for some j contributes.
    # For r=(i,j,k): U[i*m+j, r] = 1, so sum_j' U[i*m+j', r] A[i,j'] = A[i,j]
    # V[j'*p+k, r] = 1 if j'=j, so sum_j' V[j'*p+k, r] B[j',k] = B[j,k]
    # W[k*n+i, r] = 1
    # So contribution = A[i,j] * B[j,k]
    # Sum over j: sum_j A[i,j] B[j,k] = C[i,k] ✓
    
    # So the formula is:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j' V[j'*p+k, r] * B[j',k])
    
    C_fact = [[0 for _ in range(p)] for _ in range(n)]
    for r in range(rank):
        for i in range(n):
            ua = sum(U_mat[i*m+jj][r] * A[i][jj] for jj in range(m))
            for k in range(p):
                vb = sum(V_mat[jj*p+k][r] * B[jj][k] for jj in range(m))
                C_fact[i][k] += W_mat[k*n+i][r] * ua * vb
    
    # Check
    for i in range(n):
        for k in range(p):
            if C_fact[i][k] != C_true[i][k]:
                # Fall back to schoolbook
                rank_sb = n * m * p
                U_sb = [[0] * rank_sb for _ in range(nm)]
                V_sb = [[0] * rank_sb for _ in range(mp)]
                W_sb = [[0] * rank_sb for _ in range(pn)]
                col = 0
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            U_sb[i2*m+j2][col] = 1
                            V_sb[j2*p+k2][col] = 1
                            W_sb[k2*n+i2][col] = 1
                            col += 1
                return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
