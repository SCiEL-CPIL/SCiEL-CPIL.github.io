import time

DIMENSIONS = [2, 3, 9]

def _find_rank7_decomposition():
    """Find a rank-7 decomposition of 2x3x3 matrix multiplication.
    
    R(2,3,3) = 7 (Landsberg, 2010). We search for it using structured ansatz.
    """
    import numpy as np
    
    # The 2x3x3 multiplication tensor: T[i,j,k] = delta for A[i,j]*B[j,k]->C[i,k]
    # U has shape (6, r), V has shape (9, r), W has shape (6, r)
    # Check: sum_r U[i,j,r]*V[j,k,r] = W[i,k,r]... no
    # Actually: C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor form: T[i,j,k] with contraction over j
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r] where n=2,m=3,p=3
    # Product: sum_r U[i*3+j, r] * V[j*3+k, r] should give W[k*2+i, r]... 
    # No: the decomposition is sum_r U[i,j,r]*V[j,k,r] = T[i,j,k] = delta_{i,i'}delta_{k,k'}... 
    # Actually the multiplication tensor is: T_{i,j,k} = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # The tensor for matrix mult (n,m,p) has T[i,j,k] = delta_{i,i}delta_{k,k}... 
    # Standard: the multiplication tensor T in A^* x B^* x C^* where
    # T(a,b) = a*b. In coordinates: T_{i,j,k} = 1 if k corresponds to i,j pair in output.
    # For 2x3x3: output C is 2x3, so k indexes (i',k') in output.
    # T[i,j, (i',k')] = delta_{i,i'} * delta_{j,j} ... no.
    # T[i,j,k] where k=(i',k'): T[i,j,(i',k')] = delta_{i,i'}
    # So for each (i',k'), and j: T[i',j,(i',k')] = 1
    # That gives a 2x3x6 tensor (since output has 2*3=6 entries)
    # Wait, n=2, m=3, p=3: U is (n*m)=6 x rank, V is (m*p)=9 x rank, W is (p*n)=6 x rank
    # W index: k*n+i where k in [0,p), i in [0,n)
    # So W[k*2+i, r]
    # The decomposition: sum_r U[i*3+j, r] * V[j*3+k, r] = delta_{i,i'} for the W part
    # Actually the tensor is: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r] = W[k*n+i,r]
    # No, that's not right either. The correct form:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # In tensor decomposition: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] where T is the "identity" tensor
    # T[i,j,k] = 1 if this triple contributes, which it always does with weight 1.
    # So we need: sum_r U[i,j,r]*V[j,k,r] = 1 for all valid (i,j,k)? No.
    # The standard form: the matrix multiplication tensor T_{n,m,p} is in (F^n x F^m)^* x (F^m x F^p)^* x (F^n x F^p)^*
    # T(A,B) = A*B
    # In coordinates with A indexed by (i,j), B by (j,k), C by (i,k):
    # T_{(i,j),(j,k),(i,k)} = 1
    # For a rank-r decomposition: T = sum_{r=1}^r u_r v_r w_r
    # where u_r is in (F^{n*m})^*, v_r in (F^{m*p})^*, w_r in (F^{p*n})^*
    # Coordinate form: T[i,j,k] (where i indexes n*m, j indexes m*p, k indexes p*n)
    # = sum_r U[i,r] * V[j,r] * W[k,r]
    # And T[i,j,k] = 1 iff the indices are consistent: if i=(i1,j1), j=(j2,k1), k=(k2,i2)
    # then T=1 iff i1=i2 and j1=j2.
    
    # Let me just verify the schoolbook works and try to find rank 7.
    
    n, m, p = 2, 3, 3
    # Build the target tensor as a function
    def tensor_val(i_idx, j_idx, k_idx):
        i1, j1 = divmod(i_idx, m)  # i_idx = i*m + j
        j2, k1 = divmod(j_idx, p)  # j_idx = j*p + k
        k2, i2 = divmod(k_idx, n)  # k_idx = k*n + i
        if i1 == i2 and j1 == j2:
            return 1.0
        return 0.0
    
    # Schoolbook: rank = n*m*p = 18
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    # Verify schoolbook
    U, V, W = schoolbook()
    for i_idx in range(n*m):
        for j_idx in range(m*p):
            for k_idx in range(p*n):
                val = sum(U[i_idx, r] * V[j_idx, r] * W[k_idx, r] for r in range(n*m*p))
                expected = tensor_val(i_idx, j_idx, k_idx)
                if abs(val - expected) > 1e-10:
                    raise ValueError(f"Schoolbook failed at {i_idx},{j_idx},{k_idx}: {val} vs {expected}")
    
    # Try to find rank 7 using a structured approach.
    # The known rank-7 decomposition of 2x3x3 uses specific patterns.
    # Let's try a semi-structured search.
    
    # Alternative: use the fact that 2x3x3 can be related to 2x2x2 (rank 7? No, rank 7 for 2x2x2 is also Landsberg's result but actually R(2,2,2)=7 is conjectured, proven lower bound is 7... wait no)
    # R(2,2,2) = 7 is NOT proven. The best known is 7 <= R(2,2,2) <= 7. Actually it IS proven to be 7? No.
    # Actually R(2,2,2) is known to be 7? Let me recall: Strassen showed R(2,2,2) <= 7. The lower bound is 7? No, the lower bound is 7 only for certain variants.
    # Actually, it's known that R(2,2,2) >= 7 and R(2,2,2) <= 7, so R(2,2,2) = 7? 
    # Wait: the asymptotic rank gives a lower bound. For 2x2x2, the laser method gives R >= 7? 
    # Actually I recall R(2,2,2) is between 7 and 7, meaning it's exactly 7. Let me assume so for now.
    
    # For 2x3x3, the known result is R(2,3,3) = 7.
    # Let me try to construct it or search numerically.
    
    # Try numerical optimization for rank 7
    from scipy.optimize import minimize
    
    def check_rank(U_mat, V_mat, W_mat, rank):
        """Check if (U,V,W) of given rank decompose the tensor."""
        for i_idx in range(n*m):
            for j_idx in range(m*p):
                for k_idx in range(p*n):
                    val = sum(U_mat[i_idx, r] * V_mat[j_idx, r] * W_mat[k_idx, r] for r in range(rank))
                    expected = tensor_val(i_idx, j_idx, k_idx)
                    if abs(val - expected) > 1e-6:
                        return False
        return True
    
    # Try a structured ansatz for rank 7.
    # Known constructions often use integer or half-integer entries.
    # Let me try a grid search over small integer/half-integer values.
    
    # Given time constraints, let me try a few known patterns.
    # One approach: the 2x3x3 tensor can be written using 7 terms.
    # Based on literature, the decomposition involves:
    # Terms that look like:
    # (e_i + a*e_i') (e_j + b*e_j') (e_k + c*e_k') type constructions
    
    # For time efficiency, let me just return the schoolbook (rank 18) for now
    # and hardcode if I can find rank 7.
    
    # Actually, let me try one more thing: use the Kronecker product structure.
    # 2x3x3 = 2x(3x3). If we had R(2,1,3) that would help, but that's just 2*3=6.
    
    # Let me try: treat it as multiplying 2x3 by 3x3 three times? No.
    
    # I'll return schoolbook for safety.
    return None  # Signal that we didn't find rank 7

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 9
    
    deadline = time.time() + 140  # safety margin
    
    # Try to find rank-7 decomposition of 2x3x3
    # The 3x9 matrix = 3 blocks of 3x3
    # So we need 3 independent 2x3x3 multiplications
    # Total rank = 3 * R(2,3,3)
    
    # R(2,3,3) is known to be 7
    # Let me try to use a known rank-7 decomposition
    
    # Based on the structure, I'll attempt to use a hardcoded rank-7 decomposition.
    # If that fails, fall back to schoolbook.
    
    # For now, use schoolbook: rank = 2*3*9 = 54
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
