import numpy as np

def construct():
    n, m, p = 3, 3, 7
    
    # For 3x3x7 matrix multiplication, we need a tensor decomposition
    # U: (n*m) x rank = 9 x rank
    # V: (m*p) x rank = 21 x rank  
    # W: (p*n) x rank = 21 x rank
    # such that for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,j,k}
    # Wait, that's not quite right. The tensor is M_{i,j,k} where (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor T_{i,j,k} = 1 if we're computing (AB)_{i,k} from A_{i,j} and B_{j,k}
    # Actually, the standard formulation: T is a 3-tensor of size n x m x p
    # T[i,j,k] contributes to (AB)[i,k] via A[i,j] * B[j,k]
    # So we need: for each output (i,k), (AB)[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor decomposition: T = sum_r U_r ⊗ V_r ⊗ W_r
    # where U_r is in R^{n*m}, V_r in R^{m*p}, W_r in R^{p*n}
    # and T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # But wait, T[i,j,k] should be 1 for the "standard" basis element.
    # Actually, the multiplication tensor has T[i,j,k] = 1 if the term A[i,j]*B[j,k] appears in (AB)[i,k]
    # For each (i,k), the coefficient of A[i,j]*B[j,k] in (AB)[i,k] is 1.
    # So T[i,j,k] = 1 for all valid i,j,k? No, that's not right either.
    
    # Let me reconsider. The matrix multiplication tensor M_{n,m,p} is defined as:
    # M_{n,m,p} = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    # Wait, no. Let me look at this more carefully.
    # (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor is: T = sum_{i,j,k} A_{i,j} B_{j,k} * E_{i,k}
    # In tensor form: T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k}
    # So T[i,j,k] = 1 for all i,j,k? No, that would give sum_j A[i,j]B[j,k] for each (i,k)
    # which is exactly what we want.
    
    # So the tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Wait, that's not a sparse tensor. Let me re-examine.
    
    # Actually, I think the standard formulation is:
    # T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k}
    # So T[i,j,k] = 1 for all i,j,k.
    
    # A rank-r decomposition is: T = sum_{r=0}^{r-1} U_r ⊗ V_r ⊗ W_r
    # where U_r ∈ R^{n*m}, V_r ∈ R^{m*p}, W_r ∈ R^{n*p}
    # and T[i,j,k] = sum_{r} U_r[i*m+j] * V_r[j*p+k] * W_r[i*p+k]
    
    # For schoolbook: each term is U_{i,j,k} = e_{i,j}, V_{i,j,k} = e_{j,k}, W_{i,j,k} = e_{i,k}
    # So rank = n*m*p = 3*3*7 = 63.
    
    # Can we do better? For 3x3x3, the minimum rank is 23 (Bläser).
    # For 3x3x7, we might be able to do better than 63.
    
    # One approach: use a rank-23 decomposition for the 3x3x3 part and schoolbook for the rest.
    # But 3x3x7 is not simply 3x3x3 + 3x3x4.
    
    # Alternative: Use the fact that we can pad to 4x4x4 and use Strassen (rank 49),
    # but that's worse than 63? No, 49 < 63, so it would be better.
    # But padding 3x3x7 to 4x4x4 is not straightforward.
    
    # Let me try a different approach: use a known rank-23 decomposition for 3x3x3.
    # I'll hardcode the coefficients from a known source.
    
    # From the literature, a rank-23 decomposition for 3x3x3 exists.
    # Let me use the one from the paper "On the rank of the matrix multiplication tensor" 
    # or from standard references.
    
    # Actually, let me try to construct a better decomposition using a different strategy.
    # For now, let me use schoolbook but verify it's correct, then try to improve.
    
    # Schoolbook decomposition:
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Verify: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For schoolbook, this should give 1 for all i,j,k.
    
    # Let me verify:
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if val != 1:
                    raise ValueError(f"Verification failed at ({i},{j},{k}): got {val}")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
