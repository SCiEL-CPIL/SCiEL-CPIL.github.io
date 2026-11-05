import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    def schoolbook_3x3x5():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def verify_decomposition(U, V, W, n, m, p):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        rank = U.shape[1]
        # The tensor T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We check a random sample of (i,j,k) tuples
        np.random.seed(42)
        for trial in range(100):
            i = np.random.randint(0, n)
            j = np.random.randint(0, m)
            k = np.random.randint(0, p)
            expected = 1.0 if (i >= 0 and j >= 0 and k >= 0) else 0.0
            # Actually, the matrix multiplication tensor has T[i,j,k] = 1 if this is a valid (i,j,k) position
            # Wait, let me reconsider. The matrix multiplication tensor T satisfies:
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # So the bilinear form is: for each r, we have terms U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # This should reconstruct the multiplication. Let me verify by checking:
            # For fixed i,k: sum_j A[i,j]*B[j,k] = sum_j sum_r U[i*m+j,r]*W[k*n+i,r] * V[j*p+k,r]
            # = sum_r W[k*n+i,r] * (sum_j U[i*m+j,r] * V[j*p+k,r])
            # This equals sum_r W[k*n+i,r] * (U_i · V_k) where U_i is row i of U reshaped
            # Let me just do a full numerical check
            pass
        
        # Full check: for all i,j,k, sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should be 1 if (i,j,k) is in the tensor support
        # Actually, the correct check is:
        # The tensor T_{ijk} = delta (this is the structure tensor of matrix multiplication)
        # T_{ijk} = 1 if the indices correspond to a valid multiplication term, else 0
        # For matrix multiplication C = AB where A is n×m, B is m×p, C is n×p:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor is T[i,j,k] where the bilinear form is sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[k,i]
        # Wait, the convention is W[k*n+i, r], so the output index is (k, i) where k is row of C, i is col of C
        # Let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
        # So W has shape (p*n, rank), indexed by (k, i) where k in [0,p), i in [0,n)
        # U has shape (n*m, rank), indexed by (i, j) where i in [0,n), j in [0,m)  
        # V has shape (m*p, rank), indexed by (j, k) where j in [0,m), k in [0,p)
        # The decomposition should satisfy:
        # For all i in [0,n), j in [0,m), k in [0,p):
        #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if this is a "valid" term
        # But what is "valid"? In matrix multiplication, every (i,j,k) triple contributes:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # So the tensor T[i,j,k] = 1 for ALL i,j,k (since every combination of input indices contributes to some output)
        # Wait no. The tensor for matrix multiplication is:
        # T_{i,j,k} where the trilinear form is sum_{i,j,k} T_{i,j,k} x_i y_j z_k = (x·A)(B·y) or similar
        # Let me think differently. The structure tensor of matrix multiplication <n,m,p> has:
        # T_{(i,j),(j',k),(k',i')} = delta_{jj'} * delta_{kk'} * delta_{ii'}
        # No wait. Let me use the standard definition.
        # 
        # The matrix multiplication tensor M_{n,m,p} is in R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n}
        # M_{(i,j),(j',k),(k',i')} = 1 if j=j' and i=i' and k=k', else 0
        # Wait, that's not right either.
        #
        # Standard: The bilinear complexity of multiplying an n×m matrix by an m×p matrix.
        # The tensor is T ∈ R^{nm} ⊗ R^{mp} ⊗ R^{pn}
        # T_{(i,j),(j',k),(k',i')} = δ_{jj'} δ_{ii'} δ_{kk'} ... no.
        #
        # Let me use the explicit formula. C = AB where A ∈ R^{n×m}, B ∈ R^{m×p}, C ∈ R^{n×p}
        # C_{ik} = sum_j A_{ij} B_{jk}
        # The tensor T satisfies: sum_{i,j,k} T_{ijk} A_{ij} B_{jk} C_{ki} = ... 
        # Actually, the standard tensor for matrix multiplication is:
        # T_{(i,j),(j',k),(k',i')} = δ_{jj'} δ_{ii'} δ_{kk'}
        # No, that gives sum_{i,j,k} δ_{jj'}δ_{ii'}δ_{kk'} A_{ij}B_{jk}C_{ki} which is trace(ABC) not multiplication.
        #
        # Let me look at this more carefully. The bilinear form for matrix multiplication:
        # We want to compute C = AB. The "tensor" that represents this is:
        # T_{(i,j),(j',k),(k',i')} = δ_{jj'} · δ_{ii'} · δ_{kk'} ... 
        #
        # OK let me just do a numerical verification. I'll construct the full tensor and check.
        
        # The correct check: 
        # For the decomposition to be valid, we need:
        # sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = δ_{jj'} δ_{ii'} δ_{kk'}
        # for all i,j,j',k,k',i'
        # 
        # Let's verify this numerically for a small case
        rank_val = U.shape[1]
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    for k in range(p):
                        for k2 in range(p):
                            for i2 in range(n):
                                expected = 1.0 if (j == j2 and i == i2 and k == k2) else 0.0
                                actual = 0.0
                                for r in range(rank_val):
                                    actual += U[i*m+j, r] * V[j2*p+k, r] * W[k2*n+i2, r]
                                if abs(actual - expected) > 1e-9:
                                    return False
        return True

    # BSC 3x3x3 decomposition - rank 23
    # Using a known valid decomposition from the literature
    # I'll use the decomposition from Bini, Strassen, and Coppersmith
    
    # The BSC decomposition for 3x3x3 has rank 23
    # Let me use a verified set of coefficients
    
    # U_bsc: 9x23, V_bsc: 9x23, W_bsc: 9x23
    # Indices: U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]
    
    # I'll use a known rank-23 decomposition. Let me construct it using the formula.
    # Actually, let me use a simpler approach: I'll use a rank-23 decomposition that I can verify.
    
    # Known BSC 3x3x3 rank-23 decomposition coefficients:
    # This is from the original BSC paper. The 23 rank-1 terms are:
    
    U_bsc = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=float)
    
    V_bsc = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=float)
    
    W_bsc = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=float)
    
    # This is just schoolbook for 3x3x3 with rank 27, not 23. Let me use a proper BSC.
    # I'll fall back to a verified approach.
    
    # Let me use a different strategy: use the known fact that rank(3x3x3) = 23
    # and construct a valid decomposition.
    
    # Actually, let me just use schoolbook for now to ensure correctness, 
    # and try to find a better decomposition within time limits.
    
    U, V, W = schoolbook_3x3x5()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
