import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 92
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(i_off, j_off, k_off, u_vec, v_vec, w_vec):
        nonlocal col
        n_loc = len(u_vec)
        m_loc = len(v_vec)
        p_loc = len(w_vec)
        
        for i in range(n_loc):
            if u_vec[i] != 0:
                U[i_off + i][col] += u_vec[i]
        for j in range(m_loc):
            if v_vec[j] != 0:
                V[j_off + j][col] += v_vec[j]
        for k in range(p_loc):
            if w_vec[k] != 0:
                W[k_off + k][col] += w_vec[k]
        col += 1

    # Strassen 2x2x2
    def strassen_222(i_off, j_off, k_off):
        # Terms: U, V, W
        # 1: (A00+A11)(B00+B11) -> C00, C11
        add_term(i_off, j_off, k_off, [1, 1], [1, 1], [1, 1])
        # 2: (A01+A11)B00 -> C10, -C11
        add_term(i_off, j_off, k_off, [1, 1], [1, 0], [0, -1]) # Wait, C10 is index 0? No.
        # C indices: 00, 01, 10, 11.
        # W vector is for k? No, W is for the output index?
        # The tensor is T[i,j,k]. W[k] is the coefficient for k.
        # But C has 2 indices i and k.
        # In the decomposition C[i,k] = sum_j A[i,j] B[j,k].
        # The term contributes to specific (i,k) pairs.
        # My add_term assumes W is a vector of length p_loc.
        # But C is a matrix.
        # The standard tensor decomposition is T[i,j,k] = sum_s U[i,s] V[j,s] W[k,s].
        # This implies T[i,j,k] is a 3D tensor.
        # For matrix multiplication, T[i,j,k] = 1 if the term A[i,j]B[j,k] exists?
        # No, T[i,j,k] is 1 for all i,j,k.
        # And C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]? No.
        # C[i,k] = sum_j A[i,j] B[j,k].
        # This is a contraction.
        # The tensor T is defined such that T[i,j,k] = 1.
        # Then sum_s U[i,s] V[j,s] W[k,s] = 1?
        # No.
        # The decomposition is for the *multiplication tensor*.
        # The multiplication tensor T has T[i,j,k] = 1.
        # We want T[i,j,k] = sum_s U[i,s] V[j,s] W[k,s].
        # This is NOT correct for matrix multiplication.
        # Matrix multiplication is C = AB.
        # C[i,k] = sum_j A[i,j] B[j,k].
        # This is not a simple outer product sum of 1s.
        # The tensor for matrix multiplication is usually defined as T[i,j,k] = 1.
        # And the rank is the min number of outer products U_s V_s W_s that sum to T.
        # Yes, that is the definition.
        # So for Strassen:
        # M1 = (A00+A11)(B00+B11).
        # This term contributes to C00 and C11.
        # How does it map to U, V, W?
        # U[0]=1, U[1]=1.
        # V[0]=1, V[1]=1.
        # W[0]=1, W[1]=1.
        # Then sum_s U[i,s]V[j,s]W[k,s] for i=0,j=0,k=0:
        # s=1: 1*1*1 = 1.
        # This gives 1. T[0,0,0] is 1. Correct.
        # For i=0, j=0, k=1:
        # s=1: 1*1*1 = 1.
        # T[0,0,1] is 1. Correct.
        # For i=0, j=1, k=0:
        # s=1: 1*1*1 = 1.
        # T[0,1,0] is 1. Correct.
        # So W[k] is just 1 for all k involved?
        # No.
        # In Strassen, M1 is used in C00 and C11.
        # C00 = M1 + ...
        # C11 = M1 + ...
        # The tensor T[i,j,k] = 1.
        # The decomposition sums to 1.
        # So for M1, U=[1,1], V=[1,1], W=[1,1] is correct.
        
        # What about M2?
        # M2 = (A01+A11)B00.
        # U=[1,1], V=[1,0].
        # It contributes to C10 and C11 (with -1).
        # C10 = M2 + ...
        # C11 = ... - M2 ...
        # So for i=1, k=0, the contribution is +1.
        # For i=1, k=1, the contribution is -1.
        # So W[0] = 1, W[1] = -1.
        # But W is indexed by k.
        # What about i?
        # U[0]=1, U[1]=1.
        # So for i=0, j=0, k=0: U[0]V[0]W[0] = 1*1*1 = 1.
        # T[0,0,0] should be 1.
        # But M2 should NOT contribute to C00.
        # C00 does not contain M2.
        # So the sum for i=0,k=0 must be 0 from this term.
        # U[0]=1, V[0]=1, W[0]=1 -> 1.
        # This is a contradiction.
        
        # Ah, the tensor T[i,j,k] is NOT 1 for all i,j,k.
        # The tensor for matrix multiplication is defined by the indices.
        # Actually, the standard definition is T[i,j,k] = 1.
        # And the decomposition is an identity.
        # If M2 contributes to C10 and C11, it means the term is present in the expansion of those C's.
        # But the tensor decomposition is a global sum.
        # sum_s U[i,s]V[j,s]W[k,s] = 1 for ALL i,j,k.
        # If M2 has U=[1,1], V=[1,0], W=[1,-1].
        # i=0, j=0, k=0: 1*1*1 = 1.
        # i=0, j=0, k=1: 1*1*(-1) = -1.
        # i=1, j=0, k=0: 1*1*1 = 1.
        # i=1, j=0, k=1: 1*1*(-1) = -1.
        # This term adds 1 to T[0,0,0], -1 to T[0,0,1], 1 to T[1,0,0], -1 to T[1,0,1].
        # But T[i,j,k] is 1 everywhere.
        # So other terms must cancel these?
        # No, the sum of all terms must be 1.
        
        # This implies my understanding of the mapping is slightly off or the Strassen coefficients need to be distributed.
        # Actually, Strassen works because the linear combinations cancel out in the T tensor sum?
        # No, Strassen is an algorithm for C.
        # The tensor rank is the min number of multiplications.
        # The decomposition T = sum U_s V_s W_s is exact.
        
        # Let's trust the standard Strassen tensor decomposition.
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
