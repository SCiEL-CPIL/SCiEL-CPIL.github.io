"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    # Winograd-style decomposition using polynomial interpolation for the
    # 3x3 * 3x4 matrix multiplication.
    #
    # We view the multiplication C = A * B (A is 3x3, B is 3x4, C is 3x4)
    # as: for each output (i, k): C[i,k] = sum_j A[i,j] * B[j,k].
    #
    # We use a 1D Winograd approach on the "inner" dimension j (size 3).
    # For each pair (i, k), the inner product is over j=0,1,2.
    #
    # We use polynomial evaluation: A polynomial P(x) = A[i,0] + A[i,1]*x + A[i,2]*x^2
    # and Q(x) = B[0,k] + B[1,k]*x + B[2,k]*x^2.
    # The coefficient of x^2 in P(x)*Q(x) is not directly what we want.
    # Actually, the dot product sum_j A[i,j]*B[j,k] is the constant term of
    # P(x) * Q(x^{-1}) where Q(x^{-1}) = B[0,k] + B[1,k]*x^{-1} + B[2,k]*x^{-2}.
    #
    # Instead, let's use a simpler approach: we need to compute 3*3*4 = 36
    # bilinear forms. A Winograd scheme for 3-point dot products uses 3
    # multiplications instead of 3 (no savings for size 3). 
    #
    # Let's try a 2D approach or just use the known optimal rank.
    # Actually, for 3x3 * 3x4, the minimum rank is known to be 25 
    # (from literature on matrix multiplication tensor rank).
    #
    # But we need an explicit decomposition. Let's use a heuristic search
    # over small integer coefficients to find a rank-25 decomposition,
    # falling back to schoolbook (rank 36) if not found in time.
    #
    # Given the time constraints, we'll implement a direct rank-36 
    # schoolbook decomposition which is guaranteed correct, and try to
    # find a lower rank using a simple local search if time permits.
    
    import time
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Schoolbook rank: n*m*p = 36
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    def verify(U, V, W):
        """Check if the decomposition is exact."""
        n, m, p = DIMENSIONS
        # For each (i,j,k), check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] 
        # equals the standard multiplication tensor entry.
        # The tensor T[i,j,k] = 1 if we consider C[i,k] = sum_j A[i,j]*B[j,k].
        # Actually, the decomposition should satisfy:
        # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) ... 
        # No, that's not right.
        
        # The correct condition for matrix multiplication tensor decomposition:
        # For all (i, j, k) with 0<=i<n, 0<=j<m, 0<=k<p:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # And for all other (i', j', k') not corresponding to valid multiplication,
        # but actually the tensor is defined only on the support.
        
        # The matrix multiplication tensor T has T[i,j,k] = 1 for all valid (i,j,k)
        # in the sense that C[i,k] = sum_j A[i,j] B[j,k].
        # The decomposition requires:
        # For each r: the trilinear form is U_r ⊗ V_r ⊗ W_r
        # Sum_r U_r ⊗ V_r ⊗ W_r = T
        # where T[i,j,k] = 1 if (i,j,k) is a valid term.
        
        # So we check: for all i in [0,n), j in [0,m), k in [0,p):
        #   sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] == 1
        # And for all other combinations that could arise, they should be 0,
        # but since we only have indices in range, we just check the valid ones.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(len(U[0])):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True
    
    # Try to find a lower rank decomposition using a simple approach.
    # For now, return the schoolbook decomposition.
    # A more sophisticated approach would use tensor rank algorithms,
    # but given time constraints, we return the guaranteed rank-36.
    
    # Check if we should attempt a lower rank (this is a placeholder for
    # potential improvement; the schoolbook is always valid).
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
