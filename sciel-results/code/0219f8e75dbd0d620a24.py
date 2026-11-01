"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    # Construct a rank-42 decomposition for 2x3 x 3x7 matrix multiplication.
    # The tensor T_{i,k,j,l} = delta_{j,l} with n=2, m=3, p=7.
    # We use a structured decomposition with coefficients derived from the 
    # rectangular dimension mismatch to achieve rank 42 < 42 (schoolbook).
    # 
    # Strategy: Group the k indices and use a low-rank factorization of the
    # 3x7 identity-like pattern.
    
    n, m, p = DIMENSIONS
    # Schoolbook rank is n*m*p = 42.
    # For 2x3x7, we can potentially do better.
    # Known bounds: rank <= n*m*p, but rectangular cases can have lower rank.
    # Let's try a specific structured decomposition.
    
    # For a 2xm x mxp multiplication, we can think of it as two rows of outputs.
    # The tensor has 4 indices: (i, k, j, l) where i in {0,1}, k in {0..6}, j in {0..2}, l in {0..6}
    # T_{i,k,j,l} = 1 if j == l, 0 otherwise.
    # 
    # We need: sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c] = T_{i,k,j',...}
    # Actually the convention from the seed:
    # U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank
    # The decomposition reconstructs: for each (i, j, k), the product involves
    # summing over c and the internal indices.
    
    # Let me use a different approach. I'll hardcode a known low-rank decomposition
    # or derive one carefully.
    
    # For 2x3 x 3x7:
    # The multiplication tensor is a 3-fold tensor of dimensions (2*3) x (3*7) x (7*2)
    # which is 6 x 21 x 14.
    
    # One approach: use the fact that for rectangular matrices, we can sometimes
    # save by sharing computation between the two output rows (i=0 and i=1).
    
    # Let me try to construct a rank-42 first (same as schoolbook) but with
    # different coefficients, then try to reduce.
    
    # Actually, let me try a constructive approach:
    # For each output entry (i, k), the value is sum_j A_{i,j} * B_{j,k}.
    # The tensor is T[(i,k), (j,l)] = delta(j,l) * delta(i',i) * delta(k',k)
    # Wait, I need to be more careful with the indexing convention.
    
    # From the seed: U[i*m+j], V[j*p+k], W[k*n+i]
    # The product is: C[i,k] = sum_j A[i,j] B[j,k]
    # The decomposition should satisfy:
    # sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c] 
    # = 1 if (j==j' and k==k' and i==i'), else 0
    # This gives: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = delta_{j',j} delta_{k',k} delta_{i',i}
    
    # So for fixed (i,j,k), sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1
    # For (i,j,k) != (i',j',k'), the sum = 0 when we consider the full 4-tensor.
    
    # Actually, the tensor T_{i,j,k,l} (with appropriate indexing) equals
    # delta_{j,l} * delta_{k,k'} ... let me just verify the seed works and try
    # to find a lower rank.
    
    # Let me try a known technique: for 2xm x mxp, one can sometimes use rank
    # less than 2*m*p by exploiting linearity.
    
    # For n=1, m=3, p=7: rank is m*p = 21 (schoolbook is optimal for 1xm x mxp)
    # For n=2, we might be able to share.
    
    # Actually, the rank of matrix multiplication tensor for (n x m) * (m x p)
    # is at most n*m*p, but can be lower. For 2x3 x 3x7, let me try to construct
    # a rank-38 decomposition as suggested in the strategy.
    
    # Hardcoded approach: I'll use a specific construction.
    # Let me think of it as: C = A * B where A is 2x3, B is 3x7.
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # One idea: express some of the products in terms of others.
    # But since the tensor must be exact, we need an exact linear combination.
    
    # Let me try a different strategy: use the schoolbook rank 42 but with
    # half-integer coefficients that might allow for a valid lower-rank form
    # through cancellation. However, without a valid reduction, I'll stick with
    # a valid decomposition.
    
    # Let me carefully construct a rank-42 decomposition that's definitely valid,
    # matching the seed's logic but potentially with different coefficients.
    
    n, m, p = DIMENSIONS
    rank = n * m * p  # 42
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
