"""Program for matrix multiplication tensor decomposition: 2x2 by 2x16."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 16
    
    # Use Strassen's 7-term decomposition for each 2x2x2 block.
    # Group the 16 columns of B into 8 pairs of 2 columns.
    # For each pair, we have a local 2x2x2 matrix multiplication problem.
    
    num_blocks = p // 2  # 8 blocks
    rank = 7 * num_blocks  # 56
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 32 x 56
    
    # Strassen's 7 factors for 2x2 matrix multiplication.
    # The tensor decomposition requires: sum_r U[i,r]*V[j,r]*W[k,r] = delta_{j,k} * 1
    # where i is row of A/C, j is col of A/row of B, k is col of B/C.
    
    # For a 2x2x2 block, the 7 Strassen factors (u, v, w) are:
    # Each factor contributes u[i]*v[j]*w[k] to the tensor entry T[i,j,k].
    # We need sum over all 7 factors to give T[i,j,k] = 1 if j==k else 0.
    
    # Standard Strassen factors for C = A*B where A,B are 2x2:
    # M1 = (a11+a22)(b11+b22) -> u=[1,1], v=[1,1], w=[1,1]
    # M2 = (a21+a22)b11       -> u=[0,1], v=[1,0], w=[0,1]  -- WRONG
    # 
    # Let me derive the correct factors. We need 7 rank-1 tensors whose sum
    # equals the matrix multiplication tensor T where T[i,j,k] = delta_{j,k}.
    # 
    # Actually, for matrix multiplication, the tensor is:
    # T[i,j,k] = 1 for all valid i,j,k? No.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The bilinear form is: <A, B> -> C where the "structure tensor" has
    # entries T[i,j,k] such that C[i,k] = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k]
    # But this isn't quite right either.
    #
    # The correct way: The matrix multiplication map is a bilinear map
    # (R^{n x m}, R^{m x p}) -> R^{n x p}
    # A rank-1 term in the tensor decomposition corresponds to:
    # C += (u^T A) * (B v) * w^T  ... no.
    #
    # Let me use the standard formulation from algebraic complexity theory.
    # The matrix multiplication tensor M_{n,m,p} is an element of R^{n x m} ⊗ R^{m x p} ⊗ R^{n x p}
    # Its entries are: M[i,j | j',k | i',k'] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    # Wait, that's not right.
    #
    # Actually, the tensor M has entries M[(i,j), (j',k), (i',k')] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    # But we need the indices to match: j = j', i = i', k = k'.
    # So M[(i,j), (j,k), (i,k)] = 1, and 0 otherwise.
    #
    # A rank-1 decomposition term is: u ⊗ v ⊗ w where:
    # u is in R^{n*m} (indexed by (i,j))
    # v is in R^{m*p} (indexed by (j,k))  
    # w is in R^{n*p} (indexed by (i,k))
    # And the contribution is: sum_{r} u_r[(i,j)] * v_r[(j,k)] * w_r[(i,k)]
    # We need this to equal delta_{j,j'}... 
    #
    # OK, I think the issue is that for each fixed (i,j,k), the tensor entry is 1
    # (since A[i,j] * B[j,k] contributes to C[i,k]).
    # So T[(i,j),(j,k),(i,k)] = 1 for all valid i,j,k.
    #
    # A rank-1 factor r has:
    # U_r: vector in R^{n*m}, so U_r[i*m+j] for i in 0..n-1, j in 0..m-1
    # V_r: vector in R^{m*p}, so V_r[j*p+k] for j in 0..m-1, k in 0..p-1
    # W_r: vector in R^{n*p}, so W_r[k*n+i] for k in 0..p-1, i in 0..n-1
    # 
    # The constraint is: for all i,j,k:
    #   sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] = 1
    #
    # This is the tensor decomposition we need.
    
    # For the 2x2x2 block (local indices i,j,k in {0,1}), the 7 Strassen factors:
    # I'll use the known Strassen decomposition.
    
    # The 7 factors for 2x2x2 matrix multiplication tensor:
    # Factor 0: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]  (in flattened indices)
    #   U: u[0*2+0]=1, u[1*2+1]=1 -> rows 0,3
    #   V: v[0*2+0]=1, v[1*2+1]=1 -> rows 0,3 (local p=2)
    #   W: w[0*2+0]=1, w[1*2+1]=1 -> rows 0,3 (local n=2)
    
    # Let me just hardcode the 7 Strassen factors for 2x2x2.
    # Using the standard Strassen algorithm:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # For each M_r, we can write it as a product of linear forms in A, B, and C.
    # M_r = (sum_{i,j} alpha_{r,i,j} A[i,j]) * (sum_{j,k} beta_{r,j,k} B[j,k])
    # And M_r contributes to C[i,k] with coefficient gamma_{r,i,k}.
    
    # The tensor factor for M_r has:
    # U_r[i*m+j] = alpha_{r,i,j}
    # V_r[j*p+k] = beta_{r,j,k}
    # W_r[k*n+i] = gamma_{r,i,k}
    
    # And the constraint is: sum_r alpha_{r,i,j} * beta_{r,j,k} * gamma_{r,i,k} = delta_{j,k}
    # Wait, no. The constraint is that the bilinear map computed by the decomposition
    # equals the matrix multiplication map.
    
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Using the decomposition: C[i,k] = sum_r (sum_{i',j'} U_r[i'm+j'] A[i',j']) * (sum_{j',k'} V_r[j'p+k'] B[j',k']) * W_r[kn+i]
    # This doesn't seem right. Let me think again.
    
    # Actually, I think the correct interpretation is:
    # For each rank-1 term r, we have a "product" M_r = (linear form in A) * (linear form in B)
    # And C[i,k] gets a contribution of M_r * W_r[k*n+i].
    # So: C[i,k] = sum_r W_r[k*n+i] * (sum_{i',j'} U_r[i'm+j'] A[i',j']) * (sum_{j',k'} V_r[j'p+k'] B[j',k'])
    
    # For this to equal sum_j A[i,j] B[j,k], we need:
    # sum_r W_r[k*n+i] * U_r[i'm+j] * V_r[j'p+k'] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'} ... 
    # No, this is getting too complicated.
    
    # Let me just use the schoolbook decomposition which I know is correct,
    # and then try to apply Strassen to reduce the rank.
    
    # Actually, let me just implement the schoolbook for now and verify it works.
    rank = n * m * p  # 64
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
