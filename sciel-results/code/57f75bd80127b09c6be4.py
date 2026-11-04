"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Start with schoolbook decomposition
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Identify the 8 columns corresponding to the 2x2x2 block at i in {0,1}, j in {0,1}, k in {0,1}
    # The column index for (i, j, k) is i*m*p + j*p + k in the schoolbook ordering?
    # Wait, the loop order was i, j, k.
    # So index is i * (m*p) + j * p + k.
    
    block_indices = []
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [0, 1]:
                idx = i * (m * p) + j * p + k
                block_indices.append(idx)
                
    # These 8 indices will be replaced by 7 Strassen terms.
    # We will zero out the 8 original columns and add 7 new ones.
    # To keep the rank minimal, we can just reuse the 8 slots, setting one to 0,
    # or better, create a new list of columns.
    
    # Let's build the new U, V, W with rank 104.
    new_rank = rank_school - 1
    new_U = [[0] * new_rank for _ in range(n * m)]
    new_V = [[0] * new_rank for _ in range(m * p)]
    new_W = [[0] * new_rank for _ in range(p * n)]
    
    # Copy all non-block columns
    col_new = 0
    for c in range(rank_school):
        if c not in block_indices:
            for r in range(n * m):
                new_U[r][col_new] = U[r][c]
            for r in range(m * p):
                new_V[r][col_new] = V[r][c]
            for r in range(p * n):
                new_W[r][col_new] = W[r][c]
            col_new += 1
            
    # Now add the 7 Strassen terms
    # The block is A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Indices in the block:
    # A: (0,0), (0,1), (1,0), (1,1) -> global U indices: 0, 1, 5, 6 (since m=5, j=0,1)
    # Wait, U index is i*m + j.
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=1, j=0 -> 5
    # i=1, j=1 -> 6
    
    # B: (0,0), (0,1), (1,0), (1,1) -> global V indices: j*p + k
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=1, k=0 -> 7
    # j=1, k=1 -> 8
    
    # C: (0,0), (0,1), (1,0), (1,1) -> global W indices: k*n + i
    # k=0, i=0 -> 0
    # k=0, i=1 -> 3
    # k=1, i=0 -> 1
    # k=1, i=1 -> 4
    
    # Strassen terms:
    # M1 = (A00+A11)(B00+B11)
    # U: A00(0), A11(6)
    # V: B00(0), B11(8)
    # W: none (it's an intermediate product, but in tensor decomposition, 
    # each term is a rank-1 tensor u v w^T)
    # Wait, the tensor decomposition is T = sum u_k v_k w_k^T.
    # For Strassen, we express the 8 basis vectors as a linear combination of 7 new vectors.
    # The new vectors are not simply "products", they are the rank-1 tensors that sum to the block tensor.
    # Actually, the standard way is:
    # The 8 schoolbook terms are e_{000}, e_{001}, etc.
    # We replace them with 7 terms t_1...t_7 such that sum t_k = sum e_i.
    # The t_k are not necessarily rank-1 in the same way?
    # No, any tensor can be decomposed into rank-1 tensors.
    # Strassen's algorithm provides a decomposition of the $2 \times 2 \times 2$ multiplication tensor into 7 rank-1 tensors.
    
    # The 7 rank-1 tensors for the 2x2x2 block are:
    # T1: u=(1,0,1,0) in A-space? No.
    # Let's use the explicit rank-1 decompositions for the 2x2x2 tensor.
    # The tensor T is defined by T_{ijk} = 1 if i,j,k are the indices of the multiplication.
    # The 7 terms are:
    # 1. u=(1,0,1,0)^T? No, u is a vector in R^{n*m}.
    # For the block, u is supported on A indices (0,0) and (1,1).
    # u1 = e_{00} + e_{11} (in the 4-dim subspace of A)
    # v1 = e_{00} + e_{11} (in the 4-dim subspace of B)
    # w1 = e_{00} + e_{11} (in the 4-dim subspace of C)
    # This is not correct. The rank-1 tensors are specific combinations.
    
    # Let's look up the rank-1 decomposition of M_{2,2,2}.
    # It is known that M_{2,2,2} = sum_{k=1}^7 u_k v_k w_k^T.
    # The vectors are:
    # u1 = (1, 0, 1, 0)  # A00, A11
    # v1 = (1, 0, 1, 0)  # B00, B11
    # w1 = (1, 1, 0, 0)  # C00, C01? No.
    
    # Actually, it is easier to just use the linear combination of the schoolbook basis.
    # The 8 schoolbook basis vectors for the block are:
    # s1=e000, s2=e001, s3=e010, s4=e011, s5=e100, s6=e101, s7=e110, s8=e111.
    # We want to find 7 vectors t1..t7 in the span of s1..s8 such that
    # sum_{i=1}^8 s_i = sum_{k=1}^7 t_k.
    # This is always possible. But we want the t_k to be "simple" (integer/half-integer).
    # And we want the overall decomposition to be valid.
    # If we replace the 8 columns with 7 columns that are linear combinations of the original 8,
    # the sum of the new columns must equal the sum of the original 8 columns.
    # And the new columns must be rank-1 tensors?
    # No, the columns of U, V, W define the rank-1 tensors.
    # If we change the columns, we are changing the rank-1 tensors.
    # We need the new set of rank-1 tensors to still sum to the original tensor.
    
    # The simplest way is to find 7 rank-1 tensors that sum to the 2x2x2 multiplication tensor.
    # These are the Strassen tensors.
    # I will hardcode the 7 rank-1 tensors for the 2x2x2 block.
    
    # The 7 rank-1 tensors (u, v, w) for the 2x2x2 block (indices 0,1 for A, B, C):
    # Note: A indices are (i,j), B indices are (j,k), C indices are (k,i).
    # In the block, i,j,k are 0 or 1.
    
    # T1:
    # u: A(0,0)=1, A(1,1)=1
    # v: B(0,0)=1, B(1,1)=1
    # w: C(0,0)=1, C(0,1)=1, C(1,0)=1, C(1,1)=1? No.
    
    # Let's use a known list of 7 rank-1 decompositions for M_{2,2,2}.
    # From literature:
    # 1. u=(1,0,1,0), v=(1,0,1,0), w=(1,1,0,0) -> C(0,0), C(0,1)
    # 2. u=(0,1,0,1), v=(0,1,0,1), w=(1,1,0,0) -> C(0,0), C(0,1)
    # This is getting risky without a reliable source.
    
    # Alternative: Just return the schoolbook decomposition.
    # The score will be 1.0.
    # But the prompt asks for a stronger algorithm.
    # If I cannot guarantee a valid lower-rank decomposition, I should return the schoolbook one.
    
    # However, I can try a simple linear dependency check.
    # If I find that one of the schoolbook terms is a linear combination of the others,
    # I can remove it.
    # For M_{2,2,2}, e000 + e111 = e001 + e010 + e100 + e110? No.
    # The dependency is more complex.
    
    # Given the risk of invalidity, I will return the schoolbook decomposition.
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
