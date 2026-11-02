import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # For each output element C[i,k] (stored in W at index k*n+i),
    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] ... no wait.
    # The decomposition is: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # where T[i,j; j,k; i,k] = 1 (the matrix multiplication tensor)
    # So for each (i,k) pair, sum_j T[i,j; j,k] * A[i,j] * B[j,k] = C[i,k]
    # The rank-r decomposition means: for each r, we have vectors u_r, v_r, w_r
    # such that T = sum_r u_r ⊗ v_r ⊗ w_r
    # T[i,j, j,k, i,k] ... let me think about the tensor structure.
    
    # The matrix multiplication tensor M_{i,j,k} for A(i,j), B(j,k), C(i,k):
    # M[i,j; j',k; i',k'] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    # No, that's not right either.
    
    # The bilinear map: (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor is T = sum_{i,j,k} e_{i,k}^{(C)} ⊗ e_{i,j}^{(A)} ⊗ e_{j,k}^{(B)}
    # So T has components T[i,k, i,j, j,k] ... this is getting confusing.
    
    # Let me think of it as: for each r in 1..rank,
    # C += W[:,r] * (U[:,r] ⊗ V[:,r]) where the product is appropriate.
    
    # Actually, the standard formulation: 
    # C[i,k] = sum_r (sum_j U[i*m+j, r] * something) ... 
    
    # Let me use the direct verification:
    # For the decomposition to be valid, for all i, k:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * V[j*p+k, r])
    # Wait no, that's not right either.
    
    # The correct formulation: The tensor T satisfies:
    # T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r is in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}
    # And T[a, b, c] = sum_r u_r[a] * v_r[b] * w_r[c]
    # where a indexes (i,j) in A, b indexes (j',k) in B, c indexes (i',k') in C
    # and T[a,b,c] = 1 if j=j' and i=i' and k=k', else 0.
    
    # So T[i*m+j, j'*p+k, k'*n+i'] = 1 if j==j' and i==i' and k==k' else 0.
    
    # Let's verify: for each (i,j, j',k, i',k'), 
    # sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] should equal 
    # 1 if j==j' and i==i' and k==k', else 0.
    
    # This is expensive for large tensors. Let's use a smarter check.
    # Instead, verify that the bilinear map is correct:
    # For random A and B, compute C using the decomposition and check C[i,k] = sum_j A[i,j]*B[j,k].
    
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B
    
    # Compute C using the decomposition
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * V[j*p+k, r])
    # Wait, that's not how it works. Let me re-derive.
    
    # The tensor decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
    # T[a,b,c] = sum_r u_r[a] v_r[b] w_r[c]
    # The bilinear map: C = T(A, B) where (T(A,B))[c] = sum_{a,b} T[a,b,c] A[a] B[b]
    # So C[c] = sum_r w_r[c] * (sum_a u_r[a] A[a]) * (sum_b v_r[b] B[b])
    
    # Let's compute:
    # For each r:
    #   alpha_r = sum_a U[a,r] * A_flat[a]  (where A_flat[i*m+j] = A[i,j])
    #   beta_r = sum_b V[b,r] * B_flat[b]   (where B_flat[j*p+k] = B[j,k])
    #   C_flat[c] += W[c,r] * alpha_r * beta_r
    
    A_flat = A.flatten()  # length n*m, index i*m+j
    B_flat = B.flatten()  # length m*p, index j*p+k
    C_flat = np.zeros(p * n)  # index k*n+i
    
    U_arr = np.array(U)  # (n*m, rank)
    V_arr = np.array(V)  # (m*p, rank)
    W_arr = np.array(W)  # (p*n, rank)
    
    # alpha_r = U_arr^T @ A_flat
    alpha = U_arr.T @ A_flat  # length rank
    beta = V_arr.T @ B_flat   # length rank
    C_flat = W_arr @ (alpha * beta)  # length p*n
    
    C_computed = C_flat.reshape(p, n).T  # reshape to (n, p)
    
    return np.allclose(C_computed, C_expected, atol=1e-9)

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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

def _block_decomposition(n, m, p, block_sizes):
    """
    Decompose using blocks. For each block of size b (columns of B),
    use a sub-decomposition for n×m × m×b.
    """
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    col_offset = 0
    for b in block_sizes:
        # Get sub-decomposition for n×m by m×b
        sub = _get_sub_decomp(n, m, b)
        sub_U = sub["U"]  # (n*m, sub_rank)
        sub_V = sub["V"]  # (m*b, sub_rank)
        sub_W = sub["W"]  # (b*n, sub_rank)
        
        # Expand to full size
        full_U = [[0] * len(sub_U[0]) for _ in range(n * m)]
        full_V = [[0] * len(sub_V[0]) for _ in range(m * p)]
        full_W = [[0] * len(sub_W[0]) for _ in range(p * n)]
        
        for i in range(n * m):
            for r in range(len(sub_U[0])):
                full_U[i][r] = sub_U[i][r]
        
        for j in range(m):
            for k in range(b):
                for r in range(len(sub_V[0])):
                    full_V[j * p + (col_offset + k)][r] = sub_V[j * b + k][r]
        
        for k in range(b):
            for i in range(n):
                for r in range(len(sub_W[0])):
                    full_W[(col_offset + k) * n + i][r] = sub_W[k * n + i][r]
        
        U_blocks.append(full_U)
        V_blocks.append(full_V)
        W_blocks.append(full_W)
        col_offset += b
    
    # Concatenate all blocks
    total_rank = sum(len(b[0]) for b in U_blocks)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for block_U, block_V, block_W in zip(U_blocks, V_blocks, W_blocks):
        block_rank = len(block_U[0])
        for i in range(n * m):
            for r in range(block_rank):
                U[i][col_offset + r] = block_U[i][r]
        for j in range(m * p):
            for r in range(block_rank):
                V[j][col_offset + r] = block_V[j][r]
        for k in range(p * n):
            for r in range(block_rank):
                W[k][col_offset + r] = block_W[k][r]
        col_offset += block_rank
    
    return {"U": U, "V": V, "W": W}

def _get_sub_decomp(n, m, p):
    """Get a sub-decomposition for n×m by m×p."""
    if p == 1:
        # [n,m,1]: C[i] = sum_j A[i,j] B[j]
        # Rank = n*m (each output needs m terms, and they're independent)
        rank = n * m
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m)]
        W = [[0] * rank for _ in range(n)]
        col = 0
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = 1
                V[j][col] = 1
                W[i][col] = 1
                col += 1
        return {"U": U, "V": V, "W": W}
    elif p == 2:
        # [n,m,2]: Try to find a good decomposition
        # For [2,3,2], schoolbook is 12. Let's see if we can do better.
        # For now, use schoolbook.
        rank = n * m * p
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
    elif p == 3:
        # [n,m,3]: For [2,3,3], try rank 8 if possible, else schoolbook 18
        # Use schoolbook for safety
        rank = n * m * p
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
    else:
        # General case: schoolbook
        rank = n * m * p
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

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    best_result = None
    best_rank = float('inf')
    
    # Strategy 1: Schoolbook (rank 60)
    result = _schoolbook(n, m, p)
    if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        best_result = result
        best_rank = n * m * p
    
    # Strategy 2: 5 blocks of [2,3,2] (schoolbook each, rank 12 each, total 60)
    # Same as schoolbook, so skip
    
    # Strategy 3: 3 blocks of [2,3,3] + 1 block of [2,3,1]
    # If [2,3,3] can be done in rank 8, total = 3*8 + 6 = 30
    # But we need to verify. For now, use schoolbook for sub-blocks.
    
    # Let's try to find a better [2,3,2] decomposition.
    # [2,3,2]: A is 2×3, B is 3×2, C is 2×2.
    # C[0,0] = a00*b00 + a01*b10 + a02*b20
    # C[0,1] = a00*b01 + a01*b11 + a02*b21
    # C[1,0] = a10*b00 + a11*b10 + a12*b20
    # C[1,1] = a10*b01 + a11*b11 + a12*b21
    
    # Schoolbook: 12 terms. Can we do it in 11?
    # This is equivalent to [2,2,3] by transposing B.
    # [2,2,3]: A is 2×2, B is 2×3, C is 2×3.
    # I believe the rank of [2,2,3] is 8.
    
    # Let me try to construct [2,2,3] in rank 8.
    # A = [[a,b],[c,d]], B = [[e,f,g],[h,i,j]], C = [[ae+bh, af+bi, ag+bj],[ce+dh, cf+di, cg+dj]]
    
    # One approach: use 4 "Strassen-like" products for the 2×2×2 part,
    # plus additional terms for the third column.
    
    # Actually, let me just try a few known constructions and verify.
    
    # For now, let's just return the schoolbook if nothing better is found.
    if time.time() > deadline:
        if best_result:
            return best_result
        return _schoolbook(n, m, p)
    
    # Try block decomposition with various block sizes
    block_configs = [
        [2, 2, 2, 2, 2],      # 5 blocks of 2
        [3, 3, 3, 1],          # 3 blocks of 3, 1 block of 1
        [3, 3, 4],             # 2 blocks of 3, 1 block of 4
        [4, 3, 3],
        [5, 5],                # 2 blocks of 5
        [2, 3, 5],
        [2, 5, 3],
        [5, 2, 3],
        [1, 2, 3, 4],
        [1, 3, 2, 4],
        [2, 2, 3, 3],
        [3, 2, 3, 2],
    ]
    
    for config in block_configs:
        if time.time() > deadline:
            break
        if sum(config) != p:
            continue
        try:
            result = _block_decomposition(n, m, p, config)
            rank = len(result["U"][0])
            if rank < best_rank:
                if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                    best_result = result
                    best_rank = rank
        except Exception:
            pass
    
    if best_result:
        return best_result
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
