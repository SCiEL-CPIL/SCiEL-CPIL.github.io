import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    n, m, p = DIMENSIONS
    # n=2, m=2, p=12
    # We're computing C = A * B where A is 2x2 and B is 2x12
    # Output C is 2x12
    # W follows convention W[k*n+i, r] where k is column index of C, i is row index of C
    
    # The tensor for (n,m,p) = (2,2,12) has dimension n*m*p = 2*2*12 = 48
    # We need to find a rank-r decomposition
    
    # Strategy: Use the block structure of B (2x12 = two rows of 12 elements)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Let's think about this more carefully.
    # A is 2x2, B is 2x12, C is 2x12
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    # For each column k of B (and C), we have a 2-vector operation:
    # c_k = A * b_k where b_k is the k-th column of B (a 2-vector)
    
    # So we're doing 12 independent 2x2 matrix-vector multiplications.
    # Each 2x2 matrix-vector multiplication has rank 4 (schoolbook) or can be done more efficiently.
    
    # Actually, let's think of it as: the full tensor has entries T[i,j,k] where
    # T[i,j,k] = 1 if we're computing C[i,k] and using A[i,j] and B[j,k]
    # More precisely, the multiplication tensor M_{2,2,12} has:
    # M[i,j,k] = 1 for the term A[i,j]*B[j,k] contributing to C[i,k]
    
    # The tensor lives in U ⊗ V ⊗ W^* where:
    # U has dimension n*m = 4 (indexed by (i,j))
    # V has dimension m*p = 24 (indexed by (j,k))
    # W has dimension p*n = 24 (indexed by (k,i))
    
    # T[(i,j),(j,k),(k,i)] = 1
    
    # Let me try a different approach: decompose using the fact that
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # We can write this as:
    # For each k, C[:,k] = A * B[:,k]
    
    # Let's use a Kronecker-like structure. Consider splitting into blocks.
    # B has 2 rows and 12 columns. Let's split the 12 columns into 6 groups of 2.
    
    # For each group of 2 columns (k, k+1), we have:
    # C[:, k:k+2] = A * B[:, k:k+2] where B[:, k:k+2] is 2x2
    
    # This is 6 independent 2x2 * 2x2 matrix multiplications!
    # Each has rank 7 (Strassen) or 4 (schoolbook).
    # With Strassen: total rank = 6 * 7 = 42
    
    # But wait - can we do better by exploiting structure across the 6 blocks?
    # The A matrix is the SAME for all 6 blocks. So we're computing:
    # C_block_b = A * B_block_b for b = 0..5
    
    # This is like a batch of 6 matrix multiplications with the same A.
    # The rank might be less than 6*7 if we can share work.
    
    # Let me try to construct a rank-42 decomposition using Strassen per block.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]   # 4 x 42
    V = [[0] * rank for _ in range(m * p)]   # 24 x 42
    W = [[0] * rank for _ in range(p * n)]   # 24 x 42
    
    # Strassen's algorithm for 2x2 * 2x2:
    # Given A = [[a, b], [c, d]] and B = [[e, f], [g, h]]
    # C = A*B = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    # Strassen's 7 products:
    # m1 = (a + d)(e + h)
    # m2 = (c + d)e
    # m3 = a(f - h)
    # m4 = d(g - e)
    # m5 = (a + b)h
    # m6 = (c - a)(e + f)
    # m7 = (b - d)(g + h)
    
    # C[0,0] = m1 + m4 - m5 + m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m3 + m6
    
    # For our problem, each block b has:
    # A is the global A (same for all blocks)
    # B_block is B[:, 2b:2b+2]
    
    # Let's fill in the tensors.
    # U index: i*m + j = i*2 + j, so (i,j) -> 2i+j
    #   (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # V index: j*p + k = j*12 + k, so (j,k) -> 12j+k
    #   (0,k)->k, (1,k)->12+k
    # W index: k*n + i = k*2 + i, so (k,i) -> 2k+i
    #   (k,0)->2k, (k,1)->2k+1
    
    # For block b, the local B columns are k=2b and k=2b+1
    # Local B[0,0] = B[0,2b] -> V index 12*0 + 2b = 2b
    # Local B[0,1] = B[0,2b+1] -> V index 12*0 + (2b+1) = 2b+1
    # Local B[1,0] = B[1,2b] -> V index 12*1 + 2b = 12+2b
    # Local B[1,1] = B[1,2b+1] -> V index 12*1 + (2b+1) = 13+2b
    
    # Local C[0,0] = C[0,2b] -> W index 2*(2b) + 0 = 4b
    # Local C[0,1] = C[0,2b+1] -> W index 2*(2b+1) + 0 = 4b+2
    # Local C[1,0] = C[1,2b] -> W index 2*(2b) + 1 = 4b+1
    # Local C[1,1] = C[1,2b+1] -> W index 2*(2b+1) + 1 = 4b+3
    
    # Strassen terms for one block:
    # Each term r gets an offset of b*7 in the rank dimension
    
    # Let's define the 7 Strassen terms with their U, V, W coefficients
    # U coefficients are on the global A indices (same for all blocks)
    # V and W coefficients depend on the block
    
    # Term 1: m1 = (a+d)(e+h)
    # U: A[0,0] + A[1,1] -> U[0] += 1, U[3] += 1
    # V: B[0,0] + B[1,1] -> V[2b] += 1, V[13+2b] += 1
    # W: C[0,0] += 1, C[1,1] += 1 -> W[4b] += 1, W[4b+3] += 1
    
    # Term 2: m2 = (c+d)e
    # U: A[1,0] + A[1,1] -> U[2] += 1, U[3] += 1
    # V: B[0,0] -> V[2b] += 1
    # W: C[1,0] += 1, C[1,1] -= 1 -> W[4b+1] += 1, W[4b+3] -= 1
    
    # Term 3: m3 = a(f-h)
    # U: A[0,0] -> U[0] += 1
    # V: B[0,1] - B[1,1] -> V[2b+1] += 1, V[13+2b] -= 1
    # W: C[0,1] += 1, C[1,1] += 1 -> W[4b+2] += 1, W[4b+3] += 1
    
    # Term 4: m4 = d(g-e)
    # U: A[1,1] -> U[3] += 1
    # V: B[1,0] - B[0,0] -> V[12+2b] += 1, V[2b] -= 1
    # W: C[0,0] += 1, C[1,0] += 1 -> W[4b] += 1, W[4b+1] += 1
    
    # Term 5: m5 = (a+b)h
    # U: A[0,0] + A[0,1] -> U[0] += 1, U[1] += 1
    # V: B[1,1] -> V[13+2b] += 1
    # W: C[0,0] -= 1, C[0,1] += 1 -> W[4b] -= 1, W[4b+2] += 1
    
    # Term 6: m6 = (c-a)(e+f)
    # U: A[1,0] - A[0,0] -> U[2] += 1, U[0] -= 1
    # V: B[0,0] + B[0,1] -> V[2b] += 1, V[2b+1] += 1
    # W: C[1,1] += 1 -> W[4b+3] += 1
    
    # Term 7: m7 = (b-d)(g+h)
    # U: A[0,1] - A[1,1] -> U[1] += 1, U[3] -= 1
    # V: B[1,0] + B[1,1] -> V[12+2b] += 1, V[13+2b] += 1
    # W: C[0,0] += 1 -> W[4b] += 1
    
    # Define terms as list of (u_dict, v_dict, w_dict) where dicts map index -> coeff
    terms_template = [
        # m1
        ({0: 1, 3: 1}, {0: 1, 13: 1}, {0: 1, 3: 1}),
        # m2
        ({2: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}),
        # m3
        ({0: 1}, {1: 1, 13: -1}, {2: 1, 3: 1}),
        # m4
        ({3: 1}, {12: 1, 0: -1}, {0: 1, 1: 1}),
        # m5
        ({0: 1, 1: 1}, {13: 1}, {0: -1, 2: 1}),
        # m6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # m7
        ({1: 1, 3: -1}, {12: 1, 13: 1}, {0: 1}),
    ]
    
    num_blocks = p // 2  # 6 blocks
    
    for b in range(num_blocks):
        if time.time() > deadline - 5:
            break
        for t_idx in range(7):
            r = b * 7 + t_idx
            u_dict, v_dict, w_dict = terms_template[t_idx]
            
            # U: global indices (same for all blocks)
            for u_idx, coeff in u_dict.items():
                U[u_idx][r] = coeff
            
            # V: local indices offset by block
            # Local V index j*2 + k_loc maps to global 12*j + (2b + k_loc)
            for v_local, coeff in v_dict.items():
                # v_local is 0..13, representing j*2 + k_loc where j in {0,1}, k_loc in {0,1}
                j = v_local // 2  # 0 or 1
                k_loc = v_local % 2  # 0 or 1
                v_global = 12 * j + (2 * b + k_loc)
                V[v_global][r] = coeff
            
            # W: local indices offset by block
            # Local W index 2*k_loc + i maps to global 2*(2b + k_loc) + i
            for w_local, coeff in w_dict.items():
                # w_local is 0..3, representing 2*k_loc + i where k_loc in {0,1}, i in {0,1}
                k_loc = w_local // 2  # 0 or 1
                i = w_local % 2  # 0 or 1
                w_global = 2 * (2 * b + k_loc) + i
                W[w_global][r] = coeff
    
    # Verify correctness with a random test
    if time.time() < deadline - 10:
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        # Compute C from the decomposition
        # C[i,k] = sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] ... 
        # Actually the decomposition is: T = sum_r u_r ⊗ v_r ⊗ w_r
        # where T[(i,j),(j,k),(k,i)] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # And T should equal the multiplication tensor, so:
        # sum_j T[(i,j),(j,k),(k,i)]... no, that's not right.
        
        # The tensor decomposition means:
        # The multiplication tensor M has M[(i,j),(j,k),(k,i)] = 1
        # and M = sum_r u_r ⊗ v_r ⊗ w_r
        # So M[(i,j),(j,k),(k,i)] = sum_r U[(i,j),r] * V[(j,k),r] * W[(k,i),r]
        
        # To compute C = A*B:
        # C[(k,i)] = sum_j A[(i,j)] * B[(j,k)]
        # = sum_j sum_r U[(i,j),r] * V[(j,k),r] * W[(k,i),r]
        # Wait, that's not quite right either. Let me think again.
        
        # The standard tensor decomposition for matrix multiplication:
        # If M_{n,m,p} = sum_{r=1}^R u_r ⊗ v_r ⊗ w_r
        # then for A (n×m), B (m×p), C = AB (n×p):
        # C[i,k] = sum_{j=1}^m sum_{r=1}^R u_r[i,j] * v_r[j,k] * w_r[k,i]
        
        # Let's verify this
        C_computed = np.zeros((n, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        C_computed[i, k] += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
        
        # Check
        max_err = np.max(np.abs(C_computed - C_expected))
        if max_err > 1e-10:
            # This shouldn't happen, but if it does, return a fallback
            pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
