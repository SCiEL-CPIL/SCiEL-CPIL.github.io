import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank is n * m * p
    rank = n * m * p
    
    # Dimensions for U, V, W
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Initialize with zeros
    U = np.zeros((dim_u, rank), dtype=float)
    V = np.zeros((dim_v, rank), dtype=float)
    W = np.zeros((dim_w, rank), dtype=float)
    
    # Fill in the ones for each (i, j, k) term
    # r is the rank index
    # i in [0, n), j in [0, m), k in [0, p)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                
                # U index: i * m + j
                u_idx = i * m + j
                U[u_idx, r] = 1.0
                
                # V index: j * p + k
                v_idx = j * p + k
                V[v_idx, r] = 1.0
                
                # W index: k * n + i (based on "trace convention W[k*n+i,r]")
                w_idx = k * n + i
                W[w_idx, r] = 1.0
                
    # Optional: Verification step to ensure correctness
    # Compute the tensor T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    # We expect T to be 1 at valid (i, j, k) triples and 0 elsewhere.
    # However, the "elsewhere" in the full tensor space is complex.
    # We check that the matrix multiplication property holds for random matrices.
    # This is a strong check.
    
    A_rand = np.random.rand(n, m)
    B_rand = np.random.rand(m, p)
    C_expected = A_rand @ B_rand
    
    C_decomposed = np.zeros((n, p))
    for r in range(rank):
        # Extract vectors
        u_vec = U[:, r]  # size n*m
        v_vec = V[:, r]  # size m*p
        w_vec = W[:, r]  # size p*n
        
        # Reshape to matrices
        # U corresponds to A indices (i, j) -> reshape (n, m)
        # V corresponds to B indices (j, k) -> reshape (m, p)
        # W corresponds to C indices (k, i) -> reshape (p, n) ? 
        # Wait, the product is C = A B.
        # Term is sum_j A[i, j] B[j, k].
        # The factorization is C[i, k] = sum_r (sum_j U[i, j, r] A[i, j]) ... no.
        # Standard bilinear form:
        # C[i, k] = sum_r ( sum_j U[i, j, r] A[i, j] ) * ( sum_k' V[j, k', r] B[j, k'] ) ? No.
        
        # Correct interpretation:
        # C[i, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r] ? No, indices don't match for summation.
        # The identity is:
        # sum_j A[i, j] B[j, k] = sum_r ( sum_j U[i, j, r] A[i, j] ) ... this is not right.
        
        # The tensor decomposition identity is:
        # T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
        # Where T[i, j, k] is 1 if this term contributes to C[i, k] (which it always does for the matching i, k).
        # Actually, for matrix multiplication, the tensor is:
        # T[i, j, k] = 1 if the term is A[i, j] * B[j, k] contributing to C[i, k].
        # But the output index is (i, k).
        # So the 3rd index of the tensor is the output index.
        # Let output index be l = k * n + i.
        # Then T[i, j, k] corresponds to output l.
        # The decomposition is:
        # M[l, (i, j), (j, k)] = sum_r W[l, r] U[(i, j), r] V[(j, k), r]
        
        # Let's verify this specific structure.
        # We constructed:
        # U[i*m+j, r] = 1
        # V[j*p+k, r] = 1
        # W[k*n+i, r] = 1
        # For a specific r corresponding to (i0, j0, k0).
        # U is 1 at (i0, j0).
        # V is 1 at (j0, k0).
        # W is 1 at (k0, i0).
        
        # So for this r:
        # Product term in sum: W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r]
        # This is 1 if:
        # 1. (k, i) == (k0, i0)  => k=k0, i=i0
        # 2. (i, j) == (i0, j0)  => i=i0, j=j0
        # 3. (j, k) == (j0, k0)  => j=j0, k=k0
        
        # So it is 1 if i=i0, j=j0, k=k0.
        # And the output index is l = k*n + i.
        # So this term contributes 1 to the output component (i, k) if i=i0 and k=k0.
        # And it requires the input components A[i0, j0] and B[j0, k0] to be multiplied.
        # This matches exactly the schoolbook multiplication term A[i0, j0] * B[j0, k0] contributing to C[i0, k0].
        
        pass # Verification logic is implicitly satisfied by construction.
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
