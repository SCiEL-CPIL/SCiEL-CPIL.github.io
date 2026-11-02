import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s out of 150s wall-clock limit
    
    n, m, p = 2, 2, 13
    rank = 46
    
    # Strassen-based decomposition for 2x2 * 2x13
    # We use 7 Strassen products for the 2x2 block, but since the right matrix is 2x13,
    # we need to be more careful. Let's use a direct approach.
    
    # Actually, for multiplying A (2x2) by B (2x13), we have:
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # C has shape 2x13
    
    # The tensor T has indices (i*m+k, k*p+j, j*n+i) where:
    # i in [0,1] (rows of A / output), k in [0,1] (cols of A / rows of B), j in [0,12] (cols of B / output)
    # U index: i*m + k = 2*i + k, so U has 4 rows
    # V index: k*p + j = 13*k + j, so V has 26 rows
    # W index: j*n + i = 13*j + i, so W has 26 rows
    
    # Standard rank for 2x2 * 2x13 multiplication:
    # We can compute this with at most 2*2*13 = 52 scalar multiplications (naive)
    # But we can do better using Strassen-like ideas or bilinear complexity results.
    
    # The bilinear complexity of 2x2 * 2x13 matrix multiplication is known to be 46.
    # Let's construct an explicit rank-46 decomposition.
    
    # For 2x2 * 2x13, we can view this as:
    # C = A * B where A is 2x2, B is 2x13
    # C[0, j] = A[0,0]*B[0,j] + A[0,1]*B[1,j]
    # C[1, j] = A[1,0]*B[0,j] + A[1,1]*B[1,j]
    
    # We can use the fact that we only have 2x2 left matrix to reduce rank.
    # One approach: express each output entry as a linear combination of a small set of products.
    
    # Let's use the following decomposition:
    # We'll have 46 rank-1 tensors.
    
    U = np.zeros((4, rank))
    V = np.zeros((26, rank))
    W = np.zeros((26, rank))
    
    # Let's build a valid decomposition.
    # For each j in 0..12, C[0,j] = A[0,0]*B[0,j] + A[0,1]*B[1,j]
    # C[1,j] = A[1,0]*B[0,j] + A[1,1]*B[1,j]
    
    # Naive would be 4*13 = 52 products. We can save 6 products.
    
    # Strategy: Use shared products where possible.
    # For example, A[0,0]*B[0,j] appears in C[0,j], and A[1,0]*B[0,j] appears in C[1,j].
    # We can't directly share these unless we use Strassen-like combinations.
    
    # Let's use a known rank-46 construction.
    # One approach: treat the 2x2 A and group B columns.
    
    # Actually, let me construct it systematically.
    # We need 46 rank-1 terms.
    
    # Term construction:
    # For each of the 13 columns j of B, we need to compute:
    # C[0,j] = A00*B0j + A01*B1j
    # C[1,j] = A10*B0j + A11*B1j
    
    # Let's define products:
    # p_j0 = A00 * B0j  (for j=0..12) -> 13 products
    # p_j1 = A01 * B1j  (for j=0..12) -> 13 products
    # p_j2 = A10 * B0j  (for j=0..12) -> 13 products
    # p_j3 = A11 * B1j  (for j=0..12) -> 13 products
    # Total: 52. We need to reduce to 46.
    
    # We can use Strassen for the 2x2 part if we view pairs of columns.
    # For two adjacent columns, we can use 7 Strassen products instead of 8.
    # 13 columns = 6 pairs + 1 single. 6*7 + 4 = 46. That works!
    
    # Let's pair columns: (0,1), (2,3), (4,5), (6,7), (8,9), (10,11), and single 12.
    
    # For each pair (j, j+1), use Strassen:
    # M1 = (A00 + A11) * (B0j + B1j)  -- wait, Strassen is for 2x2 * 2x2.
    # For 2x2 A and 2x2 B_block (2 columns), we can use Strassen to compute 2x2 * 2x2 in 7 mults.
    # But our B_block is 2x2 (2 rows, 2 columns). Yes!
    
    # Strassen for 2x2 * 2x2:
    # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a + d)(e + h)
    # M2 = (c + d)e
    # M3 = a(f - h)
    # M4 = d(g - e)
    # M5 = (a + b)h
    # M6 = (c - a)(e + f)
    # M7 = (b - d)(g + h)
    # Result:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For pair j, j+1:
    # B_block = [[B0j, B0(j+1)], [B1j, B1(j+1)]]
    # A = [[A00, A01], [A10, A11]]
    
    # So for each pair, we use 7 products. For the single column 12, we use 4 products.
    # Total: 6*7 + 4 = 46.
    
    idx = 0
    # Process 6 pairs
    for pair in range(6):
        j = 2 * pair
        j1 = j + 1
        # Strassen 7 products
        # M1 = (A00 + A11) * (B0j + B1j)
        U[0, idx] = 1.0  # A00
        U[1, idx] = 0.0  # A01
        U[2, idx] = 1.0  # A10
        U[3, idx] = 0.0  # A11  -- wait, we need A00 + A11
        # U index: i*m+k. U[0]=A00, U[1]=A01, U[2]=A10, U[3]=A11
        # For M1: (A00 + A11) * (B0j + B1j)
        # A part: A00 + A11 -> U[0]=1, U[3]=1, U[1]=0, U[2]=0
        U[0, idx] = 1.0
        U[1, idx] = 0.0
        U[2, idx] = 0.0
        U[3, idx] = 1.0
        # B part: B0j + B1j -> V[13*0+j] = B0j, V[13*1+j] = B1j
        V[0 * p + j, idx] = 1.0
        V[1 * p + j, idx] = 1.0
        # W part: M1 contributes to C[0,j] and C[1,j1] and C[1,j] and C[0,j1]
        # From Strassen: C11 = M1 + M4 - M5 + M7, C12 = M3 + M5, C21 = M2 + M4, C22 = M1 - M2 + M3 + M6
        # C11 = C[0,j], C12 = C[0,j1], C21 = C[1,j], C22 = C[1,j1]
        # M1 contributes +1 to C[0,j] and C[1,j1]
        W[j * n + 0, idx] = 1.0   # C[0, j]
        W[j1 * n + 1, idx] = 1.0  # C[1, j1]
        idx += 1
        
        # M2 = (A10 + A11) * B0j
        U[0, idx] = 0.0
        U[1, idx] = 0.0
        U[2, idx] = 1.0  # A10
        U[3, idx] = 1.0  # A11
        V[0 * p + j, idx] = 1.0
        V[1 * p + j, idx] = 0.0
        # M2 contributes -1 to C[1,j1] and -1 to C[1,j]
        # C21 = M2 + M4, C22 = M1 - M2 + M3 + M6
        W[j * n + 1, idx] = 1.0   # C[1, j]
        W[j1 * n + 1, idx] = -1.0  # C[1, j1]
        idx += 1
        
        # M3 = A00 * (B0(j+1) - B1(j+1))
        U[0, idx] = 1.0
        U[1, idx] = 0.0
        U[2, idx] = 0.0
        U[3, idx] = 0.0
        V[0 * p + j1, idx] = 1.0
        V[1 * p + j1, idx] = -1.0
        # M3 contributes +1 to C[0,j1] and +1 to C[1,j1]
        # C12 = M3 + M5, C22 = M1 - M2 + M3 + M6
        W[j1 * n + 0, idx] = 1.0  # C[0, j1]
        W[j1 * n + 1, idx] = 1.0  # C[1, j1]
        idx += 1
        
        # M4 = A11 * (B1j - B0j)
        U[0, idx] = 0.0
        U[1, idx] = 0.0
        U[2, idx] = 0.0
        U[3, idx] = 1.0
        V[0 * p + j, idx] = -1.0
        V[1 * p + j, idx] = 1.0
        # M4 contributes +1 to C[0,j] and +1 to C[1,j]
        # C11 = M1 + M4 - M5 + M7, C21 = M2 + M4
        W[j * n + 0, idx] = 1.0  # C[0, j]
        W[j * n + 1, idx] = 1.0  # C[1, j]
        idx += 1
        
        # M5 = (A00 + A01) * B1(j+1)
        U[0, idx] = 1.0
        U[1, idx] = 1.0
        U[2, idx] = 0.0
        U[3, idx] = 0.0
        V[0 * p + j1, idx] = 0.0
        V[1 * p + j1, idx] = 1.0
        # M5 contributes -1 to C[0,j] and +1 to C[0,j1]
        # C11 = M1 + M4 - M5 + M7, C12 = M3 + M5
        W[j * n + 0, idx] = -1.0  # C[0, j]
        W[j1 * n + 0, idx] = 1.0  # C[0, j1]
        idx += 1
        
        # M6 = (A10 - A00) * (B0j + B0(j+1))
        U[0, idx] = -1.0
        U[1, idx] = 0.0
        U[2, idx] = 1.0
        U[3, idx] = 0.0
        V[0 * p + j, idx] = 1.0
        V[0 * p + j1, idx] = 1.0
        V[1 * p + j, idx] = 0.0
        V[1 * p + j1, idx] = 0.0
        # M6 contributes +1 to C[1,j1]
        # C22 = M1 - M2 + M3 + M6
        W[j1 * n + 1, idx] = 1.0  # C[1, j1]
        idx += 1
        
        # M7 = (A01 - A11) * (B1j + B1(j+1))
        U[0, idx] = 0.0
        U[1, idx] = 1.0
        U[2, idx] = 0.0
        U[3, idx] = -1.0
        V[0 * p + j, idx] = 0.0
        V[0 * p + j1, idx] = 0.0
        V[1 * p + j, idx] = 1.0
        V[1 * p + j1, idx] = 1.0
        # M7 contributes +1 to C[0,j]
        # C11 = M1 + M4 - M5 + M7
        W[j * n + 0, idx] = 1.0  # C[0, j]
        idx += 1
    
    # Single column j=12: use 4 products
    j = 12
    # A00 * B0j
    U[0, idx] = 1.0
    U[1, idx] = 0.0
    U[2, idx] = 0.0
    U[3, idx] = 0.0
    V[0 * p + j, idx] = 1.0
    V[1 * p + j, idx] = 0.0
    W[j * n + 0, idx] = 1.0  # C[0, 12]
    idx += 1
    
    # A01 * B1j
    U[0, idx] = 0.0
    U[1, idx] = 1.0
    U[2, idx] = 0.0
    U[3, idx] = 0.0
    V[0 * p + j, idx] = 0.0
    V[1 * p + j, idx] = 1.0
    W[j * n + 0, idx] = 1.0  # C[0, 12]
    idx += 1
    
    # A10 * B0j
    U[0, idx] = 0.0
    U[1, idx] = 0.0
    U[2, idx] = 1.0
    U[3, idx] = 0.0
    V[0 * p + j, idx] = 1.0
    V[1 * p + j, idx] = 0.0
    W[j * n + 1, idx] = 1.0  # C[1, 12]
    idx += 1
    
    # A11 * B1j
    U[0, idx] = 0.0
    U[1, idx] = 0.0
    U[2, idx] = 0.0
    U[3, idx] = 1.0
    V[0 * p + j, idx] = 0.0
    V[1 * p + j, idx] = 1.0
    W[j * n + 1, idx] = 1.0  # C[1, 12]
    idx += 1
    
    assert idx == rank, f"Expected {rank} terms, got {idx}"
    
    # Verify the decomposition
    # Build the tensor from U, V, W
    T_decomp = np.zeros((n * m, m * p, p * n))
    for t in range(rank):
        T_decomp[:, :, t] = np.outer(np.outer(U[:, t], V[:, t]), np.ones(1))  # placeholder
        # Actually, T[i, j, k] = U[i, t] * V[j, t] * W[k, t]
    
    T_decomp = np.zeros((n * m, m * p, p * n))
    for t in range(rank):
        for i in range(n * m):
            for j in range(m * p):
                for k in range(p * n):
                    T_decomp[i, j, k] += U[i, t] * V[j, t] * W[k, t]
    
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                T_target[i * m + k, k * p + j, j * n + i] = 1.0
    
    if not np.allclose(T_decomp, T_target, atol=1e-10):
        # If verification fails, fall back to naive rank-52
        rank = 52
        U = np.zeros((4, rank))
        V = np.zeros((26, rank))
        W = np.zeros((26, rank))
        idx = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k, idx] = 1.0
                    V[k * p + j, idx] = 1.0
                    W[j * n + i, idx] = 1.0
                    idx += 1
        assert idx == rank
    
    return {'U': U, 'V': V, 'W': W}
