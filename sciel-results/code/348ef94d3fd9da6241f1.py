"""Program for matrix multiplication 2x3x10 rank-minimal decomposition."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-55 decomposition of the 2x3x10 matrix multiplication tensor.
    
    Strategy: Split m=3 into 2+1.
    For each of the 5 blocks of size p=2:
      - Use Strassen for the 2x2x2 part (rank 7) for i in {0,1}, j in {0,1}, k in block
      - Use standard multiplication for the 2x1x2 part (rank 4) for i in {0,1}, j=2, k in block
    Total rank = 5 * (7 + 4) = 55.
    """
    import time
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    rank = 55
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]  # (6, 55)
    V = [[0] * rank for _ in range(m * p)]  # (30, 55)
    W = [[0] * rank for _ in range(p * n)]  # (20, 55)
    
    col = 0
    num_blocks = p // 2  # 5 blocks of size 2
    
    for b in range(num_blocks):
        k_offset = 2 * b  # starting k index for this block
        
        # --- Strassen part for i in {0,1}, j in {0,1}, k in {k_offset, k_offset+1} ---
        # Strassen's 7 multiplications for 2x2 matrix multiplication
        # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
        # M1 = a00 * (b00 + b11)
        # M2 = (a10 + a11) * b00
        # M3 = (a00 + a01) * b11
        # M4 = a11 * (b10 - b00)
        # M5 = a00 * (b01 + b10)
        # M6 = (a10 - a00) * (b00 + b01)
        # M7 = (a01 - a11) * (b10 + b11)
        
        # M1: U[i=0, j=0]=1, V[j=0, k=0]=1 and V[j=0, k=1]=1
        # Actually, Strassen combines sums. Let me be more careful.
        
        # In the tensor decomposition, each rank-1 term corresponds to one multiplication.
        # Strassen's algorithm uses 7 multiplications:
        # M1 = A[0,0] * (B[0,0] + B[1,1])
        # M2 = (A[1,0] + A[1,1]) * B[0,0]
        # M3 = (A[0,0] + A[0,1]) * B[1,1]
        # M4 = A[1,1] * (B[1,0] - B[0,0])
        # M5 = A[0,0] * (B[0,1] + B[1,0])
        # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
        # M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
        
        # Output:
        # C[0,0] = M1 - M4 + M7
        # C[0,1] = M5 - M7
        # C[1,0] = M2 - M6
        # C[1,1] = M1 + M3 - M5 + M6
        
        # For each rank-1 term, we need U, V, W entries.
        
        # M1: A[0,0] * (B[0,0] + B[1,1])
        # U[0*3+0][col] = 1 (i=0, j=0)
        # V[0*10 + k_offset][col] = 1 (j=0, k=k_offset)
        # V[0*10 + k_offset+1][col] = 1 (j=0, k=k_offset+1)
        # W[k_offset*2+0][col] = 1 (k=k_offset, i=0)
        # W[k_offset*2+0][col] += 1, W[(k_offset+1)*2+0][col] += 1
        U[0][col] = 1
        V[0 * 10 + k_offset][col] = 1
        V[0 * 10 + k_offset + 1][col] = 1
        W[k_offset * 2 + 0][col] = 1
        W[(k_offset + 1) * 2 + 0][col] = 1
        col += 1
        
        # M2: (A[1,0] + A[1,1]) * B[0,0]
        # U[1*3+0][col] = 1, U[1*3+1][col] = 1 (i=1, j=0 and i=1, j=1)
        # V[0*10 + k_offset][col] = 1
        # W[k_offset*2+1][col] = 1 (k=k_offset, i=1)
        U[1 * 3 + 0][col] = 1
        U[1 * 3 + 1][col] = 1
        V[0 * 10 + k_offset][col] = 1
        W[k_offset * 2 + 1][col] = 1
        col += 1
        
        # M3: (A[0,0] + A[0,1]) * B[1,1]
        # U[0*3+0][col] = 1, U[0*3+1][col] = 1 (i=0, j=0 and i=0, j=1)
        # V[1*10 + k_offset+1][col] = 1 (j=1, k=k_offset+1)
        # W[(k_offset+1)*2+0][col] = 1 (k=k_offset+1, i=0)
        U[0 * 3 + 0][col] = 1
        U[0 * 3 + 1][col] = 1
        V[1 * 10 + k_offset + 1][col] = 1
        W[(k_offset + 1) * 2 + 0][col] = 1
        col += 1
        
        # M4: A[1,1] * (B[1,0] - B[0,0])
        # U[1*3+1][col] = 1 (i=1, j=1)
        # V[1*10 + k_offset][col] = 1, V[0*10 + k_offset][col] = -1
        # W[k_offset*2+1][col] = -1 (contribution to C[1,1] with negative sign)
        U[1 * 3 + 1][col] = 1
        V[1 * 10 + k_offset][col] = 1
        V[0 * 10 + k_offset][col] = -1
        W[k_offset * 2 + 1][col] = -1
        col += 1
        
        # M5: A[0,0] * (B[0,1] + B[1,0])
        # U[0*3+0][col] = 1
        # V[0*10 + k_offset+1][col] = 1, V[1*10 + k_offset][col] = 1
        # W[k_offset*2+1][col] = -1 (contribution to C[0,1] with negative sign from C[0,1]=M5-M7)
        # Actually, let me recheck. C[0,1] = M5 - M7. So M5 contributes +1 to C[0,1].
        # C[1,1] = M1 + M3 - M5 + M6. So M5 contributes -1 to C[1,1].
        U[0 * 3 + 0][col] = 1
        V[0 * 10 + k_offset + 1][col] = 1
        V[1 * 10 + k_offset][col] = 1
        W[k_offset * 2 + 1][col] = -1  # C[1,1] gets -M5
        # But we also need C[0,1] to get +M5. C[0,1] is at index (k_offset+1)*2+1 and k_offset*2+1?
        # Wait, C[0,1] means i=0, j=1. In W, the index is k*n+i. 
        # For output C[i,j], the W index should encode which output element.
        # Actually, in the tensor decomposition, W[k*n+i, col] gives the coefficient for output element (k,i) 
        # but that doesn't seem right for matrix multiplication.
        
        # Let me reconsider the W indexing. The task says "The output factor follows the official trace convention W[k*n+i,r]".
        # This is unusual. Typically for matrix multiplication A*B=C where A is n x m, B is m x p, C is n x p,
        # the tensor is T[i,j,k] = sum_l A[i,l] * B[l,k] = C[i,k].
        # So the output indices are i and k, not i and j.
        # W[k*n+i, r] means the W matrix has rows indexed by (k,i) pairs, i.e., k*n + i.
        # So for a given rank-1 term, W[k*n+i, r] = coefficient for C[i,k] from that term.
        
        # Let me redo this more carefully.
        # For the 2x2x2 subproblem with i in {0,1}, j in {0,1}, k in {0,1}:
        # A is 2x2 (i x j), B is 2x2 (j x k), C is 2x2 (i x k).
        # T[i,j,k] = C[i,k] = sum_l A[i,l] * B[l,k]
        
        # Strassen decompositions:
        # M1 = A[0,0]*(B[0,0]+B[1,1])
        #   U: A[0,0] -> i=0,j=0
        #   V: B[0,0] and B[1,1] -> j=0,k=0 and j=1,k=1
        #   W: contributes to C[0,0] and C[1,1]
        #   C[0,0] gets +M1, C[1,1] gets +M1
        
        # Let me use a cleaner approach and define the 7 Strassen terms explicitly.
        
        pass  # This is getting complex; let me restart with a clearer formulation.
    
    # Let me restart with a clearer implementation.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    num_blocks = p // 2  # 5 blocks
    
    for b in range(num_blocks):
        k0 = 2 * b  # first k in block
        k1 = 2 * b + 1  # second k in block
        
        # Strassen for 2x2x2: i in {0,1}, j in {0,1}, k in {k0, k1}
        # M1 = A[0,0] * (B[0,0] + B[1,1])
        #   U[0][col] = 1  (i=0, j=0)
        #   V[0][col] = 1  (j=0, k=k0)
        #   V[10+k1][col] = 1  (j=1, k=k1)
        #   W[k0*2+0][col] = 1  (k=k0, i=0) -> C[0, k0]
        #   W[k1*2+1][col] = 1  (k=k1, i=1) -> C[1, k1]
        U[0][col] = 1
        V[0 * 10 + k0][col] = 1
        V[1 * 10 + k1][col] = 1
        W[k0 * 2 + 0][col] = 1
        W[k1 * 2 + 1][col] = 1
        col += 1
        
        # M2 = (A[1,0] + A[1,1]) * B[0,0]
        #   U[3][col] = 1  (i=1, j=0)
        #   U[4][col] = 1  (i=1, j=1)
        #   V[0][col] = 1  (j=0, k=k0)
        #   W[k0*2+1][col] = 1  (k=k0, i=1) -> C[1, k0]
        U[3][col] = 1
        U[4][col] = 1
        V[0 * 10 + k0][col] = 1
        W[k0 * 2 + 1][col] = 1
        col += 1
        
        # M3 = (A[0,0] + A[0,1]) * B[1,1]
        #   U[0][col] = 1  (i=0, j=0)
        #   U[1][col] = 1  (i=0, j=1)
        #   V[10+k1][col] = 1  (j=1, k=k1)
        #   W[k1*2+0][col] = 1  (k=k1, i=0) -> C[0, k1]
        U[0][col] = 1
        U[1][col] = 1
        V[1 * 10 + k1][col] = 1
        W[k1 * 2 + 0][col] = 1
        col += 1
        
        # M4 = A[1,1] * (B[1,0] - B[0,0])
        #   U[4][col] = 1  (i=1, j=1)
        #   V[10+k0][col] = 1  (j=1, k=k0)
        #   V[0][col] = -1  (j=0, k=k0)
        #   W[k0*2+1][col] = -1  (k=k0, i=1) -> C[1, k0] gets -M4
        U[4][col] = 1
        V[1 * 10 + k0][col] = 1
        V[0 * 10 + k0][col] = -1
        W[k0 * 2 + 1][col] = -1
        col += 1
        
        # M5 = A[0,0] * (B[0,1] + B[1,0])
        #   U[0][col] = 1  (i=0, j=0)
        #   V[10+k0][col] = 1  (j=1, k=k0)
        #   V[1][col] = 1  (j=0, k=k1)
        #   W[k0*2+1][col] = -1  (k=k0, i=1) -> C[1, k0] gets -M5
        #   W[k1*2+0][col] = 1  (k=k1, i=0) -> C[0, k1] gets +M5
        U[0][col] = 1
        V[1 * 10 + k0][col] = 1
        V[0 * 10 + k1][col] = 1
        W[k0 * 2 + 1][col] = -1
        W[k1 * 2 + 0][col] = 1
        col += 1
        
        # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
        #   U[3][col] = 1  (i=1, j=0)
        #   U[0][col] = -1  (i=0, j=0)
        #   V[0][col] = 1  (j=0, k=k0)
        #   V[1][col] = 1  (j=0, k=k1)
        #   W[k0*2+1][col] = 1  (k=k0, i=1) -> C[1, k0] gets +M6
        #   W[k1*2+1][col] = 1  (k=k1, i=1) -> C[1, k1] gets +M6
        U[3][col] = 1
        U[0][col] = -1
        V[0 * 10 + k0][col] = 1
        V[0 * 10 + k1][col] = 1
        W[k0 * 2 + 1][col] = 1
        W[k1 * 2 + 1][col] = 1
        col += 1
        
        # M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
        #   U[1][col] = 1  (i=0, j=1)
        #   U[4][col] = -1  (i=1, j=1)
        #   V[10+k0][col] = 1  (j=1, k=k0)
        #   V[10+k1][col] = 1  (j=1, k=k1)
        #   W[k0*2+0][col] = 1  (k=k0, i=0) -> C[0, k0] gets +M7
        #   W[k1*2+0][col] = -1  (k=k1, i=0) -> C[0, k1] gets -M7
        U[1][col] = 1
        U[4][col] = -1
        V[1 * 10 + k0][col] = 1
        V[1 * 10 + k1][col] = 1
        W[k0 * 2 + 0][col] = 1
        W[k1 * 2 + 0][col] = -1
        col += 1
        
        # Standard multiplication for j=2 (the remaining middle dimension)
        # 2x1x2: i in {0,1}, j=2, k in {k0, k1}
        # 4 terms:
        # T1: A[0,2] * B[2,k0] -> C[0,k0]
        # T2: A[0,2] * B[2,k1] -> C[0,k1]
        # T3: A[1,2] * B[2,k0] -> C[1,k0]
        # T4: A[1,2] * B[2,k1] -> C[1,k1]
        
        # T1: U[2][col]=1 (i=0,j=2), V[20+k0][col]=1 (j=2,k=k0), W[k0*2+0][col]=1
        U[2][col] = 1
        V[2 * 10 + k0][col] = 1
        W[k0 * 2 + 0][col] = 1
        col += 1
        
        # T2: U[2][col]=1, V[20+k1][col]=1, W[k1*2+0][col]=1
        U[2][col] = 1
        V[2 * 10 + k1][col] = 1
        W[k1 * 2 + 0][col] = 1
        col += 1
        
        # T3: U[5][col]=1 (i=1,j=2), V[20+k0][col]=1, W[k0*2+1][col]=1
        U[5][col] = 1
        V[2 * 10 + k0][col] = 1
        W[k0 * 2 + 1][col] = 1
        col += 1
        
        # T4: U[5][col]=1, V[20+k1][col]=1, W[k1*2+1][col]=1
        U[5][col] = 1
        V[2 * 10 + k1][col] = 1
        W[k1 * 2 + 1][col] = 1
        col += 1
        
        # Check deadline
        if time.time() > deadline:
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
