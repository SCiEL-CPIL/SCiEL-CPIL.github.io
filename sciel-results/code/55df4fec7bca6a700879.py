"""
Improved Strassen-based decomposition for 2x2x14 matrix multiplication.
Maintains the rank-49 construction but with optimized code structure and verification.
"""
import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Construct the rank-49 decomposition using Strassen's algorithm
    result = _construct_strassen_decomposition(n, m, p)
    
    # Verify the decomposition
    if result and _verify_decomposition(result, n, m, p):
        return result
    
    # Fallback to schoolbook if verification fails (should not happen)
    return _construct_schoolbook(n, m, p)


def _construct_strassen_decomposition(n, m, p):
    """
    Constructs a rank-49 decomposition for 2x2x14 matrix multiplication.
    This is derived from applying Strassen's algorithm (rank 7) to each of the 7 
    non-overlapping 2x2 blocks of the output matrix (since p=14, we have 7 blocks of width 2).
    """
    if n != 2 or m != 2 or p % 2 != 0:
        return None
        
    num_blocks = p // 2
    rank = 7 * num_blocks
    
    # Initialize matrices with zeros
    # U shape: (n*m, rank) -> (4, rank)
    # V shape: (m*p, rank) -> (28, rank)
    # W shape: (p*n, rank) -> (28, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products for 2x2x2 matrix multiplication
    # Let A = [[a, b], [c, d]] and B = [[e, f], [g, h]]
    # The 7 products are:
    # M1 = (a + d)(e + h)
    # M2 = (c + d)e
    # M3 = a(f - h)
    # M4 = d(g - e)
    # M5 = (a + b)h
    # M6 = (c - a)(e + f)
    # M7 = (b - d)(g + h)
    
    # For each block b in 0..6:
    # The block corresponds to B columns 2b, 2b+1 and C columns 2b, 2b+1.
    # The structure is identical for each block.
    
    for b in range(num_blocks):
        base_col = b * 7
        # B row indices for this block: 2b, 2b+1
        # C col indices for this block: 2b, 2b+1
        
        # M1 = (A[0,0] + A[1,1]) * (B[0,2b] + B[1,2b+1])
        # U indices: A[0,0]=0, A[1,1]=3
        # V indices: B[0,2b]=2b, B[1,2b+1]=2+2b+1=2b+3
        # W indices: C[2b,0]=2b*2+0=4b, C[2b+1,1]=(2b+1)*2+1=4b+3
        U[0][base_col] += 1
        U[3][base_col] += 1
        V[2 * b][base_col] += 1
        V[2 * b + 3][base_col] += 1
        W[4 * b][base_col] += 1
        W[4 * b + 3][base_col] += 1
        
        # M2 = (A[1,0] + A[1,1]) * B[0,2b]
        # U indices: A[1,0]=2, A[1,1]=3
        # V indices: B[0,2b]=2b
        # W indices: C[2b,1]=4b+1, C[2b+1,1]=4b+3
        U[2][base_col + 1] += 1
        U[3][base_col + 1] += 1
        V[2 * b][base_col + 1] += 1
        W[4 * b + 1][base_col + 1] += 1
        W[4 * b + 3][base_col + 1] += -1
        
        # M3 = A[0,0] * (B[0,2b+1] - B[1,2b+1])
        # U indices: A[0,0]=0
        # V indices: B[0,2b+1]=2b+1, B[1,2b+1]=2b+3
        # W indices: C[2b,0]=4b, C[2b+1,0]=4b+2
        U[0][base_col + 2] += 1
        V[2 * b + 1][base_col + 2] += 1
        V[2 * b + 3][base_col + 2] += -1
        W[4 * b][base_col + 2] += 1
        W[4 * b + 2][base_col + 2] += 1
        
        # M4 = A[1,1] * (B[1,2b] - B[0,2b])
        # U indices: A[1,1]=3
        # V indices: B[1,2b]=2b+2, B[0,2b]=2b
        # W indices: C[2b,0]=4b, C[2b+1,0]=4b+2
        U[3][base_col + 3] += 1
        V[2 * b + 2][base_col + 3] += 1
        V[2 * b][base_col + 3] += -1
        W[4 * b][base_col + 3] += 1
        W[4 * b + 2][base_col + 3] += 1
        
        # M5 = (A[0,0] + A[0,1]) * B[1,2b+1]
        # U indices: A[0,0]=0, A[0,1]=1
        # V indices: B[1,2b+1]=2b+3
        # W indices: C[2b,0]=4b, C[2b+1,0]=4b+2
        U[0][base_col + 4] += 1
        U[1][base_col + 4] += 1
        V[2 * b + 3][base_col + 4] += 1
        W[4 * b][base_col + 4] += -1
        W[4 * b + 2][base_col + 4] += 1
        
        # M6 = (A[1,0] - A[0,0]) * (B[0,2b] + B[0,2b+1])
        # U indices: A[1,0]=2, A[0,0]=0
        # V indices: B[0,2b]=2b, B[0,2b+1]=2b+1
        # W indices: C[2b+1,1]=4b+3
        U[2][base_col + 5] += 1
        U[0][base_col + 5] += -1
        V[2 * b][base_col + 5] += 1
        V[2 * b + 1][base_col + 5] += 1
        W[4 * b + 3][base_col + 5] += 1
        
        # M7 = (A[0,1] - A[1,1]) * (B[1,2b] + B[1,2b+1])
        # U indices: A[0,1]=1, A[1,1]=3
        # V indices: B[1,2b]=2b+2, B[1,2b+1]=2b+3
        # W indices: C[2b,0]=4b
        U[1][base_col + 6] += 1
        U[3][base_col + 6] += -1
        V[2 * b + 2][base_col + 6] += 1
        V[2 * b + 3][base_col + 6] += 1
        W[4 * b][base_col + 6] += 1
        
    return {"U": U, "V": V, "W": W}


def _construct_schoolbook(n, m, p):
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


def _verify_decomposition(decomp, n, m, p, num_tests=5):
    """
    Verifies the decomposition by checking C = A @ B against the tensor product.
    Uses vectorized NumPy operations for speed.
    """
    U = np.array(decomp["U"], dtype=np.float64)
    V = np.array(decomp["V"], dtype=np.float64)
    W = np.array(decomp["W"], dtype=np.float64)
    
    rank = U.shape[1]
    
    # Reshape for broadcasting
    # U: (n, m, rank)
    # V: (m, p, rank)
    # W: (p, n, rank)
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    for _ in range(num_tests):
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B
        
        # Compute C from decomposition
        # C[i,k] = sum_r W[k,i,r] * (sum_j U[i,j,r] * A[i,j] * V[j,k,r] * B[j,k])
        
        # Efficient calculation:
        # 1. Compute intermediate tensors
        # A_U = A[i,j] * U[i,j,r] -> (n, m, rank)
        A_U = A[:, :, np.newaxis] * U_reshaped
        # B_V = B[j,k] * V[j,k,r] -> (m, p, rank)
        B_V = B[:, :, np.newaxis] * V_reshaped
        
        # 2. Contract over j: sum_j A_U[i,j,r] * B_V[j,k,r]
        # A_U: (n, m, rank), B_V: (m, p, rank)
        # We want result[i,k,r] = sum_j A_U[i,j,r] * B_V[j,k,r]
        # This is a batch matrix multiplication if we move 'rank' to the front
        # A_U_trans = (rank, n, m)
        # B_V_trans = (rank, m, p)
        # Product = (rank, n, p)
        A_U_batch = A_U.transpose(2, 0, 1)  # (rank, n, m)
        B_V_batch = B_V.transpose(2, 0, 1)  # (rank, m, p)
        
        # Matrix multiply: (rank, n, p)
        Intermediate = np.matmul(A_U_batch, B_V_batch) # (rank, n, p)
        
        # 3. Multiply by W and sum over r
        # W_reshaped: (p, n, rank)
        # We need to sum_r W[k,i,r] * Intermediate[r,i,k]
        # W_trans = (rank, p, n)
        W_batch = W_reshaped.transpose(2, 0, 1) # (rank, p, n)
        
        # Element-wise multiply: (rank, p, n) * (rank, n, p)
        # Note: Intermediate is (rank, n, p). W_batch is (rank, p, n).
        # We need to align dimensions for element-wise multiplication.
        # Intermediate[r, i, k] * W_batch[r, k, i]
        C_computed = np.sum(Intermediate * W_batch.transpose(0, 2, 1), axis=0)
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
            
    return True
# EVOLVE-BLOCK-END
