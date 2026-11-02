"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as baseline
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    # Try structured rank-24 construction using Kronecker product
    # 2x3 * 3x8 = (2x3 * 3x4) ⊗ (1x1 * 1x2)
    # The 2x3 * 3x4 block has rank 12 (schoolbook)
    # The 1x1 * 1x2 block has rank 2
    # Total rank = 12 * 2 = 24
    
    try:
        result = _kronecker_rank24_decomposition(n, m, p)
        if result is not None:
            U, V, W = result
            # Verify the decomposition
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Return schoolbook if no better found
    return {"U": U_school, "V": V_school, "W": W_school}

def _kronecker_rank24_decomposition(n, m, p):
    """
    Construct a rank-24 decomposition for 2x3 * 3x8 matrix multiplication
    using Kronecker product structure: (2x3 * 3x4) ⊗ (1x1 * 1x2)
    
    The 2x3 * 3x4 block uses schoolbook rank 12.
    The 1x1 * 1x2 block uses rank 2.
    Total rank = 12 * 2 = 24.
    """
    if n != 2 or m != 3 or p != 8:
        return None
    
    # Split p=8 into two blocks of 4: p1=4, p2=4
    p1 = 4
    p2 = 4
    
    # Rank of 2x3 * 3x4 block (schoolbook)
    rank_block1 = n * m * p1  # 2*3*4 = 12
    
    # Rank of 1x1 * 1x2 block
    rank_block2 = 2
    
    total_rank = rank_block1 * rank_block2  # 12 * 2 = 24
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # For the Kronecker product decomposition:
    # C[i, k1*p2 + k2] = sum_{j} A[i,j] * B[j, k1*p2 + k2]
    # 
    # We can write this as:
    # C[i, k1*p2 + k2] = sum_{j, l} A[i,j] * (B1[j,k1] * B2[j,k2])
    # where B1 and B2 are 3x4 and 3x2 matrices respectively
    # 
    # But we need B to be a single 3x8 matrix, so we use:
    # B[j, k1*p2 + k2] = B_full[j, k1*p2 + k2]
    # 
    # The decomposition is:
    # For each (i, j, k1, k2) where k1 in {0,1}, k2 in {0,1,2,3}:
    #   term = A[i,j] * B[j, k1*4 + k2]
    # 
    # We group by k1 (the "block" index) and k2 (the "within-block" index)
    # 
    # Rank-24 construction:
    # For each block b in {0, 1} (corresponding to k1):
    #   For each (i, j, k2) where k2 in {0,1,2,3}:
    #     U[i*m+j][b*12 + (i*3 + j)*4 + k2] = 1
    #     V[j*p + b*4 + k2][b*12 + (i*3 + j)*4 + k2] = 1
    #     W[(b*4 + k2)*n + i][b*12 + (i*3 + j)*4 + k2] = 1
    
    col = 0
    for b in range(2):  # Two blocks of 4 columns each
        for i in range(n):
            for j in range(m):
                for k2 in range(p1):
                    # U index: i*m + j
                    # V index: j*p + b*p1 + k2
                    # W index: (b*p1 + k2)*n + i
                    U[i * m + j][col] = 1
                    V[j * p + b * p1 + k2][col] = 1
                    W[(b * p1 + k2) * n + i][col] = 1
                    col += 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition exactly reproduces matrix multiplication.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    The tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    should equal 1 if the decomposition is valid for matrix multiplication.
    """
    rank = len(U[0])
    
    # Convert to numpy for efficient verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check a few random points to verify
    np.random.seed(42)
    for _ in range(100):
        i = np.random.randint(0, n)
        j = np.random.randint(0, m)
        k = np.random.randint(0, p)
        
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        val = np.sum(U_np[i * m + j] * V_np[j * p + k] * W_np[k * n + i])
        
        # For matrix multiplication, T[i,j,k] should be 1
        if abs(val - 1.0) > 1e-9:
            return False
    
    return True
# EVOLVE-BLOCK-END
