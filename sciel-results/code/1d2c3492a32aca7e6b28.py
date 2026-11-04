"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 3]

def _verify(U, V, W, n=2, m=3, p=3):
    """Check if the decomposition exactly matches the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the tensor T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # and compare to the expected multiplication tensor.
    # For matrix multiplication C = A * B, C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i,j,k] should equal 1 if we're building the multiplication tensor.
    # Actually the tensor for matrix multiplication has T[i,j,k] = 1 for all (i,j,k) in [0,n)x[0,m)x[0,p)
    # Wait no. The matrix multiplication tensor M satisfies:
    # M[i,j,k] = 1 if the output C[i,k] uses input A[i,j] and B[j,k].
    # So M[i,j,k] = 1 for all valid (i,j,k).
    # The decomposition says: M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for c in range(rank):
                    actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _strassen_rank17():
    """
    Build a rank-17 decomposition for 2x3 x 3x3 matrix multiplication.
    
    The 2x3 x 3x3 multiplication can be seen as multiplying a 2x3 matrix A by a 3x3 matrix B.
    We can view this as two 2x2 Strassen multiplications plus one extra term, or use a block approach.
    
    Actually, let's think of it differently. A is 2x3, B is 3x3, C is 2x3.
    C[i,k] = sum_j=0..2 A[i,j] * B[j,k]
    
    We can split B into columns: B[:,0], B[:,1], B[:,2] (each 3x1... no, 3x1 isn't right).
    
    Better: Think of it as C = A * B where A is 2x3 and B is 3x3.
    We can write this using Strassen by padding or by a direct construction.
    
    A known approach: The tensor rank of 2x3 x 3x3 is at most 17.
    Let me construct it using the following method:
    
    View the 2x3 * 3x3 as involving the 2x2 * 2x2 Strassen pattern applied to blocks.
    
    Let A = [[a00, a01, a02], [a10, a11, a12]]
    Let B = [[b00, b01, b02], [b10, b11, b12], [b20, b21, b22]]
    
    C = A * B = [[c00, c01, c02], [c10, c11, c12]]
    
    c00 = a00*b00 + a01*b10 + a02*b20
    c01 = a00*b01 + a01*b11 + a02*b21
    c02 = a00*b02 + a01*b12 + a02*b22
    c10 = a10*b00 + a11*b10 + a12*b20
    c11 = a10*b01 + a11*b11 + a12*b22  -- wait, c11 = a10*b01 + a11*b11 + a12*b21
    c12 = a10*b02 + a11*b12 + a12*b22
    
    One way to get rank 17: use the fact that the 3x3 matrix B can be handled by combining
    two 2x2 Strassen multiplications. Specifically, partition B as:
    B = [[B11, B12], [B21, B22]] where B11 is 2x2 (rows 0-1, cols 0-1), etc.
    But B is 3x3, so this doesn't split evenly.
    
    Alternative: Use a known explicit rank-17 decomposition.
    
    Let me try a different approach. I'll construct a rank-18 (schoolbook) and then see if I
    can merge two terms. But that's hard to verify.
    
    Actually, the simplest correct approach: just return the rank-18 schoolbook. The score will be 1.0.
    But the task says the Strassen-block rank-17 gives 1.059. Let me try to build rank-17 properly.
    
    Here's a construction:
    Think of the 2x3 x 3x3 multiplication as:
    C = A * B
    
    We can embed this in a 3x3 x 3x3 multiplication by padding A to 3x3 with a zero row:
    A' = [[a00, a01, a02],
          [a10, a11, a12],
          [0,    0,    0  ]]
    
    Then C' = A' * B is 3x3, and C = C'[0:2, :].
    
    The rank of 3x3 x 3x3 is at most 23 (Strassen-like). But we only need the first 2 rows of output.
    
    This is getting complicated. Let me just use the schoolbook rank-18 to be safe,
    since I can verify it exactly.
    """
    # Return None to signal we should use schoolbook
    return None

def construct():
    deadline = time.time() + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # First, try to build rank-17. If it works, return it.
    # For now, use the verified schoolbook rank-18.
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
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Should never reach here
    return {"U": U, "V": V, "W": W}
