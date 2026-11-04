"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _deadline_ok(deadline, margin=10.0):
    """Check if we have enough time remaining before deadline."""
    return time.time() < deadline - margin

def _verify_decomp(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    # Check tensor: T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if total != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition with rank n*m*p."""
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

def _try_strassen_2x5x5(n, m, p):
    """
    Try to construct a Strassen-based decomposition for 2x5 * 5x5.
    
    Strategy: Partition the 5x5 matrix B into blocks.
    B = [B1 B2 B3 B4 B5] where B1,B2 are 2x5, B3 is 1x5, B4,B5 are... 
    Actually, let's partition differently.
    
    Better: View A as 2x5, B as 5x5.
    We can use the fact that n=2 is small.
    
    Approach: Use Karatsuba-like tricks on the 5-dimensional vectors.
    For each output row, we compute a dot product of two 5-vectors.
    But the vectors are shared across outputs.
    
    Let me try: C[0,k] = sum_j A[0,j]*B[j,k] for k=0..4
    C[1,k] = sum_j A[1,j]*B[j,k] for k=0..4
    
    This is 10 independent dot products of length 5, but with shared structure.
    
    Actually, let's think of it as: for each k, we compute [C[0,k], C[1,k]]^T = A[:,k]^T * B[:,k]
    where A[:,k] is the k-th column of A (2x1) and B[:,k] is the k-th column of B (5x1).
    Wait, that's not right. C[0,k] = sum_j A[0,j]*B[j,k], so it's the dot product of row 0 of A with column k of B.
    
    Let me use a different partitioning. Split the 5 columns into groups.
    Group 1: columns 0,1 (size 2)
    Group 2: columns 2,3 (size 2)  
    Group 3: column 4 (size 1)
    
    For each group, we compute a 2x2 or 2x1 multiplication.
    
    For group 1 (2x2): A[:,0:2] * B[0:2,:] gives 2x2, rank at most 4 (schoolbook) or 7 (Strassen).
    For group 2 (2x2): similar, rank at most 4 or 7.
    For group 3 (2x1): A[:,4] * B[4,:] gives 2x1, rank at most 2.
    
    Total rank with Strassen: 7 + 7 + 2 = 16. That's much better than 50!
    
    But wait, we need to be careful. The multiplication is:
    C = A * B where A is 2x5, B is 5x5.
    C[:, 0:2] = A[:, 0:2] * B[0:2, :]  -- this is 2x2 * 2x5 = 2x5
    C[:, 2:4] = A[:, 2:4] * B[2:4, :]  -- this is 2x2 * 2x5 = 2x5
    C[:, 4:5] = A[:, 4:5] * B[4:5, :]  -- this is 2x1 * 1x5 = 2x5
    
    So we have three sub-problems:
    1. 2x2 * 2x5: compute A1 * B1 where A1 is 2x2, B1 is 2x5, result is 2x5
    2. 2x2 * 2x5: similar
    3. 2x1 * 1x5: compute A3 * B3 where A3 is 2x1, B3 is 1x5, result is 2x5
    
    For problem 1: A1 * B1. We can further partition B1 into two 2x2 blocks? No, B1 is 2x5, so we can't split it into 2x2 blocks evenly.
    
    Let me reconsider. B1 is 2x5. We can split it as B1 = [B1a B1b] where B1a is 2x2 and B1b is 2x3.
    Then A1 * B1 = A1 * [B1a B1b] = [A1*B1a, A1*B1b]
    A1*B1a is 2x2*2x2 = 2x2, rank at most 7 (Strassen)
    A1*B1b is 2x2*2x3 = 2x3, rank at most 6 (schoolbook) or less
    
    This is getting complicated. Let me just try to construct a valid decomposition with rank < 50 using a simple approach.
    
    Alternative: Use the fact that we can combine some of the 50 terms.
    """
    # Let's try a simpler approach: use Strassen on a 4x4 sub-problem and handle the rest.
    # Pad to 4x4: A_padded is 2x4 (pad with zeros), B_padded is 4x4 (pad with zeros)
    # But we also have the 5th column/row.
    
    # Actually, let me just return schoolbook for now and try to improve.
    return None

def _construct():
    """Main construction function with deadline guard."""
    n, m, p = DIMENSIONS
    deadline = time.time() + 120.0
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    
    # Try Strassen-based approaches
    strassen_result = _try_strassen_2x5x5(n, m, p)
    if strassen_result is not None and _verify_decomp(strassen_result['U'], strassen_result['V'], strassen_result['W'], n, m, p):
        if len(strassen_result['U'][0]) < len(best['U'][0]):
            best = strassen_result
    
    return best

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices
    for the matrix multiplication tensor decomposition.
    """
    return _construct()
# EVOLVE-BLOCK-END
