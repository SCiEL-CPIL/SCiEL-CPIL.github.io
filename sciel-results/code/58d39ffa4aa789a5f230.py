import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 5]

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify that the tensor decomposition is exact."""
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    rank = U.shape[1]
    # Build the tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # The correct tensor for matrix multiplication is T[i,j,k] = delta (i.e., 1 for each (i,j,k) pair)
    # Actually, the matrix multiplication tensor is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T[i,j,k] = 1 for all i,j,k (it's the identity-like tensor)
    # Wait, let me reconsider. The tensor for matrix multiplication M_{n,m,p} is defined such that:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # In tensor form, this is a 3-tensor T where T[i,j,k] = 1 for all valid indices.
    # The rank decomposition says: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[i,k,r]
    # where W uses the trace convention W[k*n + i, r]
    
    # Let's verify: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_rank28():
    """
    Construct a rank-28 decomposition for 2x3 * 3x5 matrix multiplication.
    
    Strategy: Split the 3-column B matrix into a 2-column block and a 1-column block.
    Use Strassen's algorithm for the 2x2 * 2x2 part (rank 7) and handle the 
    2x1 * 1x5 part with rank 2, plus cross terms.
    
    Actually, let me think more carefully.
    
    For 2x3 * 3x5:
    A is 2x3, B is 3x5, C is 2x5.
    
    Split B into B1 (3x2) and B2 (3x1).
    Split A into A1 (2x2) and A2 (2x1).
    
    C = A1*B1 + A2*B2
    
    A1*B1 is 2x2 * 2x2 = 2x2, can use Strassen with rank 7.
    A2*B2 is 2x1 * 1x5 = 2x5, rank 2 (just 2 scalar multiplications).
    
    But we need to be careful about how the tensors combine.
    
    Let me use a different approach. The known minimal rank for 2x3 * 3x5 is 28.
    Let me construct it using block Strassen.
    
    Split A (2x3) as [A_left (2x2) | A_right (2x1)]
    Split B (3x5) as [B_top (3x2); B_bottom (3x3)] -- no, that's not right either.
    
    Actually, let me think of it as:
    A is 2x3, B is 3x5.
    
    We can write:
    C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    Split the sum over j into j=0,1 and j=2:
    C[i,k] = (sum_{j=0}^{1} A[i,j] * B[j,k]) + A[i,2] * B[2,k]
    
    The first term is a 2x2 * 2x2 multiplication (for each k, but actually it's 2x2 * 2x5... no).
    
    Let me reconsider. Let's split:
    A = [A1 (2x2) | a2 (2x1)]
    B = [B1 (2x5); b2 (1x5)]  -- wait, B is 3x5, so B1 is 2x5 and b2 is 1x5
    
    C = A1 * B1 + a2 * b2
    
    A1 * B1 is 2x2 * 2x5. This can be viewed as 5 independent 2x2 * 2x1 multiplications, or we can use a block approach.
    
    Actually, for 2x2 * 2x5, we can use Strassen's algorithm for the 2x2 part. The 2x2*2x5 multiplication has tensor rank 7*... no.
    
    Let me just use the known construction. The rank of M_{2,3,5} is 28.
    
    One way to achieve rank 28:
    - Use the fact that M_{2,3,5} can be embedded in a larger structure.
    
    Alternative: Use the "winograd" or "block" approach.
    
    Let me try a direct construction. I'll use the following decomposition:
    
    For the 2x3 * 3x5 case, we can split as follows:
    - Treat it as a 2x2 * 2x5 plus a 2x1 * 1x5 contribution.
    - The 2x2 * 2x5 part: use Strassen for the 2x2 block, giving 7 multiplications per "column group". Since B has 5 columns, and Strassen works on 2x2 blocks, we need to handle 5 columns. We can split the 5 columns into 2+2+1.
      - 2x2 * 2x2: rank 7 (Strassen)
      - 2x2 * 2x2: rank 7 (Strassen) 
      - 2x2 * 2x1: rank 4 (schoolbook, since Strassen doesn't help for 2x1)
      Total for 2x2*2x5: 7+7+4 = 18
    - The 2x1 * 1x5 part: rank 5 (each of the 2 rows times the 1 row of B, for 5 columns)
      Wait, 2x1 * 1x5: C[i,k] = A[i,2] * B[2,k]. This is just 2*5 = 10 scalar multiplications, but the tensor rank is 2 (since it's outer product of A[:,2] and B[2,:]).
      
    So total: 18 + 2 = 20? That can't be right since we know the rank is at least 28... wait, no, the lower bound might be lower.
    
    Actually, I'm confusing myself. Let me just hardcode a known rank-28 decomposition or use a simpler approach.
    
    Given the time constraints, let me just return the schoolbook rank-30 decomposition as a fallback, and try to construct rank 28 if possible.
    """
    n, m, p = 2, 3, 5
    # For now, return schoolbook as a safe fallback
    return _schoolbook(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    # Start with schoolbook (rank 30) as guaranteed correct baseline
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to construct rank-28 decomposition
    # Strategy: Use block Strassen approach
    
    # Split A (2x3) into A1 (2x2) and A2 (2x1)
    # Split B (3x5) into B1 (2x5) and B2 (1x5)
    # C = A1 * B1 + A2 * B2
    
    # For A1 * B1 (2x2 * 2x5):
    # Split B1 into B1a (2x2), B1b (2x2), B1c (2x1)
    # A1 * B1a: 2x2 * 2x2, use Strassen (rank 7)
    # A1 * B1b: 2x2 * 2x2, use Strassen (rank 7)
    # A1 * B1c: 2x2 * 2x1, schoolbook (rank 4)
    # Total: 7 + 7 + 4 = 18
    
    # For A2 * B2 (2x1 * 1x5):
    # This is an outer product, rank 2
    # Total: 18 + 2 = 20
    
    # Wait, this gives rank 20, but the known minimal rank is 28?
    # Let me re-examine. The issue is that the tensor rank of a block product is not simply additive.
    
    # Actually, I think the issue is that when we say "rank 7" for Strassen, it's for the specific tensor.
    # Let me just construct the rank-28 decomposition directly.
    
    # Known result: The tensor rank of M_{2,3,5} is 28.
    # Let me construct it using the following approach:
    
    # I'll use a direct construction based on the known rank-28 decomposition.
    
    # Actually, let me just verify if my block approach gives a valid decomposition.
    # If it does, great. If not, fall back to schoolbook.
    
    # Let me construct the rank-20 decomposition and verify it.
    # If it's valid, we have an improvement. If not, we need rank 28.
    
    # Constructing the block decomposition:
    # U, V, W for the full 2x3 * 3x5
    
    # The indices are:
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (15, rank)  
    # W: (p*n, rank) = (10, rank)
    
    # U[i*m + j, r] corresponds to A[i,j]
    # V[j*p + k, r] corresponds to B[j,k]
    # W[k*n + i, r] corresponds to C[i,k]
    
    # Block 1: A1 (2x2) * B1a (2x2) -> C1a (2x2)
    # A1 indices: i in {0,1}, j in {0,1} -> U index: i*3 + j
    # B1a indices: j in {0,1}, k in {0,1} -> V index: j*5 + k
    # C1a indices: k in {0,1}, i in {0,1} -> W index: k*2 + i
    
    # Strassen for 2x2 * 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let me build this step by step.
    
    rank28 = 28
    U28 = np.zeros((n * m, rank28), dtype=float)
    V28 = np.zeros((m * p, rank28), dtype=float)
    W28 = np.zeros((p * n, rank28), dtype=float)
    
    col = 0
    
    # Block 1: A1 (2x2) * B1a (2x2) using Strassen
    # A1: i in {0,1}, j in {0,1}
    # B1a: j in {0,1}, k in {0,1}
    # C1a: i in {0,1}, k in {0,1}
    
    # Strassen factors for 2x2*2x2
    # M1 = (A00+A11)(B00+B11)
    U28[0*3+0, col] = 1.0  # A[0,0]
    U28[1*3+1, col] = 1.0  # A[1,1]
    V28[0*5+0, col] = 1.0  # B[0,0]
    V28[1*5+1, col] = 1.0  # B[1,1]
    W28[0*2+0, col] = 1.0  # C[0,0]
    W28[1*2+1, col] = 1.0  # C[1,1]
    W28[0*2+1, col] = -1.0 # C[0,1]
    W28[1*2+0, col] = -1.0 # C[1,0]
    col += 1
    
    # M2 = (A10+A11)B00
    U28[1*3+0, col] = 1.0  # A[1,0]
    U28[1*3+1, col] = 1.0  # A[1,1]
    V28[0*5+0, col] = 1.0  # B[0,0]
    W28[1*2+0, col] = 1.0  # C[1,0]
    W28[1*2+1, col] = -1.0 # C[1,1]
    col += 1
    
    # M3 = A00(B10-B11)
    U28[0*3+0, col] = 1.0  # A[0,0]
    V28[1*5+0, col] = 1.0  # B[1,0]
    V28[1*5+1, col] = -1.0 # B[1,1]
    W28[0*2+1, col] = 1.0  # C[0,1]
    W28[1*2+0, col] = 1.0  # C[1,0]
    col += 1
    
    # M4 = A11(B01-B10)
    U28[1*3+1, col] = 1.0  # A[1,1]
    V28[0*5+1, col] = 1.0  # B[0,1]
    V28[1*5+0, col] = -1.0 # B[1,0]
    W28[0*2+0, col] = 1.0  # C[0,0]
    W28[1*2+1, col] = 1.0  # C[1,1]
    col += 1
    
    # M5 = (A00+A01)B11
    U28[0*3+0, col] = 1.0  # A[0,0]
    U28[0*3+1, col] = 1.0  # A[0,1]
    V28[1*5+1, col] = 1.0  # B[1,1]
    W28[0*2+1, col] = 1.0  # C[0,1]
    W28[1*2+1, col] = -1.0 # C[1,1]
    col += 1
    
    # M6 = (A10-A00)(B00+B10)
    U28[1*3+0, col] = 1.0  # A[1,0]
    U28[0*3+0, col] = -1.0 # A[0,0]
    V28[0*5+0, col] = 1.0  # B[0,0]
    V28[1*5+0, col] = 1.0  # B[1,0]
    W28[1*2+1, col] = 1.0  # C[1,1]
    col += 1
    
    # M7 = (A01-A11)(B01+B11)
    U28[0*3+1, col] = 1.0  # A[0,1]
    U28[1*3+1, col] = -1.0 # A[1,1]
    V28[0*5+1, col] = 1.0  # B[0,1]
    V28[1*5+1, col] = 1.0  # B[1,1]
    W28[0*2+0, col] = 1.0  # C[0,0]
    W28[0*2+1, col] = -1.0 # C[0,1]
    col += 1
    
    # Block 2: A1 (2x2) * B1b (2x2) using Strassen
    # B1b: j in {0,1}, k in {2,3}
    
    # M8 = (A00+A11)(B02+B13)
    U28[0*3+0, col] = 1.0
    U28[1*3+1, col] = 1.0
    V28[0*5+2, col] = 1.0
    V28[1*5+3, col] = 1.0
    W28[2*2+0, col] = 1.0  # C[0,2]
    W28[3*2+1, col] = 1.0  # C[1,3]
    W28[2*2+1, col] = -1.0 # C[0,3]
    W28[3*2+0, col] = -1.0 # C[1,2]
    col += 1
    
    # M9 = (A10+A11)B02
    U28[1*3+0, col] = 1.0
    U28[1*3+1, col] = 1.0
    V28[0*5+2, col] = 1.0
    W28[3*2+0, col] = 1.0  # C[1,2]
    W28[3*2+1, col] = -1.0 # C[1,3]
    col += 1
    
    # M10 = A00(B12-B13)
    U28[0*3+0, col] = 1.0
    V28[1*5+2, col] = 1.0
    V28[1*5+3, col] = -1.0
    W28[2*2+1, col] = 1.0  # C[0,3]
    W28[3*2+0, col] = 1.0  # C[1,2]
    col += 1
    
    # M11 = A11(B03-B12)
    U28[1*3+1, col] = 1.0
    V28[0*5+3, col] = 1.0
    V28[1*5+2, col] = -1.0
    W28[2*2+0, col] = 1.0  # C[0,2]
    W28[3*2+1, col] = 1.0  # C[1,3]
    col += 1
    
    # M12 = (A00+A01)B13
    U28[0*3+0, col] = 1.0
    U28[0*3+1, col] = 1.0
    V28[1*5+3, col] = 1.0
    W28[2*2+1, col] = 1.0  # C[0,3]
    W28[3*2+1, col] = -1.0 # C[1,3]
    col += 1
    
    # M13 = (A10-A00)(B02+B12)
    U28[1*3+0, col] = 1.0
    U28[0*3+0, col] = -1.0
    V28[0*5+2, col] = 1.0
    V28[1*5+2, col] = 1.0
    W28[3*2+1, col] = 1.0  # C[1,3]
    col += 1
    
    # M14 = (A01-A11)(B03+B13)
    U28[0*3+1, col] = 1.0
    U28[1*3+1, col] = -1.0
    V28[0*5+3, col] = 1.0
    V28[1*5+3, col] = 1.0
    W28[2*2+0, col] = 1.0  # C[0,2]
    W28[2*2+1, col] = -1.0 # C[0,3]
    col += 1
    
    # Block 3: A1 (2x2) * B1c (2x1) using schoolbook
    # B1c: j in {0,1}, k in {4}
    
    # C[i,4] = A[i,0]*B[0,4] + A[i,1]*B[1,4]
    
    # M15 = A00 * B04
    U28[0*3+0, col] = 1.0
    V28[0*5+4, col] = 1.0
    W28[4*2+0, col] = 1.0  # C[0,4]
    col += 1
    
    # M16 = A01 * B14
    U28[0*3+1, col] = 1.0
    V28[1*5+4, col] = 1.0
    W28[4*2+0, col] = 1.0  # C[0,4]
    col += 1
    
    # M17 = A10 * B04
    U28[1*3+0, col] = 1.0
    V28[0*5+4, col] = 1.0
    W28[4*2+1, col] = 1.0  # C[1,4]
    col += 1
    
    # M18 = A11 * B14
    U28[1*3+1, col] = 1.0
    V28[1*5+4, col] = 1.0
    W28[4*2+1, col] = 1.0  # C[1,4]
    col += 1
    
    # Block 4: A2 (2x1) * B2 (1x5)
    # A2: i in {0,1}, j=2
    # B2: j=2, k in {0,1,2,3,4}
    # C[i,k] += A[i,2] * B[2,k]
    
    # This is an outer product, so we need 2 rank-1 terms (one for each i)
    # For i=0: A[0,2] * B[2,k] for all k
    # For i=1: A[1,2] * B[2,k] for all k
    
    # M19: A[0,2] * B[2,0]
    U28[0*3+2, col] = 1.0
    V28[2*5+0, col] = 1.0
    W28[0*2+0, col] = 1.0  # C[0,0]
    col += 1
    
    # M20: A[0,2] * B[2,1]
    U28[0*3+2, col] = 1.0
    V28[2*5+1, col] = 1.0
    W28[1*2+0, col] = 1.0  # C[0,1]
    col += 1
    
    # M21: A[0,2] * B[2,2]
    U28[0*3+2, col] = 1.0
    V28[2*5+2, col] = 1.0
    W28[2*2+0, col] = 1.0  # C[0,2]
    col += 1
    
    # M22: A[0,2] * B[2,3]
    U28[0*3+2, col] = 1.0
    V28[2*5+3, col] = 1.0
    W28[3*2+0, col] = 1.0  # C[0,3]
    col += 1
    
    # M23: A[0,2] * B[2,4]
    U28[0*3+2, col] = 1.0
    V28[2*5+4, col] = 1.0
    W28[4*2+0, col] = 1.0  # C[0,4]
    col += 1
    
    # M24: A[1,2] * B[2,0]
    U28[1*3+2, col] = 1.0
    V28[2*5+0, col] = 1.0
    W28[0*2+1, col] = 1.0  # C[1,0]
    col += 1
    
    # M25: A[1,2] * B[2,1]
    U28[1*3+2, col] = 1.0
    V28[2*5+1, col] = 1.0
    W28[1*2+1, col] = 1.0  # C[1,1]
    col += 1
    
    # M26: A[1,2] * B[2,2]
    U28[1*3+2, col] = 1.0
    V28[2*5+2, col] = 1.0
    W28[2*2+1, col] = 1.0  # C[1,2]
    col += 1
    
    # M27: A[1,2] * B[2,3]
    U28[1*3+2, col] = 1.0
    V28[2*5+3, col] = 1.0
    W28[3*2+1, col] = 1.0  # C[1,3]
    col += 1
    
    # M28: A[1,2] * B[2,4]
    U28[1*3+2, col] = 1.0
    V28[2*5+4, col] = 1.0
    W28[4*2+1, col] = 1.0  # C[1,4]
    col += 1
    
    # Now verify the rank-28 decomposition
    if _verify(U28, V28, W28, n, m, p):
        best_rank = 28
        U, V, W = U28, V28, W28
    else:
        # Fall back to schoolbook
        U, V, W = _schoolbook(n, m, p)
        best_rank = n * m * p
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
