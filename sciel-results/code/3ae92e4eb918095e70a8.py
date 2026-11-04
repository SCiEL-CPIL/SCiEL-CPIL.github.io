"""Matrix multiplication 2x3x8 rank-minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    """
    Decompose 2x3x8 matrix multiplication into:
      - 2x2x4 block: use known rank-10 decomposition
      - 2x1x4 block: trivial rank-4 decomposition
    Total rank: 10 + 4 = 14
    
    Wait, let me reconsider. The matrix is 2x3 times 3x8 = 2x8.
    
    We can split the 3 columns of the first matrix and 3 rows of the second:
      A (2x3) * B (3x8) = A[:, :2] * B[:2, :] + A[:, 2:] * B[2:, :]
    
    But that's not quite right for tensor decomposition. Let me think differently.
    
    Actually, for a 2x3x8 multiplication, we can view it as:
      Split the middle dimension (3) into 2+1:
      - 2x2x8 subproblem (first 2 columns/rows)
      - 2x1x8 subproblem (last 1 column/row)
    
    For 2x2x8: We can split the 8 into 4+4, giving two 2x2x4 blocks.
      Known rank for 2x2x4 is 10 (from literature).
      So 2x2x8 would have rank 20.
    
    For 2x1x8: This is just scaling, rank 8.
    
    Total: 20 + 8 = 28.
    
    Alternatively, split 8 into 2+2+2+2:
      2x2x2 has rank 7 (Strassen), so 4 * 7 = 28.
      Same result.
    
    Let me try a different split. What if we split the 3 into 1+1+1?
      Three 2x1x8 blocks, each rank 8, total 24.
    
    Or split the 8 into smaller pieces more cleverly.
    
    Actually, the best known approach for rectangular matrices often uses
    the fact that rank(M_{n,m,p}) can be less than n*m*p.
    
    For 2x3x8, let me try:
      - Split 3 = 2 + 1
      - 2x2x8: split 8 = 4 + 4, each 2x2x4 has rank 10, so 20
      - 2x1x8: rank 8
      Total: 28
    
    But wait, can we do better by not splitting independently?
    
    Let me look at this more carefully. The tensor for 2x3x8 multiplication
    is the sum over j=0..2 of the 2x1x8 tensors.
    
    Each 2x1x8 tensor has rank 8 (it's just outer products).
    So the trivial rank is 3*8 = 24.
    
    Can we do better than 24? The Strassen-like improvements come from
    combining terms. For 2x2x2, Strassen gives 7 < 8.
    
    For 2x3x8, if we group as:
      (2x2x8) + (2x1x8)
    
    For 2x2x8, we can use: split 8 into four 2x2x2 blocks, each rank 7, total 28.
    Or: use the 2x2x4 rank-10 decomposition twice, total 20.
    
    So 2x2x8 has rank at most 20.
    2x1x8 has rank 8.
    Total: 28.
    
    But 28 > 24, so the trivial 24 is better!
    
    Let me reconsider. The trivial decomposition of 2x3x8 has rank 24.
    Can we beat 24?
    
    For the 2x2x2 part, Strassen saves 1 (7 vs 8). If we have four 2x2x2 blocks
    in the 2x2x8 part, we save 4, giving 32-4 = 28 for 2x2x8.
    Plus 8 for 2x1x8, total 36. Worse than 24.
    
    So the trivial 24 seems better. Let me verify: is there a known rank < 24 for 2x3x8?
    
    Actually, I should just implement the trivial rank-24 decomposition and see if
    we can improve.
    """
    start_time = time.time()
    deadline = start_time + 140  # 140 second deadline
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Start with schoolbook rank n*m*p = 48
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
    
    # Try to improve using Strassen on 2x2x2 subblocks
    # Split the 3 dimension: j=0,1 form a 2x2x8, j=2 is 2x1x8
    # For 2x2x8, split p=8 into four 2x2x2 blocks
    
    # Let's build a rank-24 decomposition by using Strassen on 2x2x2 blocks
    # where possible.
    
    # Actually, let me just try to find a better decomposition using
    # random search or structured approaches.
    
    # For now, return the schoolbook result if we're running out of time
    if time.time() > deadline - 10:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try a structured decomposition:
    # 2x3x8 = sum over j of 2x1x8
    # Each 2x1x8 is rank 8
    # Total trivial: 24
    
    # Let's try to combine the 2x2x8 part using Strassen-like methods
    # 2x2x8: split into 4 blocks of 2x2x2
    # Each 2x2x2 can be done in rank 7 (Strassen)
    # So 2x2x8 in rank 28, plus 2x1x8 in rank 8, total 36. Worse.
    
    # Alternative: use the known rank-10 for 2x2x4
    # 2x2x8 = two 2x2x4, rank 20, plus 2x1x8 rank 8, total 28. Still worse.
    
    # So the best simple approach is rank 24.
    
    # Let me implement rank 24:
    rank24 = 24
    U24 = np.zeros((n * m, rank24), dtype=float)
    V24 = np.zeros((m * p, rank24), dtype=float)
    W24 = np.zeros((p * n, rank24), dtype=float)
    
    col = 0
    for j in range(m):
        for k in range(p):
            for i in range(n):
                U24[i * m + j, col] = 1.0
                V24[j * p + k, col] = 1.0
                W24[k * n + i, col] = 1.0
                col += 1
    
    # Verify correctness
    def verify(U_mat, V_mat, W_mat, n, m, p):
        """Verify that the decomposition correctly computes matrix multiplication."""
        rank = U_mat.shape[1]
        # For random A (n x m) and B (m x p), check that
        # sum_c U[:,c] * V[:,c] * W[:,c] gives the correct product
        # In tensor form: T_{i,j,k} = sum_c U_{i*m+j,c} * V_{j*p+k,c} * W_{k*n+i,c}
        # should equal delta (which is 1 for all i,j,k in the multiplication tensor)
        # Actually, the multiplication tensor has T_{i,j,k} = 1 for all valid indices
        # when we're computing (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
        
        # Let's verify with specific values
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        AB = A @ B
        
        # Compute using the decomposition
        result = np.zeros((n, p))
        for c in range(rank):
            # U[:, c] is a vector of length n*m
            # V[:, c] is a vector of length m*p
            # W[:, c] is a vector of length p*n
            
            Uc = U_mat[:, c].reshape(n, m)
            Vc = V_mat[:, c].reshape(m, p)
            Wc = W_mat[:, c].reshape(p, n)
            
            # The contribution of term c is:
            # sum over j of (A_{i,j} * Uc_{i,j}) * (B_{j,k} * Vc_{j,k}) * ... 
            # Actually, the tensor decomposition means:
            # (A*B)_{i,k} = sum_c (sum_j A_{i,j} * U_{i*m+j,c}) * (sum_j B_{j,k} * V_{j*p+k,c}) * W_{k*n+i,c}
            # Wait, that's not right either.
            
            # The correct interpretation: the multiplication tensor T has entries
            # T[i,j,k] = 1, and we need T = sum_c U_c \otimes V_c \otimes W_c
            # where U_c is a vector in R^{n*m}, V_c in R^{m*p}, W_c in R^{p*n}
            
            # So T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            
            pass
        
        # Simpler verification: check that the tensor matches
        # T[i,j,k] should be 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c]
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True
    
    if not verify(U24, V24, W24, n, m, p):
        # Fall back to schoolbook
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U24.tolist(), "V": V24.tolist(), "W": W24.tolist()}
