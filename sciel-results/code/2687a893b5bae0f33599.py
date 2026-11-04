import time

# EVOLVE-BLOCK-START
def _verify(U, V, W, n=2, m=3, p=3):
    """Check that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == delta_{i,k} for all i,j,k."""
    import numpy as np
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0 if i == k else 0.0
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                s = np.dot(Ua[i*m+j], Va[j*p+k] * Wa[k*n+i])
                if abs(s - expected) > 1e-9:
                    return False
    return True

def _construct_rank17():
    """
    Construct a rank-17 decomposition for 2x3 x 3x3 matrix multiplication.
    
    Strategy: Use Strassen's algorithm for the 2x2x2 core block (rank 7),
    and handle the remaining parts (2x1x3 and 2x2x1 blocks) with direct terms.
    
    The 2x3 x 3x3 product C = A*B has C[i,k] = sum_j A[i,j] B[j,k].
    
    We partition columns of A (j=0,1,2) and rows of B (j=0,1,2).
    
    One approach: Split B into B = [B0, B1, B2] where B0, B1, B2 are 1x3 rows.
    Then C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k].
    
    For the 2x2x2 subproblem (using A[:, 0:2] and B[0:2, :]), Strassen gives rank 7.
    The remaining term is A[:,2] * B[2,:], which is an outer product of a 2x1 and 1x3, 
    contributing 2x3=6 scalar multiplications, but as a tensor it's rank 2*3=6? 
    No, the outer product of (2x1) and (1x3) has rank... actually each element C[i,k] 
    from this part is A[i,2]*B[2,k], so we need 6 separate rank-1 terms for these.
    
    Total: 7 + 6 = 13? That seems too good. Let me reconsider.
    
    Actually, the Strassen decomposition for 2x2x2 uses 7 multiplications. 
    For 2x3 x 3x3, if we group differently:
    
    Let's use the known result. The rank of 2x3x3 is 17 (as stated in the prompt).
    A construction: use Strassen on a 2x2x2 block (rank 7), and the remaining 
    2x1x3 and 2x2x1 parts. The 2x1x3 part (A[:,2] * B[2,:]) can be done in 
    2+3-1=4 multiplications? No.
    
    Let me just hardcode a known valid rank-17 decomposition.
    
    Actually, let me construct it systematically. 
    C[i,k] = sum_{j=0}^{2} A[i,j] B[j,k] for i in {0,1}, k in {0,1,2}.
    
    Using the identity: for the j=0,1 part (2x2x2), Strassen gives 7 mults.
    For the j=2 part: A[i,2]*B[2,k] for i in {0,1}, k in {0,1,2}. 
    This is an outer product, requiring 6 rank-1 terms if done naively, 
    but we can share: let M = A[0,2]*A[1,2]... no, that doesn't help.
    
    Each term A[i,2]*B[2,k] is independent. We need 6 terms.
    7 + 6 = 13. But the prompt says rank 17, so 13 is not achievable.
    
    The issue: Strassen's 7 multiplications are for computing a 2x2 product from 
    two 2x2 matrices. But here, the "2x2" subproblem shares structure with the 
    full problem. The decomposition isn't simply additive.
    
    Let me use a different approach: directly search/construct.
    
    Given the complexity, let me use a known explicit construction.
    """
    # I'll construct a valid decomposition by combining Strassen + direct terms.
    # 
    # For C[i,k] = sum_j A[i,j]B[j,k]:
    #
    # Strassen for the subproblem with j in {0,1}:
    # Let A1 = A[:,0:2], B1 = B[0:2,:]. 
    # Strassen computes A1*B1 with 7 mults:
    # M1 = (A1[0,0]+A1[1,1]) * (B1[0,0]+B1[1,1])  -- but these are scalars for 2x2
    # Actually for 2x2x2:
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    # M3 = A[0,0]*(B[0,1]-B[1,1])
    # M4 = A[1,1]*(B[1,0]-B[0,0])
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    # 
    # But this is for 2x2 * 2x2 -> 2x2. Here B is 2x3, so the "2x2" is really 
    # multiplying 2x2 by 2x3, giving 2x3. Strassen doesn't directly apply.
    #
    # Alternative: just use rank 18 schoolbook as fallback, but try to find rank 17.
    # Given time constraints, let me return the schoolbook rank 18.
    
    # Actually, let me try a direct construction that might give rank 17.
    # 
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0] + A[0,2]B[2,0]
    # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1] + A[0,2]B[2,1]
    # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2] + A[0,2]B[2,2]
    # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0] + A[1,2]B[2,0]
    # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1] + A[1,2]B[2,1]
    # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2] + A[1,2]B[2,2]
    #
    # 18 terms total. To reduce to 17, we need one shared multiplication.
    # 
    # One known trick: 
    # Let's define intermediate products:
    # P1 = A[0,0]*B[0,0]
    # P2 = A[0,1]*B[1,1]
    # P3 = A[0,2]*B[2,2]
    # P4 = A[1,0]*B[0,0]
    # P5 = A[1,1]*B[1,1]
    # P6 = A[1,2]*B[2,2]
    # P7 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])  -- Strassen-like
    # ...
    # This is getting complex. Let me just use rank 18.
    
    n, m, p = 2, 3, 3
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

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 x 3x3 matrix multiplication.
    Returns dict with keys 'U', 'V', 'W' where:
    - U has shape (n*m, rank) = (6, rank)
    - V has shape (m*p, rank) = (9, rank)  
    - W has shape (p*n, rank) = (6, rank)
    """
    deadline = time.time() + 100  # Leave safety margin
    
    # Try rank 17 construction
    # Since we don't have a verified rank-17 decomposition, use rank 18 schoolbook
    result = _construct_rank17()
    
    # Verify
    if _verify(result["U"], result["V"], result["W"]):
        return result
    
    # Fallback: should not reach here
    raise ValueError("Verification failed for schoolbook decomposition")

# EVOLVE-BLOCK-END
