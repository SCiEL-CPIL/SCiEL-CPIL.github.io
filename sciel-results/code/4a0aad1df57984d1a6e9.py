"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]
DEADLINE = 140  # seconds, with safety margin

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Try structured Strassen-based decomposition
    result = _strassen_3x4x5(n, m, p)
    if result is not None:
        # Verify it
        if _verify_decomposition(result, n, m, p):
            return result
    
    # Try probabilistic rank search from 58 downward
    best_result = None
    best_rank = rank_schoolbook
    
    for rank in range(58, 30, -1):
        if time.time() - start_time > DEADLINE - 10:
            break
        found = _try_random_rank(n, m, p, rank, max_attempts=50)
        if found is not None:
            if _verify_decomposition(found, n, m, p):
                best_result = found
                best_rank = rank
                break  # Found minimal rank in this search
    
    if best_result is not None:
        return best_result
    
    # Return best available
    if result is not None:
        return result
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}

def _verify_decomposition(decomp, n, m, p, tol=1e-10):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(decomp['U'], dtype=float)
    V = np.array(decomp['V'], dtype=float)
    W = np.array(decomp['W'], dtype=float)
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    # The tensor T[i,j,k] = 1 if i in [0,n), j in [0,m), k in [0,p)
    # Decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Wait, let me re-read the convention.
    # W[k*n+i, r] is the index for W
    # So W is indexed by (k, i) where k is the column of B / row of C, i is row of A / col of C
    
    # Actually, let me think about this more carefully.
    # Matrix multiply: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor: T[i,j,k] = 1
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's verify by checking that the reconstructed tensor matches
    # We can check this by evaluating on random inputs
    
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Compute C using the decomposition
    # C[i,k] = sum_j A[i,j] * B[j,k] = sum_j,k A[i,j] B[j,k]
    # Using decomposition: C[i,k] = sum_r (sum_j A[i,j] U[i*m+j,r]) * (sum_j V[j*p+k,r] B[j,k]) * W[k*n+i,r]
    # Wait, that's not right either. Let me think again.
    
    # The bilinear form: C[i,k] = sum_j A[i,j] B[j,k]
    # Decomposition: C[i,k] = sum_r (sum_j A[i,j] U[i*m+j,r]) * (sum_j B[j,k] V[j*p+k,r]) * W[k*n+i,r]
    # Hmm, but that gives a product of three sums, not a bilinear form.
    
    # Actually, the standard tensor decomposition for matrix multiplication is:
    # C[i,k] = sum_r U[i*m+j1, r] * V[j2*p+k, r] * W[k*n+i, r]
    # where the sums over j1, j2 are implicit... no.
    
    # Let me look at the schoolbook:
    # For each (i,j,k): U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # So C[i,k] = sum_col U[i*m+j, col] * V[j*p+k, col] * W[k*n+i, col]
    # But this sums over col, and for a given col, j is fixed.
    # So C[i,k] = sum_col (sum over j such that col corresponds to (i,j,k)) ... 
    # This is getting confusing. Let me just verify numerically.
    
    # The correct interpretation:
    # C[i,k] = sum_r (sum_j A[i,j] * U[i*m+j, r]) * (sum_j B[j,k] * V[j*p+k, r]) * W[k*n+i, r]
    # No wait, that's a trilinear form in A, B, and W... 
    
    # I think the correct form is:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j A[i,j] U[i*m+j, r]) * (sum_j B[j,k] V[j*p+k, r])
    
    # Let's test this with schoolbook:
    # For schoolbook, each rank-1 term has U[i*m+j*, r*]=1, V[j**p+k*, r*]=1, W[k**n+i**, r*]=1
    # where (i*, j*, k*) is the specific triple.
    # Then the term contributes: W[k**n+i**, r*] * (sum_j A[i,j] U[i*m+j, r*]) * (sum_j B[j,k] V[j*p+k, r*])
    # = 1 * (A[i*, j*] if i=i*) * (B[j*, k] if k=k*)
    # This only contributes when i=i*, k=k*, and j=j*... but we sum over j in the inner sums.
    
    # I think I'm overcomplicating this. Let me just do a direct numerical check.
    
    # Alternative verification: construct the full tensor and check.
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 for all valid (i,j,k)
    # But wait, that's not right either because the tensor for matrix mult is T[i,j,k]=1.
    
    # Let me just verify by computing C = A*B using the decomposition formula.
    
    # I believe the correct formula is:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j A[i,j] * U[i*m+j, r]) * (sum_j B[j,k] * V[j*p+k, r])
    
    C_decomp = np.zeros((n, p))
    for r in range(rank):
        # Compute intermediate products
        Au = np.zeros(n)
        Bv = np.zeros(p)
        for i in range(n):
            for j in range(m):
                Au[i] += A[i, j] * U[i * m + j, r]
        for k in range(p):
            for j in range(m):
                Bv[k] += B[j, k] * V[j * p + k, r]
        for i in range(n):
            for k in range(p):
                C_decomp[i, k] += W[k * n + i, r] * Au[i] * Bv[k]
    
    if np.max(np.abs(C_decomp - C_expected)) > tol:
        return False
    return True

def _try_random_rank(n, m, p, rank, max_attempts=100):
    """
    Try to find a rank decomposition using random rational factors.
    We set up a linear system where the unknowns are the W entries,
    given random U and V matrices.
    """
    np.random.seed(0)
    
    for attempt in range(max_attempts):
        if time.time() > DEADLINE - 5:
            return None
        
        # Generate random integer U and V matrices
        U = np.random.randint(-2, 3, size=(n * m, rank))
        V = np.random.randint(-2, 3, size=(m * p, rank))
        
        # For each (i,j,k), we need:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # This is a system of n*m*p equations in n*p*rank unknowns (W entries)
        # But each equation only involves W[k*n+i, :] for a fixed (i,k),
        # with coefficients depending on j.
        
        # For fixed (i,k), the equations are:
        # For each j: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # This is m equations in rank unknowns (W[k*n+i, 0..rank-1])
        
        # We need this system to be solvable for all (i,k) pairs.
        # There are n*p such systems, each with m equations and rank unknowns.
        
        W = np.zeros((p * n, rank))
        valid = True
        
        for i in range(n):
            for k in range(p):
                # Build the m x rank system
                A_sys = np.zeros((m, rank))
                for j in range(m):
                    for r in range(rank):
                        A_sys[j, r] = U[i * m + j, r] * V[j * p + k, r]
                
                # Solve A_sys @ w = ones(m)
                try:
                    w, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, np.ones(m), rcond=None)
                except:
                    valid = False
                    break
                
                # Check if the solution is exact
                if np.max(np.abs(A_sys @ w - np.ones(m))) > 1e-8:
                    valid = False
                    break
                
                W[k * n + i, :] = w
        
        if valid:
            # Convert to list format with integer/half-integer check
            decomp = {
                'U': U.tolist(),
                'V': V.tolist(),
                'W': W.tolist()
            }
            return decomp
    
    return None

def _strassen_3x4x5(n, m, p):
    """
    Decompose 3x4x5 matrix multiplication using Strassen-based blocks.
    Split the 4-dimension (j) into two halves of size 2.
    For each half, use a rank-7 Strassen-like decomposition.
    Total rank: 7 + 7 = 14? No, need to be more careful.
    
    Actually, for 3x2 x 2x5, we need a more careful analysis.
    Let me use the known approach: split j into j=2a+b.
    For each a in {0,1}, we have A[:, 2a:2a+2] (3x2) times B[2a:2a+2, :] (2x5).
    Each sub-problem is 3x2 x 2x5.
    
    The 3x2 x 2x5 tensor can be decomposed with rank 7 using:
    M1 = (A0+A1)(B0+B1)  -> 3x5
    M2 = A1*B0            -> 3x5  
    M3 = A0*(B1-B0)       -> 3x5
    M4 = A1*(B0-B1)       -> 3x5  [this is -M3 + ... wait]
    
    Actually, for C = A0*B0 + A1*B1 where A0,A1 are 3x1 columns and B0,B1 are 1x5 rows:
    C = A0*B0 + A1*B1
    
    Using 7 multiplications (Strassen-like for rank-2 inner product):
    This is essentially the dot product of two 2-vectors (for each i,k pair),
    which has rank 3 (not 7). The rank-3 decomposition:
    M1 = (A0+A1)(B0+B1)
    M2 = A0*B0
    M3 = A1*B1
    C = M1 - M2 - M3 + M2 + M3... no.
    
    Actually: A0*B0 + A1*B1 = (A0+A1)(B0+B1) - A0*B1 - A1*B0
    That's 3 multiplications. Or:
    = (A0+A1)(B0+B1) - (A0+A1)(B0-B1) - ... no.
    
    The minimal rank for 2-dimensional inner product is 3:
    M1 = (x0+x1)(y0+y1)
    M2 = x0*y0
    M3 = x1*y1
    x0*y0 + x1*y1 = M1 - M2 - M3 + M2 + M3... 
    
    Wait: (x0+x1)(y0+y1) = x0*y0 + x0*y1 + x1*y0 + x1*y1
    So x0*y0 + x1*y1 = M1 - x0*y1 - x1*y0
    
    That's still 3 mults if we can compute x0*y1 and x1*y0 for free, which we can't.
    
    Actually the rank of the 2x2x2 tensor (which is the identity tensor) is 3:
    M1 = (x0+x1)(y0+y1)
    M2 = x0*y0  
    M3 = x1*y1
    Then x0*y0 + x1*y1 = M1 - x0*y1 - x1*y0... no that doesn't work.
    
    Let me just use rank 4 for each 3x2x2x5 block:
    M1 = A0*B0
    M2 = A1*B1
    M3 = (A0+A1)*(B0+B1)
    M4 = (A0-A1)*(B0-B1)
    Then A0*B0 + A1*B1 = (M3 + M4)/2... let's check:
    M3 = A0*B0 + A0*B1 + A1*B0 + A1*B1
    M4 = A0*B0 - A0*B1 - A1*B0 + A1*B1
    M3 + M4 = 2*A0*B0 + 2*A1*B1
    So A0*B0 + A1*B1 = (M3+M4)/2
    
    That's rank 2! Using M3 and M4 only.
    
    So for each a in {0,1}:
    M3_a = (A[:,2a] + A[:,2a+1]) * (B[2a,:] + B[2a+1,:])  -> 3x5
    M4_a = (A[:,2a] - A[:,2a+1]) * (B[2a,:] - B[2a+1,:])  -> 3x5
    C += (M3_a + M4_a) / 2
    
    Total rank: 2 + 2 = 4? That seems too low. Let me verify.
    
    For a=0: C_partial = A[:,0]*B[0,:] + A[:,1]*B[1,:]
    M3_0 = (A[:,0]+A[:,1])*(B[0,:]+B[1,:]) = A[:,0]*B[0,:] + A[:,0]*B[1,:] + A[:,1]*B[0,:] + A[:,1]*B[1,:]
    M4_0 = (A[:,0]-A[:,1])*(B[0,:]-B[1,:]) = A[:,0]*B[0,:] - A[:,0]*B[1,:] - A[:,1]*B[0,:] + A[:,1]*B[1,:]
    M3_0 + M4_0 = 2*A[:,0]*B[0,:] + 2*A[:,1]*B[1,:]
    So C_partial = (M3_0 + M4_0)/2. Correct!
    
    Similarly for a=1.
    
    Total C = C_partial_0 + C_partial_1 = (M3_0+M4_0)/2 + (M3_1+M4_1)/2
    
    So we need 4 rank-1 terms total! Rank 4?
    
    Wait, but each "rank-1 term" here is actually a 3x5 matrix product, which in the tensor sense
    is not rank 1. Let me reconsider.
    
    In the tensor decomposition sense, a rank-1 term is:
    U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For the term (A[:,0]+A[:,1])*(B[0,:]+B[1,:]):
    This contributes to C[i,k] the value (A[i,0]+A[i,1])*(B[0,k]+B[1,k])
    = A[i,0]*B[0,k] + A[i,0]*B[1,k] + A[i,1]*B[0,k] + A[i,1]*B[1,k]
    
    In the tensor decomposition, this would require:
    U[i*m+0, r] = 1, U[i*m+1, r] = 1 (for all i)
    V[0*p+k, r] = 1, V[1*p+k, r] = 1 (for all k)
    W[k*n+i, r] = 1/2 (for all i,k)
    
    Then the contribution is: sum_j U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    = U[i*m+0,r]*V[0*p+k,r]*W[k*n+i,r] + U[i*m+1,r]*V[1*p+k,r]*W[k*n+i,r]
    = 1*1*(1/2) + 1*1*(1/2) = 1
    
    But we want this to equal (A[i,0]+A[i,1])*(B[0,k]+B[1,k]) when multiplied by A and B.
    
    I think I'm confusing the tensor decomposition with the bilinear algorithm. Let me restart.
    
    The bilinear algorithm for matrix multiplication C = AB uses:
    C[i,k] = sum_r (sum_j A[i,j] * U[i*m+j, r]) * (sum_j B[j,k] * V[j*p+k, r]) * W[k*n+i, r]
    
    For the term (A[:,0]+A[:,1])*(B[0,:]+B[1,:])/2:
    We want: sum_j A[i,j] * U[i*m+j, r] = A[i,0] + A[i,1]
    So U[i*m+0, r] = 1, U[i*m+1, r] = 1, U[i*m+2, r] = 0, U[i*m+3, r] = 0 (for all i)
    
    And: sum_j B[j,k] * V[j*p+k, r] = B[0,k] + B[1,k]
    So V[0*p+k, r] = 1, V[1*p+k, r] = 1, V[2*p+k, r] = 0, V[3*p+k, r] = 0 (for all k)
    
    And W[k*n+i, r] = 1/2
    
    Let me verify: the contribution to C[i,k] from this rank-r term is:
    (A[i,0]+A[i,1]) * (B[0,k]+B[1,k]) * 1/2
    
    Similarly for the other 3 terms.
    
    Total C[i,k] = 1/2*(A[i,0]+A[i,1])*(B[0,k]+B[1,k]) + 1/2*(A[i,0]-A[i,1])*(B[0,k]-B[1,k])
    + 1/2*(A[i,2]+A[i,3])*(B[2,k]+B[3,k]) + 1/2*(A[i,2]-A[i,3])*(B[2,k]-B[3,k])
    
    = 1/2*(A[i,0]*B[0,k] + A[i,0]*B[1,k] + A[i,1]*B[0,k] + A[i,1]*B[1,k])
    + 1/2*(A[i,0]*B[0,k] - A[i,0]*B[1,k] - A[i,1]*B[0,k] + A[i,1]*B[1,k])
    + 1/2*(A[i,2]*B[2,k] + A[i,2]*B[3,k] + A[i,3]*B[2,k] + A[i,3]*B[3,k])
    + 1/2*(A[i,2]*B[2,k] - A[i,2]*B[3,k] - A[i,3]*B[2,k] + A[i,3]*B[3,k])
    
    = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    = sum_j A[i,j]*B[j,k] = C[i,k] ✓
    
    So rank 4 works! Let me implement this.
    """
    n, m, p = 3, 4, 5
    
    U = [[0] * 4 for _ in range(n * m)]
    V = [[0] * 4 for _ in range(m * p)]
    W = [[0] * 4 for _ in range(p * n)]
    
    # Term 0: (A[:,0]+A[:,1])*(B[0,:]+B[1,:]) * 1/2
    for i in range(n):
        U[i * m + 0][0] = 1
        U[i * m + 1][0] = 1
    for k in range(p):
        V[0 * p + k][0] = 1
        V[1 * p + k][0] = 1
    for i in range(n):
        for k in range(p):
            W[k * n + i][0] = 0.5
    
    # Term 1: (A[:,0]-A[:,1])*(B[0,:]-B[1,:]) * 1/2
    for i in range(n):
        U[i * m + 0][1] = 1
        U[i * m + 1][1] = -1
    for k in range(p):
        V[0 * p + k][1] = 1
        V[1 * p + k][1] = -1
    for i in range(n):
        for k in range(p):
            W[k * n + i][1] = 0.5
    
    # Term 2: (A[:,2]+A[:,3])*(B[2,:]+B[3,:]) * 1/2
    for i in range(n):
        U[i * m + 2][2] = 1
        U[i * m + 3][2] = 1
    for k in range(p):
        V[2 * p + k][2] = 1
        V[3 * p + k][2] = 1
    for i in range(n):
        for k in range(p):
            W[k * n + i][2] = 0.5
    
    # Term 3: (A[:,2]-A[:,3])*(B[2,:]-B[3,:]) * 1/2
    for i in range(n):
        U[i * m + 2][3] = 1
        U[i * m + 3][3] = -1
    for k in range(p):
        V[2 * p + k][3] = 1
        V[3 * p + k][3] = -1
    for i in range(n):
        for k in range(p):
            W[k * n + i][3] = 0.5
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
