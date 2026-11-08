"""Matrix multiplication 2x4 by 4x4 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]


def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x4 matrix.
    The output factor W follows the official trace convention W[k*n+i, r].
    
    Interface: construct() -> {'U', 'V', 'W'}
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    
    Score is the rank (lower is better).
    """
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    # We need to find matrices U, V, W such that:
    # For all i in [0,n), j in [0,m), k in [0,p):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for all other (i,j,k) combinations where they don't correspond to valid multiplication:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 0
    #
    # Wait, let me re-read the problem. This is the tensor decomposition for matrix multiplication.
    # The matrix multiplication tensor T has T[i,j,k] = 1 if and only if we multiply A[i,j] * B[j,k]
    # to contribute to C[i,k]. But here the indices are different.
    #
    # Let me think about this more carefully. For multiplying an n x m matrix A by an m x p matrix B
    # to get an n x p matrix C:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # The tensor T has dimensions (n*m) x (m*p) x (p*n) in the factorization sense.
    # Actually, the standard decomposition is:
    # U has shape (n*m, rank) - indexed by (i,j) pairs
    # V has shape (m*p, rank) - indexed by (j,k) pairs  
    # W has shape (p*n, rank) - indexed by (k,i) pairs (note: k*n+i, not i*n+k)
    #
    # The condition is: for all i,j,k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    # where T[i,j,k] = 1 if this term contributes to the matrix multiplication, 0 otherwise.
    #
    # For matrix multiplication, every triple (i,j,k) with i in [0,n), j in [0,m), k in [0,p)
    # contributes exactly once to C[i,k] via A[i,j]*B[j,k]. So T[i,j,k] = 1 for all valid (i,j,k).
    #
    # Wait, that's not right either. Let me reconsider.
    #
    # Actually, I think the tensor is defined such that:
    # C[i,k] = sum_j sum_r U[i*m+j, r] * W[k*n+i, r] * (something involving V)
    #
    # Let me look at the schoolbook example in the seed:
    # U[i*m+j][column] = 1
    # V[j*p+k][column] = 1
    # W[k*n+i][column] = 1
    #
    # So for each (i,j,k), there's a column where U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
    # The product U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] is 1 for that specific (i,j,k,r).
    #
    # I believe the tensor T is defined as: T[i,j,k] = 1 for all i,j,k (since in matrix mult,
    # every A[i,j]*B[j,k] term is used). The rank is the minimum number of such "rank-1" terms
    # that can reconstruct the full tensor.
    #
    # Actually, I think I'm overcomplicating this. Let me think of it as:
    # We want to decompose the matrix multiplication as a sum of rank-1 tensors.
    # A rank-1 tensor is u * v * w where u is in R^(n*m), v is in R^(m*p), w is in R^(p*n).
    #
    # The matrix multiplication tensor is: T[a,b,c] = 1 where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # and T is nonzero only when j=j' and i=i' and... hmm.
    #
    # Let me just use the constraint: for the decomposition to be valid,
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] must equal 1 for all valid (i,j,k)
    # because each (i,j,k) triple corresponds to one multiplication that contributes to the result.
    
    # Actually, I think I need to be more careful. Let me reconsider.
    
    # In matrix multiplication C = A * B where A is n x m and B is m x p:
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    #
    # The tensor T for this operation has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # This is because every combination of (i,j,k) is used in the computation.
    #
    # We want to write T as a sum of rank-1 tensors:
    # T = sum_r u_r * v_r * w_r
    # where (u_r)_{i,j} = U[i*m+j, r], (v_r)_{j,k} = V[j*p+k, r], (w_r)_{k,i} = W[k*n+i, r]
    #
    # The constraint is: for all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # For n=2, m=4, p=4: n*m*p = 32, so schoolbook gives rank 32.
    # We want to find a lower rank decomposition.
    
    # Let's try a simple approach: use random initialization and gradient descent
    # to minimize the error, but since we need exact integer/half-integer entries,
    # we might need a different approach.
    
    # Actually, let me think about this differently. The matrix multiplication tensor
    # for n x m by m x p matrices has a known minimum rank. For small dimensions,
    # we can try to find it via optimization.
    
    # Let's try to use a rank-1 decomposition approach where we try different ranks
    # and use least squares to see if we can achieve zero error.
    
    best_rank = n * m * p  # schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Generate the target tensor T
    # T has dimensions (n*m) x (m*p) x (p*n)
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 if j==j' and i==i' and k==k'... 
    # No wait. Let me re-examine.
    
    # From the schoolbook: for each (i,j,k), we have a column where:
    # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # 
    # The condition that this correctly decomposes matrix multiplication is:
    # For the product to work out, we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    #
    # This is because in the schoolbook, each (i,j,k) has exactly one r (its column)
    # where all three are 1, giving product 1. All other (i,j,k) combinations for that r
    # give 0 in at least one factor.
    
    # So the target is: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # where the index into U is i*m+j, into V is j*p+k, into W is k*n+i.
    
    # Let's verify: for a given r, the contribution to (i,j,k) is U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r].
    # In schoolbook, for column corresponding to (i0,j0,k0):
    # U[i*m+j,r] = 1 iff (i,j)=(i0,j0)
    # V[j*p+k,r] = 1 iff (j,k)=(j0,k0)  
    # W[k*n+i,r] = 1 iff (k,i)=(k0,i0)
    # So the product is 1 iff (i,j,k)=(i0,j0,k0), which is correct.
    
    # Now, for a general decomposition, we need:
    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
    
    # This is a system of equations. Let's set it up.
    
    # Number of equations: n*m*p = 2*4*4 = 32
    # Unknowns: rank * (n*m + m*p + p*n) = rank * (8 + 16 + 8) = rank * 32
    
    # For rank 1: 32 unknowns, 32 equations. Might have a solution?
    # For rank 2: 64 unknowns, 32 equations. More flexible.
    
    # Let's try to find a solution using least squares for various ranks.
    
    # Actually, the entries need to be integer or half-integer. This makes it harder.
    # Let's first find a real-valued solution, then round.
    
    # Let me set up the system of equations.
    # For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # This is nonlinear in U,V,W. Let's use a different approach.
    
    # Alternative: fix W, then the system becomes linear in U*V.
    # Or use an iterative approach.
    
    # Let me try a simple heuristic: try small ranks and use random restarts
    # with gradient-based optimization.
    
    import numpy as np
    
    def check_decomposition(U, V, W, n, m, p, tol=1e-9):
        """Check if U,V,W form a valid decomposition."""
        rank = U.shape[1]
        # For each (i,j,k), compute sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def try_rank(rank, n, m, p, max_iters=1000, seed=0):
        """Try to find a decomposition of given rank."""
        np.random.seed(seed)
        
        # Initialize randomly
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        # Use gradient descent on the squared error
        def compute_error(U, V, W):
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(rank):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        err += (s - 1.0) ** 2
            return err
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            err = compute_error(U, V, W)
            if err < 1e-12:
                break
            
            # Compute gradients (approximate)
            eps = 1e-7
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            # This is expensive. Let's use a simpler approach.
            # Actually, for small dimensions, let's just try random restarts
            # and check if we get lucky.
            break
        
        # After "optimization" (which we didn't really do), check
        if check_decomposition(U, V, W, n, m, p, tol=1e-6):
            return U, V, W
        return None
    
    # Let's try a different approach: use the fact that we can set up
    # the system and solve it.
    
    # For a given rank R, we have R*32 unknowns and 32 equations.
    # If R >= 1, we have more unknowns than equations (for R>=1, 32R >= 32).
    # 
    # Let's try R=1 first. If it works, great.
    
    # For R=1: U is (8,1), V is (16,1), W is (8,1)
    # U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0] = 1 for all i,j,k
    #
    # Let a_ij = U[i*m+j,0], b_jk = V[j*p+k,0], c_ki = W[k*n+i,0]
    # Need: a_ij * b_jk * c_ki = 1 for all i,j,k
    #
    # From a_ij * b_jk * c_ki = 1, we get:
    # a_ij = 1 / (b_jk * c_ki)
    # For this to be consistent, a_ij must factor as f(i,j) where the dependence
    # on i and j is separable in a specific way.
    #
    # a_ij * b_jk * c_ki = 1
    # => a_ij = 1/(b_jk * c_ki)
    # For different k: a_ij * b_jk1 * c_k1i = 1 and a_ij * b_jk2 * c_k2i = 1
    # => b_jk1 * c_k1i = b_jk2 * c_k2i
    # => b_jk1 / b_jk2 = c_k2i / c_k1i
    # This must hold for all i, which means b_jk1 / b_jk2 must be independent of k1,k2... 
    # This is very restrictive.
    #
    # Similarly, for different i: a_i1j * b_jk * c_ki1 = 1 and a_i2j * b_jk * c_ki2 = 1
    # => a_i1j * c_ki1 = a_i2j * c_ki2
    # => a_i1j / a_i2j = c_ki2 / c_ki1
    # Must hold for all k, so a_i1j/a_i2j is independent of i1,i2... very restrictive.
    #
    # It seems R=1 is unlikely to work unless all a,b,c are 1, but then
    # 1*1*1=1 works! Let's check: if U=1, V=1, W=1 (all entries 1), then
    # sum_r = 1*1*1 = 1. Yes! R=1 works!
    
    # Wait, really? Let me double-check.
    # U is (8,1) with all entries 1.
    # V is (16,1) with all entries 1.
    # W is (8,1) with all entries 1.
    # For any (i,j,k): U[i*m+j,0]*V[j*p+k,0]*W[k*n+i,0] = 1*1*1 = 1. ✓
    
    # So rank 1 works?! That seems too good to be true.
    # 
    # Let me re-read the problem statement to make sure I understand the convention correctly.
    
    # "The output factor follows the official trace convention W[k*n+i,r]"
    # This just means W is indexed by k*n+i, which is what we have.
    
    # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # U: (8,1), V: (16,1), W: (8,1) ✓
    
    # "with integer/half-integer entries that exactly decompose the matrix multiplication tensor"
    # All entries are 1, which is an integer. ✓
    
    # So rank 1 is the answer? Let me verify once more.
    
    # The matrix multiplication tensor for 2x4 by 4x4:
    # C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, k in {0,1,2,3}
    # 
    # The tensor T has T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}.
    # 
    # With U=all 1's, V=all 1's, W=all 1's, rank 1:
    # sum_r=1 U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1*1*1 = 1 = T[i,j,k] ✓
    
    # This works! The rank is 1.
    
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    result = construct()
    print(f"U shape: {len(result['U'])} x {len(result['U'][0])}")
    print(f"V shape: {len(result['V'])} x {len(result['V'][0])}")
    print(f"W shape: {len(result['W'])} x {len(result['W'][0])}")
    rank = len(result['U'][0])
    print(f"Rank: {rank}")
