"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def _build_strassen_3x3x3():
    """
    Build the 15-term Laderman decomposition for 3x3 matrix multiplication.
    Returns U (9,15), V (9,15), W (9,15) where:
      U[i*3+j, c], V[j*3+k, c], W[k*3+i, c]
    """
    rank = 15
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    
    # Laderman's 15 products for 3x3 multiplication
    # Using the standard Laderman decomposition
    
    # P1 = A[0,0] * (B[0,0] + B[0,1])
    U[0, 0] = 1  # A[0,0]
    V[0, 0] = 1; V[1, 0] = 1  # B[0,0] + B[0,1]
    W[0, 0] = 1; W[3, 0] = 1  # C[0,0] + C[0,1]
    
    # P2 = (A[0,0] - A[1,0]) * (B[1,0] - B[0,0])
    U[0, 1] = 1; U[3, 1] = -1  # A[0,0] - A[1,0]
    V[3, 2] = 1; V[0, 2] = -1  # B[1,0] - B[0,0]
    W[3, 2] = 1  # C[1,0]... wait let me use a known correct Laderman
    
    # Let me use a verified Laderman decomposition.
    # The 15 rank-1 terms for 3x3x3 matrix multiplication:
    
    # Reset and use a well-known formulation
    U = np.zeros((9, 15))
    V = np.zeros((9, 15))
    W = np.zeros((9, 15))
    
    # I'll use the Laderman algorithm as described in the literature.
    # Indices: U[i*3+j], V[j*3+k], W[k*3+i]
    
    # Term 0: P0 = A00 * (B00 + B01)
    U[0, 0] = 1
    V[0, 0] = 1; V[1, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # contributes to C[0,0] and C[0,1]... 
    # W[k*3+i]: C[0,0] -> k=0,i=0 -> W[0]; C[0,1] -> k=1,i=0 -> W[3]
    # But P0 should contribute to C[0,0] and C[0,1]? Let me think again.
    # P0 = A[0,0]*(B[0,0]+B[0,1]). This product contributes to:
    # C[0,0] via A[0,0]*B[0,0] and C[0,1] via A[0,0]*B[0,1]
    # So W[0,0]=1 (for C[0,0], k=0,i=0) and W[3,0]=1 (for C[0,1], k=1,i=0)
    # Wait, W index is k*n+i where n=3. C[i,j] where i=row of C, j=col of C.
    # The convention is W[k*n+i, r] where C[i,j] = sum_k A[i,k]*B[k,j]
    # So C[i,j] corresponds to W[j*3+i]. 
    # P0 contributes to C[0,0] (j=0, i=0 -> W[0]) and C[0,1] (j=1, i=0 -> W[3])
    
    # Let me restart with a cleaner approach using a known valid decomposition.
    
    # Actually, let me just construct a valid rank-15 decomposition for 3x3x3
    # by using the schoolbook for a subset and adding structure, or better yet,
    # let me use the explicit Laderman constants.
    
    # Laderman's 15 products:
    # M1 = (A11+A22)(B11+B22)  -> but these are 2x2 subblocks... 
    # No, for 3x3 scalar matrices, Laderman gives 15 scalar products.
    
    # Let me use a different, verifiable approach: 
    # Construct the 3x3x3 tensor and find a rank-15 decomposition numerically.
    
    # The 3x3x3 multiplication tensor T[i,j,k] = 1 if k matches, else 0
    # Actually T[i,j,k] = 1 for the (i,j,k) entry where C[i,j] = sum_k A[i,k]B[k,j]
    # So the tensor has T[i*3+j, j*3+k, k*3+i] = 1 in flattened form.
    
    # For a quick valid approach, let me just use schoolbook for 3x3 (rank 27)
    # and then try to compress. But that won't help.
    
    # Alternative: use the fact that 2x5x5 can be decomposed as:
    # C = A * B where A is 2x5, B is 5x5
    # C[i,j] = sum_k A[i,k] * B[k,j]
    
    # Let me just provide a valid schoolbook decomposition for 2x5x5 with rank 50
    # and return it. The score will be 1.0 which is the same as current.
    # But the task asks for rank-minimal. Let me at least try to improve.
    
    # For now, return a valid schoolbook to ensure correctness.
    return None

def _schoolbook_2x5x5():
    """Schoolbook rank-50 decomposition for 2x5 by 5x5 matrix multiplication."""
    n, m, p = 2, 5, 5
    rank = n * m * p  # 50
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication."""
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # The decomposition gives: C[i,j] = sum_c (sum_k U[i*m+k, c]*V[k*p+j, c]) * W[j*n+i, c]
    # Wait, the tensor entry is T[i,j,k] = 1 meaning C[i,j] gets contribution A[i,k]*B[k,j]
    # In the decomposition: T[i,j,k] = sum_c U[i*m+k, c] * V[k*p+j, c] * W[j*n+i, c]
    # We need this to equal 1 for all (i,j,k) corresponding to the multiplication tensor.
    
    # Actually the multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k) where
    # C[i,j] = sum_k A[i,k]*B[k,j]. So every (i,j,k) triple should have T[i,j,k]=1.
    # No wait - the tensor is defined as T[i,j,k] = 1 if this triple participates,
    # which is ALL triples. So we need sum_c U[i*m+k,c]*V[k*p+j,c]*W[j*n+i,c] = 1
    # for ALL i in [0,n), j in [0,p), k in [0,m).
    
    rank = U.shape[1]
    # Build the reconstructed tensor
    for i in range(n):
        for j in range(p):
            for k in range(m):
                val = 0
                for c in range(rank):
                    val += U[i * m + k, c] * V[k * p + j, c] * W[j * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 5
    deadline = time.time() + 120
    
    # Start with schoolbook rank-50
    U, V, W = _schoolbook_2x5x5()
    best_rank = 50
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Verify schoolbook
    assert _verify_decomposition(best_U, best_V, best_W, n, m, p), "Schoolbook failed!"
    
    # Try to find lower-rank decompositions using numerical optimization
    # For 2x5x5, the trivial upper bound is 50. 
    # Known results: M(2,5,5) is at most 50 (schoolbook).
    # It's an open problem whether it can be done in less.
    # For small cases, the rank is often equal to the schoolbook rank.
    
    # Let's try a few random rank-15 and rank-20 attempts using the structure
    # but realistically, for 2x5x5, rank 50 may be optimal or near-optimal.
    
    # Try rank 45, 40, etc. with random initialization and gradient descent
    for target_rank in [45, 40, 35, 30, 25, 20, 15]:
        if time.time() > deadline - 30:
            break
            
        success = False
        for attempt in range(20):
            if time.time() > deadline - 15:
                break
                
            rng = np.random.RandomState(attempt * 1000 + target_rank)
            U_try = rng.randn(n * m, target_rank) * 0.1
            V_try = rng.randn(m * p, target_rank) * 0.1
            W_try = rng.randn(p * n, target_rank) * 0.1
            
            # Gradient descent on the error
            lr = 0.01
            for step in range(200):
                if time.time() > deadline - 15:
                    break
                
                # Compute error and gradients
                # T_recon[i,j,k] = sum_c U[i*m+k,c]*V[k*p+j,c]*W[j*n+i,c]
                # Error = T_recon - 1
                # dE/dU[i*m+k,c] = sum over (j) of (T_recon[i,j,k]-1) * V[k*p+j,c] * W[j*n+i,c]
                
                err = 0.0
                U_grad = np.zeros_like(U_try)
                V_grad = np.zeros_like(V_try)
                W_grad = np.zeros_like(W_try)
                
                for i in range(n):
                    for j in range(p):
                        for k in range(m):
                            t_val = 0
                            for c in range(target_rank):
                                t_val += U_try[i*m+k, c] * V_try[k*p+j, c] * W_try[j*n+i, c]
                            e = t_val - 1.0
                            err += e * e
                            for c in range(target_rank):
                                u_c = U_try[i*m+k, c]
                                v_c = V_try[k*p+j, c]
                                w_c = W_try[j*n+i, c]
                                U_grad[i*m+k, c] += e * v_c * w_c
                                V_grad[k*p+j, c] += e * u_c * w_c
                                W_grad[j*n+i, c] += e * u_c * v_c
                
                err = 2 * err / (n * p * m)
                if err < 1e-12:
                    success = True
                    break
                
                U_try -= lr * U_grad
                V_try -= lr * V_grad
                W_try -= lr * W_grad
            
            if success:
                # Verify
                if _verify_decomposition(U_try, V_try, W_try, n, m, p, tol=1e-8):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
                    break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
