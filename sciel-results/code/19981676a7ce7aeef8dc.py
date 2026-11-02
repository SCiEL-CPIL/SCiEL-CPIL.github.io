"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x3 * 3x5 matrix multiplication.
    
    Strategy:
    1. Use the known rank-7 (Strassen) decomposition for 3x3x3 matrix multiplication.
    2. Extend it to handle the extra two columns of the 3x5 output by adding terms.
    
    The tensor for multiplying A (n x m) by B (m x p) has entries T[i,j,k] = A[i,j]*B[j,k].
    We need to decompose this as sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    
    For 3x3x3, Strassen gives rank 7. For 3x3x5, we can view it as three 3x3x3 blocks
    in the p-dimension, but the last block only has 2 columns (since p=5).
    
    Actually, let's think more carefully. The output C is 3x5. We can split B into B1 (3x3) and B2 (3x2).
    Then C = A*B1 + A*B2.
    
    For A*B1 (3x3*3x3), we use Strassen rank 7.
    For A*B2 (3x3*3x2), we need a decomposition.
    
    The tensor for 3x3*3x2 has rank at most 3*3*2 = 18 by schoolbook, but can we do better?
    Actually, the rank of the 3x3*3x2 multiplication tensor... Let me think.
    
    Alternatively, we can just use a direct approach: since we need integer/half-integer entries,
    and we want minimal rank, let's try to construct it explicitly.
    
    For 3x3x3 Strassen, the 7 rank-1 terms are well-known. Let me recall them.
    
    Actually, let me just try a simpler approach first: use the schoolbook for the 3x3 part 
    and add 6 more terms for the 3x2 part, giving rank 9 + 6 = 15. But that's not minimal.
    
    Let me think about this differently. The multiplication tensor for m_{n,m,p} where n=3, m=3, p=5.
    
    We can write B as a 3x5 matrix. The product C[i,k] = sum_j A[i,j]*B[j,k].
    
    One approach: use the fact that for the first 3 columns of B, we can use Strassen (rank 7),
    and for the last 2 columns, we need additional terms.
    
    For the last 2 columns, we have C[:,3] and C[:,4]. These are linear combinations of the rows of A
    weighted by the columns of B2 (the last 2 columns of B).
    
    C[i,3] = sum_j A[i,j]*B[j,3]
    C[i,4] = sum_j A[i,j]*B[j,4]
    
    This is like multiplying A (3x3) by B2 (3x2). The tensor for this has entries T2[i,j,k] for k in {0,1}
    (representing the two output columns).
    
    The rank of the 3x3*3x2 multiplication tensor: I believe it's 10 or so, but let me just use
    a simple construction.
    
    Actually, let me just implement a general rank-finding approach. Since the problem size is small,
    we can try to find a low-rank decomposition using linear algebra.
    
    But first, let me try a constructive approach.
    
    For 3x3x3, Strassen's algorithm uses 7 multiplications. The rank-1 tensors are:
    
    Let me define them properly. For matrices A (3x3), B (3x3), C = A*B (3x3).
    
    Strassen's 7 products:
    M1 = (A00 + A22) * (B00 + B22)
    M2 = (A20 + A22) * B02
    M3 = A00 * (B02 - B22)
    M4 = A22 * (B20 - B00)
    M5 = (A00 + A10) * B00
    M6 = (A12 + A22) * (B20 + B22)
    M7 = (A00 - A10) * (B10 + B12)
    M8 = (A02 - A22) * (B20 + B22)  -- wait, that's 8, standard Strassen for 4x4 uses 7.
    
    Let me look up the standard 3x3 Strassen-like decomposition. Actually, the minimal rank for 3x3x3
    is known to be 7 (over infinite fields), and there is an explicit decomposition.
    
    Let me just use a known rank-7 decomposition for 3x3x3 and then extend.
    
    Actually, for simplicity and correctness, let me just use the schoolbook approach but try to
    find a lower rank by solving a system of equations.
    
    Given the time constraints, let me just return the schoolbook decomposition of rank 45,
    but that scores 1.0. I need to do better.
    
    Let me try a different approach: since we have 120 seconds, let me try to find a low-rank
    decomposition using numerical methods.
    
    The tensor T has dimensions (n*m) x (m*p) x (p*n) = 9 x 15 x 15.
    
    We want to find rank r such that T = sum_{s=1}^{r} u_s ⊗ v_s ⊗ w_s.
    
    This is a tensor rank decomposition problem. For small ranks, we can use iterative methods.
    
    Let me try a simple approach: initialize with random vectors and use ALS (alternating least squares)
    to minimize the residual.
    
    Actually, for exact decomposition with integer/half-integer entries, numerical methods may not
    converge to exact solutions. But the problem says "exactly decompose", so we need exact entries.
    
    Let me try a constructive approach based on Strassen.
    
    Known rank-7 decomposition for 3x3x3 (over any field with 2 invertible):
    
    I'll use the standard Strassen decomposition adapted to 3x3.
    
    Actually, let me just implement a search for low-rank decompositions with rational entries.
    
    Given the complexity, let me just return a known good decomposition. The rank of the 3x3x5
    multiplication tensor is at most 7 + 2*3 = 13 if we can share some terms, or 7 + 6 = 13.
    
    Let me try rank 13.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Try to construct a low-rank decomposition
    # Start with schoolbook and try to reduce
    
    # For now, let's try a constructive approach
    # Use Strassen for the 3x3x3 part and extend for the extra 2 columns
    
    # The 3x3x3 multiplication tensor can be decomposed with rank 7
    # Let's define the 7 rank-1 components
    
    # Standard Strassen for 3x3 matrices (indices 0,1,2)
    # M1 = (A[0,0]+A[2,2])*(B[0,0]+B[2,2])
    # M2 = (A[2,0]+A[2,2])*B[0,2]
    # M3 = A[0,0]*(B[0,2]-B[2,2])
    # M4 = A[2,2]*(B[2,0]-B[0,0])
    # M5 = (A[0,0]+A[1,0])*B[0,0]
    # M6 = (A[1,2]+A[2,2])*(B[2,0]+B[2,2])
    # M7 = (A[0,0]-A[1,0])*(B[1,0]+B[1,2])
    # M8 = (A[0,2]-A[2,2])*(B[2,0]+B[2,2]) -- this is 8, not 7
    
    # Let me use a verified rank-7 decomposition.
    # Actually, the minimal rank of 3x3x3 matrix multiplication is 7, but the explicit
    # decomposition is non-trivial.
    
    # For simplicity, let me just use a rank-9 decomposition (schoolbook for 3x3 is 9)
    # and add 6 for the 3x2 part, total rank 15.
    
    # But we can do better. Let me try to find a rank-13 or lower decomposition.
    
    # Actually, let me just implement a numerical search.
    
    def build_tensor():
        """Build the full multiplication tensor T[i*m+j, j*p+k, k*n+i] = delta[i,k] * 1 (for the product A[i,j]*B[j,k])."""
        # T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
        # The tensor entry is 1 if the indices correspond to A[i,j]*B[j,k] contributing to C[i,k]
        # So T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        return T
    
    T = build_tensor()
    
    # Try to find a low-rank decomposition using ALS
    best_rank = n * m * p  # Schoolbook rank
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from low to high
    for target_rank in range(10, min(46, n*m*p + 1)):
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        for attempt in range(5):
            if time.time() > deadline - 20:
                break
            
            # Initialize U, V, W randomly
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # ALS iterations
            converged = False
            for it in range(200):
                if time.time() > deadline - 15:
                    break
                
                # Update U
                # T = sum_s U[:,s] ⊗ V[:,s] ⊗ W[:,s]
                # For each a, T[a,:,:] = sum_s U[a,s] * V[:,s] * W[:,s]^T
                # So T[a] = U[a,:] @ (V * W) -- not quite, it's a tensor product
                
                # Residual: T - sum_s U[:,s] V[:,s]^T W[:,s]
                # To update U, fix V and W
                # T[a] = sum_s U[a,s] * (V[:,s] * W[:,s])  -- this is not standard matrix mult
                
                # Actually, T[a, b, c] = sum_s U[a,s] * V[b,s] * W[c,s]
                # For fixed b,c: T[:, b, c] = U @ diag(V[b,:]) @ diag(W[c,:])^T -- no
                # T[:, b, c] = sum_s U[:,s] * V[b,s] * W[c,s]
                # = U @ (V[b,:] * W[c,:])  -- element-wise product of V[b,:] and W[c,:]
                
                # So for each (b,c), T[:,b,c] = U @ (V[b,:] * W[c,:])
                # This is a linear system in U.
                
                # Let's form the matrix M_U where M_U[(b,c), s] = V[b,s] * W[c,s]
                # Then T.reshape(n*m, m*p*p*n) = U @ M_U^T
                # So U = T.reshape(n*m, -1) @ pinv(M_U^T)
                
                M_U = np.zeros((m*p * p*n, target_rank))
                for s in range(target_rank):
                    M_U[:, s] = (V[:, s].reshape(-1, 1) * W[:, s].reshape(1, -1)).flatten()
                
                T_flat = T.reshape(n*m, -1)
                try:
                    U_new = T_flat @ np.linalg.pinv(M_U.T)
                except:
                    U_new = U
                
                # Update V
                M_V = np.zeros((n*m * p*n, target_rank))
                for s in range(target_rank):
                    M_V[:, s] = (U[:, s].reshape(-1, 1) * W[:, s].reshape(1, -1)).flatten()
                T_flat_v = T.transpose(1, 0, 2).reshape(m*p, -1)
                try:
                    V_new = T_flat_v @ np.linalg.pinv(M_V.T)
                except:
                    V_new = V
                
                # Update W
                M_W = np.zeros((n*m * m*p, target_rank))
                for s in range(target_rank):
                    M_W[:, s] = (U[:, s].reshape(-1, 1) * V[:, s].reshape(1, -1)).flatten()
                T_flat_w = T.transpose(2, 0, 1).reshape(p*n, -1)
                try:
                    W_new = T_flat_w @ np.linalg.pinv(M_W.T)
                except:
                    W_new = W
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence
                # Compute residual
                T_recon = np.einsum('as,bs,cs->abc', U, V, W)
                residual = np.linalg.norm(T - T_recon) / np.linalg.norm(T)
                
                if residual < 1e-10:
                    converged = True
                    break
            
            if converged:
                # Check if entries are close to integers or half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                T_recon_rounded = np.einsum('as,bs,cs->abc', U_rounded, V_rounded, W_rounded)
                residual_rounded = np.linalg.norm(T - T_recon_rounded) / np.linalg.norm(T)
                
                if residual_rounded < 1e-8:
                    best_rank = target_rank
                    best_U = U_rounded
                    best_V = V_rounded
                    best_W = W_rounded
                    break
        
        if best_rank == target_rank:
            break
    
    if best_U is not None:
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook decomposition
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
# EVOLVE-BLOCK-END
