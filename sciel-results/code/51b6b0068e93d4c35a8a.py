import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # safety margin before 120s limit
    
    # Try to build a low-rank decomposition using Strassen-style block decomposition
    # For 3x4x6: use rank 68 via structured decomposition
    
    # Schoolbook rank is 72. We can save 4 by using Strassen on 2x2 sub-blocks.
    # Strategy: decompose into 2x2 Strassen blocks + schoolbook for remaining
    
    # The multiplication tensor T[i,j,k] = 1 if C[i,k] gets A[i,j]*B[j,k]
    # Indexing: A index = i*m+j, B index = j*p+k, C index = k*n+i
    
    # Build the full tensor
    A_dim = n * m  # 12
    B_dim = m * p  # 24
    C_dim = p * n  # 18
    
    # Use a structured decomposition:
    # Split the 4x6 B matrix into 2x2, 2x2, 2x2 column blocks
    # Split the 3x4 A matrix into 2x2, 2x2, 1x4 row blocks (or 2x2, 1x4)
    # Actually, let's use a direct approach with known low-rank algorithms
    
    # For 3x4x6, a known bound is rank <= 68
    # Let's construct it explicitly using block Strassen
    
    # Block decomposition:
    # A (3x4) = [A11 A12; A21 A22] where A11,A12 are 2x2, A21,A22 are 1x2
    # B (4x6) = [B11 B12 B13; B21 B22 B23] where each is 2x2
    # C (3x6) = [C11 C12 C13; C21 C22 C23] where C1* are 2x2, C2* are 1x2
    
    # C11 = A11*B11 + A12*B21 (2x2)
    # C12 = A11*B12 + A12*B22 (2x2)
    # C13 = A11*B13 + A12*B23 (2x2)
    # C21 = A21*B11 + A22*B21 (1x2)
    # C22 = A21*B12 + A22*B22 (1x2)
    # C23 = A21*B13 + A22*B23 (1x2)
    
    # For each 2x2 result, use Strassen (rank 7)
    # For each 1x2 result, use schoolbook (rank 2)
    # Total: 3*7 + 3*2 = 21 + 6 = 27... that seems too low, must be wrong
    
    # The issue: Strassen computes C = A*B for 2x2 matrices using 7 products
    # But here C11 = A11*B11 + A12*B21 is a sum of two 2x2 products
    # We need to compute both A11*B11 and A12*B21, which share structure
    
    # Better approach: use the full tensor decomposition directly
    # Let's use a verified rank-68 decomposition
    
    # Actually, let's use a simpler guaranteed approach:
    # Rank 71 or 70 by merging some schoolbook terms
    
    # Safest: return schoolbook rank 72 (score 1.0) but try to improve
    
    # Let me implement a correct Strassen-based block decomposition
    # For the 2x2x2 tensor, the 7 rank-1 terms are:
    
    def strassen_2x2x2():
        """Return 7 (u, v, w) vectors for 2x2x2 matrix multiplication tensor.
        Basis order: (00, 01, 10, 11) for each mode.
        """
        # Verified Strassen decomposition
        terms = []
        # M1 = (A00+A11)(B00+B11) -> contributes to C00 and C11
        # The rank-1 tensor: u⊗v⊗w where
        # u picks A00 and A11, v picks B00 and B11, w picks C00 and C11
        # But we need: sum of u_i v_j w_k = 1 at (i,j,k) where the product appears
        
        # Correct formulation from literature:
        # Term 1: u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1)
        # This gives: (00,00,00),(00,11,00),(11,00,00),(11,11,00),
        #           (00,00,11),(00,11,11),(11,00,11),(11,11,11)
        # But we only want (00,00,00) and (11,11,11) for M1's direct contribution
        # The other terms come from the other M's
        
        # I'll use a known correct set. After careful derivation:
        # The 7 terms for 2x2x2 (indices 0=A00, 1=A01, 2=A10, 3=A11):
        
        # t1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]
        # t2: u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,0]  
        # t3: u=[1,0,0,0], v=[0,0,1,-1], w=[0,1,0,0]
        # t4: u=[0,0,0,1], v=[0,1,-1,0], w=[0,0,1,0]
        # t5: u=[1,0,1,0], v=[0,0,0,1], w=[0,1,0,0]
        # t6: u=[0,0,-1,1], v=[1,0,1,0], w=[0,0,1,0]
        # t7: u=[0,1,0,-1], v=[0,0,1,1], w=[1,0,0,0]
        
        # Verify t1: sum u_i v_j w_k
        # (0,0,0): 1*1*1=1
        # (0,3,0): 1*1*1=1  -> A00*B11 contributes to C00? In Strassen M1=(A00+A11)(B00+B11)
        # This contributes to C00 and C11, but the cross terms A00*B11 and A11*B00 
        # should cancel with other terms.
        
        # This is getting complex. Let me use a different strategy:
        # Use rank 72 schoolbook (guaranteed correct) and attempt local improvement
        
        pass
    
    # Given the complexity and risk of incorrect Strassen vectors,
    # let's use a verified approach: build the tensor and use SVD-based 
    # approximation to find a lower-rank decomposition, then round.
    
    T = np.zeros((A_dim, B_dim, C_dim), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try to find a low-rank decomposition by iterative projection
    # Start with rank 68 and try to solve
    
    best_rank = n * m * p  # 72
    best_U = best_V = best_W = None
    
    for target_rank in range(60, best_rank + 1):
        if time.time() > deadline:
            break
        
        # Try to find a decomposition of this rank
        # Use randomized SVD-like approach on matricizations
        
        # Matricize T as (A_dim * B_dim) x C_dim
        T_mat = T.reshape(A_dim * B_dim, C_dim)
        
        # Get top target_rank singular vectors
        try:
            U_mat, s, Vt = np.linalg.svd(T_mat, full_matrices=False)
            if target_rank > len(s):
                continue
            U_mat = U_mat[:, :target_rank]
            Vt = Vt[:target_rank, :]
        except:
            continue
        
        # Now we have T ≈ U_mat @ diag(s) @ Vt
        # But we need T = sum_r u_r ⊗ v_r ⊗ w_r
        # This is a 3-way decomposition, not 2-way
        
        # Alternative: use PARAFAC-like iteration
        # Initialize U, V, W randomly
        np.random.seed(42 + target_rank)
        U = np.random.randn(A_dim, target_rank) * 0.1
        V = np.random.randn(B_dim, target_rank) * 0.1
        W = np.random.randn(C_dim, target_rank) * 0.1
        
        # Alternate minimization
        for it in range(50):
            if time.time() > deadline:
                break
            
            # Update U: for each a, U[a,:] should satisfy
            # sum_r U[a,r] * (V[:,r] ⊗ W[:,r]) = T[a,:,:].reshape(-1)
            # This is a linear system: (V ⊙ W)^T U[a,:] = T[a,:,:].reshape(-1)
            # where V ⊙ W is Khatri-Rao product
            
            # Khatri-Rao of V (B_dim x R) and W (C_dim x R) is (B_dim*C_dim) x R
            V_kr_W = np.kron(V.T, W.T).T  # shape (B_dim*C_dim, R)
            # Actually: V_kr_W[:,r] = V[:,r] ⊗ W[:,r]
            V_kr_W = np.zeros((B_dim * C_dim, target_rank))
            for r in range(target_rank):
                V_kr_W[:, r] = np.kron(V[:, r], W[:, r])
            
            # For each a: solve V_kr_W^T U[a,:] = T[a,:,:].reshape(-1)
            # Use least squares
            T_flat = T.reshape(A_dim, B_dim * C_dim)
            try:
                U = np.linalg.lstsq(V_kr_W.T, T_flat.T, rcond=None)[0].T
            except:
                break
            
            # Update V
            U_kr_W = np.zeros((A_dim * C_dim, target_rank))
            for r in range(target_rank):
                U_kr_W[:, r] = np.kron(U[:, r], W[:, r])
            T_flat_V = T.transpose(1, 0, 2).reshape(B_dim, A_dim * C_dim)
            try:
                V = np.linalg.lstsq(U_kr_W.T, T_flat_V.T, rcond=None)[0].T
            except:
                break
            
            # Update W
            U_kr_V = np.zeros((A_dim * B_dim, target_rank))
            for r in range(target_rank):
                U_kr_W[:, r] = np.kron(U[:, r], V[:, r])
            T_flat_W = T.transpose(2, 0, 1).reshape(C_dim, A_dim * B_dim)
            try:
                W = np.linalg.lstsq(U_kr_V.T, T_flat_W.T, rcond=None)[0].T
            except:
                break
        
        # Check if decomposition is exact
        result = np.einsum('ar,br,cr->abc', U, V, W)
        if np.allclose(result, T, atol=1e-6, rtol=1e-8):
            best_rank = target_rank
            best_U, best_V, best_W = U, V, W
            break
    
    # If we found a better rank, use it; otherwise use schoolbook
    if best_rank < n * m * p and best_U is not None:
        # Round to integers/half-integers if possible
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2
        
        # Verify rounded version
        result = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
        if np.allclose(result, T, atol=1e-6, rtol=1e-8):
            best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
        
        U_list = [[best_U[i, j] for j in range(best_rank)] for i in range(A_dim)]
        V_list = [[best_V[i, j] for j in range(best_rank)] for i in range(B_dim)]
        W_list = [[best_W[i, j] for j in range(best_rank)] for i in range(C_dim)]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fall back to schoolbook rank 72
        rank = n * m * p
        U = [[0] * rank for _ in range(A_dim)]
        V = [[0] * rank for _ in range(B_dim)]
        W = [[0] * rank for _ in range(C_dim)]
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
