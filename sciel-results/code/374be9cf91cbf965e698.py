"""Find rank-minimal exact tensor decomposition for 2x4 by 4x7 matrix multiplication."""
import numpy as np
import time
from itertools import combinations, product
from scipy.optimize import linprog

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    start_time = time.time()
    deadline = 145  # Leave 5s margin before 150s kill

    # The tensor T has shape (n*m, m*p, p*n)
    # Indices: a = i*m + j, b = j*p + k, c = k*n + i
    # T[a, b, c] = 1 if valid (i,j,k), else 0.
    
    T_full = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_full[a, b, c] = 1.0

    def verify_decomposition(U, V, W, T_target, atol=1e-9):
        """Verify that sum_r U[:,r] V[:,r] W[:,r] == T_target"""
        r = U.shape[1]
        T_approx = np.zeros_like(T_target)
        for r_idx in range(r):
            T_approx += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
        return np.allclose(T_target, T_approx, atol=atol)

    def clean_matrix(M):
        """Round to nearest half-integer if close"""
        M_rounded = np.round(M * 2) / 2
        if np.allclose(M, M_rounded, atol=1e-6):
            return M_rounded
        M_int = np.round(M)
        if np.allclose(M, M_int, atol=1e-6):
            return M_int
        return M

    # Strategy: Systematic search over structured sparse decompositions
    # 
    # Key insight: The tensor decomposes into two blocks (i=0 and i=1) that share the V factor.
    # For each block i, we need: T_i[a', b, c'] = sum_r U_i[a', r] * V[b, r] * W_i[c', r]
    # where a' = j (0..m-1), c' = k (0..p-1) for fixed i.
    #
    # The 1x4x7 tensor T_1 has rank exactly 28 (it's 7 independent dot products of length 4).
    # For 2x4x7, if we could perfectly share V between the two blocks, we might get rank < 56.
    #
    # Approach: Try ranks from a lower bound up, using structured initialization.
    # Lower bound: The tensor has a "slice rank" lower bound. For matrix mult n×m×p,
    # rank >= max(n*m, m*p, p*n) is NOT true, but rank >= n*m*p / max(n,m,p) is a known bound.
    # Actually, the trivial bound is n*m*p = 56. Let's search from 40 upward.
    
    # First, let's establish the schoolbook solution as baseline (rank 56)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Verify schoolbook
    assert verify_decomposition(U_sb, V_sb, W_sb, T_full)
    
    # Now search for lower rank using structured approaches
    # 
    # Method 1: For each rank r, try to find a decomposition where:
    #   - U has block structure: U[i*m:(i+1)*m, :] for i=0,1
    #   - W has block structure: W[k*n:(k+1)*n, :] for each k
    #   - V is shared
    #
    # We'll use a hybrid: fix sparse patterns for U and W, solve for V linearly,
    # then check if the result is consistent.
    
    def try_structured_rank(r, max_patterns=5000):
        """
        Try to find a rank-r decomposition with structured sparsity.
        For each candidate, we fix which entries of U and W are nonzero (sparse pattern),
        then solve for V and the nonzero entries of U and W.
        """
        if time.time() > deadline:
            return None
        
        # The tensor is sparse: only n*m*p = 56 entries are 1.
        # For a rank-r decomposition, we need r terms.
        # 
        # Key structural approach: 
        # For 1xm x 1xp (dot product), rank is m (trivial) or potentially less with Winograd.
        # For 1x4x7, the 7 dot products are independent, so rank >= 7*1 = 7? No.
        # Actually, each dot product of length 4 requires rank 4 (no known improvement for 1D).
        # So 1x4x7 has rank 28.
        # 
        # For 2x4x7, the two "rows" of A are independent inputs, and the two "rows" of C
        # are independent outputs. The only sharing is through B.
        # 
        # Let's try a different approach: use the fact that the tensor is a sum of 
        # elementary tensors and search for a minimal set.
        
        # Approach: Greedy/Iterative method
        # 1. Start with the 56 elementary tensors
        # 2. Try to merge pairs of terms that can be combined
        # 3. Use linear algebra to check if a set of r terms can span the tensor
        
        # More practical: Use ALS with structured initialization based on the block structure
        
        np.random.seed(42 + r)
        
        best_err = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        num_restarts = 30
        max_iters = 200
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                return None
            
            # Initialize with block-aware structure
            # U: shape (8, r) - 2 blocks of 4
            # V: shape (28, r) - 4 blocks of 7
            # W: shape (14, r) - 7 blocks of 2
            
            # Initialize V with block structure: each block of 7 columns might be sparse
            V = np.zeros((m*p, r))
            U = np.zeros((n*m, r))
            W = np.zeros((p*n, r))
            
            # Random sparse initialization with block awareness
            for r_idx in range(r):
                # Each term should contribute to a subset of the tensor
                # Randomly assign which (i,j,k) combinations this term covers
                # Simplified: random dense with small values
                V[:, r_idx] = np.random.randn(m*p) * 0.1
                U[:, r_idx] = np.random.randn(n*m) * 0.1
                W[:, r_idx] = np.random.randn(p*n) * 0.1
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Update U: For each a, T[a] = sum_r U[a,r] (V[:,r] ⊗ W[:,r])
                # Flatten: T[a].reshape(-1) = U[a] @ M_U.T
                M_U = np.zeros((r, m*p * p*n))
                for r_idx in range(r):
                    M_U[r_idx, :] = np.kron(V[:, r_idx], W[:, r_idx])
                
                T_U = T_full.reshape(n*m, -1)
                
                try:
                    # Use lstsq for better conditioning
                    U_new, _, _, _ = np.linalg.lstsq(M_U.T, T_U.T, rcond=None)
                    U_new = U_new.T
                except:
                    return None
                
                # Update V
                M_V = np.zeros((r, n*m * p*n))
                for r_idx in range(r):
                    M_V[r_idx, :] = np.kron(U[:, r_idx], W[:, r_idx])
                
                T_V = T_full.transpose(1, 0, 2).reshape(m*p, -1)
                
                try:
                    V_new, _, _, _ = np.linalg.lstsq(M_V.T, T_V.T, rcond=None)
                    V_new = V_new.T
                except:
                    return None
                    
                # Update W
                M_W = np.zeros((r, n*m * m*p))
                for r_idx in range(r):
                    M_W[r_idx, :] = np.kron(U[:, r_idx], V[:, r_idx])
                
                T_W = T_full.transpose(2, 0, 1).reshape(p*n, -1)
                
                try:
                    W_new, _, _, _ = np.linalg.lstsq(M_W.T, T_W.T, rcond=None)
                    W_new = W_new.T
                except:
                    return None
                    
                U, V, W = U_new, V_new, W_new
                
                # Calculate error
                T_approx = np.zeros_like(T_full)
                for r_idx in range(r):
                    T_approx += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
                    
                err = np.linalg.norm(T_full - T_approx) / (np.linalg.norm(T_full) + 1e-15)
                
                if err < best_err:
                    best_err = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    
                if err < 1e-10:
                    break
                # Early termination if not improving
                if it > 50 and err > 0.1:
                    break
        
        if best_err < 1e-9 and best_U is not None:
            U = clean_matrix(best_U)
            V = clean_matrix(best_V)
            W = clean_matrix(best_W)
            
            if verify_decomposition(U, V, W, T_full):
                return U, V, W, best_err
            elif verify_decomposition(best_U, best_V, best_W, T_full, atol=1e-6):
                return best_U, best_V, best_W, best_err
        
        return None

    # Also try a completely different approach: 
    # Exploit the fact that for 1x4x7, we can use Winograd-like transforms
    # For dot product of length 4: a1*b1 + a2*b2 + a3*b3 + a4*b4
    # Winograd for length 4: F = [[1,0,0,0],[1,1,0,0],[1,-1,0,0],[1,0,1,0],[1,0,-1,0]]
    # Actually, standard Winograd for length 4 uses 5 multipliers instead of 4? No, that's not right.
    # For 1D convolution of length 4 with length 4, Winograd uses 5 multipliers.
    # But for a single dot product (length 4 x length 4), it's just 4 multipliers.
    # Winograd helps with convolutions, not single dot products.
    
    # Let's try ranks from 40 to 55
    for r in range(40, 56):
        if time.time() > deadline:
            break
        
        res = try_structured_rank(r)
        if res is not None:
            U, V, W, err = res
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fallback: Schoolbook rank 56
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
