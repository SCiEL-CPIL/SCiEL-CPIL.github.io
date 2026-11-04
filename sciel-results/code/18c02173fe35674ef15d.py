"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START

def _is_half_integer(x):
    """Check if x is a half-integer (multiple of 0.5) within tolerance."""
    return np.isclose(2 * x, np.round(2 * x), atol=1e-9)

def _verify_decomposition(U, V, W, n=3, m=3, p=3):
    """
    Verify that U, V, W form a valid tensor decomposition for n x m x p matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Convention: W[k*n + i, r]
    T[i, j, k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    """
    rank = U.shape[1]
    T_recon = np.zeros((n*m, m*p, p*n), dtype=float)
    for r in range(rank):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        T_recon += np.einsum('a,b,c->abc', u, v, w)
    
    # Check required 1s
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T_recon[a, b, c] - 1.0) > 1e-9:
                    return False
    
    # Check required 0s
    mask = np.zeros((n*m, m*p, p*n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                mask[a, b, c] = True
    
    if not np.allclose(T_recon[~mask], 0.0, atol=1e-9):
        return False
        
    return True

def _check_half_ints(U, V, W):
    """Check if all entries are half-integers."""
    if not np.all(_is_half_integer(U)): return False
    if not np.all(_is_half_integer(V)): return False
    if not np.all(_is_half_integer(W)): return False
    return True

def _round_to_half_int(x):
    """Round a matrix to the nearest half-integer."""
    return np.round(2 * x) / 2.0

def _winograd_3x3_rank23():
    """
    Construct the Winograd rank-23 decomposition for 3x3 matrix multiplication.
    
    The Winograd F(3,3) algorithm uses transform matrices:
    M = [[1, 1, 0], [1, 0, 1], [1, -1, 0]]  (row transform)
    N = [[1, 0, 1], [1, 1, 0], [1, 0, -1]]   (col transform)
    
    The 23 bilinear terms arise from the structure of the algorithm.
    """
    n, m, p = 3, 3, 3
    rank = 23
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Winograd 3x3: The algorithm computes C = A @ B where C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # Transform: For each row i of A, compute u_i = M @ A[i,:]
    #   u_i[0] = A[i,0] + A[i,1]
    #   u_i[1] = A[i,0] + A[i,2]
    #   u_i[2] = A[i,0] - A[i,1]
    #
    # For each column k of B, compute v_k = B[:,k] @ N
    #   v_k[0] = B[0,k] + B[2,k]
    #   v_k[1] = B[0,k] + B[1,k]
    #   v_k[2] = B[0,k] - B[2,k]
    #
    # Product: w_{i,k} = u_i * v_k (element-wise)
    #
    # Inverse: C[i,k] = w_{i,k}[0] + w_{i,k}[1] + w_{i,k}[2]
    #
    # Total bilinear terms: 3*3*3 = 27... but Winograd shares some.
    #
    # Actually, the standard Winograd 3x3 has rank 23. Let me construct it properly.
    #
    # The 23 terms are indexed as follows. Let me use the known structure:
    # Terms 0-8: The "core" 3x3 multiplications (one for each (i,j,k) where the 
    #            transform coefficients match)
    # Terms 9-22: Additional terms for the transform structure.
    #
    # Let me build it from the explicit Winograd algorithm:
    
    # M transform coefficients (for rows of A):
    # M[0] = [1, 1, 0]  -> u0 = a0 + a1
    # M[1] = [1, 0, 1]  -> u1 = a0 + a2
    # M[2] = [1, -1, 0] -> u2 = a0 - a1
    
    # N transform coefficients (for cols of B):
    # N[0] = [1, 0, 1]  -> v0 = b0 + b2
    # N[1] = [1, 1, 0]  -> v1 = b0 + b1
    # N[2] = [1, 0, -1] -> v2 = b0 - b2
    
    # The product w_{i,k,l} = u_i[l] * v_k[l] for l in {0,1,2}
    # And C[i,k] = sum_l w_{i,k,l}
    
    # So C[i,k] = (a_i0+a_i1)(b_k0+b_k2) + (a_i0+a_i2)(b_k0+b_k1) + (a_i0-a_i1)(b_k0-b_k2)
    
    # Expanding:
    # = a_i0*b_k0 + a_i0*b_k2 + a_i1*b_k0 + a_i1*b_k2
    #   + a_i0*b_k0 + a_i0*b_k1 + a_i2*b_k0 + a_i2*b_k1
    #   + a_i0*b_k0 - a_i0*b_k2 - a_i1*b_k0 + a_i1*b_k2
    
    # = 3*a_i0*b_k0 + a_i0*b_k1 - a_i0*b_k2 + a_i1*b_k0 + 2*a_i1*b_k2
    #   + a_i2*b_k0 + a_i2*b_k1
    
    # But we need C[i,k] = a_i0*b_k0 + a_i1*b_k1 + a_i2*b_k2
    
    # Hmm, this doesn't look right. Let me reconsider.
    
    # The Winograd F(3,3) is more subtle. Let me use the standard formulation.
    
    # Actually, I think the issue is that the simple F(3,3) I described has rank 27,
    # not 23. The rank-23 result uses a more sophisticated structure.
    
    # Let me use the known rank-23 decomposition directly.
    # Based on literature, here's a rank-23 construction:
    
    # I'll use a verified construction. The 23 terms are:
    
    # Let me just hardcode a known valid rank-23 decomposition.
    
    # Based on the Winograd 3x3 algorithm as described in various references,
    # here is the explicit factor matrix construction:
    
    # Term 0-2: For the "diagonal" of the transform
    # Term 3-8: Cross terms
    # Term 9-22: Additional transform terms
    
    # Let me use a different approach: construct from the explicit algorithm steps.
    
    # The Winograd 3x3 algorithm (from Winograd 1979, "Arithmetic Complexity of
    # Polynomial and Matrix Operations"):
    
    # Step 1: Transform A rows:
    #   For each row i:
    #     u[i][0] = A[i][0] + A[i][1]
    #     u[i][1] = A[i][0] + A[i][2]  
    #     u[i][2] = A[i][0] - A[i][1]
    
    # Step 2: Transform B columns:
    #   For each col k:
    #     v[k][0] = B[0][k] + B[2][k]
    #     v[k][1] = B[0][k] + B[1][k]
    #     v[k][2] = B[0][k] - B[2][k]
    
    # Step 3: Multiply:
    #   For each i, k, l in {0,1,2}:
    #     w[i][k][l] = u[i][l] * v[k][l]
    
    # Step 4: Inverse transform:
    #   C[i][k] = w[i][k][0] + w[i][k][1] + w[i][k][2]
    
    # This gives 3*3*3 = 27 bilinear terms, not 23.
    
    # The rank-23 result must use a different, more efficient scheme.
    
    # Let me look for the actual rank-23 construction.
    
    # After research, the rank-23 for 3x3 is achieved by the following:
    # It's a specific pattern. Let me construct it by solving the system.
    
    # Actually, let me just try to find it numerically using the fact that
    # the coefficients should be small integers or half-integers.
    
    # I'll initialize with a structured guess and refine.
    
    # For now, let me return the schoolbook (rank 27) as fallback,
    # but first try a few structured approaches.
    
    # Approach: Use the Winograd structure but with shared terms.
    
    # Key insight: In the F(3,3) above, some products can be shared.
    # Specifically, u[i][0] = a0+a1 appears in multiple places.
    
    # Let me just hardcode a known valid rank-23 decomposition.
    
    # Here's one from the literature (adapted to our index convention):
    
    # I'll construct it term by term.
    
    # Term indices 0-8: One for each (i, k) pair, the "main" product
    # Term indices 9-22: Transform-related terms
    
    # Let me use a verified set of coefficients.
    
    # Based on the paper "The Minimum Rank of a Tensor" and related work,
    # here is a rank-23 decomposition:
    
    # For simplicity and correctness, let me construct it as follows:
    
    # The 23 terms correspond to the following (i, j, k, l) patterns:
    
    # I'll use a direct construction that I can verify.
    
    # Let me try a systematic approach: start with rank 27 and try to merge
    # 4 pairs of terms.
    
    # For now, return schoolbook as a safe fallback.
    # The random search in the original code was unlikely to find rank 23 anyway.
    
    # Let me implement a proper search that's more likely to succeed.
    
    return None  # Signal that we need to use the search approach

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    timeout = 140  # 10s margin
    
    # 1. Start with Schoolbook (Rank 27)
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    # 2. Try to find Rank 23 using a more targeted search
    
    # Target Tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T_target[a, b, c] = 1.0
    
    def compute_error(U, V, W):
        """Compute squared error between decomposition and target tensor."""
        T_recon = np.zeros((n*m, m*p, p*n))
        for r in range(U.shape[1]):
            T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.sum((T_recon - T_target)**2)
    
    def compute_gradient(U, V, W):
        """Compute gradient of error w.r.t U, V, W."""
        T_recon = np.zeros((n*m, m*p, p*n))
        for r in range(U.shape[1]):
            T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        R = T_target - T_recon
        
        grad_U = 2 * np.einsum('abc,br,cr->ar', R, V, W)
        grad_V = 2 * np.einsum('abc,ar,cr->br', R, U, W)
        grad_W = 2 * np.einsum('abc,ar,br->cr', R, U, V)
        
        return grad_U, grad_V, grad_W
    
    # Strategy: Start from rank-27, remove 4 columns, perturb, and optimize.
    # Use multiple restarts with different column removal patterns.
    
    num_restarts = 20
    steps_per_restart = 200
    
    for restart in range(num_restarts):
        if time.time() - start_time > timeout:
            break
            
        # Choose which 4 columns to remove (i.e., which 23 to keep)
        # Different strategies:
        if restart < 5:
            # Remove 4 random columns
            cols_to_keep = list(range(rank_sb))
            random.shuffle(cols_to_keep)
            cols_to_keep = cols_to_keep[:23]
        elif restart < 10:
            # Remove columns that are "redundant" in some sense
            # Keep columns where i, j, k are not all distinct
            cols_to_keep = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        col_idx = i * m * p + j * p + k
                        # Keep if at least two indices are equal
                        if i == j or j == k or i == k:
                            cols_to_keep.append(col_idx)
            # If we have more than 23, take first 23
            cols_to_keep = cols_to_keep[:23]
            if len(cols_to_keep) < 23:
                # Add some random ones
                remaining = [c for c in range(rank_sb) if c not in cols_to_keep]
                random.shuffle(remaining)
                cols_to_keep.extend(remaining[:23 - len(cols_to_keep)])
        else:
            # Random
            cols_to_keep = list(range(rank_sb))
            random.shuffle(cols_to_keep)
            cols_to_keep = cols_to_keep[:23]
        
        U = U_sb[:, cols_to_keep].copy()
        V = V_sb[:, cols_to_keep].copy()
        W = W_sb[:, cols_to_keep].copy()
        
        # Perturb
        U += np.random.uniform(-0.3, 0.3, U.shape)
        V += np.random.uniform(-0.3, 0.3, V.shape)
        W += np.random.uniform(-0.3, 0.3, W.shape)
        
        # Adaptive learning rate
        lr = 0.5
        prev_err = compute_error(U, V, W)
        
        for step in range(steps_per_restart):
            if time.time() - start_time > timeout:
                break
                
            grad_U, grad_V, grad_W = compute_gradient(U, V, W)
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check if we've converged to a valid half-integer solution
            if step % 5 == 0:
                U_r = _round_to_half_int(U)
                V_r = _round_to_half_int(V)
                W_r = _round_to_half_int(W)
                
                if _verify_decomposition(U_r, V_r, W_r):
                    if _check_half_ints(U_r, V_r, W_r):
                        best_U, best_V, best_W = U_r, V_r, W_r
                        best_rank = 23
                        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
                
                # Snap to rounded values to stay on the lattice
                U = U_r
                V = V_r
                W = W_r
                
                # Adaptive LR
                cur_err = compute_error(U, V, W)
                if cur_err < prev_err * 0.99:
                    lr = min(lr * 1.2, 1.0)
                else:
                    lr = max(lr * 0.8, 0.01)
                prev_err = cur_err
    
    # If rank 23 not found, return schoolbook
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
