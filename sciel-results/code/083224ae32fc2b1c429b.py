import time
import sys
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for matrix multiplication
    of a 2x3 matrix by a 3x6 matrix.
    
    Strategy:
    1. The tensor M_{2,3,6} is the direct sum of two M_{2,3,3} tensors.
    2. We search for a rank-10 decomposition of M_{2,3,3}.
    3. If found, we duplicate it to get a rank-20 decomposition of M_{2,3,6}.
    4. We verify the result using half-integer rounding.
    5. If rank 10 fails, we try rank 11, 12, etc.
    6. Fallback to schoolbook rank 36.
    """
    start_time = time.time()
    deadline = start_time + 140  # 140 second deadline with margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_sb = [[0.0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0.0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0.0] * rank_schoolbook for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1.0
                V_sb[j * p + k][column] = 1.0
                W_sb[k * n + i][column] = 1.0
                column += 1
    
    def verify_decomposition(U, V, W, tol=1e-8):
        """Verify if U, V, W form an exact decomposition of the multiplication tensor."""
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(R))
                    if abs(actual - 1.0) > tol:
                        return False
        return True

    def verify_decomp_small(n_s, m_s, p_s, U, V, W, tol=1e-8):
        """Verify decomposition for small dimensions."""
        R = U.shape[1]
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    actual = sum(U[i*m_s+j, r] * V[j*p_s+k, r] * W[k*n_s+i, r] for r in range(R))
                    if abs(actual - 1.0) > tol:
                        return False
        return True

    def find_decomp_small(n_s, m_s, p_s, target_rank, max_iters=500, lr=0.01, seeds=5):
        """Find a rank-target_rank decomposition for M_{n_s, m_s, p_s}."""
        for seed in range(seeds):
            if time.time() > deadline - 5:
                return None
            
            rng = np.random.RandomState(seed)
            U = rng.randn(n_s * m_s, target_rank) * 0.5
            V = rng.randn(m_s * p_s, target_rank) * 0.5
            W = rng.randn(p_s * n_s, target_rank) * 0.5
            
            best_loss = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # Compute loss and gradients
                loss = 0.0
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n_s):
                    for j in range(m_s):
                        for k in range(p_s):
                            val = sum(U[i*m_s+j, r] * V[j*p_s+k, r] * W[k*n_s+i, r] for r in range(target_rank))
                            diff = val - 1.0
                            loss += diff * diff
                            for r in range(target_rank):
                                grad_U[i*m_s+j, r] += 2 * diff * V[j*p_s+k, r] * W[k*n_s+i, r]
                                grad_V[j*p_s+k, r] += 2 * diff * U[i*m_s+j, r] * W[k*n_s+i, r]
                                grad_W[k*n_s+i, r] += 2 * diff * U[i*m_s+j, r] * V[j*p_s+k, r]
                
                loss /= (n_s * m_s * p_s)
                
                if loss < best_loss:
                    best_loss = loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check convergence
                if loss < 1e-10:
                    # Try to round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    if verify_decomp_small(n_s, m_s, p_s, U_r, V_r, W_r, tol=1e-6):
                        return (U_r, V_r, W_r)
                
                # Adaptive learning rate
                if it % 50 == 0 and it > 0:
                    lr *= 0.9
            
            # Final check after loop
            if best_loss < 1e-6:
                U_r = np.round(best_U * 2) / 2
                V_r = np.round(best_V * 2) / 2
                W_r = np.round(best_W * 2) / 2
                if verify_decomp_small(n_s, m_s, p_s, U_r, V_r, W_r, tol=1e-6):
                    return (U_r, V_r, W_r)
        
        return None

    # Try to find rank 10 decomposition for M_{2,3,3}
    # M_{2,3,6} is M_{2,3,3} + M_{2,3,3}
    n_s, m_s, p_s = 2, 3, 3
    
    best_rank = rank_schoolbook
    best_decomp = (U_sb, V_sb, W_sb)
    
    # Try ranks 10, 11, 12 for the sub-problem
    for sub_rank in [10, 11, 12]:
        if time.time() > deadline - 10:
            break
        
        result = find_decomp_small(n_s, m_s, p_s, sub_rank, max_iters=300, seeds=5)
        if result is not None:
            U_sub, V_sub, W_sub = result
            
            # Construct M_{2,3,6} decomposition
            # U is same for both halves? No, U is for A (2x3).
            # V is for B (3x6). B is [B1, B2].
            # V_sub is 3x3. V_full should be 3x6.
            # V_full[:, 0:3] = V_sub, V_full[:, 3:6] = V_sub
            # W is for C (2x6). C is [C1, C2].
            # W_sub is 3x2 (p*n = 3*2).
            # W_full should be 6x2 (p*n = 6*2).
            # W_full[0:3, :] = W_sub, W_full[3:6, :] = W_sub
            
            total_rank = 2 * sub_rank
            
            U_full = np.zeros((n * m, total_rank))
            V_full = np.zeros((m * p, total_rank))
            W_full = np.zeros((p * n, total_rank))
            
            # U is the same for both blocks, so we just repeat the columns?
            # No, the tensor is a direct sum.
            # T = T1 + T2.
            # T1 uses indices for B1, C1. T2 uses indices for B2, C2.
            # The U matrix corresponds to A. A is shared.
            # In the tensor decomposition, each rank-1 component is u * v * w.
            # For T1, v is in the first 3 cols of B, w is in first 3 rows of C.
            # For T2, v is in the last 3 cols of B, w is in last 3 rows of C.
            # The u vectors are the same (from A).
            # So U_full[:, 0:sub_rank] = U_sub
            # U_full[:, sub_rank:2*sub_rank] = U_sub
            
            U_full[:, 0:sub_rank] = U_sub
            U_full[:, sub_rank:2*sub_rank] = U_sub
            
            # V_full:
            # V_sub is 3x3 (m x p_s).
            # V_full is 3x6 (m x p).
            # First 3 cols of V_full come from V_sub.
            # Last 3 cols of V_full come from V_sub.
            # But the rank components are separate.
            # Component r (0..sub_rank-1) uses V_sub[:, r] for B1.
            # Component r+sub_rank uses V_sub[:, r] for B2.
            
            V_full[:, 0:sub_rank] = V_sub
            # For the second half, the V part is zero for the first 3 cols, and V_sub for the last 3?
            # No.
            # Let's trace indices.
            # B is 3x6. Indices (j, k) where j in 0..2, k in 0..5.
            # B1 is k in 0..2. B2 is k in 3..5.
            # V_full[j, r] for r in 0..sub_rank-1:
            #   This component corresponds to T1.
            #   It should be non-zero only for k in 0..2?
            #   Actually, V is a matrix of size m x p.
            #   For T1, the vector v is in R^6. It has non-zeros in first 3 positions.
            #   v[j, r] = V_sub[j, r] for j in 0..2.
            #   v[j, r] = 0 for j in 0..2, k in 3..5? No, V is m x p.
            #   V[j, k].
            #   For r < sub_rank:
            #     V_full[j, k] = V_sub[j, k] if k < 3, else 0.
            #   For r >= sub_rank:
            #     V_full[j, k] = V_sub[j, k-3] if k >= 3, else 0.
            
            for r in range(sub_rank):
                for j in range(m):
                    V_full[j, r] = V_sub[j, r]
                    V_full[j, r + sub_rank] = 0
                    # Wait, V_full is m x p.
                    # V_full[j, k]
                    # For r in 0..sub_rank-1:
                    #   V_full[j, 0:3] = V_sub[j, r]
                    #   V_full[j, 3:6] = 0
                    # For r in sub_rank..2*sub_rank-1:
                    #   V_full[j, 0:3] = 0
                    #   V_full[j, 3:6] = V_sub[j, r-sub_rank]
                    
            # Let's re-construct V_full properly
            V_full = np.zeros((m * p, total_rank))
            for r in range(sub_rank):
                for j in range(m):
                    V_full[j * p + 0, r] = V_sub[j * p_s + 0, r]
                    V_full[j * p + 1, r] = V_sub[j * p_s + 1, r]
                    V_full[j * p + 2, r] = V_sub[j * p_s + 2, r]
                    V_full[j * p + 3, r] = 0
                    V_full[j * p + 4, r] = 0
                    V_full[j * p + 5, r] = 0
                    
                    V_full[j * p + 0, r + sub_rank] = 0
                    V_full[j * p + 1, r + sub_rank] = 0
                    V_full[j * p + 2, r + sub_rank] = 0
                    V_full[j * p + 3, r + sub_rank] = V_sub[j * p_s + 0, r]
                    V_full[j * p + 4, r + sub_rank] = V_sub[j * p_s + 1, r]
                    V_full[j * p + 5, r + sub_rank] = V_sub[j * p_s + 2, r]
            
            # W_full:
            # W is p x n.
            # W_sub is 3 x 2.
            # W_full is 6 x 2.
            # For r in 0..sub_rank-1:
            #   W_full[0:3, :] = W_sub[:, r]
            #   W_full[3:6, :] = 0
            # For r in sub_rank..2*sub_rank-1:
            #   W_full[0:3, :] = 0
            #   W_full[3:6, :] = W_sub[:, r-sub_rank]
            
            W_full = np.zeros((p * n, total_rank))
            for r in range(sub_rank):
                for k in range(p_s):
                    for i in range(n_s):
                        W_full[k * n_s + i, r] = W_sub[k * n_s + i, r]
                        W_full[(k + p_s) * n_s + i, r] = 0
                        
                        W_full[k * n_s + i, r + sub_rank] = 0
                        W_full[(k + p_s) * n_s + i, r + sub_rank] = W_sub[k * n_s + i, r]
            
            # Verify the full decomposition
            if verify_decomposition(U_full, V_full, W_full, tol=1e-6):
                best_rank = total_rank
                best_decomp = (U_full, V_full, W_full)
                break
    
    if best_rank < rank_schoolbook:
        U_out = best_decomp[0].tolist()
        V_out = best_decomp[1].tolist()
        W_out = best_decomp[2].tolist()
        return {"U": U_out, "V": V_out, "W": W_out}
    else:
        return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
