import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave margin for 150s limit

    # Target: 2x3x6
    # Strategy: Split B (3x6) into B1 (3x3) and B2 (3x3).
    # C = A * B1 + A * B2.
    # This is two independent 2x3x3 multiplications.
    # Rank of 2x3x3 is 9.
    # Total rank = 9 + 9 = 18.
    
    # We need a rank-9 decomposition for 2x3x3.
    # U_block (6x9), V_block (9x9), W_block (6x9)
    # such that sum_r U_block[:,r] V_block[:,r] W_block[:,r] = T_block
    
    # T_block indices:
    # A: (i, j) -> a = i*3 + j (0..5)
    # B: (j, k) -> b = j*3 + k (0..8)
    # C: (i, k) -> c = i*3 + k (0..5)
    
    T_block = np.zeros((6, 9, 6))
    for i in range(2):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = i * 3 + k
                T_block[a, b, c] = 1.0

    # Precompute reshaped T for ALS
    # For U update: T[a, (b,c)] -> shape (6, 54)
    T_reshape_U = T_block.reshape(6, 54)
    # For V update: T[b, (a,c)] -> shape (9, 36)
    T_reshape_V = T_block.transpose(1, 0, 2).reshape(9, 36)
    # For W update: T[c, (a,b)] -> shape (6, 54)
    T_reshape_W = T_block.transpose(2, 0, 1).reshape(6, 54)

    rank_target = 9
    best_U = None
    best_V = None
    best_W = None
    found = False
    
    max_restarts = 50 # More restarts are cheap now
    max_iters = 50

    for r in range(max_restarts):
        if time.time() > deadline - 10:
            break
            
        # Random initialization
        U = np.random.randn(6, rank_target)
        V = np.random.randn(9, rank_target)
        W = np.random.randn(6, rank_target)
        
        # Normalize to avoid scaling issues
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Update U
            # M_U shape (54, R). M_U[(b,c), r] = V[b,r] * W[c,r]
            # V is (9, R), W is (6, R)
            # We need to compute outer product of V and W along r?
            # M_U[:, r] = V[:, r] \otimes W[:, r]
            # This is a reshape of V * W?
            # V[:, r] is (9,). W[:, r] is (6,).
            # We want a (54, R) matrix.
            # Let's compute M_U = np.einsum('br,cr->bcr', V, W).reshape(54, R)
            M_U = np.einsum('br,cr->bcr', V, W).reshape(54, rank_target)
            # Solve T_reshape_U (6, 54) = M_U (54, R) @ U^T (R, 6)
            # U^T = lstsq(M_U, T_reshape_U.T)
            U_T, _, _, _ = np.linalg.lstsq(M_U, T_reshape_U.T, rcond=None)
            U = U_T.T
            
            # Update V
            # M_V shape (36, R). M_V[(a,c), r] = U[a,r] * W[c,r]
            M_V = np.einsum('ar,cr->acr', U, W).reshape(36, rank_target)
            V_T, _, _, _ = np.linalg.lstsq(M_V, T_reshape_V.T, rcond=None)
            V = V_T.T
            
            # Update W
            # M_W shape (54, R). M_W[(a,b), r] = U[a,r] * V[b,r]
            M_W = np.einsum('ar,br->abr', U, V).reshape(54, rank_target)
            W_T, _, _, _ = np.linalg.lstsq(M_W, T_reshape_W.T, rcond=None)
            W = W_T.T
            
            # Check error
            # T_rec = sum_r U[:,r] V[:,r] W[:,r]
            T_rec = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T_rec - T_block)
            
            if err < 1e-8:
                # Try to round to half-integers
                # The solution might be scaled.
                # Check if U, V, W are close to half-integers.
                # First, check if they are integers.
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                
                # Verify integer solution
                T_rec_int = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
                err_int = np.linalg.norm(T_rec_int - T_block)
                
                if err_int < 1e-5:
                    best_U = U_int
                    best_V = V_int
                    best_W = W_int
                    found = True
                    break
                
                # Try half-integers
                U_half = np.round(U * 2) / 2
                V_half = np.round(V * 2) / 2
                W_half = np.round(W * 2) / 2
                
                T_rec_half = np.einsum('ar,br,cr->abc', U_half, V_half, W_half)
                err_half = np.linalg.norm(T_rec_half - T_block)
                
                if err_half < 1e-5:
                    best_U = U_half
                    best_V = V_half
                    best_W = W_half
                    found = True
                    break
            
            if err < 1e-10:
                # Converged to a real solution, but not rounded yet?
                # Maybe the scaling is off.
                # If err is very small, we can try to scale the factors.
                # But ALS preserves scale somewhat.
                # Let's just keep searching for a rounded one.
                pass
                
        if found:
            break

    if found:
        # Construct full rank-18 solution
        U_full = np.zeros((6, 18))
        V_full = np.zeros((18, 18))
        W_full = np.zeros((12, 18))
        
        U_full[:, 0:9] = best_U
        U_full[:, 9:18] = best_U
        
        V_full[0:9, 0:9] = best_V
        V_full[9:18, 9:18] = best_V
        
        W_full[0:6, 0:9] = best_W
        W_full[6:12, 9:18] = best_W
        
        # Final verification
        T_target = np.zeros((6, 18, 12))
        for i in range(2):
            for j in range(3):
                for k in range(6):
                    a = i * 3 + j
                    b = j * 6 + k
                    c = i * 6 + k
                    T_target[a, b, c] = 1.0
                    
        T_full = np.einsum('ar,br,cr->abc', U_full, V_full, W_full)
        err_full = np.linalg.norm(T_full - T_target)
        
        if err_full < 1e-5:
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Fallback: Schoolbook Rank 36
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
