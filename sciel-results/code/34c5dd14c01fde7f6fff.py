import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 135

    n, m, p = 3, 3, 8
    schoolbook_rank = n * m * p  # 72

    # Build schoolbook as fallback
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = schoolbook_rank

    def check_full(U, V, W, rank):
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_np[i * m + j, :] * V_np[j * p + k, :] * W_np[k * n + i, :])
                    if abs(s - 1) > 1e-10:
                        return False
        return True

    # Try to find a low-rank decomposition for 3x3x3
    # Known: rank 8 is achievable for 3x3x3
    # Let's try a random search with structure
    
    def try_333_decomp(rank, max_iters=5000):
        """Try to find a rank-rank decomposition for 3x3x3."""
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            # Random initialization with small values
            U = np.random.randn(9, rank) * 0.5
            V = np.random.randn(9, rank) * 0.5
            W = np.random.randn(9, rank) * 0.5
            
            # Gradient descent to minimize sum of (sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - delta(i,j,k))^2
            # where delta is 1 if this is a valid term
            # Actually, we want: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            # Wait, for 3x3x3: n=m=p=3
            # The tensor T[i,j,k] = 1 for all i,j,k in range(3)
            # We need: T[i,j,k] = sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r]
            
            for inner_iter in range(200):
                if time.time() > deadline - 5:
                    return None
                # Compute gradients
                # F = sum over (i,j,k) of (pred[i,j,k] - 1)^2
                # pred[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
                
                # Use simple gradient step
                lr = 0.01 / (1 + it * 0.001)
                
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            pred = np.sum(U[i*3+j, :] * V[j*3+k, :] * W[k*3+i, :])
                            err = pred - 1.0
                            for r in range(rank):
                                grad_U[i*3+j, r] += 2 * err * V[j*3+k, r] * W[k*3+i, r]
                                grad_V[j*3+k, r] += 2 * err * U[i*3+j, r] * W[k*3+i, r]
                                grad_W[k*3+i, r] += 2 * err * U[i*3+j, r] * V[j*3+k, r]
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
            
            # Check if we found a valid decomposition
            valid = True
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        pred = np.sum(U[i*3+j, :] * V[j*3+k, :] * W[k*3+i, :])
                        if abs(pred - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return U.tolist(), V.tolist(), W.tolist()
        
        return None

    # Try rank 8 for 3x3x3
    result_333 = try_333_decomp(8, max_iters=3000)
    
    if result_333 is not None:
        U333, V333, W333 = result_333
        # Now extend to 3x3x8
        # We have two 3x3x3 blocks (k=0,1,2 and k=3,4,5) and one 3x3x2 block (k=6,7)
        # For the 3x3x2 part, we can use rank 6 (schoolbook for 3x3x2)
        
        rank_333 = 8
        rank_332 = 6  # schoolbook for 3x3x2
        total_rank = 2 * rank_333 + rank_332  # 22
        
        U_ext = [[0] * total_rank for _ in range(n * m)]
        V_ext = [[0] * total_rank for _ in range(m * p)]
        W_ext = [[0] * total_rank for _ in range(p * n)]
        
        # First 3x3x3 block: k in {0,1,2}
        for i in range(n):
            for j in range(m):
                for r in range(rank_333):
                    U_ext[i * m + j][r] = U333[i * m + j][r]
                    # V: j*p+k where k in 0,1,2 -> indices j*8+k
                    V_ext[j * p + k][r] = V333[j * 3 + k][r] if k < 3 else 0
                    # Wait, V333 is for 3x3, so V333[j*3+k, r] for k in 0,1,2
                    # But in the extended problem, V index is j*p+k = j*8+k
                    # For k in 0,1,2: V_ext[j*8+k, r] = V333[j*3+k, r]
                    # For k in 3,4,5: different block
                    # For k in 6,7: third block
                    pass
        
        # This is getting complicated. Let me restructure.
        
        # Actually, let me just do it more carefully
        U_ext = [[0] * total_rank for _ in range(n * m)]
        V_ext = [[0] * total_rank for _ in range(m * p)]
        W_ext = [[0] * total_rank for _ in range(p * n)]
        
        # Block 1: k in {0,1,2}, uses columns 0..7
        for i in range(n):
            for j in range(m):
                for r in range(rank_333):
                    U_ext[i * m + j][r] = U333[i * m + j][r]
        for j in range(m):
            for k in range(3):
                for r in range(rank_333):
                    V_ext[j * p + k][r] = V333[j * 3 + k][r]
        for k in range(3):
            for i in range(n):
                for r in range(rank_333):
                    W_ext[k * n + i][r] = W333[k * 3 + i][r]
        
        # Block 2: k in {3,4,5}, uses columns 8..15
        offset = rank_333
        for i in range(n):
            for j in range(m):
                for r in range(rank_333):
                    U_ext[i * m + j][offset + r] = U333[i * m + j][r]
        for j in range(m):
            for k in range(3):
                for r in range(rank_333):
                    V_ext[j * p + (k + 3)][offset + r] = V333[j * 3 + k][r]
        for k in range(3):
            for i in range(n):
                for r in range(rank_333):
                    W_ext[(k + 3) * n + i][offset + r] = W333[k * 3 + i][r]
        
        # Block 3: k in {6,7}, schoolbook rank 6, uses columns 16..21
        offset2 = 2 * rank_333
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(2):
                    U_ext[i * m + j][offset2 + col] = 1
                    V_ext[j * p + (k + 6)][offset2 + col] = 1
                    W_ext[(k + 6) * n + i][offset2 + col] = 1
                    col += 1
        
        if check_full(U_ext, V_ext, W_ext, total_rank):
            if total_rank < best_rank:
                best_result = {"U": U_ext, "V": V_ext, "W": W_ext}
                best_rank = total_rank
    else:
        # Fallback: try a simpler approach
        # Use rank 27 = 3*3*3 for each 3x3x3 block? No, that's schoolbook.
        # Let's just return schoolbook
        pass

    return best_result
