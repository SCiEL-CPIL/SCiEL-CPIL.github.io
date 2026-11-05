import numpy as np
import time
from fractions import Fraction
import itertools

DIMENSIONS = [2, 3, 5]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                s = 0.0
                for r in range(rank):
                    s += U_arr[idx_u, r] * V_arr[idx_v, r] * W_arr[idx_w, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_rank28():
    """Build the known rank-28 decomposition."""
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: j=2, k=0..4 (10 terms)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 2: j=0,1, k=4 (4 terms)
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
            
    # Part 3: Strassen for j=0,1, k=0,1 (7 terms)
    U[0][r] = 1; U[4][r] = 1
    V[0][r] = 1; V[6][r] = 1
    W[0][r] = 1; W[3][r] = 1
    r += 1
    U[3][r] = 1; U[4][r] = 1
    V[0][r] = 1
    W[1][r] = 1; W[3][r] = -1
    r += 1
    U[0][r] = 1
    V[1][r] = 1; V[6][r] = -1
    W[2][r] = 1; W[3][r] = 1
    r += 1
    U[4][r] = 1
    V[5][r] = 1; V[0][r] = -1
    W[0][r] = 1; W[1][r] = 1
    r += 1
    U[0][r] = 1; U[1][r] = 1
    V[6][r] = 1
    W[0][r] = -1; W[2][r] = 1
    r += 1
    U[3][r] = 1; U[0][r] = -1
    V[0][r] = 1; V[1][r] = 1
    W[3][r] = 1
    r += 1
    U[1][r] = 1; U[4][r] = -1
    V[5][r] = 1; V[6][r] = 1
    W[0][r] = 1
    r += 1
    
    # Part 4: Strassen for j=0,1, k=2,3 (7 terms)
    U[0][r] = 1; U[4][r] = 1
    V[2][r] = 1; V[8][r] = 1
    W[4][r] = 1; W[7][r] = 1
    r += 1
    U[3][r] = 1; U[4][r] = 1
    V[2][r] = 1
    W[5][r] = 1; W[7][r] = -1
    r += 1
    U[0][r] = 1
    V[3][r] = 1; V[8][r] = -1
    W[6][r] = 1; W[7][r] = 1
    r += 1
    U[4][r] = 1
    V[7][r] = 1; V[2][r] = -1
    W[4][r] = 1; W[5][r] = 1
    r += 1
    U[0][r] = 1; U[1][r] = 1
    V[8][r] = 1
    W[4][r] = -1; W[6][r] = 1
    r += 1
    U[3][r] = 1; U[0][r] = -1
    V[2][r] = 1; V[3][r] = 1
    W[7][r] = 1
    r += 1
    U[1][r] = 1; U[4][r] = -1
    V[7][r] = 1; V[8][r] = 1
    W[4][r] = 1
    r += 1
    
    return U, V, W

def _build_tensor_from_decomposition(U, V, W, n, m, p):
    """Build the 3D tensor T[i,j,k] from the decomposition."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i,j,k] = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
    return T

def _try_reduce_rank(U, V, W, n, m, p, target_rank, time_limit=60.0):
    """
    Try to find a decomposition with target_rank using iterative optimization.
    Uses a smooth optimization approach: minimize the sum of squared residuals
    while encouraging sparsity in the factor matrices.
    """
    start_time = time.time()
    T = _build_tensor_from_decomposition(U, V, W, n, m, p)
    
    # Initialize with random values scaled appropriately
    np.random.seed(42)
    U_opt = np.random.randn(n * m, target_rank) * 0.5
    V_opt = np.random.randn(m * p, target_rank) * 0.5
    W_opt = np.random.randn(p * n, target_rank) * 0.5
    
    # Use Adam-like optimizer with momentum
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    mU = np.zeros_like(U_opt)
    vU = np.zeros_like(U_opt)
    mV = np.zeros_like(V_opt)
    vV = np.zeros_like(V_opt)
    mW = np.zeros_like(W_opt)
    vW = np.zeros_like(W_opt)
    
    t = 0
    best_loss = float('inf')
    best_U = U_opt.copy()
    best_V = V_opt.copy()
    best_W = W_opt.copy()
    
    for iteration in range(50000):
        if time.time() - start_time > time_limit:
            break
        
        t += 1
        lr_t = lr * np.sqrt(1 - beta2**t) / (1 - beta1**t)
        
        # Compute loss and gradients
        # Loss = sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - T[i,j,k])^2
        U_curr = U_opt.copy()
        V_curr = V_opt.copy()
        W_curr = W_opt.copy()
        
        # Compute predictions
        pred = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    pred[i,j,k] = np.dot(U_curr[i*m+j, :], V_curr[j*p+k, :] * W_curr[k*n+i, :])
        
        residual = pred - T
        loss = np.sum(residual**2)
        
        if loss < best_loss:
            best_loss = loss
            best_U = U_curr.copy()
            best_V = V_curr.copy()
            best_W = W_curr.copy()
            
        if best_loss < 1e-12:
            break
        
        # Compute gradients
        dU = np.zeros_like(U_opt)
        dV = np.zeros_like(V_opt)
        dW = np.zeros_like(W_opt)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    res = residual[i,j,k]
                    # d/dU[i*m+j,r] = 2 * res * V[j*p+k,r] * W[k*n+i,r]
                    dU[i*m+j, :] += 2 * res * V_curr[j*p+k, :] * W_curr[k*n+i, :]
                    # d/dV[j*p+k,r] = 2 * res * U[i*m+j,r] * W[k*n+i,r]
                    dV[j*p+k, :] += 2 * res * U_curr[i*m+j, :] * W_curr[k*n+i, :]
                    # d/dW[k*n+i,r] = 2 * res * U[i*m+j,r] * V[j*p+k,r]
                    dW[k*n+i, :] += 2 * res * U_curr[i*m+j, :] * V_curr[j*p+k, :]
        
        # Update with Adam
        mU = beta1 * mU + (1 - beta1) * dU
        vU = beta2 * vU + (1 - beta2) * dU**2
        U_opt -= lr_t * mU / (np.sqrt(vU) + eps)
        
        mV = beta1 * mV + (1 - beta1) * dV
        vV = beta2 * vV + (1 - beta2) * dV**2
        V_opt -= lr_t * mV / (np.sqrt(vV) + eps)
        
        mW = beta1 * mW + (1 - beta1) * dW
        vW = beta2 * vW + (1 - beta2) * dW**2
        W_opt -= lr_t * mW / (np.sqrt(vW) + eps)
        
        # Periodically try to snap to simple values (0, ±1, ±0.5)
        if iteration % 1000 == 0 and best_loss < 0.1:
            # Try rounding to nearest half-integer
            U_rounded = np.round(U_opt * 2) / 2
            V_rounded = np.round(V_opt * 2) / 2
            W_rounded = np.round(W_opt * 2) / 2
            
            pred_r = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        pred_r[i,j,k] = np.dot(U_rounded[i*m+j, :], V_rounded[j*p+k, :] * W_rounded[k*n+i, :])
            loss_r = np.sum((pred_r - T)**2)
            
            if loss_r < 1e-9:
                return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), True
        
        # Check if we're making progress
        if iteration % 5000 == 0 and best_loss > 1e-6:
            # Try a different random restart
            np.random.seed(iteration)
            U_opt = np.random.randn(n * m, target_rank) * 0.3
            V_opt = np.random.randn(m * p, target_rank) * 0.3
            W_opt = np.random.randn(p * n, target_rank) * 0.3
            mU = np.zeros_like(U_opt)
            vU = np.zeros_like(U_opt)
            mV = np.zeros_like(V_opt)
            vV = np.zeros_like(V_opt)
            mW = np.zeros_like(W_opt)
            vW = np.zeros_like(W_opt)
            t = 0
    
    # Final check
    if best_loss < 1e-9:
        return best_U.tolist(), best_V.tolist(), best_W.tolist(), True
    
    return None, None, None, False

def _try_structural_reduction(U, V, W, n, m, p, target_rank, time_limit=30.0):
    """
    Try structural approaches: exploit the block structure of the 2x3x5 problem.
    The key insight: we can try different ways to partition the k=0..4 dimension.
    """
    start_time = time.time()
    T = _build_tensor_from_decomposition(U, V, W, n, m, p)
    
    # Try different block structures for the k dimension
    # Current: k=0,1 (Strassen 7), k=2,3 (Strassen 7), k=4 (schoolbook 4), j=2 (schoolbook 10)
    # Alternative: try to share more structure
    
    # Approach: use the fact that for each j, we have a 2x2xp_j subproblem
    # For j=0: need to compute 2x2 @ 2x3 (k=0,1,4) -> rank at least 10
    # For j=1: same structure
    # For j=2: 2x2 @ 2x5 -> rank at least 10
    
    # The minimum possible rank is max(10, 10, 10) = 10 if we could share perfectly,
    # but we can't because the k indices differ.
    
    # Let's try a more careful approach: enumerate possible rank-27 decompositions
    # by trying to merge pairs of terms from the rank-28 solution
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = len(U[0])
    
    # Try removing each pair of terms and see if the remaining 26 terms + 1 new term works
    # This is expensive, so we'll try a smarter approach
    
    # Key insight: if two rank-1 terms have similar structure, we might be able to
    # replace them with a single term using the identity:
    # (a+b)(c+d)(e+f) = ac*e + ad*f + bc*f + bd*e
    # But this increases rank, not decreases it.
    
    # Better approach: use the fact that the tensor has a specific algebraic structure
    # and try to find a decomposition with the minimal number of terms
    
    # For now, let's try the optimization approach with the target rank
    result = _try_reduce_rank(U, V, W, n, m, p, target_rank, time_limit)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Start with the known rank-28 decomposition
    U, V, W = _build_rank28()
    
    # Verify the base decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Base decomposition verification failed")
    
    best_U, best_V, best_W = U, V, W
    best_rank = len(U[0])
    
    # Try to reduce rank from 27 down
    for target_rank in range(27, 14, -1):
        if time.time() > deadline - 30:
            break
        
        print(f"Trying rank {target_rank}...")
        U_opt, V_opt, W_opt, success = _try_structural_reduction(U, V, W, n, m, p, target_rank, time_limit=20.0)
        
        if success:
            if _verify_decomposition(U_opt, V_opt, W_opt, n, m, p):
                best_U, best_V, best_W = U_opt, V_opt, W_opt
                best_rank = target_rank
                print(f"Found rank {target_rank} decomposition!")
                break
        else:
            print(f"Rank {target_rank} failed")
    
    return {"U": best_U, "V": best_V, "W": best_W}
