"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    # Build the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This should equal 1 for all (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _strassen_2x2_decomposition():
    """
    Return the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    U shape (4, 7), V shape (4, 7), W shape (4, 7)
    Indices: U[i*2+j, c] for i,j in {0,1}, c in {0..6}
             V[j*2+k, c] for j,k in {0,1}, c in {0..6}
             W[k*2+i, c] for k,i in {0,1}, c in {0..6}
    """
    # Strassen's 7 products for 2x2 multiplication
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    #
    # M1 = (a11 + a22)(b11 + b22)
    # M2 = (a21 + a22)b11
    # M3 = a11(b12 - b22)
    # M4 = a22(b21 - b11)
    # M5 = (a11 + a12)b22
    # M6 = (a21 - a11)(b11 + b12)
    # M7 = (a12 - a22)(b21 + b22)
    #
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    # U rows: (i,j) pairs -> [0,0], [0,1], [1,0], [1,1]
    # V rows: (j,k) pairs -> [0,0], [0,1], [1,0], [1,1]
    # W rows: (k,i) pairs -> [0,0], [0,1], [1,0], [1,1]
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a11 + a22)(b11 + b22)
    # U: a11 (row 0) gets +1, a22 (row 3) gets +1
    # V: b11 (row 0) gets +1, b22 (row 3) gets +1
    # W: contributes to c11 (+1), c22 (+1)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # c11 and c22
    
    # M2 = (a21 + a22)b11
    # U: a21 (row 2) gets +1, a22 (row 3) gets +1
    # V: b11 (row 0) gets +1
    # W: contributes to c21 (+1), c22 (-1)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1  # c21 and c22
    
    # M3 = a11(b12 - b22)
    # U: a11 (row 0) gets +1
    # V: b12 (row 1) gets +1, b22 (row 3) gets -1
    # W: contributes to c12 (+1), c22 (+1)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1  # c12 and c22
    
    # M4 = a22(b21 - b11)
    # U: a22 (row 3) gets +1
    # V: b21 (row 2) gets +1, b11 (row 0) gets -1
    # W: contributes to c11 (+1), c21 (+1)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1  # c11 and c21
    
    # M5 = (a11 + a12)b22
    # U: a11 (row 0) gets +1, a12 (row 1) gets +1
    # V: b22 (row 3) gets +1
    # W: contributes to c11 (-1), c12 (+1)
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1  # c11 and c12
    
    # M6 = (a21 - a11)(b11 + b12)
    # U: a21 (row 2) gets +1, a11 (row 0) gets -1
    # V: b11 (row 0) gets +1, b12 (row 1) gets +1
    # W: contributes to c22 (+1)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # c22
    
    # M7 = (a12 - a22)(b21 + b22)
    # U: a12 (row 1) gets +1, a22 (row 3) gets -1
    # V: b21 (row 2) gets +1, b22 (row 3) gets +1
    # W: contributes to c11 (+1)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # c11
    
    return U.tolist(), V.tolist(), W.tolist()

def _try_strassen_block(n, m, p, deadline):
    """
    Try to use Strassen's algorithm with block decomposition.
    
    For 3x5 * 5x6, we can partition as:
    - Split rows of A (3x5) into blocks: [2x5, 1x5]
    - Split columns of B (5x6) into blocks: [5x2, 5x4]
    
    Then C (3x6) is partitioned into:
    - [2x2, 2x4; 1x2, 1x4]
    
    The 2x2 block of C = (2x5) * (5x2) can use Strassen if we further split.
    But 5 and 2 don't divide evenly for 2x2 Strassen.
    
    Alternative: Use a direct approach with the known rank-7 Strassen for the
    overlapping parts and schoolbook for the rest.
    
    Let's try a different strategy: use the fact that we can decompose the
    multiplication into smaller subproblems.
    
    Actually, let's try a more systematic approach:
    Partition A into row blocks and B into column blocks.
    
    For A (3x5): split into A1 (2x5) and A2 (1x5)
    For B (5x6): split into B1 (5x3) and B2 (5x3)
    
    Then C (3x6) = [A1*B1, A1*B2; A2*B1, A2*B2]
    
    Each subproblem:
    - A1*B1: 2x5 * 5x3 -> 2x3
    - A1*B2: 2x5 * 5x3 -> 2x3
    - A2*B1: 1x5 * 5x3 -> 1x3
    - A2*B2: 1x5 * 5x3 -> 1x3
    
    For 2x5 * 5x3, we can use a rank-? decomposition.
    The schoolbook rank is 2*5*3 = 30.
    
    This doesn't immediately help. Let me try a different approach.
    
    Let's just try to find a good decomposition using numerical optimization
    with better initialization and constraints.
    """
    return None

def _try_rank_r_optimized(n, m, p, r, max_time=8, seed=42):
    """
    Try to find a rank-r decomposition using optimized gradient descent
    with multiple random restarts and adaptive learning rate.
    """
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    best_loss = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    # Try multiple restarts
    num_restarts = min(5, max(1, int((max_time - (time.time() - start_time)) / 2)))
    
    for restart in range(num_restarts):
        if time.time() - start_time > max_time - 1:
            break
        
        # Initialize with small random values
        scale = 0.5 / np.sqrt(r)
        U = rng.randn(n*m, r) * scale
        V = rng.randn(m*p, r) * scale
        W = rng.randn(p*n, r) * scale
        
        lr = 0.1
        best_local_loss = float('inf')
        no_improve_count = 0
        
        for iteration in range(3000):
            if time.time() - start_time > max_time - 0.5:
                break
            
            # Compute loss and gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            loss = 0.0
            for i in range(n):
                for j in range(m):
                    ui = U[i*m+j, :]
                    for k in range(p):
                        vj = V[j*p+k, :]
                        wk = W[k*n+i, :]
                        val = np.dot(ui, vj * wk)
                        err = val - 1.0
                        loss += err * err
                        
                        # Gradients
                        grad_U[i*m+j, :] += 2 * err * vj * wk
                        grad_V[j*p+k, :] += 2 * err * ui * wk
                        grad_W[k*n+i, :] += 2 * err * ui * vj
            
            # Adaptive learning rate
            if loss < best_local_loss * 0.95:
                lr = min(lr * 1.1, 1.0)
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > 50:
                    lr *= 0.9
                    no_improve_count = 0
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            if loss < best_local_loss:
                best_local_loss = loss
                best_local_U = U.copy()
                best_local_V = V.copy()
                best_local_W = W.copy()
            
            if best_local_loss < 1e-10:
                break
        
        if best_local_loss < best_loss:
            best_loss = best_local_loss
            best_U = best_local_U
            best_V = best_local_V
            best_W = best_local_W
        
        if best_loss < 1e-8:
            break
    
    if best_loss < 1e-6 and best_U is not None:
        # Try rounding to half-integers
        for round_factor in [2, 4, 10]:
            U_rounded = np.round(best_U * round_factor) / round_factor
            V_rounded = np.round(best_V * round_factor) / round_factor
            W_rounded = np.round(best_W * round_factor) / round_factor
            
            if _verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
                return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
        
        # Also try without rounding (in case exact real values work)
        if _verify_decomposition(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p):
            return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Lower bound for rank is max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # Try ranks from just above lower bound up to schoolbook
    # Focus on the most promising range first
    
    # Try a range of ranks, starting from lower values
    # Based on known results for small matrix multiplication,
    # ranks in the 40-70 range might be achievable
    
    ranks_to_try = list(range(35, 65)) + list(range(65, 91, 5))
    # Sort by likelihood (try middle range first)
    ranks_to_try = sorted(set(ranks_to_try), key=lambda r: abs(r - 50))
    
    for r in ranks_to_try:
        if time.time() > deadline - 15:
            break
        
        if r < 30:  # Below theoretical lower bound
            continue
        
        if r >= best_rank:
            continue
        
        result = _try_rank_r_optimized(n, m, p, r, max_time=10, seed=42 + r)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_result = {"U": U, "V": V, "W": W}
            # Don't break - try to find even better
            # But stop if we've hit the lower bound
            if r <= 32:
                break
    
    return best_result
# EVOLVE-BLOCK-END
