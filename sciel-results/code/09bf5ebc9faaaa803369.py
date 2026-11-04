import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

def get_strassen_2x2():
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Term 1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # Term 3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # Term 4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # Term 5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # Term 6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Term 7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def construct_4x4x4_strassen():
    """Construct rank-49 decomposition for 4x4x4 matrix multiplication via Kronecker product of 2x2 Strassen."""
    U2, V2, W2 = get_strassen_2x2()
    
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    I = (i1*2+i2)*4 + (j1*2+j2)
                    u_idx1 = i1*2+j1
                    u_idx2 = i2*2+j2
                    for t1 in range(7):
                        for t2 in range(7):
                            R = t1*7+t2
                            U4[I][R] = U2[u_idx1][t1] * U2[u_idx2][t2]
                    
                    for k1 in range(2):
                        for k2 in range(2):
                            J = (j1*2+j2)*4 + (k1*2+k2)
                            v_idx1 = j1*2+k1
                            v_idx2 = j2*2+k2
                            for t1 in range(7):
                                for t2 in range(7):
                                    R = t1*7+t2
                                    V4[J][R] = V2[v_idx1][t1] * V2[v_idx2][t2]
                    
                    for k1 in range(2):
                        for k2 in range(2):
                            K = (k1*2+k2)*4 + (i1*2+i2)
                            w_idx1 = k1*2+i1
                            w_idx2 = k2*2+i2
                            for t1 in range(7):
                                for t2 in range(7):
                                    R = t1*7+t2
                                    W4[K][R] = W2[w_idx1][t1] * W2[w_idx2][t2]
    
    return U4, V4, W4

def verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the n x m x p matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for t in range(rank):
                    s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def search_4x4x3_low_rank(deadline, seed=42):
    """Search for a low-rank decomposition of the 4x4x3 matrix multiplication tensor.
    Returns (U, V, W, rank) or None if not found within deadline."""
    n, m, p = 4, 4, 3
    rng = np.random.RandomState(seed)
    
    # Start with schoolbook rank 48 and try to reduce
    # The tensor T[i,j,k] = 1 for all (i,j,k) in [0,4)x[0,4)x[0,3)
    # We want: sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] = 1 for all (i,j,k)
    
    best_U, best_V, best_W, best_rank = None, None, None, 48
    
    # Try various ranks from low to high
    for target_rank in range(24, 49):
        if time.time() > deadline:
            break
        
        # Multiple random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Random initialization with small integers
            U = np.zeros((16, target_rank))
            V = np.zeros((12, target_rank))
            W = np.zeros((12, target_rank))
            
            # Initialize with random values in {-1, 0, 1}
            for t in range(target_rank):
                U[:, t] = rng.choice([-1, 0, 1], size=16)
                V[:, t] = rng.choice([-1, 0, 1], size=12)
                W[:, t] = rng.choice([-1, 0, 1], size=12)
            
            # Iterative refinement: minimize residual
            # Residual[i,j,k] = sum_t U[i,j,t]*V[j,k,t]*W[k,i,t] - 1
            for iteration in range(100):
                if time.time() > deadline:
                    break
                
                # Compute current residual
                residual = np.zeros((4, 4, 3))
                for t in range(target_rank):
                    u_t = U.reshape(4, 4, -1)[:, :, t]
                    v_t = V.reshape(4, 3, -1)[:, :, t]
                    w_t = W.reshape(3, 4, -1)[:, :, t]
                    # residual[i,j,k] += U[i,j,t] * V[j,k,t] * W[k,i,t]
                    for i in range(4):
                        for j in range(4):
                            for k in range(3):
                                residual[i, j, k] += U[i*4+j, t] * V[j*3+k, t] * W[k*4+i, t]
                
                residual -= 1.0
                
                # Check if we've converged
                max_res = np.max(np.abs(residual))
                if max_res < 1e-9:
                    # Found exact decomposition!
                    return U.astype(int).tolist(), V.astype(int).tolist(), W.astype(int).tolist(), target_rank
                
                # Update U: for each (i,j), we want to minimize
                # sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] - residual[i,j,k]
                # This is a linear system: for fixed j, U[i,j,:] * (V[j,:,t]*W[:,i,t]) = target
                # Let's do gradient descent
                lr = 0.1 / (1 + iteration)
                
                # Update U
                dU = np.zeros_like(U)
                for i in range(4):
                    for j in range(4):
                        for k in range(3):
                            for t in range(target_rank):
                                dU[i*4+j, t] += residual[i,j,k] * V[j*3+k, t] * W[k*4+i, t]
                U -= lr * dU
                
                # Update V
                dV = np.zeros_like(V)
                for i in range(4):
                    for j in range(4):
                        for k in range(3):
                            for t in range(target_rank):
                                dV[j*3+k, t] += residual[i,j,k] * U[i*4+j, t] * W[k*4+i, t]
                V -= lr * dV
                
                # Update W
                dW = np.zeros_like(W)
                for i in range(4):
                    for j in range(4):
                        for k in range(3):
                            for t in range(target_rank):
                                dW[k*4+i, t] += residual[i,j,k] * U[i*4+j, t] * V[j*3+k, t]
                W -= lr * dW
                
                # Round to nearest integer/half-integer periodically
                if iteration % 10 == 0 and max_res < 1.0:
                    # Try rounding
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    # Check if rounded version is better
                    res_rounded = 0.0
                    for i in range(4):
                        for j in range(4):
                            for k in range(3):
                                s = 0.0
                                for t in range(target_rank):
                                    s += U_rounded[i*4+j, t] * V_rounded[j*3+k, t] * W_rounded[k*4+i, t]
                                res_rounded += (s - 1.0)**2
                    
                    res_current = np.sum(residual**2)
                    if res_rounded < res_current:
                        U, V, W = U_rounded, V_rounded, W_rounded
            
            # After refinement, check if we have a good solution
            residual_final = np.zeros((4, 4, 3))
            for t in range(target_rank):
                for i in range(4):
                    for j in range(4):
                        for k in range(3):
                            residual_final[i, j, k] += U[i*4+j, t] * V[j*3+k, t] * W[k*4+i, t]
            residual_final -= 1.0
            
            if np.max(np.abs(residual_final)) < 1e-9:
                return U.astype(int).tolist(), V.astype(int).tolist(), W.astype(int).tolist(), target_rank
            
            # Track best
            if best_U is None or target_rank < best_rank:
                best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), target_rank
    
    return None  # No low-rank decomposition found

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Part 1: 4x4x4 Strassen, rank 49
    U4, V4, W4 = construct_4x4x4_strassen()
    
    # Part 2: Try to find low-rank decomposition for 4x4x3
    U2, V2, W2, rank2 = None, None, None, 48
    
    if time.time() < deadline - 30:  # Leave time for assembly
        result = search_4x4x3_low_rank(deadline - 30, seed=42)
        if result is not None:
            U2, V2, W2, rank2 = result
    
    if U2 is None:
        # Fall back to schoolbook for 4x4x3
        rank2 = 48
        U2 = [[0]*rank2 for _ in range(16)]
        V2 = [[0]*rank2 for _ in range(12)]
        W2 = [[0]*rank2 for _ in range(12)]
        
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    U2[i*4+j][col] = 1
                    V2[j*3+k][col] = 1
                    W2[k*4+i][col] = 1
                    col += 1
    
    total_rank = 49 + rank2
    
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(28)]
    W = [[0]*total_rank for _ in range(28)]
    
    # Fill U
    for i in range(16):
        for t in range(49):
            U[i][t] = U4[i][t]
        for t in range(rank2):
            U[i][49+t] = U2[i][t]
    
    # Fill V: V4 row j*4+k -> V row j*7+k (k=0..3)
    for j in range(4):
        for k in range(4):
            v4_row = j*4+k
            v_row = j*7+k
            for t in range(49):
                V[v_row][t] = V4[v4_row][t]
    
    # V2 row j*3+k' -> V row j*7+(4+k') (k'=0..2)
    for j in range(4):
        for k in range(3):
            v2_row = j*3+k
            v_row = j*7+(4+k)
            for t in range(rank2):
                V[v_row][49+t] = V2[v2_row][t]
    
    # Fill W: W4 row k*4+i -> W row k*4+i (k=0..3)
    for k in range(4):
        for i in range(4):
            w4_row = k*4+i
            w_row = k*4+i
            for t in range(49):
                W[w_row][t] = W4[w4_row][t]
    
    # W2 row k'*4+i -> W row (4+k')*4+i (k'=0..2)
    for k in range(3):
        for i in range(4):
            w2_row = k*4+i
            w_row = (4+k)*4+i
            for t in range(rank2):
                W[w_row][49+t] = W2[w2_row][t]
    
    # Quick verification
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        ok = True
        for i in range(4):
            for j in range(4):
                for k in range(7):
                    s = 0
                    for t in range(total_rank):
                        s += U[i*4+j][t] * V[j*7+k][t] * W[k*4+i][t]
                    if abs(s - 1) > 1e-9:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if not ok:
            # Fallback to full schoolbook
            rank = 4*4*7
            U_fb = [[0]*rank for _ in range(16)]
            V_fb = [[0]*rank for _ in range(28)]
            W_fb = [[0]*rank for _ in range(28)]
            c = 0
            for i in range(4):
                for j in range(4):
                    for k in range(7):
                        U_fb[i*4+j][c] = 1
                        V_fb[j*7+k][c] = 1
                        W_fb[k*4+i][c] = 1
                        c += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
