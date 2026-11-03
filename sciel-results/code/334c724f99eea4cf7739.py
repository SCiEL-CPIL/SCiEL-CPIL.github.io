"""Program to find rank-minimal exact tensor decomposition for 4x5 * 5x5 matrix multiplication."""

import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return U, V, W for 2x2 * 2x2 -> 2x2 with rank 7."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11 + A22)(B11 + B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[2, 0] = 1; W[1, 0] = -1; W[3, 0] = -1
    
    # M2 = (A21 + A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = A11(B21 - B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[0, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1
    
    # M4 = A22(B12 - B21)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[1, 3] = 1
    
    # M5 = (A11 + A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[3, 4] = -1
    
    # M6 = (A21 - A11)(B11 + B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12 - A22)(B21 + B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[2, 6] = 1
    
    return U, V, W


def _validate(U, V, W, tol=1e-9):
    """Validate that U, V, W form a valid tensor decomposition."""
    n, m, p = 4, 5, 5
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True


def _try_rank_r(R, n, m, p, time_budget=60.0):
    """
    Try to find a rank-R decomposition using randomized initialization 
    and gradient-based optimization. Returns (U, V, W) or None.
    """
    t_start = time.time()
    
    # Number of constraints: n*m*p
    num_constraints = n * m * p
    # Total parameters: 3 * R * (n*m + m*p + p*n) / 3 = R * (n*m + m*p + p*n)
    # Actually: U has n*m*R, V has m*p*R, W has p*n*R
    # Total params: R * (n*m + m*p + p*n)
    
    np.random.seed(42)
    
    for attempt in range(5):
        if time.time() - t_start > time_budget:
            break
        
        # Initialize with small random values
        U = np.random.randn(n*m, R) * 0.5
        V = np.random.randn(m*p, R) * 0.5
        W = np.random.randn(p*n, R) * 0.5
        
        # Use a simple gradient descent approach
        lr = 0.01
        best_loss = float('inf')
        
        for step in range(500):
            if time.time() - t_start > time_budget:
                break
            
            # Compute loss and gradients
            loss = 0.0
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Compute dot product
                        prod = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                        val = np.sum(prod)
                        diff = val - 1.0
                        loss += diff * diff
                        
                        # Gradient with respect to U[i*m+j, :]
                        # d/dU = 2 * diff * V[j*p+k, :] * W[k*n+i, :]
                        grad_U[i*m+j, :] += 2 * diff * V[j*p+k, :] * W[k*n+i, :]
                        # Gradient with respect to V[j*p+k, :]
                        grad_V[j*p+k, :] += 2 * diff * U[i*m+j, :] * W[k*n+i, :]
                        # Gradient with respect to W[k*n+i, :]
                        grad_W[k*n+i, :] += 2 * diff * U[i*m+j, :] * V[j*p+k, :]
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            if loss < best_loss:
                best_loss = loss
                if best_loss < 1e-10:
                    # Check if we've found a valid solution
                    if _validate(U, V, W):
                        return U, V, W
        
        # Check final result
        if _validate(U, V, W):
            return U, V, W
    
    return None


def _structured_decomposition():
    """
    Build a structured decomposition using block decomposition.
    Split 4x5*5x5 into blocks and use Strassen where beneficial.
    """
    n, m, p = 4, 5, 5
    
    U_list = []
    V_list = []
    W_list = []
    
    # Strategy: Split into 2x2 blocks for the 4x4 part, handle the 5th row/col separately
    # i in [0,4), j in [0,5), k in [0,5)
    
    # Block 1: i in [0,2), j in [0,2), k in [0,2) -> 2x2*2x2 -> Strassen rank 7
    # Block 2: i in [0,2), j in [0,2), k in [2,4) -> 2x2*2x2 -> Strassen rank 7
    # Block 3: i in [0,2), j in [0,2), k in [4,5) -> 2x2*2x1 -> rank 4
    # Block 4: i in [0,2), j in [2,4), k in [0,2) -> 2x2*2x2 -> Strassen rank 7
    # Block 5: i in [0,2), j in [2,4), k in [2,4) -> 2x2*2x2 -> Strassen rank 7
    # Block 6: i in [0,2), j in [2,4), k in [4,5) -> 2x2*2x1 -> rank 4
    # Block 7: i in [0,2), j in [4,5), k in [0,2) -> 2x1*2x2 -> rank 4
    # Block 8: i in [0,2), j in [4,5), k in [2,4) -> 2x1*2x2 -> rank 4
    # Block 9: i in [0,2), j in [4,5), k in [4,5) -> 2x1*1x2 -> rank 2
    # Block 10: i in [2,4), j in [0,2), k in [0,2) -> 2x2*2x2 -> Strassen rank 7
    # Block 11: i in [2,4), j in [0,2), k in [2,4) -> 2x2*2x2 -> Strassen rank 7
    # Block 12: i in [2,4), j in [0,2), k in [4,5) -> 2x2*2x1 -> rank 4
    # Block 13: i in [2,4), j in [2,4), k in [0,2) -> 2x2*2x2 -> Strassen rank 7
    # Block 14: i in [2,4), j in [2,4), k in [2,4) -> 2x2*2x2 -> Strassen rank 7
    # Block 15: i in [2,4), j in [2,4), k in [4,5) -> 2x2*2x1 -> rank 4
    # Block 16: i in [2,4), j in [4,5), k in [0,2) -> 2x1*2x2 -> rank 4
    # Block 17: i in [2,4), j in [4,5), k in [2,4) -> 2x1*2x2 -> rank 4
    # Block 18: i in [2,4), j in [4,5), k in [4,5) -> 2x1*1x2 -> rank 2
    
    i_splits = [(0, 2), (2, 2)]
    j_splits = [(0, 2), (2, 2), (4, 1)]
    k_splits = [(0, 2), (2, 2), (4, 1)]
    
    for i_off, i_sz in i_splits:
        for j_off, j_sz in j_splits:
            for k_off, k_sz in k_splits:
                if i_sz == 2 and j_sz == 2 and k_sz == 2:
                    # Use Strassen
                    Us, Vs, Ws = _strassen_2x2x2()
                    U_block = np.zeros((n*m, 7))
                    V_block = np.zeros((m*p, 7))
                    W_block = np.zeros((p*n, 7))
                    for r in range(7):
                        for ii in range(2):
                            for jj in range(2):
                                U_block[(i_off+ii)*m + (j_off+jj), r] = Us[ii*2+jj, r]
                        for jj in range(2):
                            for kk in range(2):
                                V_block[(j_off+jj)*p + (k_off+kk), r] = Vs[jj*2+kk, r]
                        for kk in range(2):
                            for ii in range(2):
                                W_block[(k_off+kk)*n + (i_off+ii), r] = Ws[kk*2+ii, r]
                    U_list.append(U_block)
                    V_list.append(V_block)
                    W_list.append(W_block)
                else:
                    # Schoolbook
                    rank = i_sz * j_sz * k_sz
                    U_block = np.zeros((n*m, rank))
                    V_block = np.zeros((m*p, rank))
                    W_block = np.zeros((p*n, rank))
                    c = 0
                    for ii in range(i_sz):
                        for jj in range(j_sz):
                            for kk in range(k_sz):
                                U_block[(i_off+ii)*m + (j_off+jj), c] = 1.0
                                V_block[(j_off+jj)*p + (k_off+kk), c] = 1.0
                                W_block[(k_off+kk)*n + (i_off+ii), c] = 1.0
                                c += 1
                    U_list.append(U_block)
                    V_list.append(V_block)
                    W_list.append(W_block)
    
    # Combine
    total_rank = sum(U.shape[1] for U in U_list)
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    c = 0
    for U_b, V_b, W_b in zip(U_list, V_list, W_list):
        r = U_b.shape[1]
        U[:, c:c+r] = U_b
        V[:, c:c+r] = V_b
        W[:, c:c+r] = W_b
        c += r
    
    return U, V, W


def _schoolbook():
    """Return the standard schoolbook decomposition with rank n*m*p."""
    n, m, p = 4, 5, 5
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1
                V[j*p+k, c] = 1
                W[k*n+i, c] = 1
                c += 1
    return U, V, W


def construct():
    n, m, p = 4, 5, 5
    t0 = time.time()
    
    # First, build the structured decomposition (rank 85)
    U_struct, V_struct, W_struct = _structured_decomposition()
    rank_struct = U_struct.shape[1]
    
    # Validate the structured decomposition
    if not _validate(U_struct, V_struct, W_struct):
        # Fallback to schoolbook
        U, V, W = _schoolbook()
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    best_U, best_V, best_W = U_struct, V_struct, W_struct
    best_rank = rank_struct
    
    # Try to find a lower rank decomposition using optimization
    # Start from rank_struct - 1 and go down
    t_opt_start = time.time()
    opt_time_budget = 90.0  # Leave 30s margin
    
    for target_rank in range(rank_struct - 1, max(rank_struct - 15, 70), -1):
        if time.time() - t_opt_start > opt_time_budget:
            break
        
        result = _try_rank_r(target_rank, n, m, p, time_budget=min(opt_time_budget - (time.time() - t_opt_start), 10.0))
        
        if result is not None:
            U, V, W = result
            if _validate(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break
    
    # Final validation
    if not _validate(best_U, best_V, best_W):
        # Fallback to schoolbook
        U, V, W = _schoolbook()
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
