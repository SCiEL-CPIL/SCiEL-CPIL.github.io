import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _schoolbook_decomposition(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
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
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition exactly represents matrix multiplication."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = np.sum(U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :])
                if abs(actual - expected) > tol:
                    return False
    return True

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    U: (4, 7) -> indices i*2+j
    V: (4, 7) -> indices j*2+k
    W: (4, 7) -> indices k*2+i
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Col 0: M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1
    U[3, 0] = 1
    V[0, 0] = 1
    V[3, 0] = 1
    W[0, 0] = 1
    W[3, 0] = 1
    
    # Col 1: M2 = (A10+A11)B00
    U[2, 1] = 1
    U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    W[3, 1] = -1
    
    # Col 2: M3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1
    V[3, 2] = -1
    W[1, 2] = 1
    W[3, 2] = 1
    
    # Col 3: M4 = A11(B00-B01)
    U[3, 3] = 1
    V[1, 3] = 1
    V[0, 3] = -1
    W[0, 3] = 1
    W[2, 3] = 1
    
    # Col 4: M5 = (A00+A01)B11
    U[0, 4] = 1
    U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1
    W[1, 4] = 1
    
    # Col 5: M6 = (A10-A00)(B00+B01)
    U[1, 5] = 1
    U[0, 5] = -1
    V[0, 5] = 1
    V[2, 5] = 1
    W[3, 5] = 1
    
    # Col 6: M7 = (A01-A11)(B10+B11)
    U[2, 6] = 1
    U[3, 6] = -1
    V[1, 6] = 1
    V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _construct_rank_23():
    """
    Construct rank-23 decomposition:
    - For j in {0,1}: use 3 Strassen 2x2x2 blocks (one per pair of k-values) -> 3*7 = 21
    - For j=2: use rank-2 decomposition for 2x1x6 -> 2
    Total rank = 23
    """
    n, m, p = DIMENSIONS
    rank_total = 23
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    col_offset = 0
    Strassen_U, Strassen_V, Strassen_W = _strassen_2x2x2()
    
    # For j in {0,1}, k in {0..5}: use Strassen blocks
    for block_idx in range(3):
        k_start = block_idx * 2
        for c in range(7):
            # U: i in {0,1}, j in {0,1} -> U[i*3+j, col]
            for i in range(2):
                for j in range(2):
                    U[i * 3 + j, col_offset] = Strassen_U[i * 2 + j, c]
            # V: j in {0,1}, k in {k_start, k_start+1} -> V[j*6+k, col]
            for j in range(2):
                for k in range(2):
                    V[j * 6 + (k_start + k), col_offset] = Strassen_V[j * 2 + k, c]
            # W: k in {k_start, k_start+1}, i in {0,1} -> W[k*2+i, col]
            for k in range(2):
                for i in range(2):
                    W[(k_start + k) * 2 + i, col_offset] = Strassen_W[k * 2 + i, c]
            col_offset += 1
    
    # For j=2: rank-2 decomposition of 2x1x6
    # U[2*3+i, c] for i in {0,1}, c in {0,1}
    # V[2*6+k, c] for k in {0..5}, c in {0,1}
    # W[k*2+i, c] for k in {0..5}, i in {0,1}, c in {0,1}
    # Need: sum_c U[6+i,c] * V[12+k,c] * W[2k+i,c] = 1 for all i,k
    
    # Solution: U[6,0]=1, U[7,1]=1, all others 0
    #           V[12+k,0]=1, V[12+k,1]=1 for all k
    #           W[2k+i,0]=1, W[2k+i,1]=1 for all k,i
    # Then for i=0: U[6,0]*V[12+k,0]*W[2k,0] + U[6,1]*V[12+k,1]*W[2k,1] = 1*1*1 + 0 = 1
    # For i=1: U[7,0]*V[12+k,0]*W[2k+1,0] + U[7,1]*V[12+k,1]*W[2k+1,1] = 0 + 1*1*1 = 1
    
    U[6, col_offset] = 1      # i=0, j=2
    U[7, col_offset + 1] = 1  # i=1, j=2
    for k in range(6):
        V[12 + k, col_offset] = 1
        V[12 + k, col_offset + 1] = 1
        W[2 * k + 0, col_offset] = 1
        W[2 * k + 1, col_offset] = 1
        W[2 * k + 0, col_offset + 1] = 1
        W[2 * k + 1, col_offset + 1] = 1
    
    col_offset += 2
    
    return U, V, W

def _try_rank_22_search(deadline):
    """
    Try to find a rank-22 decomposition.
    Strategy: Use numpy vectorized gradient descent on the tensor rank problem.
    The loss is sum over all (i,j,k) of (sum_c U[i*3+j,c]*V[j*6+k,c]*W[k*2+i,c] - 1)^2
    """
    n, m, p = DIMENSIONS
    rank = 22
    
    best_result = None
    best_loss = float('inf')
    
    for seed in range(20):
        if time.time() > deadline - 30:
            break
        
        rng = np.random.RandomState(seed)
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
        
        # Scale
        U /= np.sqrt(rank)
        V /= np.sqrt(rank)
        W /= np.sqrt(rank)
        
        lr = 0.05
        for iteration in range(2000):
            if time.time() > deadline - 25:
                break
            
            # Compute loss and gradients using vectorized operations
            # Build index arrays
            # For each (i,j,k), compute val = sum_c U[i*3+j,c] * V[j*6+k,c] * W[k*2+i,c]
            
            # Precompute index mappings
            i_idx = np.repeat(np.arange(n), m * p)
            j_idx = np.tile(np.repeat(np.arange(m), p), n)
            k_idx = np.tile(np.arange(p), n * m)
            
            u_idx = i_idx * m + j_idx  # shape (n*m*p,)
            v_idx = j_idx * p + k_idx  # shape (n*m*p,)
            w_idx = k_idx * n + i_idx  # shape (n*m*p,)
            
            # Compute all values: shape (n*m*p, rank)
            U_sel = U[u_idx, :]  # (36, rank)
            V_sel = V[v_idx, :]  # (36, rank)
            W_sel = W[w_idx, :]  # (36, rank)
            
            # val = sum over c of U_sel * V_sel * W_sel
            val = np.sum(U_sel * V_sel * W_sel, axis=1)  # (36,)
            err = val - 1.0  # (36,)
            
            loss = np.sum(err * err)
            
            if loss < best_loss:
                best_loss = loss
                if loss < 1e-10:
                    break
            
            # Gradients
            # dL/dU[u_idx[r], c] += 2 * err[r] * V_sel[r,c] * W_sel[r,c]
            # But multiple (i,j,k) can map to same u_idx, so we need to accumulate
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            # For U: grad_U[u_idx[r], c] += 2 * err[r] * V_sel[r,c] * W_sel[r,c]
            # This is: grad_U = np.zeros((n*m, rank))
            # For each r: grad_U[u_idx[r], :] += 2 * err[r] * V_sel[r, :] * W_sel[r, :]
            
            # Vectorized: use np.add.at
            dU = 2 * err[:, np.newaxis] * V_sel * W_sel  # (36, rank)
            dV = 2 * err[:, np.newaxis] * U_sel * W_sel  # (36, rank)
            dW = 2 * err[:, np.newaxis] * U_sel * V_sel  # (36, rank)
            
            np.add.at(grad_U, u_idx, dU)
            np.add.at(grad_V, v_idx, dV)
            np.add.at(grad_W, w_idx, dW)
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            if loss < 1e-12:
                break
            
            lr *= 0.999
        
        if best_loss < 1e-9:
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-6):
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                break
    
    return best_result

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # First, try the rank-23 construction
    U, V, W = _construct_rank_23()
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-8):
        # Try to find rank 22
        result_22 = _try_rank_22_search(deadline)
        if result_22 is not None:
            return result_22
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    return _schoolbook_decomposition(n, m, p)
# EVOLVE-BLOCK-END
