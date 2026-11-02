"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p, tol=1e-8):
    """Vectorized verification using numpy."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    # Build tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # Use einsum for efficiency
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    # T[i,j,k] = sum_r Ur[i,j,r] * Vr[j,k,r] * Wr[k,i,r]
    # This is a contraction over r with indices (i,j,r), (j,k,r), (k,i,r)
    # Use einsum: 'ijr,jkr,kir->ijk'
    T = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
    if np.max(np.abs(T - 1.0)) > tol:
        return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _try_block_strassen(n, m, p):
    """
    Try a block decomposition using Strassen for 2x2 blocks.
    Partition A (4x5) into 2x2 and 2x3 blocks, B (5x5) into 2x2, 2x3, 3x2, 3x3.
    Use Strassen (rank 7) for the 2x2*2x2 products, schoolbook for others.
    
    A = [A11(2x2) A12(2x3); A21(2x2) A22(2x3)]
    B = [B11(2x2) B12(2x3); B21(3x2) B22(3x3)]
    
    C11(2x2) = A11*B11 + A12*B21
    C12(2x3) = A11*B12 + A12*B22
    C21(2x2) = A21*B11 + A22*B21
    C22(2x3) = A21*B12 + A22*B22
    
    For A11*B11 and A21*B11 (2x2*2x2), use Strassen rank 7 each.
    For A12*B21 (2x3*3x2), schoolbook rank 6.
    For A11*B12 (2x2*2x3), schoolbook rank 6.
    For A12*B22 (2x3*3x3), schoolbook rank 9.
    For A22*B21 (2x3*3x2), schoolbook rank 6.
    For A21*B12 (2x2*2x3), schoolbook rank 6.
    For A22*B22 (2x3*3x3), schoolbook rank 9.
    
    Total: 7+6+7+6+6+9+6+9 = 56? No, that's not right.
    Let me recount. The products are:
    1. A11*B11: 2x2*2x2 -> Strassen rank 7
    2. A12*B21: 2x3*3x2 -> schoolbook rank 6
    3. A11*B12: 2x2*2x3 -> schoolbook rank 6
    4. A12*B22: 2x3*3x3 -> schoolbook rank 9
    5. A21*B11: 2x2*2x2 -> Strassen rank 7
    6. A22*B21: 2x3*3x2 -> schoolbook rank 6
    7. A21*B12: 2x2*2x3 -> schoolbook rank 6
    8. A22*B22: 2x3*3x3 -> schoolbook rank 9
    
    Total rank = 7+6+6+9+7+6+6+9 = 56
    
    But wait - the Strassen decomposition for 2x2*2x2 doesn't directly give
    a rank-7 decomposition of the tensor in the same coordinate system.
    I need to carefully map the Strassen bilinear forms to the global indices.
    
    This is complex. Let me try a different approach: just use the full
    Strassen decomposition for the entire 4x5 * 5x5 by padding to 6x6 or
    using a recursive block approach.
    
    Actually, let me try a simpler structural approach: use the fact that
    for 4x5*5x5, we can write it as a sum of products of submatrices and
    use known low-rank decompositions for each.
    
    For now, let me just implement the schoolbook and a random search.
    """
    return None  # Not implemented yet

def _random_search(n, m, p, target_rank, max_iters=200, deadline=None):
    """Try random integer/half-integer matrices to find a valid decomposition."""
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        U = [[random.choice([-2,-1,0,1,2]) for _ in range(target_rank)] for _ in range(n*m)]
        V = [[random.choice([-2,-1,0,1,2]) for _ in range(target_rank)] for _ in range(m*p)]
        W = [[random.choice([-2,-1,0,1,2]) for _ in range(target_rank)] for _ in range(p*n)]
        if _verify_decomposition(U, V, W, n, m, p):
            return U, V, W
    return None

def _gradient_descent(n, m, p, target_rank, max_iters=5000, lr=0.01, deadline=None):
    """
    Use gradient descent to find a low-rank decomposition.
    Minimize sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
    """
    U = np.random.randn(n*m, target_rank) * 0.1
    V = np.random.randn(m*p, target_rank) * 0.1
    W = np.random.randn(p*n, target_rank) * 0.1
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute gradient
        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # dT/dU[i*m+j,r] = V[j*p+k,r] * W[k*n+i,r]
        # Loss = sum_{i,j,k} (T[i,j,k] - 1)^2
        # dLoss/dU[i*m+j,r] = 2 * sum_k (T[i,j,k]-1) * V[j*p+k,r] * W[k*n+i,r]
        
        # Compute T
        Ur = U.reshape(n, m, target_rank)
        Vr = V.reshape(m, p, target_rank)
        Wr = W.reshape(p, n, target_rank)
        T = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
        err = T - 1.0
        
        # Gradients
        dU = 2 * np.einsum('ijk,jkr,kir->ijr', err, Vr, Wr)
        dV = 2 * np.einsum('ijk,ijr,kir->jkr', err, Ur, Wr)
        dW = 2 * np.einsum('ijk,ijr,jkr->kir', err, Ur, Vr)
        
        # Update
        U -= lr * dU.reshape(n*m, target_rank)
        V -= lr * dV.reshape(m*p, target_rank)
        W -= lr * dW.reshape(p*n, target_rank)
        
        # Periodic check
        if it % 100 == 0:
            T_check = np.einsum('ijr,jkr,kir->ijk', 
                               U.reshape(n,m,target_rank), 
                               V.reshape(m,p,target_rank), 
                               W.reshape(p,n,target_rank))
            max_err = np.max(np.abs(T_check - 1.0))
            if max_err < 1e-6:
                # Round to nearest half-integer and verify
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                T_final = np.einsum('ijr,jkr,kir->ijk',
                                    U_rounded.reshape(n,m,target_rank),
                                    V_rounded.reshape(m,p,target_rank),
                                    W_rounded.reshape(p,n,target_rank))
                if np.max(np.abs(T_final - 1.0)) < 1e-8:
                    return (U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist())
    
    return None

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Schoolbook baseline: rank = 100
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    deadline = time.time() + 130  # Leave 20s margin for the 150s limit
    
    # Try gradient descent for various ranks
    for target_rank in [70, 75, 80, 85, 90]:
        if time.time() > deadline - 30:
            break
        
        result = _gradient_descent(n, m, p, target_rank, max_iters=3000, lr=0.005, deadline=deadline - 30)
        if result is not None:
            U_r, V_r, W_r = result
            if _verify_decomposition(U_r, V_r, W_r, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U_r, V_r, W_r
                break  # Found a valid lower-rank decomposition
    
    # If gradient descent didn't find anything, try random search for slightly lower ranks
    if best_rank == n * m * p:
        for target_rank in [90, 85, 80]:
            if time.time() > deadline - 10:
                break
            result = _random_search(n, m, p, target_rank, max_iters=50, deadline=deadline - 10)
            if result is not None:
                U_r, V_r, W_r = result
                best_rank = target_rank
                best_U, best_V, best_W = U_r, V_r, W_r
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
