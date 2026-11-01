"""Matrix multiplication 2,2,11 - rank-minimal decomposition via structured search."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 11]

def _build_target_tensor():
    """Build the multiplication tensor T of shape (n*m, m*p, p*n)."""
    n, m, p = 2, 2, 11
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, T=None):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    if T is None:
        T = _build_target_tensor()
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T, T_decomp, atol=1e-9)

def _build_strassen_blocks():
    """
    Build a decomposition using Strassen for 2x2 blocks.
    For 2x2 * 2x11: use Strassen on pairs of columns.
    
    11 columns = 5 pairs + 1 single column
    Each Strassen block (2x2 * 2x2) has rank 7
    Each single column (2x2 * 2x1) has rank 2 (schoolbook)
    
    Total: 5*7 + 2 = 37
    """
    n, m, p = 2, 2, 11
    num_pairs = p // 2  # 5 pairs
    remainder = p % 2   # 1 remaining column
    
    rank = num_pairs * 7 + remainder * 2
    U = np.zeros((n * m, rank), dtype=np.float64)  # (4, rank)
    V = np.zeros((m * p, rank), dtype=np.float64)  # (22, rank)
    W = np.zeros((p * n, rank), dtype=np.float64)  # (22, rank)
    
    # Strassen terms for 2x2 * 2x2:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # U coefficients (index = i*2+j for A[i,j]):
    strassen_U = [
        [1, 0, 0, 1],  # P1: A00+A11
        [0, 0, 1, 1],  # P2: A10+A11
        [1, 0, 0, 0],  # P3: A00
        [0, 0, 0, 1],  # P4: A11
        [1, 1, 0, 0],  # P5: A00+A01
        [-1, 0, 1, 0], # P6: A10-A00
        [0, 1, -1, 0], # P7: A01-A11
    ]
    
    # V coefficients (index = j*2+k for B[j,k]):
    strassen_V = [
        [1, 0, 0, 1],  # P1: B00+B11
        [1, 0, 0, 0],  # P2: B00
        [0, 0, 1, -1], # P3: B10-B11
        [-1, 0, 0, 1], # P4: B01-B00
        [0, 0, 0, 1],  # P5: B11
        [1, 0, 0, 1],  # P6: B00+B01
        [0, 0, 1, 1],  # P7: B10+B11
    ]
    
    # W reconstruction coefficients for C[i,k]:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    strassen_W = [
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, -1, 1, 0, 0, 1, 0],   # C11
    ]
    
    col = 0
    for b in range(num_pairs):
        for t in range(7):
            # U: same for all blocks
            U[:, col] = strassen_U[t]
            
            # V: block b uses columns 2b and 2b+1
            for j in range(2):
                for kp in range(2):
                    local_idx = j * 2 + kp
                    global_idx = j * p + (2 * b + kp)
                    V[global_idx, col] = strassen_V[t][local_idx]
            
            # W: block b uses output columns 2b and 2b+1
            for i in range(2):
                for kp in range(2):
                    local_idx = i * 2 + kp
                    global_idx = (2 * b + kp) * n + i
                    W[global_idx, col] = strassen_W[local_idx][t]
            
            col += 1
    
    # Handle remainder column (column 10)
    if remainder == 1:
        # Schoolbook for single column: C[i,10] = A[i,0]*B[0,10] + A[i,1]*B[1,10]
        # Term 1: A[i,0]*B[0,10]
        U[0, col] = 1  # A00
        U[2, col] = 1  # A10
        V[0 * p + 10, col] = 1  # B[0,10]
        W[10 * n + 0, col] = 1  # C[0,10]
        W[10 * n + 1, col] = 1  # C[1,10]
        col += 1
        
        # Term 2: A[i,1]*B[1,10]
        U[1, col] = 1  # A01
        U[3, col] = 1  # A11
        V[1 * p + 10, col] = 1  # B[1,10]
        W[10 * n + 0, col] = 1  # C[0,10]
        W[10 * n + 1, col] = 1  # C[1,10]
        col += 1
    
    return U, V, W

def _build_schoolbook():
    """Build the schoolbook rank-44 decomposition."""
    n, m, p = 2, 2, 11
    rank = n * m * p  # 44
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W

def _als_cp_decomposition(T, rank, max_iter=200, tol=1e-10, seed=42):
    """
    CP decomposition via alternating least squares.
    T: 3D tensor of shape (I, J, K)
    Returns U, V, W matrices of shapes (I, rank), (J, rank), (K, rank)
    """
    I, J, K = T.shape
    rng = np.random.RandomState(seed)
    
    # Initialize with random Gaussian
    U = rng.randn(I, rank)
    V = rng.randn(J, rank)
    W = rng.randn(K, rank)
    
    for it in range(max_iter):
        # Update U: for each r, solve least squares
        for r in range(rank):
            # T contracted with V[:, r] and W[:, r]
            vec = np.einsum('ijk,j,k->i', T, V[:, r], W[:, r])
            denom = np.dot(V[:, r], V[:, r]) * np.dot(W[:, r], W[:, r])
            if abs(denom) > 1e-15:
                U[:, r] = vec / denom
            else:
                U[:, r] = vec
        
        # Update V
        for r in range(rank):
            vec = np.einsum('ijk,i,k->j', T, U[:, r], W[:, r])
            denom = np.dot(U[:, r], U[:, r]) * np.dot(W[:, r], W[:, r])
            if abs(denom) > 1e-15:
                V[:, r] = vec / denom
            else:
                V[:, r] = vec
        
        # Update W
        for r in range(rank):
            vec = np.einsum('ijk,i,j->k', T, U[:, r], V[:, r])
            denom = np.dot(U[:, r], U[:, r]) * np.dot(V[:, r], V[:, r])
            if abs(denom) > 1e-15:
                W[:, r] = vec / denom
            else:
                W[:, r] = vec
        
        # Check convergence
        T_decomp = np.einsum('ir,jr,kr->ijk', U, V, W)
        err = np.linalg.norm(T - T_decomp) / (np.linalg.norm(T) + 1e-15)
        if err < tol:
            break
    
    return U, V, W, err

def _round_to_half_integers(U, V, W, T, max_rounds=5):
    """
    Try to round the ALS solution to nearest half-integers while maintaining exactness.
    """
    I, J, K = T.shape
    rank = U.shape[1]
    
    # First, try direct rounding
    U_rounded = np.round(U * 2) / 2
    V_rounded = np.round(V * 2) / 2
    W_rounded = np.round(W * 2) / 2
    
    T_decomp = np.einsum('ir,jr,kr->ijk', U_rounded, V_rounded, W_rounded)
    if np.allclose(T, T_decomp, atol=1e-9):
        return U_rounded, V_rounded, W_rounded
    
    # Try iterative refinement: round one matrix at a time
    for iteration in range(max_rounds):
        # Round U
        U_try = np.round(U * 2) / 2
        T_decomp = np.einsum('ir,jr,kr->ijk', U_try, V, W)
        if np.allclose(T, T_decomp, atol=1e-9):
            U = U_try
            continue
        
        # Round V
        V_try = np.round(V * 2) / 2
        T_decomp = np.einsum('ir,jr,kr->ijk', U, V_try, W)
        if np.allclose(T, T_decomp, atol=1e-9):
            V = V_try
            continue
        
        # Round W
        W_try = np.round(W * 2) / 2
        T_decomp = np.einsum('ir,jr,kr->ijk', U, V, W_try)
        if np.allclose(T, T_decomp, atol=1e-9):
            W = W_try
            continue
        
        # If none worked, try rounding all together with small perturbations
        break
    
    return U, V, W

def _try_rank_reduction_als(T, start_rank, target_rank, time_limit=60):
    """
    Try to find a rank decomposition using ALS from start_rank down to target_rank.
    """
    start_time = time.time()
    I, J, K = T.shape
    
    for rank in range(start_rank, target_rank - 1, -1):
        if time.time() - start_time > time_limit:
            break
        
        # Try multiple random initializations
        best_U, best_V, best_W = None, None, None
        best_err = np.inf
        
        for seed in range(5):
            if time.time() - start_time > time_limit:
                break
            
            U, V, W, err = _als_cp_decomposition(T, rank, max_iter=100, tol=1e-12, seed=seed)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Check if we got an exact solution
        if best_err < 1e-10:
            # Try to round to half-integers
            U_rounded, V_rounded, W_rounded = _round_to_half_integers(best_U, best_V, best_W, T)
            
            T_decomp = np.einsum('ir,jr,kr->ijk', U_rounded, V_rounded, W_rounded)
            if np.allclose(T, T_decomp, atol=1e-9):
                return U_rounded, V_rounded, W_rounded, rank
        
        # If ALS converged to near-zero error but rounding failed, try with original
        if best_err < 1e-10:
            T_decomp = np.einsum('ir,jr,kr->ijk', best_U, best_V, best_W)
            if np.allclose(T, T_decomp, atol=1e-9):
                # This is valid but may not have half-integer entries
                # Check if entries are close to half-integers
                all_half = True
                for mat in [best_U, best_V, best_W]:
                    rounded = np.round(mat * 2) / 2
                    if not np.allclose(mat, rounded, atol=1e-6):
                        all_half = False
                        break
                if all_half:
                    return best_U, best_V, best_W, rank
    
    return None, None, None, None

def construct():
    start_time = time.time()
    DEADLINE = 140  # seconds, leave margin before 150s limit
    
    T = _build_target_tensor()
    
    # First, try the Strassen-based rank-37 construction
    U, V, W = _build_strassen_blocks()
    
    if _verify(U.tolist(), V.tolist(), W.tolist(), T):
        # Now try to reduce rank further using ALS
        if time.time() - start_time < DEADLINE - 30:
            U_als, V_als, W_als, rank_als = _try_rank_reduction_als(T, 37, 25, time_limit=min(30, DEADLINE - (time.time() - start_time)))
            if U_als is not None and rank_als < 37:
                return {'U': U_als.tolist(), 'V': V_als.tolist(), 'W': W_als.tolist()}
        
        # Return rank 37 if no improvement found
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    # Fall back to schoolbook rank 44
    U, V, W = _build_schoolbook()
    if _verify(U.tolist(), V.tolist(), W.tolist(), T):
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    # Should not reach here
    raise RuntimeError("No valid decomposition found")
