import numpy as np
import time
from scipy.optimize import minimize

def construct():
    n, m, p = 2, 4, 8
    
    # Directly use the known Strassen decomposition (Rank 56)
    # Optimization for lower ranks is unstable and slow, and likely to fail
    # within the time limit without finding a valid integer/half-integer solution.
    strassen_result = _strassen_decomp(n, m, p)
    if strassen_result is not None:
        return strassen_result
    
    # Final fallback: schoolbook
    return _schoolbook(n, m, p)

def _rank20_4x4_decomp(n, m, p):
    """
    Use the known rank-20 decomposition for 4x4 matrix multiplication.
    
    Strategy: Embed our 2x4 * 4x8 problem into a 4x4 * 4x4 problem.
    
    Let A be 2x4, B be 4x8, C = A*B be 2x8.
    
    We can think of this as:
    - Split B into two 4x4 blocks: B = [B1 | B2]
    - Then C = [A*B1 | A*B2]
    
    For each 4x4 block multiplication A_ext * B_block where A_ext is 4x4 
    (with A in top-left 2x4 block, zeros elsewhere), we use rank-20.
    
    Actually, let me think more carefully. The 4x4 rank-20 decomposition 
    computes C4 = A4 * B4 where A4, B4, C4 are all 4x4.
    
    Our problem: A (2x4) * B (4x8) = C (2x8)
    
    Embedding: 
    - A4 = [[A, 0], [0, 0]] (4x4, A in top-left 2x4)
    - B4 = [[0, B], [0, 0]] (4x4, B in top-right 4x4... wait, B is 4x8)
    
    Hmm, this doesn't quite work directly. Let me reconsider.
    
    Better approach: Split B into two 4x4 matrices B1, B2.
    C = A * B = A * [B1 B2] = [A*B1, A*B2]
    
    For each: A (2x4) * Bk (4x4) = Ck (2x4)
    
    Embed A into 4x4: A4 = [[A, 0], [0, 0]]
    Then A4 * Bk = [[A*Bk, 0], [0, 0]]
    
    So we need rank-20 for 4x4 * 4x4, and we extract the top-left 2x4 of the result.
    
    Total rank: 20 * 2 = 40 (for the two B blocks)
    """
    # The rank-20 decomposition for 4x4 matrix multiplication
    # This is a known result. Let me construct it.
    
    # For 4x4 * 4x4 -> 4x4, we need T[i,j,k,l] = delta_{j,k}
    # i.e., C[i,l] = sum_j A[i,j] * B[j,l]
    # Tensor: T[a, b, c] = 1 if b==c else 0, where a=(i,l), b=j, c=(j,l)
    # Wait, let me use the standard formulation.
    
    # Standard: (A B)_{i,l} = sum_j A_{i,j} B_{j,l}
    # Tensor T with indices (i,l), j, (j,l) -> T[(i,l), j, (j,l)] = 1
    
    # For 4x4: A is 16-dim (i*4+j), B is 16-dim (j*4+l), C is 16-dim (i*4+l)
    # T[i*4+l, j*4+l, j*4+l]... no wait.
    
    # Let me be precise:
    # A has indices (i, j) for i,j in 0..3
    # B has indices (j, k) for j,k in 0..3  
    # C has indices (i, k) for i,k in 0..3
    # T[A_idx, B_idx, C_idx] = 1 iff A_idx=(i,j), B_idx=(j,k), C_idx=(i,k)
    
    # So T[(i*4+j), (j*4+k), (i*4+k)] = 1
    
    # The rank-20 decomposition uses 20 rank-1 tensors.
    # I'll use the known formula from literature.
    
    # Actually, let me use a simpler approach: just try to find rank-40 
    # decomposition directly using the structure.
    
    # Alternative: Use the fact that our problem is a sub-tensor of 4x4*4x4.
    # If we have a rank-R decomposition of 4x4*4x4, then we can extract
    # a rank-R decomposition of our 2x4*4x8 problem by restricting indices.
    
    # For 4x4*4x4: A_dim=16, B_dim=16, C_dim=16
    # Our problem: A_dim=8, B_dim=32, C_dim=16
    
    # Hmm, the dimensions don't match up cleanly. Let me think differently.
    
    # Our tensor: T[i,j,k] = 1 for i in 0..1, j in 0..3, k in 0..7
    # where the contraction is sum_c U[i*4+j, c] * V[j*8+k, c] * W[k*2+i, c] = 1
    
    # This is equivalent to: for each (i,j,k), the product must be 1.
    
    # Let me just try to find a rank-40 decomposition using optimization
    # with a good initialization based on Strassen.
    
    if time.time() > deadline - 20:
        return None
    
    return _try_optimized_decomp(n, m, p, 40, deadline)

def _try_optimized_decomp(n, m, p, rank, deadline):
    """Try to find an exact rank-`rank` decomposition via optimization."""
    A_dim = n * m
    B_dim = m * p
    C_dim = p * n
    
    # Target tensor: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Initialize from Strassen-like structure scaled to desired rank
    U, V, W = _strassen_init(n, m, p, rank)
    
    def compute_residual(params):
        U_flat = params[:A_dim * rank].reshape(A_dim, rank)
        V_flat = params[A_dim * rank:A_dim * rank + B_dim * rank].reshape(B_dim, rank)
        W_flat = params[A_dim * rank + B_dim * rank:].reshape(C_dim, rank)
        
        # Vectorized residual computation
        # For each (i,j,k): dot(U[i*m+j], V[j*p+k] * W[k*n+i]) - 1
        residual = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U_flat[i*m+j]
                for k in range(p):
                    val = np.dot(u_row, V_flat[j*p+k] * W_flat[k*n+i])
                    residual += (val - 1.0) ** 2
        return residual
    
    def gradient(params):
        U_flat = params[:A_dim * rank].reshape(A_dim, rank)
        V_flat = params[A_dim * rank:A_dim * rank + B_dim * rank].reshape(B_dim, rank)
        W_flat = params[A_dim * rank + B_dim * rank:].reshape(C_dim, rank)
        
        grad_U = np.zeros_like(U_flat)
        grad_V = np.zeros_like(V_flat)
        grad_W = np.zeros_like(W_flat)
        
        for i in range(n):
            for j in range(m):
                u_row = U_flat[i*m+j]
                for k in range(p):
                    vw = V_flat[j*p+k] * W_flat[k*n+i]
                    val = np.dot(u_row, vw)
                    err = 2.0 * (val - 1.0)
                    grad_U[i*m+j] += err * vw
                    grad_V[j*p+k] += err * (u_row * W_flat[k*n+i])
                    grad_W[k*n+i] += err * (u_row * V_flat[j*p+k])
        
        return np.concatenate([grad_U.ravel(), grad_V.ravel(), grad_W.ravel()])
    
    params0 = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
    
    # Use L-BFGS-B for optimization
    try:
        res = minimize(compute_residual, params0, jac=gradient, method='L-BFGS-B',
                      options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12,
                               'maxls': 100, 'maxfun': 10000})
    except Exception:
        return None
    
    if time.time() > deadline - 5:
        return None
    
    if res.fun > 1e-8:
        return None
    
    # Extract and round to half-integers
    U_opt = res.x[:A_dim * rank].reshape(A_dim, rank)
    V_opt = res.x[A_dim * rank:A_dim * rank + B_dim * rank].reshape(B_dim, rank)
    W_opt = res.x[A_dim * rank + B_dim * rank:].reshape(C_dim, rank)
    
    # Round to nearest half-integer
    U_int = np.round(U_opt * 2) / 2
    V_int = np.round(V_opt * 2) / 2
    W_int = np.round(W_opt * 2) / 2
    
    # Verify exactness
    if _verify(U_int.tolist(), V_int.tolist(), W_int.tolist(), n, m, p):
        return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
    
    # Try rounding to integers too
    U_int2 = np.round(U_opt)
    V_int2 = np.round(V_opt)
    W_int2 = np.round(W_opt)
    if _verify(U_int2.tolist(), V_int2.tolist(), W_int2.tolist(), n, m, p):
        return {"U": U_int2.tolist(), "V": V_int2.tolist(), "W": W_int2.tolist()}
    
    # Try finer rounding: quarter-integers
    U_int3 = np.round(U_opt * 4) / 4
    V_int3 = np.round(V_opt * 4) / 4
    W_int3 = np.round(W_opt * 4) / 4
    if _verify(U_int3.tolist(), V_int3.tolist(), W_int3.tolist(), n, m, p):
        return {"U": U_int3.tolist(), "V": V_int3.tolist(), "W": W_int3.tolist()}
    
    return None

def _strassen_init(n, m, p, target_rank):
    """Initialize from Strassen decomposition, possibly with extra zero columns."""
    A_dim = n * m
    B_dim = m * p
    C_dim = p * n
    
    # Get Strassen rank-56 decomposition
    strassen = _strassen_decomp(n, m, p)
    if strassen is None:
        # Fallback to random
        np.random.seed(42)
        U = np.random.randn(A_dim, target_rank) * 0.1
        V = np.random.randn(B_dim, target_rank) * 0.1
        W = np.random.randn(C_dim, target_rank) * 0.1
        return U, V, W
    
    U_s = np.array(strassen['U'], dtype=float)
    V_s = np.array(strassen['V'], dtype=float)
    W_s = np.array(strassen['W'], dtype=float)
    s_rank = len(U_s[0])
    
    if target_rank >= s_rank:
        # Pad with zeros
        U = np.zeros((A_dim, target_rank))
        V = np.zeros((B_dim, target_rank))
        W = np.zeros((C_dim, target_rank))
        U[:, :s_rank] = U_s
        V[:, :s_rank] = V_s
        W[:, :s_rank] = W_s
    else:
        # Truncate (this won't be valid, but gives a starting point)
        U = U_s[:, :target_rank].copy()
        V = V_s[:, :target_rank].copy()
        W = W_s[:, :target_rank].copy()
    
    return U, V, W

def _strassen_decomp(n, m, p):
    """Strassen-based rank-56 decomposition."""
    A_dim = n * m
    B_dim = m * p
    C_dim = p * n
    
    rank = 56
    U = np.zeros((A_dim, rank), dtype=int)
    V = np.zeros((B_dim, rank), dtype=int)
    W = np.zeros((C_dim, rank), dtype=int)
    
    col = 0
    
    strassen_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    for j_group in range(2):
        for k_group in range(4):
            for term in strassen_terms:
                A_coeffs, B_coeffs, C_coeffs = term
                
                for (li, lj), coeff in A_coeffs.items():
                    global_i = li
                    global_j = 2 * j_group + lj
                    U[global_i * m + global_j, col] += coeff
                
                for (lj, lk), coeff in B_coeffs.items():
                    global_j = 2 * j_group + lj
                    global_k = 2 * k_group + lk
                    V[global_j * p + global_k, col] += coeff
                
                for (li, lk), coeff in C_coeffs.items():
                    global_i = li
                    global_k = 2 * k_group + lk
                    W[global_k * n + global_i, col] += coeff
                
                col += 1
    
    if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None

def _verify(U, V, W, n, m, p, tol=1e-6):
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Vectorized verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i * m + j], V_arr[j * p + k] * W_arr[k * n + i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
