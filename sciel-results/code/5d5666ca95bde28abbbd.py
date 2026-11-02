"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _build_schoolbook():
    n, m, p = DIMENSIONS
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

def _verify_decomposition(U, V, W, n, m, p):
    """Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(total - 1.0) > 1e-6:
                    return False
    return True

def _verify_integer_half_integer(U, V, W, tol=1e-9):
    """Check that all entries are integer or half-integer (multiple of 0.5)."""
    for name, M in [("U", U), ("V", V), ("W", W)]:
        for row in M:
            for val in row:
                if abs(val * 2 - round(val * 2)) > tol:
                    return False
    return True

def _hadamard_2x2():
    """Return the 2x2 Hadamard matrix."""
    return [[1, 1], [1, -1]]

def _hadamard_4x4():
    """Return the 4x4 Hadamard matrix using Kronecker product of 2x2 Hadamards."""
    H2 = _hadamard_2x2()
    H4 = [[0] * 4 for _ in range(4)]
    for i in range(2):
        for j in range(2):
            for a in range(2):
                for b in range(2):
                    H4[i*2+a][j*2+b] = H2[i][j] * H2[a][b]
    return H4

def _build_4x4x4_hadamard():
    """Build a 4x4x4 decomposition using Hadamard transform approach.
    
    The idea is to use the Hadamard transform to diagonalize the multiplication
    tensor, potentially reducing the rank below Strassen's 49.
    
    For 4x4x4, we can use the fact that the Hadamard transform H satisfies
    H * H^T = 4 * I, which allows us to express matrix multiplication in a
    transformed basis where fewer terms may be needed.
    
    This approach constructs the decomposition by:
    1. Applying Hadamard transforms to A, B, and C
    2. Exploiting the sparsity structure in the transformed domain
    3. Converting back to the original basis
    """
    n = 4
    H = _hadamard_4x4()
    H_inv = [[val / 4.0 for val in row] for row in H]
    
    # The rank will be determined by the number of non-zero terms in the
    # transformed multiplication. For now, we'll use a systematic construction.
    
    # Initialize with schoolbook rank and try to optimize
    rank = 49  # Start with Strassen rank
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    # Construct using Hadamard-inspired patterns
    # Each term corresponds to a combination of Hadamard basis vectors
    
    # For each pair of Hadamard basis indices, create a term
    # The Hadamard transform has 4 basis vectors, so we can have 4*4 = 16
    # independent combinations, but we need to account for the full multiplication
    
    # Use a systematic approach: for each (i, j) pair in the transformed domain,
    # create a term that contributes to the multiplication
    
    # Actually, let's use a different strategy: construct the decomposition
    # directly using the algebraic structure of the Hadamard transform
    
    # The key insight: if we can find a set of rank-1 tensors that sum to the
    # multiplication tensor, we're done. The Hadamard transform gives us a
    # natural set of candidates.
    
    # For now, return the Strassen-based decomposition as a fallback
    return _build_4x4x4_strassen()

def _build_4x4x4_strassen():
    """Build a 4x4x4 decomposition using recursive Strassen (rank 49).
    
    The key insight: for 4x4x4, we decompose into 2x2x2 blocks.
    Each 2x2x2 block uses 7 Strassen terms.
    The outer 2x2x2 structure also uses 7 terms.
    Total: 7 * 7 = 49 terms.
    
    For each outer term (t1) and inner term (t2):
    - U factor: product of outer U (which selects 2x2 blocks of A) and inner U (which is the Strassen U for a 2x2 block)
    - Similarly for V and W.
    """
    n = 4
    terms_2x2 = _strassen_2x2()
    
    U = [[0] * 49 for _ in range(16)]  # 4*4 = 16
    V = [[0] * 49 for _ in range(16)]  # 4*4 = 16
    W = [[0] * 49 for _ in range(16)]  # 4*4 = 16
    
    for t1, term1 in enumerate(terms_2x2):
        for t2, term2 in enumerate(terms_2x2):
            idx = t1 * 7 + t2
            
            # U: 4x4 matrix, indexed by (i,j) where i,j in 0..3
            # term1['U'] gives the 2x2 block structure for the outer decomposition
            # term2['U'] gives the inner 2x2 Strassen U pattern
            for (bi, bj), coeff_u in term1['U'].items():
                for (li, lj), coeff_v in term2['U'].items():
                    i = bi * 2 + li
                    j = bj * 2 + lj
                    U[i*4 + j][idx] += coeff_u * coeff_v
            
            # V: 4x4 matrix, indexed by (j,k) where j,k in 0..3
            for (bj, bk), coeff_v in term1['V'].items():
                for (lj, lk), coeff_w in term2['V'].items():
                    j = bj * 2 + lj
                    k = bk * 2 + lk
                    V[j*4 + k][idx] += coeff_v * coeff_w
            
            # W: 4x4 matrix, indexed by (k,i) where k,i in 0..3
            for (bk, bi), coeff_w in term1['W'].items():
                for (lk, li), coeff_u2 in term2['W'].items():
                    k = bk * 2 + lk
                    i = bi * 2 + li
                    W[k*4 + i][idx] += coeff_w * coeff_u2
    
    return {"U": U, "V": V, "W": W}

def _strassen_2x2():
    """Return the 7 terms of Strassen's 2x2x2 matrix multiplication."""
    terms = []
    terms.append({'U': {(0,0): 1, (1,1): 1}, 'V': {(0,0): 1, (1,1): 1}, 'W': {(0,0): 1, (1,1): 1}})
    terms.append({'U': {(1,0): 1, (1,1): 1}, 'V': {(0,0): 1}, 'W': {(1,0): 1}})
    terms.append({'U': {(0,0): 1}, 'V': {(1,0): 1, (1,1): -1}, 'W': {(0,1): 1, (1,1): -1}})
    terms.append({'U': {(1,1): 1}, 'V': {(0,1): 1, (0,0): -1}, 'W': {(0,0): -1, (1,0): 1}})
    terms.append({'U': {(0,0): 1, (0,1): 1}, 'V': {(1,1): 1}, 'W': {(0,1): 1, (1,1): 1}})
    terms.append({'U': {(1,0): 1, (0,0): -1}, 'V': {(0,0): 1, (0,1): 1}, 'W': {(0,0): 1, (1,0): 1}})
    terms.append({'U': {(0,1): 1, (1,1): -1}, 'V': {(1,0): 1, (1,1): 1}, 'W': {(0,0): 1, (1,1): 1}})
    return terms

def _build_4x4x5():
    """Build a 4x4x5 decomposition: 4x4x4 (rank 49) + 4x4x1 (rank 4) = rank 53.
    
    The 4x4x4 part handles C[0:4, 0:4] = A @ B[0:4, 0:4].
    The 4x4x1 part handles C[0:4, 4] = A @ B[0:4, 4].
    """
    n, m, p = DIMENSIONS
    
    # Get 4x4x4 part
    base = _build_4x4x4_strassen()
    rank_base = 49
    
    # Total rank: 49 + 4 = 53
    rank = rank_base + 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Copy 4x4x4 part
    for i in range(n * m):
        for r in range(rank_base):
            U[i][r] = base["U"][i][r]
    for j in range(m * 4):  # First 4 columns of B (indices 0..15)
        for r in range(rank_base):
            V[j][r] = base["V"][j][r]
    for k in range(4 * n):  # First 4 rows of C (k=0..3, indices 0..15)
        for r in range(rank_base):
            W[k][r] = base["W"][k][r]
    
    # Add 4x4x1 part: C[i,4] = sum_j A[i,j] * B[j,4] for i=0..3
    # This requires m=4 terms (one per j), using schoolbook for this slice
    for j in range(m):
        r = rank_base + j  # One term per j
        for i in range(n):
            U[i*m + j][r] = 1
        V[j*p + 4][r] = 1  # B[j,4]
        for i in range(n):
            W[4*n + i][r] = 1  # C[i,4] where k=4, so index = 4*4+i = 16+i
    
    return {"U": U, "V": V, "W": W}

def _build_4x4x5_winslow():
    """Try a Winograd-inspired approach for 4x4x5 using Hadamard transforms.
    
    The Winograd framework maps matrix multiplication to a convolution problem.
    By using Hadamard transforms, we can exploit algebraic symmetries to
    potentially reduce the tensor rank.
    
    For 4x4x5, we can:
    1. Treat the 4x4x4 part using a Hadamard-based decomposition
    2. Handle the 4x4x1 part separately
    3. Look for opportunities to merge or optimize terms
    """
    n, m, p = DIMENSIONS
    
    # Try the Hadamard-based approach for the 4x4x4 part
    # For now, fall back to Strassen if Hadamard doesn't yield improvement
    base = _build_4x4x4_hadamard()
    rank_base = len(base["U"][0])
    
    # If Hadamard gives same or better rank, use it; otherwise use Strassen
    if rank_base > 49:
        base = _build_4x4x4_strassen()
        rank_base = 49
    
    # Total rank
    rank = rank_base + 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Copy base part
    for i in range(n * m):
        for r in range(rank_base):
            U[i][r] = base["U"][i][r]
    for j in range(m * 4):
        for r in range(rank_base):
            V[j][r] = base["V"][j][r]
    for k in range(4 * n):
        for r in range(rank_base):
            W[k][r] = base["W"][k][r]
    
    # Add 4x4x1 part
    for j in range(m):
        r = rank_base + j
        for i in range(n):
            U[i*m + j][r] = 1
        V[j*p + 4][r] = 1
        for i in range(n):
            W[4*n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}

def _try_optimize_rank(rank, n, m, p, max_iter=2000, seed=42):
    """Try to find a decomposition of given rank using gradient optimization."""
    np.random.seed(seed)
    
    U = np.random.randn(n*m, rank) * 0.5
    V = np.random.randn(m*p, rank) * 0.5
    W = np.random.randn(p*n, rank) * 0.5
    
    def loss_and_grad(params):
        U_p = params[:n*m*rank].reshape(n*m, rank)
        V_p = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W_p = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_p[i*m+j, r] * V_p[j*p+k, r] * W_p[k*n+i, r]
                    loss += (s - 1.0) ** 2
        
        return loss
    
    params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    from scipy.optimize import minimize
    result = minimize(loss_and_grad, params, method='Nelder-Mead', 
                      options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10})
    
    if result.fun < 1e-4:
        U_opt = result.x[:n*m*rank].reshape(n*m, rank)
        V_opt = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W_opt = result.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        U_list = U_opt.tolist()
        V_list = V_opt.tolist()
        W_list = W_opt.tolist()
        if _verify_decomposition(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin
    
    n, m, p = DIMENSIONS
    
    # Try the Winograd/Hadamard-based decomposition first
    best_result = None
    try:
        result = _build_4x4x5_winslow()
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            best_result = result
    except Exception:
        pass
    
    # If that didn't work, try the standard Strassen-based approach
    if best_result is None:
        try:
            result = _build_4x4x5()
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                best_result = result
        except Exception:
            pass
    
    # Try to find lower rank through optimization
    if best_result is not None and time.time() < deadline - 60:
        current_rank = len(best_result["U"][0])
        for target_rank in [current_rank - 1, current_rank - 2, current_rank - 3]:
            if time.time() > deadline - 30:
                break
            if target_rank < 1:
                continue
            try:
                result = _try_optimize_rank(target_rank, n, m, p, max_iter=1000, seed=target_rank)
                if result is not None:
                    best_result = result
                    break
            except Exception:
                pass
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    return _build_schoolbook()
# EVOLVE-BLOCK-END
