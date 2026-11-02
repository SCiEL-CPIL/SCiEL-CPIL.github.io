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

def _strassen_2x2():
    """Return the 7 terms of Strassen's 2x2x2 matrix multiplication.
    Each term is (U_indices, V_indices, W_indices) where indices are (row, col) pairs.
    U is 2x2 (4 entries), V is 2x2 (4 entries), W is 2x2 (4 entries).
    """
    # For 2x2x2, we have 7 multiplications
    # Term 0: M1 = (A00+A11)(B00+B11)
    # Term 1: M2 = (A10+A11)B00
    # Term 2: M3 = A00(B10-B11)
    # Term 3: M4 = A11(B01-B00)
    # Term 4: M5 = (A00+A01)B11
    # Term 5: M6 = (A10-A00)(B00+B01)
    # Term 6: M7 = (A01-A11)(B10+B11)
    
    # U factors (for A matrix, 2x2):
    # M1: A00+A11 -> (0,0):1, (1,1):1
    # M2: A10+A11 -> (1,0):1, (1,1):1
    # M3: A00 -> (0,0):1
    # M4: A11 -> (1,1):1
    # M5: A00+A01 -> (0,0):1, (0,1):1
    # M6: A10-A00 -> (1,0):1, (0,0):-1
    # M7: A01-A11 -> (0,1):1, (1,1):-1
    
    # V factors (for B matrix, 2x2):
    # M1: B00+B11 -> (0,0):1, (1,1):1
    # M2: B00 -> (0,0):1
    # M3: B10-B11 -> (1,0):1, (1,1):-1
    # M4: B01-B00 -> (0,1):1, (0,0):-1
    # M5: B11 -> (1,1):1
    # M6: B00+B01 -> (0,0):1, (0,1):1
    # M7: B10+B11 -> (1,0):1, (1,1):1
    
    # W factors (for C matrix, 2x2):
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 + M6
    
    terms = []
    # Term 0: M1
    terms.append({
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    # Term 1: M2
    terms.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,0): 1},
        'W': {(1,0): 1}
    })
    # Term 2: M3
    terms.append({
        'U': {(0,0): 1},
        'V': {(1,0): 1, (1,1): -1},
        'W': {(0,1): 1, (1,1): -1}
    })
    # Term 3: M4
    terms.append({
        'U': {(1,1): 1},
        'V': {(0,1): 1, (0,0): -1},
        'W': {(0,0): -1, (1,0): 1}
    })
    # Term 4: M5
    terms.append({
        'U': {(0,0): 1, (0,1): 1},
        'V': {(1,1): 1},
        'W': {(0,1): 1, (1,1): 1}
    })
    # Term 5: M6
    terms.append({
        'U': {(1,0): 1, (0,0): -1},
        'V': {(0,0): 1, (0,1): 1},
        'W': {(0,0): 1, (1,0): 1}
    })
    # Term 6: M7
    terms.append({
        'U': {(0,1): 1, (1,1): -1},
        'V': {(1,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    
    return terms

def _build_4x4x4_strassen():
    """Build a 4x4x4 decomposition using recursive Strassen (rank 49)."""
    n = 4
    terms_2x2 = _strassen_2x2()
    
    # For 4x4x4, we use 2x2x2 of 2x2x2 blocks
    # This gives 7*7 = 49 terms
    U = [[0] * 49 for _ in range(16)]  # 4*4 = 16
    V = [[0] * 49 for _ in range(16)]  # 4*4 = 16
    W = [[0] * 49 for _ in range(16)]  # 4*4 = 16
    
    for t1, term1 in enumerate(terms_2x2):
        for t2, term2 in enumerate(terms_2x2):
            idx = t1 * 7 + t2
            
            # U: 4x4 matrix, indexed by (i,j) where i,j in 0..3
            # Block (bi, bj) where bi = i//2, bj = j//2
            # Within block, local indices (li, lj) where li = i%2, lj = j%2
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

def _build_4x4x5():
    """Build a 4x4x5 decomposition: 4x4x4 (rank 49) + 4x4x1 (rank 4) = rank 53."""
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
    for j in range(m * 4):  # First 4 columns of B
        for r in range(rank_base):
            V[j][r] = base["V"][j][r]
    for k in range(4 * n):  # First 4 rows of C (k=0..3)
        for r in range(rank_base):
            W[k][r] = base["W"][k][r]
    
    # Add 4x4x1 part: C[:,4] = A @ B[:,4]
    # For each i in 0..3, C[i,4] = sum_j A[i,j] * B[j,4]
    # We need 4 terms, one per j (not per i), since each term handles one j
    for j in range(m):
        r = rank_base + j  # One term per j
        for i in range(n):
            U[i*m + j][r] = 1
        V[j*p + 4][r] = 1  # B[j,4]
        for i in range(n):
            W[4*n + i][r] = 1  # C[i,4] where k=4, so k*n+i = 4*4+i = 16+i
    
    return {"U": U, "V": V, "W": W}

def _try_optimize_rank(rank, n, m, p, max_iter=2000, seed=42):
    """Try to find a decomposition of given rank using gradient optimization."""
    np.random.seed(seed)
    
    # Initialize
    U = np.random.randn(n*m, rank) * 0.5
    V = np.random.randn(m*p, rank) * 0.5
    W = np.random.randn(p*n, rank) * 0.5
    
    def loss_and_grad(params):
        U_p = params[:n*m*rank].reshape(n*m, rank)
        V_p = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W_p = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        loss = 0.0
        # Compute loss
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
        
        # Verify
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
    
    # Try the Strassen-based decomposition (rank 53)
    best_result = None
    try:
        result = _build_4x4x5()
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            best_result = result
    except Exception:
        pass
    
    # Try to find lower rank through optimization
    if time.time() < deadline - 60:
        for target_rank in [52, 51, 50]:
            if time.time() > deadline - 30:
                break
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
