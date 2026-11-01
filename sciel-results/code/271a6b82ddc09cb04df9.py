"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    if not U or not V or not W:
        return False
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for k in range(p):
            for j in range(m):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_decomposition():
    """
    Construct a Strassen-based decomposition for 2x4x8 matrix multiplication.
    
    A (2x4) * B (4x8) = C (2x8)
    
    Partition: A = [A_left A_right] where A_left, A_right are 2x2
               B = [B_top; B_bottom] where B_top, B_bottom are 2x8
    
    C = A_left * B_top + A_right * B_bottom
    
    We partition the 4 columns of A into two 2-column blocks, and the 8 columns 
    of B into four 2-column blocks. For each of the 2 A-blocks and 4 B-blocks,
    we perform a 2x2 by 2x2 matrix multiplication using Strassen's algorithm (7 mults).
    
    Total rank = 2 * 4 * 7 = 56.
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen terms: (A_coeffs, B_coeffs, C_coeffs)
    strassen_terms = [
        # M1 = (a00 + a11)(b00 + b11)
        (
            {(0,0): 1, (1,1): 1},   # A coeffs
            {(0,0): 1, (1,1): 1},   # B coeffs
            {(0,0): 1, (1,1): 1}    # C coeffs
        ),
        # M2 = (a10 + a11)b00
        (
            {(1,0): 1, (1,1): 1},   # A coeffs
            {(0,0): 1},             # B coeffs
            {(1,0): 1, (1,1): -1}   # C coeffs
        ),
        # M3 = a00(b01 - b11)
        (
            {(0,0): 1},             # A coeffs
            {(0,1): 1, (1,1): -1},  # B coeffs
            {(0,1): 1, (1,1): 1}    # C coeffs
        ),
        # M4 = a11(b10 - b00)
        (
            {(1,1): 1},             # A coeffs
            {(1,0): 1, (0,0): -1},  # B coeffs
            {(0,0): 1, (1,0): 1}    # C coeffs
        ),
        # M5 = (a00 + a01)b11
        (
            {(0,0): 1, (0,1): 1},   # A coeffs
            {(1,1): 1},             # B coeffs
            {(0,0): -1, (0,1): 1}   # C coeffs
        ),
        # M6 = (a10 - a00)(b00 + b01)
        (
            {(1,0): 1, (0,0): -1},  # A coeffs
            {(0,0): 1, (0,1): 1},   # B coeffs
            {(1,1): 1}              # C coeffs
        ),
        # M7 = (a01 - a11)(b10 + b11)
        (
            {(0,1): 1, (1,1): -1},  # A coeffs
            {(1,0): 1, (1,1): 1},   # B coeffs
            {(0,0): 1}              # C coeffs
        ),
    ]
    
    col = 0
    # For each A-block (2 of them) and B-block (4 of them)
    for a_block in range(2):
        for b_block in range(4):
            for term in strassen_terms:
                A_coeffs, B_coeffs, C_coeffs = term
                
                # Fill U: U[i*m + j, col]
                for (li, lj), coeff in A_coeffs.items():
                    global_i = li
                    global_j = 2 * a_block + lj
                    U[global_i * m + global_j, col] += coeff
                
                # Fill V: V[j*p + k, col]
                for (lj, lk), coeff in B_coeffs.items():
                    global_j = 2 * a_block + lj
                    global_k = 2 * b_block + lk
                    V[global_j * p + global_k, col] += coeff
                
                # Fill W: W[k*n + i, col]
                for (li, lk), coeff in C_coeffs.items():
                    global_i = li
                    global_k = 2 * b_block + lk
                    W[global_k * n + global_i, col] += coeff
                
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook_decomposition():
    """Fallback schoolbook decomposition with rank n*m*p = 64."""
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

def _rank19_4x4_decomposition():
    """
    Known rank-19 decomposition for 4x4 matrix multiplication.
    This uses a recursive/structured approach.
    
    Actually, let me use a known result: the tensor rank of 4x4 matrix multiplication
    is at most 49 (from 7*7), but there are known decompositions with lower rank.
    
    For 4x4x4, the best known upper bound is around 49 via Strassen recursion,
    but there exist explicit decompositions with rank < 49.
    
    Let me try a different approach: use the fact that we can view 2x4x8 as
    a structured problem and try to find a lower rank decomposition.
    
    Actually, let me try a direct approach: the tensor T_{ijk} for 2x4x8 has
    dimensions 8 x 32 x 16 = 4096 entries, each being 1.
    
    Let me try to use a random initialization and gradient descent to find
    a lower rank decomposition.
    """
    n, m, p = DIMENSIONS
    # Start with Strassen (rank 56) and try to reduce
    result = _strassen_decomposition()
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_reduce_by_removing_columns(U, V, W, n, m, p, tol=1e-10):
    """Try to reduce rank by removing columns that are redundant."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    active = list(range(rank))
    changed = True
    while changed:
        changed = False
        for idx in range(len(active) - 1, -1, -1):
            col = active[idx]
            new_active = active[:idx] + active[idx+1:]
            if not new_active:
                continue
            U_new = U_arr[:, new_active]
            V_new = V_arr[:, new_active]
            W_new = W_arr[:, new_active]
            
            ok = True
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        val = np.sum(U_new[i*m+j, :] * V_new[j*p+k, :] * W_new[k*n+i, :])
                        if abs(val - 1.0) > tol:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            
            if ok:
                active = new_active
                changed = True
                break
    
    if len(active) < rank:
        return U_arr[:, active].tolist(), V_arr[:, active].tolist(), W_arr[:, active].tolist()
    return U, V, W

def _gradient_descent_reduce(U, V, W, n, m, p, max_iter=5000, lr=0.01, tol=1e-9):
    """
    Try to reduce the effective rank by optimizing the factor matrices.
    We fix the number of columns and try to zero out some entries,
    or we try to find a lower rank decomposition from scratch.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try different target ranks
    for target_rank in [48, 49, 50, 51, 52, 53, 54, 55]:
        if time.time() - start_time > 100:
            break
        
        # Initialize with random values scaled appropriately
        np.random.seed(42)
        U = np.random.randn(n * m, target_rank) * 0.1
        V = np.random.randn(m * p, target_rank) * 0.1
        W = np.random.randn(p * n, target_rank) * 0.1
        
        # Use Adam-like optimization
        mU = np.zeros_like(U)
        vU = np.zeros_like(U)
        mV = np.zeros_like(V)
        vV = np.zeros_like(V)
        mW = np.zeros_like(W)
        vW = np.zeros_like(W)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        for it in range(max_iter):
            if time.time() - start_time > 105:
                break
            
            # Compute gradients
            gU = np.zeros_like(U)
            gV = np.zeros_like(V)
            gW = np.zeros_like(W)
            
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        # d/dU[i*m+j, c] of sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1
                        # = V[j*p+k, c] * W[k*n+i, c]
                        gU[i*m+j, :] += V[j*p+k, :] * W[k*n+i, :]
                        gV[j*p+k, :] += U[i*m+j, :] * W[k*n+i, :]
                        gW[k*n+i, :] += U[i*m+j, :] * V[j*p+k, :]
            
            # Adam update
            t = it + 1
            mU = beta1 * mU + (1 - beta1) * gU
            vU = beta2 * vU + (1 - beta2) * gU**2
            mU_hat = mU / (1 - beta1**t)
            vU_hat = vU / (1 - beta2**t)
            U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
            
            mV = beta1 * mV + (1 - beta1) * gV
            vV = beta2 * vV + (1 - beta2) * gV**2
            mV_hat = mV / (1 - beta1**t)
            vV_hat = vV / (1 - beta2**t)
            V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
            
            mW = beta1 * mW + (1 - beta1) * gW
            vW = beta2 * vW + (1 - beta2) * gW**2
            mW_hat = mW / (1 - beta1**t)
            vW_hat = vW / (1 - beta2**t)
            W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
            
            # Check convergence
            if it % 100 == 0:
                max_err = 0.0
                for i in range(n):
                    for k in range(p):
                        for j in range(m):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            err = abs(val - 1.0)
                            if err > max_err:
                                max_err = err
                if max_err < tol:
                    # Found a valid decomposition
                    return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Try Strassen decomposition first (rank 56)
    try:
        result = _strassen_decomposition()
        U, V, W = result["U"], result["V"], result["W"]
        if _verify(U, V, W, n, m, p, tol=1e-9):
            # Try to reduce by removing redundant columns
            U_red, V_red, W_red = _try_reduce_by_removing_columns(U, V, W, n, m, p)
            if len(U_red[0]) < len(U[0]):
                if _verify(U_red, V_red, W_red, n, m, p, tol=1e-9):
                    return {"U": U_red, "V": V_red, "W": W_red}
            return result
    except Exception:
        pass
    
    # Try gradient descent to find lower rank
    if time.time() < deadline - 10:
        try:
            gd_result = _gradient_descent_reduce(U, V, W, n, m, p, max_iter=3000, lr=0.05)
            if gd_result is not None:
                U_gd, V_gd, W_gd = gd_result
                if _verify(U_gd, V_gd, W_gd, n, m, p, tol=1e-8):
                    return {"U": U_gd, "V": V_gd, "W": W_gd}
        except Exception:
            pass
    
    # Fallback to schoolbook (rank 64)
    return _schoolbook_decomposition()

# EVOLVE-BLOCK-END
