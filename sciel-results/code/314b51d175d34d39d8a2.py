"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction

def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W, n=2, m=3, p=3, tol=1e-9):
    """Verify that the tensor decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _als_decompose(target_rank, n=2, m=3, p=3, max_iters=200, time_budget=60.0):
    """Use ALS to find a rank decomposition."""
    start_time = time.time()
    deadline = start_time + time_budget
    
    T = np.ones((n, m, p))
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        np.random.seed(restart * 1000 + target_rank)
        
        U = np.random.randn(n * m, target_rank)
        V = np.random.randn(m * p, target_rank)
        W = np.random.randn(p * n, target_rank)
        
        # Normalize columns
        for r in range(target_rank):
            norms = [np.linalg.norm(U[:, r]), np.linalg.norm(V[:, r]), np.linalg.norm(W[:, r])]
            if any(norm < 1e-10 for norm in norms):
                continue
            total_norm = np.prod(norms) ** (1/3)
            if total_norm > 1e-10:
                U[:, r] /= (norms[0] / total_norm)
                V[:, r] /= (norms[1] / total_norm)
                W[:, r] /= (norms[2] / total_norm)
        
        best_error = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, target_rank))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(A.T, T[i, j, :].reshape(-1, 1), rcond=None)[0].flatten()
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    B = np.zeros((n, target_rank))
                    for i in range(n):
                        B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(B.T, T[:, j, k].reshape(-1, 1), rcond=None)[0].flatten()
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    C = np.zeros((m, target_rank))
                    for j in range(m):
                        C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(C.T, T[i, :, k].reshape(-1, 1), rcond=None)[0].flatten()
                    except:
                        pass
            
            # Compute error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(target_rank):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        error += (val - 1.0) ** 2
            error = np.sqrt(error)
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if error < 1e-8:
                break
        
        if best_error < 1e-6:
            return best_U, best_V, best_W, best_error
    
    return None, None, None, np.inf

def _try_rational_rounding(U, V, W, max_denom=10):
    """Try to round the solution to rationals with small denominators."""
    U_f = np.array(U)
    V_f = np.array(V)
    W_f = np.array(W)
    
    # Try rounding to integers first
    U_int = np.round(U_f)
    V_int = np.round(V_f)
    W_int = np.round(W_f)
    
    if _verify(U_int.tolist(), V_int.tolist(), W_int.tolist()):
        return U_int, V_int, W_int
    
    # Try half-integers
    U_half = np.round(2 * U_f) / 2.0
    V_half = np.round(2 * V_f) / 2.0
    W_half = np.round(2 * W_f) / 2.0
    
    if _verify(U_half.tolist(), V_half.tolist(), W_half.tolist()):
        return U_half, V_half, W_half
    
    # Try rationals with small denominators
    for denom in range(2, max_denom + 1):
        U_r = np.round(U_f * denom) / denom
        V_r = np.round(V_f * denom) / denom
        W_r = np.round(W_f * denom) / denom
        
        if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
            return U_r, V_r, W_r
    
    return None, None, None

def _pad_to_333(U, V, W, n=2, m=3, p=3):
    """Pad 2x3x3 factors to 3x3x3 by adding zero rows to U and W."""
    rank = U.shape[1]
    U_padded = np.zeros((3 * m, rank))
    V_padded = np.zeros((m * p, rank))
    W_padded = np.zeros((p * 3, rank))
    
    for r in range(rank):
        U_padded[0:m, r] = U[:, r]
        V_padded[:, r] = V[:, r]
        W_padded[0:n*p, r] = W[:n*p, r]
    
    return U_padded, V_padded, W_padded

def _unpad_from_333(U3, V3, W3, n=2, m=3, p=3):
    """Unpad 3x3x3 factors back to 2x3x3."""
    rank = U3.shape[1]
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r in range(rank):
        U[:, r] = U3[0:m, r]
        V[:, r] = V3[:, r]
        W[:, r] = W3[0:n*p, r]
    
    return U, V, W

def _als_decompose_333(target_rank, max_iters=200, time_budget=60.0):
    """Use ALS to find a rank decomposition for 3x3x3."""
    start_time = time.time()
    deadline = start_time + time_budget
    
    n, m, p = 3, 3, 3
    T = np.ones((n, m, p))
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        np.random.seed(restart * 1000 + target_rank + 777)
        
        U = np.random.randn(n * m, target_rank)
        V = np.random.randn(m * p, target_rank)
        W = np.random.randn(p * n, target_rank)
        
        # Normalize columns
        for r in range(target_rank):
            norms = [np.linalg.norm(U[:, r]), np.linalg.norm(V[:, r]), np.linalg.norm(W[:, r])]
            if any(norm < 1e-10 for norm in norms):
                continue
            total_norm = np.prod(norms) ** (1/3)
            if total_norm > 1e-10:
                U[:, r] /= (norms[0] / total_norm)
                V[:, r] /= (norms[1] / total_norm)
                W[:, r] /= (norms[2] / total_norm)
        
        best_error = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, target_rank))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(A.T, T[i, j, :].reshape(-1, 1), rcond=None)[0].flatten()
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    B = np.zeros((n, target_rank))
                    for i in range(n):
                        B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(B.T, T[:, j, k].reshape(-1, 1), rcond=None)[0].flatten()
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    C = np.zeros((m, target_rank))
                    for j in range(m):
                        C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(C.T, T[i, :, k].reshape(-1, 1), rcond=None)[0].flatten()
                    except:
                        pass
            
            # Compute error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(target_rank):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        error += (val - 1.0) ** 2
            error = np.sqrt(error)
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if error < 1e-8:
                break
        
        if best_error < 1e-6:
            return best_U, best_V, best_W, best_error
    
    return None, None, None, np.inf

def _try_rational_rounding_333(U, V, W, max_denom=10):
    """Try to round the 3x3x3 solution to rationals with small denominators."""
    U_f = np.array(U)
    V_f = np.array(V)
    W_f = np.array(W)
    
    # Try rounding to integers first
    U_int = np.round(U_f)
    V_int = np.round(V_f)
    W_int = np.round(W_f)
    
    if _verify_333(U_int.tolist(), V_int.tolist(), W_int.tolist()):
        return U_int, V_int, W_int
    
    # Try half-integers
    U_half = np.round(2 * U_f) / 2.0
    V_half = np.round(2 * V_f) / 2.0
    W_half = np.round(2 * W_f) / 2.0
    
    if _verify_333(U_half.tolist(), V_half.tolist(), W_half.tolist()):
        return U_half, V_half, W_half
    
    # Try rationals with small denominators
    for denom in range(2, max_denom + 1):
        U_r = np.round(U_f * denom) / denom
        V_r = np.round(V_f * denom) / denom
        W_r = np.round(W_f * denom) / denom
        
        if _verify_333(U_r.tolist(), V_r.tolist(), W_r.tolist()):
            return U_r, V_r, W_r
    
    return None, None, None

def _verify_333(U, V, W, n=3, m=3, p=3, tol=1e-9):
    """Verify that the 3x3x3 tensor decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    # First try direct ALS on 2x3x3
    for target_rank in [14, 15, 16, 17]:
        if time.time() > deadline - 30.0:
            break
        
        U, V, W, error = _als_decompose(target_rank, time_budget=25.0)
        
        if U is not None and error < 1e-6:
            # Try to round to rationals
            U_r, V_r, W_r = _try_rational_rounding(U, V, W)
            if U_r is not None:
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            
            # If rounding fails, check if the floating point solution is valid
            if _verify(U.tolist(), V.tolist(), W.tolist()):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try tensor padding approach: solve 3x3x3 then unpad
    if time.time() < deadline - 20.0:
        for target_rank in [19, 20, 21, 22, 23]:
            if time.time() > deadline - 20.0:
                break
            
            U3, V3, W3, error3 = _als_decompose_333(target_rank, time_budget=15.0)
            
            if U3 is not None and error3 < 1e-6:
                # Unpad to 2x3x3
                U, V, W = _unpad_from_333(U3, V3, W3)
                
                # Verify the unpadded solution
                if _verify(U.tolist(), V.tolist(), W.tolist()):
                    # Try to round to rationals
                    U_r, V_r, W_r = _try_rational_rounding(U, V, W)
                    if U_r is not None:
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    return _build_schoolbook()
# EVOLVE-BLOCK-END
