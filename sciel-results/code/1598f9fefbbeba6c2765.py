import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]
n, m, p = DIMENSIONS

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    max_err = np.max(np.abs(T - 1.0))
    return max_err < tol, max_err

def _verify_exact(U, V, W, n, m, p):
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    rank = len(U[0])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c]
                if val != 1:
                    return False
    return True

def _schoolbook():
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

def _try_polish(U, V, W, n, m, p, max_denom=1000):
    U_list = U.tolist() if hasattr(U, 'tolist') else U
    V_list = V.tolist() if hasattr(V, 'tolist') else V
    W_list = W.tolist() if hasattr(W, 'tolist') else W
    
    for denom in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
        U_frac = [[Fraction(x).limit_denominator(denom) for x in row] for row in U_list]
        V_frac = [[Fraction(x).limit_denominator(denom) for x in row] for row in V_list]
        W_frac = [[Fraction(x).limit_denominator(denom) for x in row] for row in W_list]
        
        if _verify_exact(U_frac, V_frac, W_frac, n, m, p):
            return U_frac, V_frac, W_frac
    return None

def _gradient_descent(n, m, p, r, max_time=5, seed=42):
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    U = rng.randn(n * m, r) * 0.5
    V = rng.randn(m * p, r) * 0.5
    W = rng.randn(p * n, r) * 0.5
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.01
    momentum = 0.9
    vU = np.zeros_like(U)
    vV = np.zeros_like(V)
    vW = np.zeros_like(W)
    
    for iteration in range(10000):
        if time.time() - start_time > max_time:
            break
        
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        err = np.sum((T - 1.0) ** 2)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if err < 1e-12:
            break
        
        dL_dU_3d = 2 * np.einsum('ijk,jkc,kic->ijc', T - 1.0, V_3d, W_3d)
        dL_dV_3d = 2 * np.einsum('ijk,ijc,kic->jkc', T - 1.0, U_3d, W_3d)
        dL_dW_3d = 2 * np.einsum('ijk,ijc,jkc->kic', T - 1.0, U_3d, V_3d)
        
        gU = dL_dU_3d.reshape(n * m, r)
        gV = dL_dV_3d.reshape(m * p, r)
        gW = dL_dW_3d.reshape(p * n, r)
        
        vU = momentum * vU + (1 - momentum) * gU
        vV = momentum * vV + (1 - momentum) * gV
        vW = momentum * vW + (1 - momentum) * gW
        
        U -= lr * vU
        V -= lr * vV
        W -= lr * vW
        
        if iteration % 100 == 0:
            lr *= 0.99
    
    ok, final_err = _verify_fast(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p, tol=1e-8)
    if ok or final_err < 1e-6:
        return best_U, best_V, best_W, r
    
    return None

def _als_iterate(n, m, p, r, max_iters=100, tol=1e-6, seed=42):
    """
    Alternating Least Squares for tensor decomposition.
    Fix two factors, solve for the third by minimizing ||T - U ⊗ V ⊗ W||_F.
    """
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, r) * 0.5
    V = rng.randn(m * p, r) * 0.5
    W = rng.randn(p * n, r) * 0.5
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        # Solve for U: for each (i,j), minimize ||sum_c U[i,j,c] V[j,k,c] W[k,i,c] - 1||^2
        # This is a linear system in U[i,j,c] for fixed c... actually it's more complex.
        # Let me use the standard ALS formulation.
        
        # For each c, we have a rank-1 tensor U[:, :, c] ⊗ V[:, :, c] ⊗ W[:, :, c]
        # The total tensor is sum_c of these.
        
        # To update U: fix V and W. For each (i,j), we want to find U[i,j,:] such that
        # sum_c U[i,j,c] * (V and W contracted appropriately) ≈ 1
        # This is: for each (i,j), solve min_U sum_k (sum_c U[i,j,c] * B_c[i,j,k])^2 - 2 * sum_k (sum_c U[i,j,c] * B_c[i,j,k])
        # where B_c[i,j,k] = V[j,k,c] * W[k,i,c]
        
        # Actually, let me just do gradient descent with better initialization.
        break
    
    return None

def _strassen_based_init(n, m, p, r, seed=42):
    """
    Initialize using Strassen-like structure for better convergence.
    """
    rng = np.random.RandomState(seed)
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # Try to build a structured initialization
    # Split the 4x4 middle into 2x2 blocks
    # Use Strassen's 7 multiplications per 2x2 block
    
    col = 0
    # For each 2x2 block in the middle dimension
    for bi in range(2):  # row block of B
        for bk in range(2):  # col block of B
            # Strassen's 7 terms
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            
            # For block (bi, bk), B rows are 2*bi, 2*bi+1; cols are 2*bk, 2*bk+1
            b_row0 = 2 * bi
            b_row1 = 2 * bi + 1
            b_col0 = 2 * bk
            b_col1 = 2 * bk + 1
            
            # Strassen coefficients
            strassen_terms = [
                # (A_coeffs, B_coeffs) where A is 2x2, B is 2x2
                # M1 = (A00+A11)(B00+B11)
                (([1,0,0,1], [1,0,0,1])),
                # M2 = (A10+A11)B00
                (([0,1,0,1], [1,0,0,0])),
                # M3 = A00(B01-B11)
                (([1,0,0,0], [0,1,0,-1])),
                # M4 = A11(B10-B00)
                (([0,0,0,1], [0,0,1,-1])),
                # M5 = (A00+A01)B11
                (([1,1,0,0], [0,0,0,1])),
                # M6 = (A10-A00)(B00+B01)
                (([-1,1,0,0], [1,1,0,0])),
                # M7 = (A01-A11)(B10+B11)
                (([0,1,-1,0], [0,0,1,1])),
            ]
            
            for a_coeffs, b_coeffs in strassen_terms:
                if col >= r:
                    break
                
                # A is 3x2 (rows 0,1,2 of A, cols corresponding to this block)
                # For Strassen, we need A to be 2x2, but we have 3x2
                # So we apply Strassen to each pair of rows of A
                # This means for 3 rows, we need 2 Strassen applications per block
                # Total: 4 blocks * 2 * 7 = 56 terms, which is more than 48
                
                # Alternative: treat each row of A separately with 4 multiplications
                # Total: 3 * 4 * 4 = 48 (schoolbook)
                
                # Let's just use schoolbook initialization with random perturbation
                pass
    
    # Fallback: schoolbook + random perturbation
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col < r:
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
    
    # Add small random perturbation to escape local minima
    U += rng.randn(n * m, r) * 0.01
    V += rng.randn(m * p, r) * 0.01
    W += rng.randn(p * n, r) * 0.01
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145
    
    result = _schoolbook()
    best_rank = n * m * p
    
    # Try ranks from 20 to 30
    for r in range(20, 31):
        if time.time() > deadline - 10:
            break
        
        for seed in range(10):
            if time.time() > deadline - 5:
                break
            
            # Try gradient descent with different initializations
            res = _gradient_descent(n, m, p, r, max_time=3, seed=seed)
            if res is not None:
                U, V, W, rank = res
                if rank < best_rank:
                    best_rank = rank
                    polished = _try_polish(U, V, W, n, m, p)
                    if polished is not None:
                        U_f, V_f, W_f = polished
                        result = {"U": U_f, "V": V_f, "W": W_f}
                    else:
                        result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    break
        
        if best_rank <= 22:
            break
    
    return result
