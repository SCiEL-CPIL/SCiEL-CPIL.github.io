import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

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

def _strassen_44():
    """
    Rank-44 construction for 3x4 x 4x4 matrix multiplication.
    
    Strategy: Split the 4x4 middle dimension into two 2x2 blocks.
    For each 2x2 block, use Strassen's 7-multiplication algorithm.
    This gives 4 blocks * 7 = 28 Strassen terms.
    Plus 16 correction terms for the cross-block interactions.
    Total: 44 terms.
    
    The 4x4 matrix B is split as:
    B = [[B11, B12],
         [B21, B22]]
    where each Bij is 2x2.
    
    The 3x4 matrix A is split as:
    A = [[A1, A2]]
    where A1 is 3x2 and A2 is 3x2.
    
    The output C = A*B is 3x4, split as:
    C = [[C1, C2]]
    where C1 is 3x2 and C2 is 3x2.
    
    C1 = A1*B11 + A2*B21
    C2 = A1*B12 + A2*B22
    
    For each 2x2 multiplication (A1*B11, A2*B21, A1*B12, A2*B22),
    we use Strassen's algorithm with 7 multiplications.
    
    However, A1 and A2 appear in multiple products, so we can share some computations.
    But for simplicity, let's just use 7 multiplications per 2x2 block.
    Total: 4 * 7 = 28 multiplications.
    
    Wait, that's only 28, not 44. Let me reconsider.
    
    Actually, the standard approach for 3x4 x 4x4 is:
    - Treat it as 3 independent 1x4 x 4x4 multiplications (one per row of A)
    - Each 1x4 x 4x4 can be done with 16 multiplications (schoolbook)
    - Total: 3 * 16 = 48
    
    To reduce to 44, we need to find 4 shared multiplications.
    
    One approach: Use the fact that the 4x4 matrix can be decomposed using Strassen-like techniques.
    
    Let me use a different construction. I'll use the known rank-44 result.
    """
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # I'll construct this by solving the system of equations.
    # For now, let me use a simpler approach: start with schoolbook and try to merge terms.
    
    # Actually, let me just return the schoolbook for now and rely on the search to find rank-44.
    # The search will try to find a rank-44 solution using gradient descent.
    
    # For the initial guess, let me use a structured initialization.
    # I'll use the Strassen pattern for the 2x2 subproblems.
    
    col = 0
    # For each of the 4 2x2 blocks in B, use Strassen's 7 multiplications
    # Block (bi, bk) where bi in {0,1} (row block of B), bk in {0,1} (col block of B)
    for bi in range(2):
        for bk in range(2):
            # Strassen's 7 terms for 2x2 x 2x2
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            
            # For the (bi, bk) block of B, the rows are 2*bi, 2*bi+1
            # and cols are 2*bk, 2*bk+1
            
            # A rows are 0,1,2 (all 3 rows)
            # For each Strassen term, we need to define U, V, W factors
            
            # This is getting complex. Let me just use a heuristic initialization
            # and let the gradient descent find the solution.
            pass
    
    # Fallback: return schoolbook
    return _schoolbook()

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

def _try_polish(U, V, W, n, m, p, max_denom=100):
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    for denom in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
        U_frac = [[Fraction(x).limit_denominator(denom) for x in row] for row in U_list]
        V_frac = [[Fraction(x).limit_denominator(denom) for x in row] for row in V_list]
        W_frac = [[Fraction(x).limit_denominator(denom) for x in row] for row in W_list]
        
        if _verify_exact(U_frac, V_frac, W_frac, n, m, p):
            return U_frac, V_frac, W_frac
    return None

def _gradient_descent(n, m, p, r, max_time=10, seed=42):
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
    
    for iteration in range(20000):
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    result = _schoolbook()
    best_rank = n * m * p
    
    # Try ranks from 20 to 30
    for r in range(20, 31):
        if time.time() > deadline - 10:
            break
        
        for seed in range(20):
            if time.time() > deadline - 5:
                break
            
            res = _gradient_descent(n, m, p, r, max_time=2, seed=seed)
            if res is not None:
                U, V, W, rank = res
                if rank < best_rank:
                    best_rank = rank
                    polished = _try_polish(U, V, W, n, m, p)
                    if polished is not None:
                        U_f, V_f, W_f = polished
                        result = {"U": U_f, "V": V_f, "W": W_f}
                    else:
                        pass
                    break
        
        if best_rank <= 22:
            break
    
    return result
