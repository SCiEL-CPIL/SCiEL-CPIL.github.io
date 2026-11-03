import time
import numpy as np
from fractions import Fraction

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    
    T_true = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for l in range(p):
                a_idx = i * m + j
                b_idx = j * p + l
                c_idx = l * n + i
                T_true[a_idx, b_idx, c_idx] = 1.0
    
    return np.allclose(T_decomp, T_true, atol=1e-10)

def _build_schoolbook(n, m, p):
    """Build the standard schoolbook rank-n*m*p decomposition."""
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
    return U, V, W, rank

def _build_strassen_2x3x8(n, m, p):
    """
    Build a rank-44 decomposition using Strassen for the 2x2x2 blocks.
    
    Structure:
    - j=0: 16 schoolbook terms (2 i-values * 8 k-values)
    - j=1,2: 4 blocks of 2x2x2, each using Strassen (7 terms) = 28 terms
    Total: 16 + 28 = 44
    """
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Part 1: j=0, schoolbook for 2x8
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Part 2: j=1,2 in 4 blocks of 2x2x2 using Strassen
    for block in range(4):
        k0 = 2 * block
        k1 = 2 * block + 1
        
        # Strassen for 2x2x2:
        # m1 = (a00+a11)(b00+b11)
        U[0 * m + 1][col] = 1; U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1; V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1; W[k1 * n + 1][col] = 1
        col += 1
        
        # m2 = (a10+a11)b00
        U[1 * m + 1][col] = 1; U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1; W[k1 * n + 1][col] = -1
        col += 1
        
        # m3 = a00(b01-b11)
        U[0 * m + 1][col] = 1
        V[1 * p + k1][col] = 1; V[2 * p + k1][col] = -1
        W[k1 * n + 0][col] = 1; W[k1 * n + 1][col] = 1
        col += 1
        
        # m4 = a11(b10-b00)
        U[1 * m + 2][col] = 1
        V[2 * p + k0][col] = 1; V[1 * p + k0][col] = -1
        W[k0 * n + 0][col] = 1; W[k0 * n + 1][col] = 1
        col += 1
        
        # m5 = (a00+a01)b11
        U[0 * m + 1][col] = 1; U[0 * m + 2][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = -1; W[k1 * n + 0][col] = 1
        col += 1
        
        # m6 = (a10-a00)(b00+b01)
        U[1 * m + 1][col] = 1; U[0 * m + 1][col] = -1
        V[1 * p + k0][col] = 1; V[1 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # m7 = (a01-a11)(b10+b11)
        U[0 * m + 2][col] = 1; U[1 * m + 2][col] = -1
        V[2 * p + k0][col] = 1; V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1
    
    return U, V, W, rank

def _try_gradient_descent(rank, n, m, p, timeout=10, seed=42):
    """Try to find a decomposition of given rank using gradient descent."""
    start = time.time()
    np.random.seed(seed)
    
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for l in range(p):
                T[i*m+j, j*p+l, l*n+i] = 1.0
    
    lr = 0.005
    for step in range(3000):
        if time.time() - start > timeout:
            break
        T_cur = np.einsum('ar,br,cr->abc', U, V, W)
        dT = T_cur - T
        dU = np.einsum('abc,br,cr->ar', dT, V, W)
        dV = np.einsum('abc,ar,cr->br', dT, U, W)
        dW = np.einsum('abc,ar,br->cr', dT, U, V)
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    T_final = np.einsum('ar,br,cr->abc', U, V, W)
    err = np.max(np.abs(T_final - T))
    if err < 1e-8:
        return U, V, W, err
    return None, None, None, err

def _try_algebraic_structure(rank, n, m, p, timeout=10):
    """
    Try to find a low-rank decomposition by exploiting the tensor structure.
    The key insight: for fixed j, the sub-tensor T[:,j,:] is a matrix of shape (n*p, n*p)
    that factors as outer products. We try to share factors across j values.
    """
    start = time.time()
    
    # For the 2x3x8 case, try a structured approach:
    # Split into j=0 part (rank 16) and j=1,2 part (try to get below 28)
    
    # Try rank 43, 42, 41 with different initializations
    for attempt in range(5):
        if time.time() - start > timeout:
            break
        
        # Initialize with the rank-44 solution and zero out one column
        U44, V44, W44, _ = _build_strassen_2x3x8(n, m, p)
        
        U = np.array(U44, dtype=float)[:, :rank]
        V = np.array(V44, dtype=float)[:, :rank]
        W = np.array(W44, dtype=float)[:, :rank]
        
        # Add small random perturbation to encourage exploration
        np.random.seed(attempt * 100 + rank)
        U += np.random.randn(*U.shape) * 0.01
        V += np.random.randn(*V.shape) * 0.01
        W += np.random.randn(*W.shape) * 0.01
        
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for l in range(p):
                    T[i*m+j, j*p+l, l*n+i] = 1.0
        
        lr = 0.002
        for step in range(2000):
            if time.time() - start > timeout:
                break
            T_cur = np.einsum('ar,br,cr->abc', U, V, W)
            dT = T_cur - T
            dU = np.einsum('abc,br,cr->ar', dT, V, W)
            dV = np.einsum('abc,ar,cr->br', dT, U, W)
            dW = np.einsum('abc,ar,br->cr', dT, U, V)
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        T_final = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(T_final - T))
        if err < 1e-8:
            return U, V, W, err
    
    return None, None, None, float('inf')

def construct():
    n, m, p = 2, 3, 8
    deadline = time.time() + 140.0  # 10s safety margin before 150s kill
    
    # Start with schoolbook (rank 48) as fallback
    U_school, V_school, W_school, rank_school = _build_schoolbook(n, m, p)
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Try the rank-44 Strassen-based construction
    U44, V44, W44, rank44 = _build_strassen_2x3x8(n, m, p)
    
    if _verify_decomposition(U44, V44, W44, n, m, p):
        best_result = {"U": U44, "V": V44, "W": W44}
        best_rank = rank44
    
    # Try to find even lower rank solutions
    # Focus on ranks 43, 42, 41
    for r in range(best_rank - 1, max(best_rank - 4, 30), -1):
        if time.time() > deadline - 25:
            break
        
        # Try gradient descent from random init
        U_r, V_r, W_r, err = _try_gradient_descent(r, n, m, p, timeout=8, seed=r * 7)
        if U_r is not None and err < 1e-8:
            if _verify_decomposition(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                best_rank = r
                break
        
        if time.time() > deadline - 20:
            break
        
        # Try algebraic structure approach
        U_r, V_r, W_r, err = _try_algebraic_structure(r, n, m, p, timeout=8)
        if U_r is not None and err < 1e-8:
            if _verify_decomposition(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                best_rank = r
                break
    
    return best_result
