import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

def _strassen_2x2():
    """
    Returns the Strassen 7-multiplication decomposition for 2x2 matrix multiplication.
    A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    C = A*B = [[c11, c12], [c21, c22]]
    
    7 products:
    m1 = (a11+a22)(b11+b22)
    m2 = (a21+a22)b11
    m3 = a11(b12-b22)
    m4 = a22(b21-b11)
    m5 = (a11+a12)b22
    m6 = (a21-a11)(b11+b12)
    m7 = (a12-a22)(b21+b22)
    
    c11 = m1 + m4 - m5 + m7
    c12 = m3 + m5
    c21 = m2 + m4
    c22 = m1 - m2 + m3 + m6
    
    We return U, V, W matrices for this decomposition.
    For 2x2x2, we have:
    - Input A has 4 elements: a11, a12, a21, a22 -> indices 0,1,2,3
    - Input B has 4 elements: b11, b12, b21, b22 -> indices 0,1,2,3
    - Output C has 4 elements: c11, c12, c21, c22 -> indices 0,1,2,3
    
    U: (4, 7), V: (4, 7), W: (4, 7)
    Each column r corresponds to one product m_r.
    U[u, r] is the coefficient of A[u] in the first factor of m_r
    V[v, r] is the coefficient of B[v] in the second factor of m_r
    W[w, r] is the coefficient of C[w] in the output (how m_r contributes to c_w)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # m1 = (a11+a22)(b11+b22)
    # A coefficients: a11=1, a22=1
    # B coefficients: b11=1, b22=1
    # C contributions: c11=+1, c22=-1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = -1
    
    # m2 = (a21+a22)b11
    # A coefficients: a21=1, a22=1
    # B coefficients: b11=1
    # C contributions: c21=+1, c22=-1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # m3 = a11(b12-b22)
    # A coefficients: a11=1
    # B coefficients: b12=1, b22=-1
    # C contributions: c12=+1, c22=+1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # m4 = a22(b21-b11)
    # A coefficients: a22=1
    # B coefficients: b21=1, b11=-1
    # C contributions: c11=+1, c21=+1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # m5 = (a11+a12)b22
    # A coefficients: a11=1, a12=1
    # B coefficients: b22=1
    # C contributions: c11=-1, c12=+1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # m6 = (a21-a11)(b11+b12)
    # A coefficients: a21=1, a11=-1
    # B coefficients: b11=1, b12=1
    # C contributions: c22=+1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # m7 = (a12-a22)(b21+b22)
    # A coefficients: a12=1, a22=-1
    # B coefficients: b21=1, b22=1
    # C contributions: c11=+1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify the decomposition exactly."""
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    recon = np.einsum('ur,vr,wr->uvw', U, V, W)
    target = np.zeros((size_u, size_v, size_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target[u, v, w] = 1.0
    
    if not np.allclose(recon, target, atol=1e-9):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    # Strategy: Use Strassen-like divide and conquer
    # For 3x4x5, we can try to partition into blocks and use known decompositions
    
    # Approach 1: Try to use the known best rank for 3x4x5
    # Literature suggests rank around 52-56 for 3x4x5
    
    # Let's try a constructive approach using block partitioning
    
    # Partition A (3x4) into blocks:
    # Row 0: [A00, A01] where A00 is 1x2, A01 is 1x2
    # Row 1: [A10, A11] where A10 is 1x2, A11 is 1x2
    # Row 2: [A20, A21] where A20 is 1x2, A21 is 1x2
    
    # Partition B (4x5) into blocks:
    # Col 0-1: [B00, B01] where B00 is 2x2, B01 is 2x3
    # Col 2-3: [B10, B11] where B10 is 2x2, B11 is 2x3
    
    # This is getting complex. Let me try a different approach:
    # Use the fact that we can embed smaller known decompositions
    
    # For now, let's try to find a solution using ALS with better initialization
    # based on the Strassen structure
    
    # Build target tensor support
    non_zeros = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                non_zeros.append((u, v, w))
    
    # Precompute for efficient ALS
    u_to_vw = {u: [] for u in range(size_u)}
    for u, v, w in non_zeros:
        u_to_vw[u].append((v, w))
    
    def solve_als(rank, max_iter=200, tol=1e-12, seed=42):
        rng = np.random.RandomState(seed)
        U = rng.randn(size_u, rank)
        V = rng.randn(size_v, rank)
        W = rng.randn(size_w, rank)
        
        full_err = float('inf')
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            # Update U
            VtV = V.T @ V
            WtW = W.T @ W
            G = VtV * WtW
            
            U_new = np.zeros((size_u, rank))
            for u in range(size_u):
                if not u_to_vw[u]:
                    continue
                b = np.zeros(rank)
                for v, w in u_to_vw[u]:
                    b += V[v] * W[w]
                try:
                    U_new[u] = np.linalg.solve(G, b)
                except np.linalg.LinAlgError:
                    U_new[u] = np.linalg.lstsq(G, b, rcond=None)[0]
                    
            # Update V
            UtU = U_new.T @ U_new
            WtW = W.T @ W
            G_v = UtU * WtW
            
            V_new = np.zeros((size_v, rank))
            for v in range(size_v):
                j = v // p
                k = v % p
                b = np.zeros(rank)
                for i in range(n):
                    u = i * m + j
                    w = k * n + i
                    b += U_new[u] * W[w]
                try:
                    V_new[v] = np.linalg.solve(G_v, b)
                except np.linalg.LinAlgError:
                    V_new[v] = np.linalg.lstsq(G_v, b, rcond=None)[0]
                    
            # Update W
            UtU = U_new.T @ U_new
            VtV = V_new.T @ V_new
            G_w = UtU * VtV
            
            W_new = np.zeros((size_w, rank))
            for w in range(size_w):
                k = w // n
                i = w % n
                b = np.zeros(rank)
                for j in range(m):
                    u = i * m + j
                    v = j * p + k
                    b += U_new[u] * V_new[v]
                try:
                    W_new[w] = np.linalg.solve(G_w, b)
                except np.linalg.LinAlgError:
                    W_new[w] = np.linalg.lstsq(G_w, b, rcond=None)[0]
                    
            U, V, W = U_new, V_new, W_new
            
            # Check convergence every 10 iterations
            if it % 10 == 0 or it == max_iter - 1:
                recon = np.einsum('ur,vr,wr->uvw', U, V, W)
                target = np.zeros((size_u, size_v, size_w))
                for u, v, w in non_zeros:
                    target[u, v, w] = 1.0
                full_err = np.linalg.norm(recon - target)
                if full_err < tol:
                    return U, V, W, full_err
                    
        return U, V, W, full_err
    
    # Try ranks from 52 to 59
    best_solution = None
    best_rank = n * m * p
    
    for rank in range(52, best_rank):
        if time.time() > deadline - 10:
            break
            
        num_restarts = 15 if rank <= 54 else 8
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                    break
                    
            U, V, W, res = solve_als(rank, max_iter=100, tol=1e-10, seed=42 + restart)
            if res < 1e-8:
                # Round to nearest 0.5
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                if _verify(U_r, V_r, W_r, n, m, p):
                    best_rank = rank
                    best_solution = {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
                    break
        if best_solution:
            break
            
    if best_solution:
        return best_solution

    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((size_u, rank))
    V = np.zeros((size_v, rank))
    W = np.zeros((size_w, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U[u, col] = 1
                V[v, col] = 1
                W[w, col] = 1
                col += 1
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
