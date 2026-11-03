import time
import numpy as np
from itertools import product

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

def _build_sliding_strassen_2x2x8():
    """
    Build a sliding-window Strassen decomposition for 2x2x8.
    
    For columns 0-7, we compute C[:, 0:8] = A[:, 0:2] @ B[0:2, 0:8].
    
    We use a sliding window of width 2 over the 8 columns of B.
    For each pair of adjacent output columns (2k, 2k+1), we apply Strassen
    to the 2x2 sub-problem, but share the linear combinations of B's rows
    across adjacent windows where possible.
    
    Standard Strassen for 2x2x2 uses 7 products. For 4 independent blocks: 28.
    With sliding window sharing, we aim to reduce this.
    
    The key idea: for block k (columns 2k, 2k+1) and block k+1 (columns 2k+2, 2k+3),
    the B matrices share no columns, so standard Strassen doesn't directly share.
    However, we can reformulate using a different partition.
    
    Alternative: Use the fact that we're multiplying 2x2 by 2x8.
    We can use a recursive/iterative approach.
    
    Let's try: decompose B[0:2, 0:8] into 4 blocks of 2x2.
    For each block, use Strassen. That's 4*7 = 28.
    
    To reduce, we can try to combine adjacent Strassen computations.
    
    Actually, let's try a different approach: use a 2x2x4 Strassen-like scheme
    for the first 4 columns, then another for the last 4.
    
    Or: use the "baby-step giant-step" idea.
    
    For now, let's just build the standard 28-term Strassen for 2x2x8 and see
    if we can find a lower-rank decomposition via optimization.
    """
    n, m, p = 2, 2, 8
    U = [[0] * 28 for _ in range(n * m)]
    V = [[0] * 28 for _ in range(m * p)]
    W = [[0] * 28 for _ in range(p * n)]
    
    col = 0
    for block in range(4):
        # m1 = (a00+a11)(b00+b11)
        U[0][col] = 1; U[3][col] = 1
        V[0*8 + block*2][col] = 1; V[1*8 + block*2+1][col] = 1
        W[(block*2)*2 + 0][col] = 1; W[(block*2+1)*2 + 1][col] = 1
        col += 1
        # m2 = (a10+a11)b00
        U[2][col] = 1; U[3][col] = 1
        V[0*8 + block*2][col] = 1
        W[(block*2)*2 + 1][col] = 1; W[(block*2+1)*2 + 1][col] = -1
        col += 1
        # m3 = a00(b01-b11)
        U[0][col] = 1
        V[0*8 + block*2+1][col] = 1; V[1*8 + block*2+1][col] = -1
        W[(block*2+1)*2 + 0][col] = 1; W[(block*2+1)*2 + 1][col] = 1
        col += 1
        # m4 = a11(b10-b00)
        U[3][col] = 1
        V[1*8 + block*2][col] = 1; V[0*8 + block*2][col] = -1
        W[(block*2)*2 + 0][col] = 1; W[(block*2)*2 + 1][col] = 1
        col += 1
        # m5 = (a00+a01)b11
        U[0][col] = 1; U[1][col] = 1
        V[1*8 + block*2+1][col] = 1
        W[(block*2)*2 + 0][col] = -1; W[(block*2+1)*2 + 0][col] = 1
        col += 1
        # m6 = (a10-a00)(b00+b01)
        U[2][col] = 1; U[0][col] = -1
        V[0*8 + block*2][col] = 1; V[0*8 + block*2+1][col] = 1
        W[(block*2+1)*2 + 1][col] = 1
        col += 1
        # m7 = (a01-a11)(b10+b11)
        U[1][col] = 1; U[3][col] = -1
        V[1*8 + block*2][col] = 1; V[1*8 + block*2+1][col] = 1
        W[(block*2)*2 + 0][col] = 1
        col += 1
    
    return U, V, W

def _try_rank(rank, n=2, m=3, p=8, timeout=10):
    """Try to find a decomposition of given rank using random initialization + gradient descent."""
    start = time.time()
    np.random.seed(rank * 1000 + n * m * p)
    
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for l in range(p):
                T[i*m+j, j*p+l, l*n+i] = 1.0
    
    lr = 0.01
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
    if err < 1e-6:
        return U, V, W, err
    return None, None, None, err

def construct():
    n, m, p = 2, 3, 8
    deadline = time.time() + 115
    
    # Strategy: decompose into 2x2x8 (cols 0-1) + 2x1x8 (col 2)
    # Part A: 2x2x8 via Strassen = 28 terms
    # Part B: 2x1x8 via schoolbook = 16 terms
    # Total: 44 terms (same as before, but different structure)
    
    # Let's try to find a better decomposition for the 2x2x8 part.
    # The known lower bound for 2x2x8 is at most 28 (Strassen).
    # Can we do better? The tensor rank of 2x2x8 is not well-studied.
    
    # For now, use the standard approach and try to optimize.
    
    # Build the 44-term solution
    U44 = [[0] * 44 for _ in range(n * m)]
    V44 = [[0] * 44 for _ in range(m * p)]
    W44 = [[0] * 44 for _ in range(p * n)]
    
    col = 0
    # Part 1: schoolbook for column 0 (A[:,0] @ B[0,:])
    for i in range(2):
        for k in range(8):
            U44[i * 3 + 0][col] = 1
            V44[0 * 8 + k][col] = 1
            W44[k * 2 + i][col] = 1
            col += 1
    
    # Part 2: Strassen for 4 blocks of 2x2 (A[:,1:3] @ B[1:3,:])
    for block in range(4):
        # m1 = (a00+a11)(b00+b11)
        U44[1][col] = 1; U44[5][col] = 1
        V44[1*8 + block*2][col] = 1; V44[2*8 + block*2+1][col] = 1
        W44[(block*2)*2 + 0][col] = 1; W44[(block*2+1)*2 + 1][col] = 1
        col += 1
        # m2 = (a10+a11)b00
        U44[4][col] = 1; U44[5][col] = 1
        V44[1*8 + block*2][col] = 1
        W44[(block*2)*2 + 1][col] = 1; W44[(block*2+1)*2 + 1][col] = -1
        col += 1
        # m3 = a00(b01-b11)
        U44[1][col] = 1
        V44[1*8 + block*2+1][col] = 1; V44[2*8 + block*2+1][col] = -1
        W44[(block*2+1)*2 + 0][col] = 1; W44[(block*2+1)*2 + 1][col] = 1
        col += 1
        # m4 = a11(b10-b00)
        U44[5][col] = 1
        V44[2*8 + block*2][col] = 1; V44[1*8 + block*2][col] = -1
        W44[(block*2)*2 + 0][col] = 1; W44[(block*2)*2 + 1][col] = 1
        col += 1
        # m5 = (a00+a01)b11
        U44[1][col] = 1; U44[2][col] = 1
        V44[2*8 + block*2+1][col] = 1
        W44[(block*2)*2 + 0][col] = -1; W44[(block*2+1)*2 + 0][col] = 1
        col += 1
        # m6 = (a10-a00)(b00+b01)
        U44[4][col] = 1; U44[1][col] = -1
        V44[1*8 + block*2][col] = 1; V44[1*8 + block*2+1][col] = 1
        W44[(block*2+1)*2 + 1][col] = 1
        col += 1
        # m7 = (a01-a11)(b10+b11)
        U44[2][col] = 1; U44[5][col] = -1
        V44[2*8 + block*2][col] = 1; V44[2*8 + block*2+1][col] = 1
        W44[(block*2)*2 + 0][col] = 1
        col += 1
    
    valid44 = _verify_decomposition(U44, V44, W44, n, m, p)
    
    if not valid44:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = (U44, V44, W44)
    
    # Try to find lower rank solutions via optimization
    for r in range(43, 35, -1):
        if time.time() > deadline - 15:
            break
        U_r, V_r, W_r, err = _try_rank(r, n, m, p, timeout=12)
        if U_r is not None and err < 1e-6:
            if _verify_decomposition(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                best_result = (U_r.tolist(), V_r.tolist(), W_r.tolist())
                break
    
    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
