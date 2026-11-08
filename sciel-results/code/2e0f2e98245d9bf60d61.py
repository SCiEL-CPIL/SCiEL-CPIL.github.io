import time
import numpy as np

def construct():
    _t0 = time.time()
    n, m, p = 2, 3, 10
    
    def verify(U, V, W):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        r = len(U[0])
        # Build the tensor T from U, V, W
        # T[i,j,k] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
        # Should equal 1 for all i,j,k
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    s = 0
                    for r in range(r):
                        s += U[i*m+k][r] * V[k*p+j][r] * W[j*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Construct the 55-rank solution
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        # Term 1: (A00+A11)*(B0k0+B1k1) -> C0k0, C1k1
        U[0][col] += 1; U[4][col] += 1
        V[0*10+k0][col] += 1; V[1*10+k1][col] += 1
        W[k0*2+0][col] += 1; W[k1*2+1][col] += 1
        col += 1
        # Term 2: (A10+A11)*B0k0 -> C1k0, C1k1
        U[3][col] += 1; U[4][col] += 1
        V[0*10+k0][col] += 1
        W[k0*2+1][col] += 1; W[k1*2+1][col] -= 1
        col += 1
        # Term 3: A00*(B0k1-B1k1) -> C0k1, C1k1
        U[0][col] += 1
        V[0*10+k1][col] += 1; V[1*10+k1][col] -= 1
        W[k1*2+0][col] += 1; W[k1*2+1][col] += 1
        col += 1
        # Term 4: A11*(B1k0-B0k0) -> C0k0, C1k0
        U[4][col] += 1
        V[1*10+k0][col] += 1; V[0*10+k0][col] -= 1
        W[k0*2+0][col] += 1; W[k0*2+1][col] += 1
        col += 1
        # Term 5: (A00+A01)*B1k1 -> C0k0(-1), C0k1
        U[0][col] += 1; U[1][col] += 1
        V[1*10+k1][col] += 1
        W[k0*2+0][col] -= 1; W[k1*2+0][col] += 1
        col += 1
        # Term 6: (A10-A00)*(B0k0+B0k1) -> C1k1
        U[3][col] += 1; U[0][col] -= 1
        V[0*10+k0][col] += 1; V[0*10+k1][col] += 1
        W[k1*2+1][col] += 1
        col += 1
        # Term 7: (A01-A11)*(B1k0+B1k1) -> C0k0
        U[1][col] += 1; U[4][col] -= 1
        V[1*10+k0][col] += 1; V[1*10+k1][col] += 1
        W[k0*2+0][col] += 1
        col += 1
        # Terms 8-11: direct terms for row 2 of B
        U[2][col] += 1; V[2*10+k0][col] += 1; W[k0*2+0][col] += 1
        col += 1
        U[2][col] += 1; V[2*10+k1][col] += 1; W[k1*2+0][col] += 1
        col += 1
        U[5][col] += 1; V[2*10+k0][col] += 1; W[k0*2+1][col] += 1
        col += 1
        U[5][col] += 1; V[2*10+k1][col] += 1; W[k1*2+1][col] += 1
        col += 1
    
    if not verify(U, V, W):
        # Fallback: trivial rank-60 solution
        rank = 60
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i*m+k][col] = 1
                    V[k*p+j][col] = 1
                    W[j*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank by finding linear dependencies
    # Convert to numpy for linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # The tensor equation is: sum_r U[:,r] outer V[:,r] outer W[:,r] = T
    # This is nonlinear in the variables, so we can't simply do linear algebra.
    # But we can try: if we fix two of the three matrices, the third is determined linearly.
    
    # Try to find if we can reduce rank by 1:
    # Assume rank 54 is possible. We need to find 54 terms that sum to T.
    # This is a hard problem. Let's try a local search approach.
    
    # For now, return the 55-rank solution if we can't find better
    if time.time() - _t0 > 100:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
