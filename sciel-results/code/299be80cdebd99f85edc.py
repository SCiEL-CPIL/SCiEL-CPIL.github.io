"""Optimized search for a rank-minimal exact tensor decomposition of 3x3 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 3]

# Known rank for 3x3 matrix multiplication is 23 (Landsberg).
# We will attempt to find a decomposition of rank 23 or lower.

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    # The tensor T has indices (i, j, k) for i in 0..n-1, j in 0..m-1, k in 0..p-1
    # T[i, j, k] = 1 for all i, j, k.
    # The decomposition is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i, j, k]
    
    # We can use the known rank-23 decomposition for 3x3 matrix multiplication.
    # One such decomposition is given by the work of Strassen and others.
    # Here is a specific rank-23 decomposition.
    
    # Since finding a new rank-23 decomposition from scratch via search is hard,
    # and the task asks for "rank-minimal", which is 23, we can return a known rank-23 decomposition.
    
    # However, the problem states "Find a rank-minimal exact tensor decomposition".
    # If we can just return a known one, that would be ideal.
    
    # Let's try to construct a rank-23 decomposition using known results.
    # The standard rank-23 decomposition for 3x3 matrices is complex to write out manually.
    
    # Alternative approach: Use numerical optimization to find a low-rank decomposition.
    # But exactness is required ("integer/half-integer entries that exactly decompose").
    
    # Given the complexity, let's return the schoolbook decomposition of rank 27 first,
    # but try to improve it using a search if time permits.
    
    # Actually, the known rank for 3x3 is 23. Let's see if we can find it.
    # A common approach is to use the "tensor rank" algorithms, but they are computationally expensive.
    
    # For now, let's return the schoolbook decomposition of rank 27, which is guaranteed to work.
    # If we have time, we can try to optimize.
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    result = schoolbook()
    
    # Check if we have time to try to find a better decomposition
    elapsed = time.time() - start_time
    if elapsed > 50:
        return result
    
    # Try to find a rank-23 decomposition using known algorithms or search
    # This is a placeholder for a more sophisticated search
    # For now, we'll just return the schoolbook result
    
    return result

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    Uses Strassen's 7-term algorithm for the 2x2 core and optimized border terms.
    Attempts ranks 20-25, then 26 as fallback.
    """
    n, m, p = DIMENSIONS
    
    def idx_u(i, j):
        return i * m + j
        
    def idx_v(j, k):
        return j * p + k
        
    def idx_w(k, i):
        return k * n + i
    
    def verify(U, V, W, R):
        U_r = U.reshape((n, m, R))
        V_r = V.reshape((m, p, R))
        W_r = W.reshape((p, n, R))
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        T_target = np.ones((n, m, p))
        return np.allclose(T_recon, T_target, atol=1e-9)
    
    def build_strassen(U, V, W):
        """Build the 7 Strassen terms in columns 0-6."""
        r = 0
        U[idx_u(0, 0), r] = 1
        U[idx_u(1, 1), r] = 1
        V[idx_v(0, 0), r] = 1
        V[idx_v(1, 1), r] = 1
        W[idx_w(0, 0), r] = 1
        W[idx_w(1, 1), r] = 1
        
        r = 1
        U[idx_u(1, 0), r] = 1
        U[idx_u(1, 1), r] = 1
        V[idx_v(0, 0), r] = 1
        W[idx_w(0, 1), r] = 1
        W[idx_w(1, 1), r] = -1
        
        r = 2
        U[idx_u(0, 0), r] = 1
        V[idx_v(0, 1), r] = 1
        V[idx_v(1, 1), r] = -1
        W[idx_w(1, 0), r] = 1
        W[idx_w(1, 1), r] = 1
        
        r = 3
        U[idx_u(1, 1), r] = 1
        V[idx_v(1, 0), r] = 1
        V[idx_v(0, 0), r] = -1
        W[idx_w(0, 0), r] = 1
        W[idx_w(0, 1), r] = 1
        
        r = 4
        U[idx_u(0, 0), r] = 1
        U[idx_u(0, 1), r] = 1
        V[idx_v(1, 1), r] = 1
        W[idx_w(0, 0), r] = -1
        W[idx_w(1, 0), r] = 1
        
        r = 5
        U[idx_u(1, 0), r] = 1
        U[idx_u(0, 0), r] = -1
        V[idx_v(0, 0), r] = 1
        V[idx_v(0, 1), r] = 1
        W[idx_w(1, 1), r] = 1
        
        r = 6
        U[idx_u(0, 1), r] = 1
        U[idx_u(1, 1), r] = -1
        V[idx_v(1, 0), r] = 1
        V[idx_v(1, 1), r] = 1
        W[idx_w(0, 0), r] = 1
    
    # --- Try rank 25: Strassen(7) + 18 optimized border terms ---
    # Strategy: combine border terms where possible to reduce count
    # The 20 border terms (where at least one of i,j,k is 2) can be reduced
    # by combining terms that share U or V factors.
    
    def try_rank(R):
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        build_strassen(U, V, W)
        
        col = 7
        # i=2 row terms (7 terms, some combined):
        # a20*(b00+b01) -> c20, c21
        U[idx_u(2, 0), col] = 1; V[idx_v(0, 0), col] = 1; V[idx_v(0, 1), col] = 1; W[idx_w(0, 2), col] = 1; W[idx_w(1, 2), col] = 1; col += 1
        # a20*b02 -> c22
        U[idx_u(2, 0), col] = 1; V[idx_v(0, 2), col] = 1; W[idx_w(2, 2), col] = 1; col += 1
        # a21*(b10+b11) -> c20, c21
        U[idx_u(2, 1), col] = 1; V[idx_v(1, 0), col] = 1; V[idx_v(1, 1), col] = 1; W[idx_w(0, 2), col] = 1; W[idx_w(1, 2), col] = 1; col += 1
        # a21*b12 -> c22
        U[idx_u(2, 1), col] = 1; V[idx_v(1, 2), col] = 1; W[idx_w(2, 2), col] = 1; col += 1
        # a22*b20 -> c20
        U[idx_u(2, 2), col] = 1; V[idx_v(2, 0), col] = 1; W[idx_w(0, 2), col] = 1; col += 1
        # a22*b21 -> c21
        U[idx_u(2, 2), col] = 1; V[idx_v(2, 1), col] = 1; W[idx_w(1, 2), col] = 1; col += 1
        # a22*b22 -> c22
        U[idx_u(2, 2), col] = 1; V[idx_v(2, 2), col] = 1; W[idx_w(2, 2), col] = 1; col += 1
        # j=2 column terms (3 terms, combined):
        # (a00+a10)*b20 -> c02, c12
        U[idx_u(0, 0), col] = 1; U[idx_u(1, 0), col] = 1; V[idx_v(2, 0), col] = 1; W[idx_w(0, 0), col] = 1; W[idx_w(0, 1), col] = 1; col += 1
        # (a01+a11)*b21 -> c02, c12
        U[idx_u(0, 1), col] = 1; U[idx_u(1, 1), col] = 1; V[idx_v(2, 1), col] = 1; W[idx_w(1, 0), col] = 1; W[idx_w(1, 1), col] = 1; col += 1
        # (a02+a12)*b22 -> c02, c12
        U[idx_u(0, 2), col] = 1; U[idx_u(1, 2), col] = 1; V[idx_v(2, 2), col] = 1; W[idx_w(2, 0), col] = 1; W[idx_w(2, 1), col] = 1; col += 1
        # k=2 terms (8 terms):
        # c02 = a00*b02 + a01*b12 + a02*b22 (already handled a02*b22 above)
        # Need: a00*b02, a01*b12 for c02
        # c12 = a10*b02 + a11*b12 + a12*b22 (already handled a12*b22 above)
        # Need: a10*b02, a11*b12 for c12
        # c22 = a20*b02 + a21*b12 + a22*b22 (a22*b22 done, a20*b02 done, a21*b12 done)
        # Wait, a20*b02 and a21*b12 were in the i=2 terms above!
        # Let me recheck: in i=2 terms, we have a20*b02 -> c22 and a21*b12 -> c22. Good.
        # So for k=2, we only need: a00*b02->c02, a01*b12->c02, a10*b02->c12, a11*b12->c12
        # Can we combine? (a00+a10)*b02 -> c02, c12 and (a01+a11)*b12 -> c02, c12
        U[idx_u(0, 0), col] = 1; U[idx_u(1, 0), col] = 1; V[idx_v(0, 2), col] = 1; W[idx_w(2, 0), col] = 1; W[idx_w(2, 1), col] = 1; col += 1
        U[idx_u(0, 1), col] = 1; U[idx_u(1, 1), col] = 1; V[idx_v(1, 2), col] = 1; W[idx_w(2, 0), col] = 1; W[idx_w(2, 1), col] = 1; col += 1
        
        if col > R:
            return None
        if verify(U, V, W, R):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return None
    
    # Try rank 20 (7 + 13 border terms) - likely won't work but try
    for R in [20, 21, 22, 23, 24, 25]:
        result = try_rank(R)
        if result is not None:
            return result
    
    # --- Fallback: rank 26 (Strassen 7 + 19 separate border terms) ---
    R = 26
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    build_strassen(U, V, W)
    
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U[idx_u(i, j), col] = 1
                V[idx_v(j, k), col] = 1
                W[idx_w(k, i), col] = 1
                col += 1
    
    if verify(U, V, W, R):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # --- Final fallback: schoolbook rank 27 ---
    R = 27
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[idx_u(i, j), col] = 1
                V[idx_v(j, k), col] = 1
                W[idx_w(k, i), col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
    
    # --- Fallback: rank 26 (Strassen 7 + 19 separate border terms) ---
    R = 26
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    build_strassen(U, V, W)
    
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U[idx_u(i, j), col] = 1
                V[idx_v(j, k), col] = 1
                W[idx_w(k, i), col] = 1
                col += 1
    
    if verify(U, V, W, R):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # --- Final fallback: schoolbook rank 27 ---
    R = 27
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[idx_u(i, j), col] = 1
                V[idx_v(j, k), col] = 1
                W[idx_w(k, i), col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
