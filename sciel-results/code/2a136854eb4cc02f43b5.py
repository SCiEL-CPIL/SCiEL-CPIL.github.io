"""Matrix multiplication 3,4,8 rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 4, 8

    # Schoolbook fallback
    def schoolbook(n, m, p):
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
        return U, V, W, rank

    # Verify a decomposition
    def verify(U, V, W, n, m, p, tol=1e-6):
        r = len(U[0])
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Build full tensor and check only the "compatible" entries
        # T[i*m+j, j*p+k, k*n+i] should be 1 for all i,j,k
        # All other entries should be 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        # Check incompatible entries are 0
        # For each (a, b, c) triple, check if it's compatible
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    i_c = c % n
                    k_c = c // n
                    if j_a != j_b or k_b != k_c or i_a != i_c:
                        val = 0.0
                        for cc in range(r):
                            val += U_np[a, cc] * V_np[b, cc] * W_np[c, cc]
                        if abs(val) > tol:
                            return False
        return True

    # Strategy: Use tensor product structure
    # 3x4x8 = 3x(3+1)x(4*2)
    # Decompose as: (3x3x4) tensor (1x1x2) + (3x1x4) tensor (1x1x2)
    # Or more carefully:
    # j = j3 + delta where j3 in {0,1,2} and delta in {0,1}
    # k = 4*k4 + k2 where k4 in {0,1}, k2 in {0,1,2,3}
    
    # Let me try a different approach: direct structured search
    # Start with schoolbook and try to merge terms
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook(n, m, p)
    
    # Try to find a lower rank decomposition using the structure
    # Key idea: for fixed i, the row of C is A[i,:] @ B
    # C[i,k] = sum_j A[i,j] * B[k,j]
    
    # Let's try to use a 3x3x3 rank-7 decomposition as a building block
    # Known: 3x3x3 has rank 23 (Landsberg), but let's use a simpler approach
    
    # Actually, let me try a block decomposition:
    # Split B into two 4x4 blocks: B_top (k=0..3) and B_bot (k=4..7)
    # C_top[i,k] = sum_j A[i,j] * B_top[k,j]  (3x4x4 multiplication)
    # C_bot[i,k] = sum_j A[i,j] * B_bot[k,j]  (3x4x4 multiplication)
    
    # If each 3x4x4 has rank r, total rank is 2r
    # Schoolbook 3x4x4: rank 48, total 96
    
    # Let me try to find a better 3x4x4 decomposition
    # 3x4x4 = 3x(3+1)x4 = (3x3x4) + (3x1x4)
    # 3x3x4: can use Strassen-like for 3x3x3 part
    # 3x1x4: rank 12 (schoolbook)
    
    # For 3x3x3, known rank is 23 (Landsberg) or we can use 27 (schoolbook)
    # Let me try to construct a 3x3x3 rank-7 decomposition (Strassen-style)
    
    # Actually, for 2x2x2, Strassen gives rank 7.
    # For 3x3x3, let me use a block approach:
    # Embed 3x3x3 in 4x4x4 and use known results, or just use schoolbook.
    
    # Let me try a different strategy: randomized search
    # Start with rank r and try to solve the nonlinear system
    
    def try_random_rank(r, n, m, p, max_iters=200, seed=42):
        """Try to find a rank-r decomposition via random initialization and gradient descent."""
        rng = np.random.RandomState(seed)
        U = rng.randn(n * m, r)
        V = rng.randn(m * p, r)
        W = rng.randn(p * n, r)
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            # Compute loss: sum of squares of (T_compatible - 1) and (T_incompatible - 0)
            loss = 0.0
            # Compatible entries
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        loss += (val - 1.0)**2
            # Incompatible entries (sample a subset for speed)
            # For now, skip incompatible check during optimization
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_loss < 1e-8:
                break
            
            # Simple gradient step (approximate)
            lr = 0.01 / (1 + it * 0.01)
            # Compute gradients numerically (expensive but simple)
            # For now, just do a random perturbation
            U += lr * rng.randn(*U.shape) * np.sqrt(best_loss)
            V += lr * rng.randn(*V.shape) * np.sqrt(best_loss)
            W += lr * rng.randn(*W.shape) * np.sqrt(best_loss)
        
        # Verify
        if verify(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p):
            return best_U.tolist(), best_V.tolist(), best_W.tolist(), r
        return None
    
    # Try various ranks
    best_result = (U_sb, V_sb, W_sb, rank_sb)
    
    for r in [90, 88, 86, 84, 80, 78, 76, 72, 68, 64, 60, 56, 52, 48]:
        if time.time() > deadline - 10:
            break
        result = try_random_rank(r, n, m, p, max_iters=50, seed=r)
        if result is not None:
            best_result = result
            break
    
    U, V, W, rank = best_result
    
    # Final verification
    if not verify(U, V, W, n, m, p):
        # Fall back to schoolbook
        U, V, W, rank = schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
