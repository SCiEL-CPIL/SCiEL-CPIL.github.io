"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n=2, m=2, p=4):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    T_recon = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    if not np.allclose(T, T_recon, atol=1e-9):
        return False
    return True

def _strassen_rank14():
    """Construct the rank-14 decomposition using shared Strassen."""
    n, m, p = 2, 2, 4
    rank = 14
    U = np.zeros((n*m, rank))  # 4 x 14
    V = np.zeros((m*p, rank))  # 8 x 14
    W = np.zeros((p*n, rank))  # 8 x 14
    
    # Left half (k=0,1)
    r = 0
    U[0, r] = 1; U[3, r] = 1
    V[0, r] = 1; V[5, r] = 1
    W[0, r] = 1; W[3, r] = 1
    
    r = 1
    U[2, r] = 1; U[3, r] = 1
    V[0, r] = 1
    W[2, r] = 1; W[3, r] = -1
    
    r = 2
    U[0, r] = 1
    V[4, r] = 1; V[5, r] = -1
    W[1, r] = 1; W[3, r] = 1
    
    r = 3
    U[3, r] = 1
    V[1, r] = 1; V[0, r] = -1
    W[0, r] = 1; W[2, r] = 1
    
    r = 4
    U[0, r] = 1; U[1, r] = 1
    V[5, r] = 1
    W[1, r] = 1; W[0, r] = -1
    
    r = 5
    U[2, r] = 1; U[0, r] = -1
    V[0, r] = 1; V[1, r] = 1
    W[3, r] = 1
    
    r = 6
    U[1, r] = 1; U[3, r] = -1
    V[4, r] = 1; V[5, r] = 1
    W[0, r] = 1
    
    # Right half (k=2,3)
    r = 7
    U[0, r] = 1; U[3, r] = 1
    V[2, r] = 1; V[7, r] = 1
    W[4, r] = 1; W[7, r] = 1
    
    r = 8
    U[2, r] = 1; U[3, r] = 1
    V[2, r] = 1
    W[6, r] = 1; W[7, r] = -1
    
    r = 9
    U[0, r] = 1
    V[6, r] = 1; V[7, r] = -1
    W[5, r] = 1; W[7, r] = 1
    
    r = 10
    U[3, r] = 1
    V[3, r] = 1; V[2, r] = -1
    W[4, r] = 1; W[6, r] = 1
    
    r = 11
    U[0, r] = 1; U[1, r] = 1
    V[7, r] = 1
    W[5, r] = 1; W[4, r] = -1
    
    r = 12
    U[2, r] = 1; U[0, r] = -1
    V[2, r] = 1; V[3, r] = 1
    W[7, r] = 1
    
    r = 13
    U[1, r] = 1; U[3, r] = -1
    V[6, r] = 1; V[7, r] = 1
    W[4, r] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _rank13_attempt():
    """
    Try to find a rank-13 decomposition.
    
    Key insight: For 2x2 x 2x4, the tensor T[i,j,k] = 1 for all (i,j,k).
    The bilinear complexity is the minimum rank of a tensor decomposition.
    
    We model this as: find rank R such that there exist U (4xR), V (8xR), W (8xR)
    with sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k).
    
    Approach: Start from rank-14 Strassen and try to merge two terms by
    expressing one as a linear combination of the others plus a new combined term.
    
    Alternative: Use the fact that the tensor is "all ones" in a certain sense.
    For 2x2 x 2x2, Strassen gives rank 7. For 2x2 x 2x4, we can view it as
    two 2x2 x 2x2 blocks sharing the same A matrix.
    
    The question is whether we can do better than 14 = 7+7 by sharing more.
    
    Let's try a direct algebraic approach. Suppose we have rank R decomposition.
    The tensor T has 16 entries (2*2*4), all equal to 1.
    
    We can try to find a rank-13 solution by starting with rank-14 and
    attempting to eliminate one term while adjusting others.
    """
    n, m, p = 2, 2, 4
    R = 13
    
    # Start from rank-14 and try to combine two terms
    # The idea: if we can find a linear dependency among the 14 rank-1 terms
    # that allows us to merge two into one, we get rank 13.
    
    # Actually, let's try a different approach: random search with structured initialization
    # Initialize with the rank-14 solution, zero out one column, and try to
    # recover using the remaining 13 columns via least squares.
    
    base = _strassen_rank14()
    U14 = np.array(base["U"], dtype=float)
    V14 = np.array(base["V"], dtype=float)
    W14 = np.array(base["W"], dtype=float)
    
    # Try each possible column to remove
    for remove_col in range(14):
        U13 = np.delete(U14, remove_col, axis=1)
        V13 = np.delete(V14, remove_col, axis=1)
        W13 = np.delete(W14, remove_col, axis=1)
        
        # Now try to solve for adjusted U13, V13, W13 such that
        # sum_{r=0}^{12} U13[i*m+j,r] * V13[j*p+k,r] * W13[k*n+i,r] = 1
        
        # This is a system of 16 equations in 4*13 + 8*13 + 8*13 = 286 unknowns
        # But it's nonlinear. Use iterative refinement.
        
        # Initialize with the truncated solution
        U = U13.copy()
        V = V13.copy()
        W = W13.copy()
        
        # Iterate: fix two, solve for third
        for iteration in range(500):
            # Update U
            for i in range(n):
                for j in range(m):
                    # Want: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for k=0,1,2,3
                    A_mat = np.zeros((p, R))
                    for k in range(p):
                        for r in range(R):
                            A_mat[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    b = np.ones(p)
                    u, _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                    U[i*m+j, :] = u
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A_mat = np.zeros((n, R))
                    for i in range(n):
                        for r in range(R):
                            A_mat[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    b = np.ones(n)
                    v, _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                    V[j*p+k, :] = v
            
            # Update W
            for k in range(p):
                for i in range(n):
                    A_mat = np.zeros((m, R))
                    for j in range(m):
                        for r in range(R):
                            A_mat[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    b = np.ones(m)
                    w, _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                    W[k*n+i, :] = w
            
            # Check error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(R))
                        error += (val - 1.0)**2
            if error < 1e-12:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Start with rank-14 Strassen construction
    best = _strassen_rank14()
    best_rank = 14
    
    # Verify the rank-14 solution
    if not _verify(best["U"], best["V"], best["W"], n, m, p):
        # Fallback to schoolbook
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
        return {"U": U, "V": W, "W": W}
    
    # Try rank-13
    if time.time() < deadline - 20:
        result13 = _rank13_attempt()
        if result13 is not None:
            if _verify(result13["U"], result13["V"], result13["W"], n, m, p):
                best = result13
                best_rank = 13
    
    # Try rank-12 with random restarts
    if best_rank > 12 and time.time() < deadline - 10:
        for target_rank in [12]:
            for attempt in range(30):
                if time.time() > deadline - 10:
                    break
                np.random.seed(attempt * 1000 + 42)
                U = np.random.randn(n*m, target_rank) * 0.5
                V = np.random.randn(m*p, target_rank) * 0.5
                W = np.random.randn(p*n, target_rank) * 0.5
                
                for it in range(300):
                    if time.time() > deadline - 10:
                        break
                    # Update U
                    for i in range(n):
                        for j in range(m):
                            A_mat = np.zeros((p, target_rank))
                            for k in range(p):
                                for r in range(target_rank):
                                    A_mat[k, r] = V[j*p+k, r] * W[k*n+i, r]
                            b = np.ones(p)
                            U[i*m+j, :] = np.linalg.lstsq(A_mat, b, rcond=None)[0]
                    
                    # Update V
                    for j in range(m):
                        for k in range(p):
                            A_mat = np.zeros((n, target_rank))
                            for i in range(n):
                                for r in range(target_rank):
                                    A_mat[i, r] = U[i*m+j, r] * W[k*n+i, r]
                            b = np.ones(n)
                            V[j*p+k, :] = np.linalg.lstsq(A_mat, b, rcond=None)[0]
                    
                    # Update W
                    for k in range(p):
                        for i in range(n):
                            A_mat = np.zeros((m, target_rank))
                            for j in range(m):
                                for r in range(target_rank):
                                    A_mat[j, r] = U[i*m+j, r] * V[j*p+k, r]
                            b = np.ones(m)
                            W[k*n+i, :] = np.linalg.lstsq(A_mat, b, rcond=None)[0]
                    
                    # Check error
                    error = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(target_rank))
                                error += (val - 1.0)**2
                    if error < 1e-10:
                        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                            best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                            best_rank = target_rank
                        break
            
            if best_rank == target_rank:
                break
    
    return best
# EVOLVE-BLOCK-END
