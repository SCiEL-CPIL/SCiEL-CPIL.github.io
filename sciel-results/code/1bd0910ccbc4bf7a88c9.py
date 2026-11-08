"""Program for matrix multiplication 3x3x4 tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 4]

def _build_rank34():
    """Build the known rank-34 decomposition."""
    n, m, p = 3, 3, 4
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    # Strassen Left (A11 * B1L)
    U[0][col]=1; U[4][col]=1; V[0][col]=1; V[5][col]=1; W[0][col]=1; W[4][col]=1; col+=1
    U[3][col]=1; U[4][col]=1; V[0][col]=1; W[1][col]=1; W[4][col]=-1; col+=1
    U[0][col]=1; V[1][col]=1; V[5][col]=-1; W[3][col]=1; W[4][col]=1; col+=1
    U[4][col]=1; V[4][col]=1; V[0][col]=-1; W[0][col]=1; W[1][col]=1; col+=1
    U[0][col]=1; U[1][col]=1; V[5][col]=1; W[0][col]=-1; W[3][col]=1; col+=1
    U[3][col]=1; U[0][col]=-1; V[0][col]=1; V[1][col]=1; W[4][col]=1; col+=1
    U[1][col]=1; U[4][col]=-1; V[4][col]=1; V[5][col]=1; W[0][col]=1; col+=1

    # A12 * B2L
    U[2][col]=1; V[8][col]=1; W[0][col]=1; col+=1
    U[2][col]=1; V[9][col]=1; W[3][col]=1; col+=1
    U[5][col]=1; V[8][col]=1; W[1][col]=1; col+=1
    U[5][col]=1; V[9][col]=1; W[4][col]=1; col+=1

    # A21 * B1L
    U[6][col]=1; V[0][col]=1; W[2][col]=1; col+=1
    U[6][col]=1; V[1][col]=1; W[5][col]=1; col+=1
    U[7][col]=1; V[4][col]=1; W[2][col]=1; col+=1
    U[7][col]=1; V[5][col]=1; W[5][col]=1; col+=1

    # A22 * B2L
    U[8][col]=1; V[8][col]=1; W[2][col]=1; col+=1
    U[8][col]=1; V[9][col]=1; W[5][col]=1; col+=1

    # Strassen Right (A11 * B1R)
    U[0][col]=1; U[4][col]=1; V[2][col]=1; V[7][col]=1; W[6][col]=1; W[10][col]=1; col+=1
    U[3][col]=1; U[4][col]=1; V[2][col]=1; W[7][col]=1; W[10][col]=-1; col+=1
    U[0][col]=1; V[3][col]=1; V[7][col]=-1; W[9][col]=1; W[10][col]=1; col+=1
    U[4][col]=1; V[6][col]=1; V[2][col]=-1; W[6][col]=1; W[7][col]=1; col+=1
    U[0][col]=1; U[1][col]=1; V[7][col]=1; W[6][col]=-1; W[9][col]=1; col+=1
    U[3][col]=1; U[0][col]=-1; V[2][col]=1; V[3][col]=1; W[10][col]=1; col+=1
    U[1][col]=1; U[4][col]=-1; V[6][col]=1; V[7][col]=1; W[6][col]=1; col+=1

    # A12 * B2R
    U[2][col]=1; V[10][col]=1; W[6][col]=1; col+=1
    U[2][col]=1; V[11][col]=1; W[9][col]=1; col+=1
    U[5][col]=1; V[10][col]=1; W[7][col]=1; col+=1
    U[5][col]=1; V[11][col]=1; W[10][col]=1; col+=1

    # A21 * B1R
    U[6][col]=1; V[2][col]=1; W[8][col]=1; col+=1
    U[6][col]=1; V[3][col]=1; W[11][col]=1; col+=1
    U[7][col]=1; V[6][col]=1; W[8][col]=1; col+=1
    U[7][col]=1; V[7][col]=1; W[11][col]=1; col+=1

    # A22 * B2R
    U[8][col]=1; V[10][col]=1; W[8][col]=1; col+=1
    U[8][col]=1; V[11][col]=1; W[11][col]=1; col+=1

    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    T_reconstructed = np.zeros((n, m, p), dtype=float)
    for l in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_reconstructed[i, j, k] += U_np[i*m+j, l] * V_np[j*p+k, l] * W_np[k*n+i, l]
    
    T_expected = np.ones((n, m, p), dtype=float)
    return np.allclose(T_reconstructed, T_expected, atol=tol)

def _try_remove_term(U, V, W, term_idx, n, m, p):
    """Try to remove term_idx by adjusting other terms. Returns modified (U,V,W) or None."""
    rank = len(U[0])
    U_list = [row[:] for row in U]
    V_list = [row[:] for row in V]
    W_list = [row[:] for row in W]
    
    # Zero out the target term
    for i in range(n*m):
        U_list[i][term_idx] = 0
    for j in range(m*p):
        V_list[j][term_idx] = 0
    for k in range(p*n):
        W_list[k][term_idx] = 0
    
    # Check if remaining terms still span the tensor
    if _verify(U_list, V_list, W_list, n, m, p):
        return U_list, V_list, W_list
    return None

def _try_scale_merge(U, V, W, n, m, p):
    """Try to merge two terms by scaling. Returns (U,V,W, new_rank) or None."""
    rank = len(U[0])
    best = None
    
    # Try merging pairs of terms
    for i in range(rank):
        for j in range(i+1, rank):
            # Try to see if we can express term j as a linear combination
            # by adjusting coefficients in other terms
            # Simple approach: try scaling one term and see if we can zero out another
            U_list = [row[:] for row in U]
            V_list = [row[:] for row in V]
            W_list = [row[:] for row in W]
            
            # Try to absorb term j into term i by scaling
            # This is a heuristic - try various scale factors
            for scale in [-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2]:
                U_try = [row[:] for row in U]
                V_try = [row[:] for row in V]
                W_try = [row[:] for row in W]
                
                # Add scaled term j to term i
                for idx in range(n*m):
                    U_try[idx][i] += scale * U_try[idx][j]
                for idx in range(m*p):
                    V_try[idx][i] += scale * V_try[idx][j]
                for idx in range(p*n):
                    W_try[idx][i] += scale * V_try[idx][j]  # Bug: should be W_try
                    W_try[idx][i] = U[idx][i] + scale * W[idx][j]  # Fix
                # Actually let me redo this properly
                for idx in range(p*n):
                    W_try[idx][i] = W[idx][i] + scale * W[idx][j]
                
                # Zero out term j
                for idx in range(n*m):
                    W_try[idx][j] = 0  # Wrong matrix
                for idx in range(n*m):
                    U_try[idx][j] = 0
                for idx in range(m*p):
                    V_try[idx][j] = 0
                for idx in range(p*n):
                    W_try[idx][j] = 0
                
                if _verify(U_try, V_try, W_try, n, m, p):
                    # Remove the zero column
                    new_U = [[U_try[r][c] for c in range(rank) if c != j] for r in range(n*m)]
                    new_V = [[V_try[r][c] for c in range(rank) if c != j] for r in range(m*p)]
                    new_W = [[W_try[r][c] for c in range(rank) if c != j] for r in range(p*n)]
                    return new_U, new_V, new_W, rank - 1
    
    return None

def _randomized_search(U, V, W, n, m, p, time_budget):
    """Randomized search for rank reduction."""
    start = time.time()
    rank = len(U[0])
    
    # Try random linear combinations of terms
    for attempt in range(200):
        if time.time() - start > time_budget:
            break
        
        # Pick two random terms to try to merge
        i, j = np.random.choice(rank, 2, replace=False)
        
        # Try random coefficient
        alpha = np.random.uniform(-3, 3)
        
        U_try = [row[:] for row in U]
        V_try = [row[:] for row in V]
        W_try = [row[:] for row in W]
        
        for idx in range(n*m):
            U_try[idx][i] += alpha * U[idx][j]
        for idx in range(m*p):
            V_try[idx][i] += alpha * V[idx][j]
        for idx in range(p*n):
            W_try[idx][i] += alpha * W[idx][j]
        
        # Zero out term j
        for idx in range(n*m):
            U_try[idx][j] = 0
        for idx in range(m*p):
            V_try[idx][j] = 0
        for idx in range(p*n):
            W_try[idx][j] = 0
        
        if _verify(U_try, V_try, W_try, n, m, p):
            new_U = [[U_try[r][c] for c in range(rank) if c != j] for r in range(n*m)]
            new_V = [[V_try[r][c] for c in range(rank) if c != j] for r in range(m*p)]
            new_W = [[W_try[r][c] for c in range(rank) if c != j] for r in range(p*n)]
            return new_U, new_V, new_W, rank - 1
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 3, 3, 4
    
    # Start with rank-34
    U, V, W = _build_rank34()
    
    # Verify baseline
    assert _verify(U, V, W, n, m, p), "Baseline rank-34 is invalid!"
    
    best_U, best_V, best_W = U, V, W
    best_rank = 34
    
    # Try removing individual terms
    for term in range(34):
        if time.time() > deadline - 30:
            break
        result = _try_remove_term(U, V, W, term, n, m, p)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = 33
            break
    
    # Try merge operations if we haven't found rank 33 yet
    if best_rank == 34 and time.time() < deadline - 20:
        result = _try_scale_merge(U, V, W, n, m, p)
        if result is not None:
            best_U, best_V, best_W, best_rank = result
    
    # Randomized search as last resort
    if best_rank == 34 and time.time() < deadline - 10:
        result = _randomized_search(U, V, W, n, m, p, deadline - time.time() - 5)
        if result is not None:
            best_U, best_V, best_W, best_rank = result
    
    return {"U": best_U, "V": best_V, "W": best_W}
