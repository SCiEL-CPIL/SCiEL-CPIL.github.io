"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import itertools
import random
import numpy as np

def _verify_factors(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] equals the MM tensor."""
    rank = len(U[0])
    for i, j, k in itertools.product(range(n), range(m), range(p)):
        s = 0
        for c in range(rank):
            s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
        if abs(s - 1) > tol:
            return False
    return True

def _schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W}

def _find_rank19_5x5x5():
    """Attempt to find a rank-19 decomposition for 5x5x5 matrix multiplication
    with half-integer entries using randomized search with local improvement."""
    n = m = p = 5
    rank = 19
    
    best_score = float('inf')
    best_factors = None
    
    for seed in range(50):
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize with random half-integers
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        for c in range(rank):
            U[:, c] = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=n*m)
            V[:, c] = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=m*p)
            W[:, c] = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=p*n)
        
        # Compute error
        def compute_error(U, V, W):
            err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(rank):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        err += (s - 1)**2
            return err
        
        err = compute_error(U, V, W)
        
        # Local search: try flipping individual entries
        for iteration in range(200):
            improved = False
            # Pick a random matrix, row, column
            mat_idx = random.randint(0, 2)
            if mat_idx == 0:
                r = random.randint(0, n*m-1)
                c = random.randint(0, rank-1)
                old_val = U[r, c]
                new_val = old_val + random.choice([-1.0, 1.0])
                U[r, c] = new_val
            elif mat_idx == 1:
                r = random.randint(0, m*p-1)
                c = random.randint(0, rank-1)
                old_val = V[r, c]
                new_val = old_val + random.choice([-1.0, 1.0])
                V[r, c] = new_val
            else:
                r = random.randint(0, p*n-1)
                c = random.randint(0, rank-1)
                old_val = W[r, c]
                new_val = old_val + random.choice([-1.0, 1.0])
                W[r, c] = new_val
            
            new_err = compute_error(U, V, W)
            if new_err < err:
                err = new_err
                improved = True
            else:
                # Revert
                if mat_idx == 0:
                    U[r, c] = old_val
                elif mat_idx == 1:
                    V[r, c] = old_val
                else:
                    W[r, c] = old_val
            
            if not improved:
                break
            
            if err < 1e-10:
                break
        
        if err < best_score:
            best_score = err
            best_factors = (U.copy(), V.copy(), W.copy())
        
        if best_score < 1e-10:
            break
    
    if best_factors is not None and best_score < 1e-9:
        U, V, W = best_factors
        # Round to half-integers
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        # Verify after rounding
        if _verify_factors(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _slice_to_4x5x5(U5, V5, W5, n5=5, m5=5, p5=5, n4=4):
    """Slice a 5x5x5 decomposition to 4x5x5 by removing the 5th index in first dim."""
    rank = len(U5[0])
    
    # U: shape (25, rank) -> keep rows where i < 4 (i*m5+j for i in 0..3)
    U4 = []
    for i in range(n4):
        for j in range(m5):
            row_idx = i * m5 + j
            U4.append(U5[row_idx][:])
    
    # V: shape (25, rank) -> keep all rows (m5=5, p5=5)
    V4 = [row[:] for row in V5]
    
    # W: shape (25, rank) -> keep rows where i < 4 (k*n5+i for i in 0..3)
    W4 = []
    for k in range(p5):
        for i in range(n4):
            row_idx = k * n5 + i
            W4.append(W5[row_idx][:])
    
    return U4, V4, W4

def construct():
    n, m, p = DIMENSIONS
    
    # Try to get rank-19 via slicing from 5x5x5
    factors_5 = _find_rank19_5x5x5()
    if factors_5 is not None:
        U5, V5, W5 = factors_5
        U4, V4, W4 = _slice_to_4x5x5(U5, V5, W5)
        # Verify the sliced decomposition
        if _verify_factors(U4, V4, W4, n, m, p):
            return {"U": U4, "V": V4, "W": W4}
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
