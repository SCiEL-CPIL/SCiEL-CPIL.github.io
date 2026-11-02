import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 5, 6]

def _find_333_rank23(deadline):
    """Try to find a rank-23 decomposition for 3x3 x 3x3."""
    n = 3
    R = 23
    # Valid entries: (i*3+k, k*3+j, j*3+i) for i,k,j in 0..2
    valid = []
    for i in range(n):
        for k in range(n):
            for j in range(n):
                valid.append((i*n+k, k*n+j, j*n+i))
    
    # Invalid entries
    all_entries = set()
    for a in range(n*n):
        for b in range(n*n):
            for c in range(n*n):
                all_entries.add((a,b,c))
    valid_set = set(valid)
    invalid = [(a,b,c) for (a,b,c) in all_entries if (a,b,c) not in valid_set]
    
    def residuals(x):
        U = x[:n*n*R].reshape(n*n, R)
        V = x[n*n*R:2*n*n*R].reshape(n*n, R)
        W = x[2*n*n*R:].reshape(n*n, R)
        res = []
        for a,b,c in valid:
            res.append(np.sum(U[a]*V[b]*W[c]) - 1.0)
        for a,b,c in invalid:
            res.append(np.sum(U[a]*V[b]*W[c]))
        return res
    
    # Initialize with schoolbook-like solution
    x0 = np.zeros(3*n*n*R)
    for idx, (a,b,c) in enumerate(valid[:R]):
        x0[a*R+idx] = 1.0
        x0[n*n*R + b*R+idx] = 1.0
        x0[2*n*n*R + c*R+idx] = 1.0
    
    if time.time() > deadline - 10:
        return None
    
    result = least_squares(residuals, x0, method='lm', max_nfev=1000)
    
    if result.cost < 1e-8:
        U = result.x[:n*n*R].reshape(n*n, R)
        V = result.x[n*n*R:2*n*n*R].reshape(n*n, R)
        W = result.x[2*n*n*R:].reshape(n*n, R)
        return (U, V, W)
    return None

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Try to find a rank-23 decomposition for 3x3 x 3x3
    result_333 = _find_333_rank23(deadline)
    
    if result_333 is not None and time.time() < deadline - 5:
        U333, V333, W333 = result_333
        # Use the 3+2 block split
        # A (3x5) = [A1 (3x3) | A2 (3x2)]
        # B (5x6) = [B1 (3x6); B2 (2x6)]
        # B1 = [B1a (3x3) | B1b (3x3)]
        # C = A1*B1a + A1*B1b + A2*B2
        # Rank: 23 + 23 + 36 = 82
        
        R = 82
        U = [[0]*R for _ in range(n*m)]
        V = [[0]*R for _ in range(m*p)]
        W = [[0]*R for _ in range(p*n)]
        
        # A1*B1a: 3x3 x 3x3, rank 23
        # i in 0..2, k in 0..2, j in 0..2
        for r in range(23):
            for i in range(3):
                for k in range(3):
                    U[i*5+k][r] = U333[i*3+k][r]
                    V[k*6+j][r] = V333[k*3+j][r]  # j in 0..2 for B1a
                    W[j*3+i][r] = W333[j*3+i][r]
        
        # A1*B1b: 3x3 x 3x3, rank 23 (columns 3..5 of B)
        for r in range(23):
            for i in range(3):
                for k in range(3):
                    U[i*5+k][23+r] = U333[i*3+k][r]
                    V[k*6+3+j][23+r] = V333[k*3+j][r]  # j in 0..2 for B1b
                    W[(3+j)*3+i][23+r] = W333[j*3+i][r]
        
        # A2*B2: 3x2 x 2x6, schoolbook rank 36
        col = 46
        for i in range(3):
            for k in range(2, 5):  # k=2,3,4
                for j in range(6):
                    U[i*5+k][col] = 1
                    V[k*6+j][col] = 1
                    W[j*3+i][col] = 1
                    col += 1
        
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k][col] = 1
                    V[k * p + j][col] = 1
                    W[j * n + i][col] = 1
                    col += 1
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        
        return {"U": U, "V": V, "W": W}
