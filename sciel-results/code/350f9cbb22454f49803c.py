import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin
    
    # Build the tensor T[a,b,c]
    # a = i*m+j, b = j*p+k, c = k*n+i
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a,b,c] = 1.0
    
    # Try ranks from 15 to 29
    for rank in range(15, 30):
        if time.time() > deadline - 15:
            break
        
        result = try_find_decomposition(T, n, m, p, rank)
        if result is not None:
            U, V, W = result
            U_list = [[float(Fraction(U[i][j]).limit_denominator(2)) for j in range(rank)] for i in range(n*m)]
            V_list = [[float(Fraction(V[i][j]).limit_denominator(2)) for j in range(rank)] for i in range(m*p)]
            W_list = [[float(Fraction(W[i][j]).limit_denominator(2)) for j in range(rank)] for i in range(p*n)]
            
            if verify(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook rank 30
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
    return {"U": U, "V": V, "W": W}

def verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                if abs(val - 1.0) > 1e-9:
                    return False
    
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                expected = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if a == i*m+j and b == j*p+k and c == k*n+i:
                                expected = 1.0
                                break
                        if expected != 0: break
                    if expected != 0: break
                if expected != 0: continue
                
                val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                if abs(val) > 1e-9:
                    return False
    
    return True

def try_find_decomposition(T, n, m, p, rank):
    """Try to find a rank-r decomposition by fixing U,V and solving for W."""
    np.random.seed(42)
    
    n_am = n * m
    n_bmp = m * p
    n_cpn = p * n
    
    max_attempts = 80
    for attempt in range(max_attempts):
        if time.time() > 110:
            break
        
        # Initialize U and V with small half-integer values
        U = np.random.randint(-2, 3, size=(n_am, rank)) * 0.5
        V = np.random.randint(-2, 3, size=(n_bmp, rank)) * 0.5
        
        # Ensure no all-zero rows
        for i in range(n_am):
            if np.all(U[i] == 0):
                U[i, 0] = 0.5
        for i in range(n_bmp):
            if np.all(V[i] == 0):
                V[i, 0] = 0.5
        
        W = np.zeros((n_cpn, rank))
        
        valid = True
        for c in range(n_cpn):
            # Build M_c: (n_am * n_bmp) x rank, M_c[(a,b), r] = U[a,r] * V[b,r]
            M_c = np.zeros((n_am * n_bmp, rank))
            for a in range(n_am):
                for b in range(n_bmp):
                    M_c[a * n_bmp + b, :] = U[a, :] * V[b, :]
            
            rhs = T[:, :, c].flatten()
            
            try:
                W_c, residuals, rank_m, sv = np.linalg.lstsq(M_c, rhs, rcond=None)
            except:
                valid = False
                break
            
            # Check residual
            if residuals is not None and len(residuals) > 0 and residuals[0] > 1e-8:
                valid = False
                break
            
            # Check if W_c is approximately half-integer
            W_c_rounded = np.round(W_c * 2) / 2
            if np.max(np.abs(W_c - W_c_rounded)) > 1e-8:
                valid = False
                break
            
            W[c, :] = W_c_rounded
        
        if valid:
            U_list = [[float(U[i][j]) for j in range(rank)] for i in range(n_am)]
            V_list = [[float(V[i][j]) for j in range(rank)] for i in range(n_bmp)]
            W_list = [[float(W[i][j]) for j in range(rank)] for i in range(n_cpn)]
            
            if verify(U_list, V_list, W_list, n, m, p):
                return (U_list, V_list, W_list)
    
    return None
