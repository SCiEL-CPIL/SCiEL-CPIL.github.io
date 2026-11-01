import numpy as np
from fractions import Fraction
import time

def _verify(U, V, W, n, m, p, tol=1e-9):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(float(s) - 1) > tol:
                    return False
    return True

def _construct_rank19(n, m, p):
    rank = 19
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Strassen for cols 0-1
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[1][1] += 1; W[3][1] -= 1
    
    U[0][2] += 1
    V[1][2] += 1; V[3][2] -= 1
    W[2][2] += 1; W[3][2] += 1
    
    U[3][3] += 1
    V[2][3] += 1; V[0][3] -= 1
    W[0][3] += 1; W[1][3] += 1
    
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[0][4] -= 1; W[2][4] += 1
    
    U[2][5] += 1; U[0][5] -= 1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1
    
    U[1][6] += 1; U[3][6] -= 1
    V[2][6] += 1; V[3][6] += 1
    W[0][6] += 1; W[3][6] += 1
    
    # Schoolbook for cols 2-4
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(2, p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _als_rank18(n, m, p, max_trials=10, max_iter=300, deadline=None):
    R = 18
    for trial in range(max_trials):
        if deadline and time.time() > deadline - 3:
            break
        
        np.random.seed(42 + trial)
        U = np.random.randn(n*m, R) * 0.1
        V = np.random.randn(m*p, R) * 0.1
        W = np.random.randn(p*n, R) * 0.1
        
        for it in range(max_iter):
            if deadline and time.time() > deadline - 3:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    A_mat = np.array([V[j*p+k] * W[k*n+i] for k in range(p)])
                    ATA = A_mat.T @ A_mat + 1e-6 * np.eye(R)
                    ATb = A_mat.T @ np.ones(p)
                    try:
                        U[i*m+j] = np.linalg.solve(ATA, ATb)
                    except:
                        U[i*m+j] = np.linalg.lstsq(ATA, ATb, rcond=None)[0]
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A_mat = np.array([U[i*m+j] * W[k*n+i] for i in range(n)])
                    ATA = A_mat.T @ A_mat + 1e-6 * np.eye(R)
                    ATb = A_mat.T @ np.ones(n)
                    try:
                        V[j*p+k] = np.linalg.solve(ATA, ATb)
                    except:
                        V[j*p+k] = np.linalg.lstsq(ATA, ATb, rcond=None)[0]
            
            # Update W
            for i in range(n):
                for k in range(p):
                    A_mat = np.array([U[i*m+j] * V[j*p+k] for j in range(m)])
                    ATA = A_mat.T @ A_mat + 1e-6 * np.eye(R)
                    ATb = A_mat.T @ np.ones(m)
                    try:
                        W[k*n+i] = np.linalg.solve(ATA, ATb)
                    except:
                        W[k*n+i] = np.linalg.lstsq(ATA, ATb, rcond=None)[0]
            
            # Check residual
            residual = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        residual += (s - 1)**2
            
            if residual < 1e-12:
                # Try to rationalize
                for max_denom in [1, 2, 4, 8, 16, 32]:
                    U_frac = [[Fraction(U[i*m+j][r]).limit_denominator(max_denom) for r in range(R)] for i in range(n*m)]
                    V_frac = [[Fraction(V[j*p+k][r]).limit_denominator(max_denom) for r in range(R)] for j in range(m*p)]
                    W_frac = [[Fraction(W[k*n+i][r]).limit_denominator(max_denom) for r in range(R)] for k in range(p*n)]
                    
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = sum(U_frac[i*m+j][r] * V_frac[j*p+k][r] * W_frac[k*n+i][r] for r in range(R))
                                if s != 1:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    return None

def construct():
    n, m, p = 2, 2, 5
    start_time = time.time()
    deadline = start_time + 140
    
    # Fallback: schoolbook rank 20
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    best_result = {"U": U, "V": V, "W": W}
    
    # Try rank 19
    if time.time() < deadline:
        r19 = _construct_rank19(n, m, p)
        if _verify(r19["U"], r19["V"], r19["W"], n, m, p):
            best_result = r19
    
    # Try rank 18
    if time.time() < deadline - 10:
        r18 = _als_rank18(n, m, p, deadline=deadline)
        if r18 is not None and _verify(r18["U"], r18["V"], r18["W"], n, m, p):
            best_result = r18
    
    return best_result
