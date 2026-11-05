import numpy as np
from fractions import Fraction
import time
import itertools
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 5]

def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0])
    U_arr = np.array([[float(x) for x in row] for row in U], dtype=float)
    V_arr = np.array([[float(x) for x in row] for row in V], dtype=float)
    W_arr = np.array([[float(x) for x in row] for row in W], dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                s = 0.0
                for r in range(rank):
                    s += U_arr[idx_u, r] * V_arr[idx_v, r] * W_arr[idx_w, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_rank28():
    """Build the known rank-28 decomposition."""
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: j=2, all k (10 terms)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 2: j=0,1, k=4 (4 terms)
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
            
    # Part 3: j=0,1, k=0,1 using Strassen (7 terms)
    # M1
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    # M2
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    W[0 * n + 1][r] = 1
    W[1 * n + 1][r] = -1
    r += 1
    # M3
    U[0 * m + 0][r] = 1
    V[0 * p + 1][r] = 1
    V[1 * p + 1][r] = -1
    W[1 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    # M4
    U[1 * m + 1][r] = 1
    V[1 * p + 0][r] = 1
    V[0 * p + 0][r] = -1
    W[0 * n + 0][r] = 1
    W[0 * n + 1][r] = 1
    r += 1
    # M5
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = -1
    W[1 * n + 0][r] = 1
    r += 1
    # M6
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 0][r] = 1
    V[0 * p + 1][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    # M7
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    r += 1
    
    # Part 4: j=0,1, k=2,3 using Strassen (7 terms)
    # M1
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    # M2
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    W[2 * n + 1][r] = 1
    W[3 * n + 1][r] = -1
    r += 1
    # M3
    U[0 * m + 0][r] = 1
    V[0 * p + 3][r] = 1
    V[1 * p + 3][r] = -1
    W[3 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    # M4
    U[1 * m + 1][r] = 1
    V[1 * p + 2][r] = 1
    V[0 * p + 2][r] = -1
    W[2 * n + 0][r] = 1
    W[2 * n + 1][r] = 1
    r += 1
    # M5
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = -1
    W[3 * n + 0][r] = 1
    r += 1
    # M6
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 2][r] = 1
    V[0 * p + 3][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    # M7
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    r += 1
    
    return U, V, W

def _try_rank26_direct():
    """
    Try to find a rank-26 decomposition directly.
    
    Strategy: The rank-28 solution has structure:
    - 10 terms for j=2 (k=0..4)
    - 4 terms for j=0,1 k=4
    - 7 terms for j=0,1 k=0,1 (Strassen)
    - 7 terms for j=0,1 k=2,3 (Strassen)
    
    To get rank 26, we need to save 2 terms.
    
    Key idea: Can we merge some of the j=2 terms with the Strassen terms?
    Or use a different decomposition for the 2x3 x 3x5 case?
    
    Let me try: what if we use a 3x3 Strassen-like approach?
    
    Alternative: Use the fact that for the k=4 column, we have 4 terms
    (i=0,1 for j=0,1). What if we can express these using fewer terms
    by combining with other structure?
    
    Actually, let me try a completely different approach:
    Use a rank-7 method for 2x2, but apply it differently.
    
    For 2x3 x 3x5:
    - Column 4 of B: need A[:,0:2] * B[0:2,4] -> 4 multiplications
    - Columns 0-3 of B: need A[:,0:2] * B[0:2,0:4] -> 8 multiplications
    - Column 2 of A with all B: need A[2,:] * B[:,0:5] -> 10 multiplications
    
    Total schoolbook: 4 + 8 + 10 = 22? No, that's not right.
    
    Let me reconsider. The standard approach:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    For each (i,k), we sum over j=0,1,2.
    
    The rank-28 solution above is valid. Let me try to find rank 26 or 25.
    
    One approach: use a "laser method" or "Coppersmith-Winograd" style
    but for such small dimensions, direct search is more practical.
    
    Let me try: can we do the j=0,1 part (2x2 x 2x5) in rank 13 instead of 14?
    2x2 x 2x5: schoolbook is 2*2*5=20, Strassen-based is...
    
    Actually, let me try a numerical search for rank 26.
    """
    n, m, p = 2, 3, 5
    nm, mp, pn = n*m, m*p, p*n
    target_rank = 26
    
    # Parameterize: U (6x26), V (15x26), W (10x26)
    # Total params: 26*(6+15+10) = 26*31 = 806
    
    def _objective(params):
        U = params[:target_rank*nm].reshape(target_rank, nm).T
        V = params[target_rank*nm:target_rank*nm+target_rank*mp].reshape(target_rank, mp).T
        W = params[target_rank*nm+target_rank*mp:].reshape(target_rank, pn).T
        
        residual = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    residual += (s - 1.0)**2
        return residual
    
    start_time = time.time()
    best_result = None
    best_obj = np.inf
    
    # Try multiple initializations with different seeds
    for trial in range(50):
        if time.time() - start_time > 90:
            break
        
        np.random.seed(trial * 31 + 7)
        x0 = np.random.randn(target_rank * (nm + mp + pn)) * 0.3
        
        try:
            result = minimize(_objective, x0, method='L-BFGS-B', 
                            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if result.fun < best_obj:
                best_obj = result.fun
                if result.fun < 1e-8:
                    U = result.x[:target_rank*nm].reshape(target_rank, nm).T
                    V = result.x[target_rank*nm:target_rank*nm+target_rank*mp].reshape(target_rank, mp).T
                    W = result.x[target_rank*nm+target_rank*mp:].reshape(target_rank, pn).T
                    
                    # Try to round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    # Check if rounded version works
                    res_rounded = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = np.dot(U_r[i*m+j], V_r[j*p+k] * W_r[k*n+i])
                                res_rounded += (s - 1.0)**2
                    
                    if res_rounded < 1e-8:
                        U_frac = [[Fraction(int(np.round(val * 2)), 2) for val in row] for row in U_r]
                        V_frac = [[Fraction(int(np.round(val * 2)), 2) for val in row] for row in V_r]
                        W_frac = [[Fraction(int(np.round(val * 2)), 2) for val in row] for row in W_r]
                        
                        if _verify_decomposition(U_frac, V_frac, W_frac, n, m, p):
                            return (U_frac, V_frac, W_frac)
                    else:
                        # Use unrounded but still try to verify
                        U_frac = [[Fraction(int(np.round(val * 2)), 2) for val in row] for row in U]
                        V_frac = [[Fraction(int(np.round(val * 2)), 2) for val in row] for row in V]
                        W_frac = [[Fraction(int(np.round(val * 2)), 2) for val in row] for row in W]
                        
                        if _verify_decomposition(U_frac, V_frac, W_frac, n, m, p):
                            return (U_frac, V_frac, W_frac)
        except:
            continue
    
    return None

def _try_rank25_direct():
    """Try to find a rank-25 decomposition."""
    n, m, p = 2, 3, 5
    nm, mp, pn = n*m, m*p, p*n
    target_rank = 25
    
    def _objective(params):
        U = params[:target_rank*nm].reshape(target_rank, nm).T
        V = params[target_rank*nm:target_rank*nm+target_rank*mp].reshape(target_rank, mp).T
        W = params[target_rank*nm+target_rank*mp:].reshape(target_rank, pn).T
        
        residual = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    residual += (s - 1.0)**2
        return residual
    
    start_time = time.time()
    best_result = None
    best_obj = np.inf
    
    for trial in range(30):
        if time.time() - start_time > 60:
            break
        
        np.random.seed(trial * 41 + 13)
        x0 = np.random.randn(target_rank * (nm + mp + pn)) * 0.3
        
        try:
            result = minimize(_objective, x0, method='L-BFGS-B', 
                            options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if result.fun < best_obj:
                best_obj = result.fun
                if result.fun < 1e-8:
                    U = result.x[:target_rank*nm].reshape(target_rank, nm).T
                    V = result.x[target_rank*nm:target_rank*nm+target_rank*mp].reshape(target_rank, mp).T
                    W = result.x[target_rank*nm+target_rank*mp:].reshape(target_rank, pn).T
                    
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    res_rounded = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = np.dot(U_r[i*m+j], V_r[j*p+k] * W_r[k*n+i])
                                res_rounded += (s - 1.0)**2
                    
                    if res_rounded < 1e-8:
                        U_frac = [[Fraction(int(np.round(val * 2)), 2) for val in row] for row in U_r]
                        V_frac = [[Fraction(int(np.round(val * 2)), 2) for val in row] for row in V_r]
                        W_frac = [[Fraction(int(np.round(val * 2)), 2) for val in row] for row in W_r]
                        
                        if _verify_decomposition(U_frac, V_frac, W_frac, n, m, p):
                            return (U_frac, V_frac, W_frac)
                    else:
                        U_frac = [[Fraction(int(np.round(val * 2)), 2) for val in row] for row in U]
                        V_frac = [[Fraction(int(np.round(val * 2)), 2) for val in row] for row in V]
                        W_frac = [[Fraction(int(np.round(val * 2)), 2) for val in row] for row in W]
                        
                        if _verify_decomposition(U_frac, V_frac, W_frac, n, m, p):
                            return (U_frac, V_frac, W_frac)
        except:
            continue
    
    return None

def _try_structured_rank26():
    """
    Try a structured rank-26 decomposition.
    
    Idea: Use the rank-28 structure but try to merge 2 terms.
    
    In the rank-28 solution:
    - Terms 0-9: j=2, various (i,k)
    - Terms 10-13: j=0,1 k=4, various (i)
    - Terms 14-20: Strassen for j=0,1 k=0,1
    - Terms 21-27: Strassen for j=0,1 k=2,3
    
    What if we can express the k=4 part (4 terms) using 2 fewer terms
    by combining with some of the Strassen terms?
    
    Alternative: use a different factorization.
    
    Let me try: what if we handle the problem as:
    C = A * B where A is 2x3, B is 3x5
    
    Write A = [A1 | A2] where A1 is 2x2, A2 is 2x1
    Write B = [B1; B2] where B1 is 2x5, B2 is 1x5
    
    C = A1*B1 + A2*B2
    
    A2*B2: 2x1 * 1x5 = 10 scalar multiplications, rank 10
    A1*B1: 2x2 * 2x5, rank ?
    
    For 2x2 * 2x5:
    Schoolbook: 2*2*5 = 20
    Strassen-based: use 7 multiplications for the 2x2 part, but...
    
    Actually, 2x2 * 2x5 can be done as:
    Split B1 into two 2x2 blocks and a 2x1 block: B1 = [B11 | B12 | B13]
    where B11 is 2x2, B12 is 2x2, B13 is 2x1
    
    A1*B1 = A1*B11 + A1*B12 + A1*B13
    
    A1*B11: 2x2*2x2, rank 7 (Strassen)
    A1*B12: 2x2*2x2, rank 7 (Strassen)  
    A1*B13: 2x2*2x1, rank 4 (schoolbook) or less?
    
    2x2*2x1: C[i] = A[i,0]*B[0] + A[i,1]*B[1] for i=0,1
    This is 4 multiplications. Can we do it in 3? 
    C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    
    Using Strassen-like: 
    M1 = A[0,0]*B[0]
    M2 = A[0,1]*B[1]
    M3 = (A[0,0]+A[1,0])*(B[0]+B[1])
    M4 = A[1,0]*B[0]
    M5 = A[1,1]*B[1]
    
    C[0] = M1 + M2
    C[1] = M4 + M5
    But M3 is not used... so we still need 4.
    
    Actually for 2x2*2x1, the rank is 4 (it's just 2 independent bilinear forms
    each requiring 2 terms, and they're independent).
    
    So total: 10 + 7 + 7 + 4 = 28. Same as before.
    
    To get 26, we need to save 2 somewhere.
    
    What if we use a different split?
    
    Alternative: handle A1*B1 (2x2*2x5) in rank 13 instead of 18?
    2x2*2x5: is rank 13 possible?
    
    The border rank of 2x2*2x5... let me think.
    2x2*2x2 has rank 7.
    2x2*2x5 = 2x2*2x2 + 2x2*2x2 + 2x2*2x1
    If we could share some structure...
    
    Actually, let me just try the numerical search more aggressively.
    """
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # Leave 10s margin
    
    # First, build the known rank-28 solution as fallback
    U28, V28, W28 = _build_rank28()
    
    # Verify
    if not _verify_decomposition(U28, V28, W28, n, m, p):
        raise ValueError("Rank-28 decomposition verification failed")
    
    best_U = U28
    best_V = V28
    best_W = W28
    best_rank = 28
    
    # Try to find lower rank decompositions
    remaining = deadline - (time.time() - start_time)
    if remaining > 50:
        # Try rank 26
        result = _try_rank26_direct()
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = 26
    
    remaining = deadline - (time.time() - start_time)
    if remaining > 40 and best_rank > 25:
        # Try rank 25
        result = _try_rank25_direct()
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = 25
    
    return {"U": best_U, "V": best_V, "W": best_W}
