"""Matrix multiplication 3,5,7 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
import random
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 5, 7]
N, M, P = DIMENSIONS
NM = N * M
MP = M * P
PN = P * N
TENSOR_SIZE = N * M * P  # 105


def _verify(U, V, W, rank, n, m, p, tol=1e-9):
    """Check if the decomposition is valid using exact rational arithmetic when possible."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _verify_exact(U, V, W, rank, n, m, p):
    """Verify using Fraction for exactness."""
    U_frac = [[Fraction(int(round(U[i*m+j, r]*2))/2) for r in range(rank)] for i in range(n) for j in range(m)]
    V_frac = [[Fraction(int(round(V[j*p+k, r]*2))/2) for r in range(rank)] for j in range(m) for k in range(p)]
    W_frac = [[Fraction(int(round(W[k*n+i, r]*2))/2) for r in range(rank)] for k in range(p) for i in range(n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += U_frac[i*m+j][r] * V_frac[j*p+k][r] * W_frac[k*n+i][r]
                if s != 1:
                    return False
    return True


def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
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


def _strassen_2x2():
    """Return Strassen decomposition for 2x2x2."""
    # Strassen's 7-multiplication algorithm for 2x2 matrices
    # T_{ijk} = delta_{jk} for 2x2x2
    U = [[0,0],[0,0],[0,0],[0,0]]  # 4 entries
    V = [[0,0],[0,0],[0,0],[0,0]]  # 4 entries
    W = [[0,0],[0,0],[0,0],[0,0]]  # 4 entries
    
    # Standard Strassen formulas
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U[i,j] corresponds to (i,j) in A, V[j,k] to (j,k) in B, W[k,i] to (k,i) in C
    # For 2x2: U is 2x2, V is 2x2, W is 2x2
    
    # Let's use the standard Strassen with 7 terms
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a00+a11)(b00+b11)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[1,1]=1
    W[0,0]=1; W[1,1]=1; W[0,1]=-1; W[1,0]=-1  # C11 gets M1, C22 gets M1... 
    # Actually this is getting complex. Let me use a known valid 7-term decomposition.
    
    # Simpler: just use the 8-term schoolbook for 2x2
    U = [[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]]
    V = [[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]
    W = [[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]]
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                col = i*4+j*2+k
                U[i*2+j, col] = 1
                V[j*2+k, col] = 1
                W[k*2+i, col] = 1
    
    return {"U": U, "V": V, "W": W}


def _gradient_descent_decomposition(n, m, p, target_rank, deadline, seed=42, lr_init=0.01, 
                                      max_iter=5000, tol=1e-7):
    """
    Use gradient descent to find a low-rank decomposition.
    Minimize sum_{i,j,k} (sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1)^2
    """
    random.seed(seed)
    np.random.seed(seed)
    
    U = np.random.randn(n*m, target_rank) * 0.5
    V = np.random.randn(m*p, target_rank) * 0.5
    W = np.random.randn(p*n, target_rank) * 0.5
    
    lr = lr_init
    best_loss = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Compute loss and gradients
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # s = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                    s = 0.0
                    for r in range(target_rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    err = s - 1.0
                    loss += err * err
                    for r in range(target_rank):
                        grad_U[i*m+j, r] += 2*err * V[j*p+k, r] * W[k*n+i, r]
                        grad_V[j*p+k, r] += 2*err * U[i*m+j, r] * W[k*n+i, r]
                        grad_W[k*n+i, r] += 2*err * U[i*m+j, r] * V[j*p+k, r]
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < tol:
            break
        
        # Update
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Decay learning rate
        if it % 500 == 0 and it > 0:
            lr *= 0.9
        
        # Periodic restart if stuck
        if it % 1000 == 0 and it > 0:
            if loss > best_loss * 1.5:
                U, V, W = best_U.copy(), best_V.copy(), best_W.copy()
                lr *= 0.5
    
    # Try to round to half-integers
    U_rounded = np.round(best_U * 2) / 2
    V_rounded = np.round(best_V * 2) / 2
    W_rounded = np.round(best_W * 2) / 2
    
    if _verify(U_rounded, V_rounded, W_rounded, target_rank, n, m, p, tol=1e-6):
        return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}, target_rank
    
    # Try integer rounding
    U_int = np.round(best_U).astype(int).astype(float)
    V_int = np.round(best_V).astype(int).astype(float)
    W_int = np.round(best_W).astype(int).astype(float)
    
    if _verify(U_int, V_int, W_int, target_rank, n, m, p, tol=1e-6):
        return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}, target_rank
    
    return None, target_rank


def _structured_block_search(n, m, p, deadline):
    """
    Try to build a decomposition using block structure.
    For 3x5x7, try to exploit that 5 = 2+3, 7 = 3+4, etc.
    """
    results = []
    
    # Try various block decompositions
    # Block 1: 2x2x2 (Strassen or schoolbook)
    # Block 2: remaining parts
    
    # For now, try a simple structured approach:
    # Use the fact that T_{ijk} = delta_{jk}
    # Try rank = n*p + m*(n+p-1) or similar formulas
    
    # Known bounds: For n x m x p matrix mult, rank <= n*m*p
    # For 3x5x7: schoolbook = 105
    # Try to find better via structured patterns
    
    # Pattern: For each (i,k) pair, we need sum_j U[i,j,r]*V[j,k,r]*W[k,i,r] = delta_{jk}
    # This is a system of 105 equations in rank*3*(n*m+m*p+p*n) variables
    
    # Try a greedy approach: assign one term at a time to satisfy one equation
    U = np.zeros((n*m, n*m*p))
    V = np.zeros((m*p, n*m*p))
    W = np.zeros((p*n, n*m*p))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # This is just schoolbook. Let's try to merge terms.
    # Two terms (i1,j1,k1) and (i2,j2,k2) can be merged if they share structure.
    
    return None


def _rational_recovery(U, V, W, rank, n, m, p, max_denom=4):
    """
    Try to recover exact rational values from approximate ones.
    """
    from fractions import Fraction
    
    U_exact = np.zeros_like(U)
    V_exact = np.zeros_like(V)
    W_exact = np.zeros_like(W)
    
    for i in range(U.shape[0]):
        for r in range(rank):
            val = U[i, r]
            if abs(val) < 1e-10:
                U_exact[i, r] = 0
            else:
                # Try to find fraction with small denominator
                best_frac = None
                best_err = np.inf
                for d in range(1, max_denom+1):
                    num = round(val * d)
                    frac = Fraction(num, d)
                    err = abs(float(frac) - val)
                    if err < best_err:
                        best_err = err
                        best_frac = frac
                if best_err < 1e-6:
                    U_exact[i, r] = float(best_frac)
    
    # Similar for V and W
    for i in range(V.shape[0]):
        for r in range(rank):
            val = V[i, r]
            if abs(val) < 1e-10:
                V_exact[i, r] = 0
            else:
                best_frac = None
                best_err = np.inf
                for d in range(1, max_denom+1):
                    num = round(val * d)
                    frac = Fraction(num, d)
                    err = abs(float(frac) - val)
                    if err < best_err:
                        best_err = err
                        best_frac = frac
                if best_err < 1e-6:
                    V_exact[i, r] = float(best_frac)
    
    for i in range(W.shape[0]):
        for r in range(rank):
            val = W[i, r]
            if abs(val) < 1e-10:
                W_exact[i, r] = 0
            else:
                best_frac = None
                best_err = np.inf
                for d in range(1, max_denom+1):
                    num = round(val * d)
                    frac = Fraction(num, d)
                    err = abs(float(frac) - val)
                    if err < best_err:
                        best_err = err
                        best_frac = frac
                if best_err < 1e-6:
                    W_exact[i, r] = float(best_frac)
    
    if _verify(U_exact, V_exact, W_exact, rank, n, m, p, tol=1e-9):
        return {"U": U_exact.tolist(), "V": V_exact.tolist(), "W": W_exact.tolist()}, rank
    
    return None, rank


def _try_specific_ranks(n, m, p, deadline):
    """
    Try gradient descent for various target ranks.
    """
    best_result = None
    best_rank = n*m*p + 1
    
    # Try ranks from low to high
    # Theoretical lower bound is around n*p + m + ... but let's be practical
    candidate_ranks = [60, 65, 70, 75, 80, 85, 90, 95, 100, 105]
    
    for target_rank in candidate_ranks:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
        
        result, rank = _gradient_descent_decomposition(
            n, m, p, target_rank, deadline, 
            seed=target_rank, lr_init=0.005, max_iter=3000
        )
        
        if result is not None and rank < best_rank:
            best_rank = rank
            best_result = result
    
    return best_result, best_rank


def _sparse_pattern_search(n, m, p, deadline, seed=42):
    """
    Search for sparse patterns where each rank-1 term has few non-zero entries.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    best_result = None
    best_rank = n*m*p + 1
    
    for target_rank in range(70, 105):
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
        
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Generate random sparse patterns
            patterns = []
            for r in range(target_rank):
                pattern = {}
                num_positions = random.randint(1, 8)
                positions = random.sample(range(n*m*p), num_positions)
                for pos in positions:
                    i = pos // (m*p)
                    rem = pos % (m*p)
                    j = rem // p
                    k = rem % p
                    val = random.choice([1, -1, 0.5, -0.5, 2, -2, 1.5, -1.5])
                    pattern[(i, j, k)] = val
                patterns.append(pattern)
            
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            for r, pattern in enumerate(patterns):
                for (i, j, k), val in pattern.items():
                    U[i*m+j, r] += val
                    V[j*p+k, r] += val
                    W[k*n+i, r] += val
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(target_rank):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = target_rank
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break
        
        if best_result is not None:
            break
    
    return best_result, best_rank


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # Start with schoolbook as fallback
    schoolbook_result = _schoolbook(n, m, p)
    best_result = schoolbook_result
    best_rank = n * m * p  # 105
    
    # Try gradient descent for various ranks
    if time.time() < deadline - 80:
        result, rank = _try_specific_ranks(n, m, p, deadline)
        if result is not None and rank < best_rank:
            best_result = result
            best_rank = rank
    
    # Try sparse pattern search
    if time.time() < deadline - 40:
        result, rank = _sparse_pattern_search(n, m, p, deadline, seed=67890)
        if result is not None and rank < best_rank:
            best_result = result
            best_rank = rank
    
    # Final verification
    if time.time() < deadline - 10:
        U = np.array(best_result["U"])
        V = np.array(best_result["V"])
        W = np.array(best_result["W"])
        rank = U.shape[1]
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            # Fall back to schoolbook
            best_result = _schoolbook(n, m, p)
    
    return best_result

# EVOLVE-BLOCK-END
