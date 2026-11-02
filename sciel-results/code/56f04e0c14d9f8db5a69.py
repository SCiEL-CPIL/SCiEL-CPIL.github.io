import time
import random
from fractions import Fraction
from itertools import product

n, m, p = 4, 4, 7
DEADLINE = time.time() + 140.0

def _verify(U, V, W):
    """Verify exact decomposition using Fraction for correctness."""
    r = len(U[0])
    # Check all (i,j,k) entries equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(r):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    # Check all forbidden entries equal 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            if i2 == i and j2 == j and k2 == k:
                                continue
                            s = Fraction(0)
                            for c in range(r):
                                s += Fraction(U[i*m+j2][c]) * Fraction(V[j*p+k2][c]) * Fraction(W[k*n+i2][c])
                            if s != 0:
                                return False
    return True

def _verify_fast(U, V, W):
    """Fast float-based verification for search."""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-6:
                    return False
    # Spot-check forbidden entries
    for _ in range(200):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        i2 = random.randint(0, n-1)
        j2 = random.randint(0, m-1)
        k2 = random.randint(0, p-1)
        if i2 == i and j2 == j and k2 == k:
            continue
        s = 0.0
        for c in range(r):
            s += U[i*m+j2][c] * V[j*p+k2][c] * W[k*n+i2][c]
        if abs(s) > 1e-6:
            return False
    return True

def _solve_linear_system(A, b):
    """Solve Ax=b using Gaussian elimination with fractions. Returns solution or None."""
    N = len(b)
    M = len(A)
    if M < N:
        return None
    # Use fraction-based elimination for exactness
    aug = [[Fraction(x) for x in row] + [Fraction(b[i])] for i, row in enumerate(A)]
    rank = 0
    for col in range(N):
        piv = None
        for r in range(rank, M):
            if aug[r][col] != 0:
                piv = r
                break
        if piv is None:
            continue
        aug[rank], aug[piv] = aug[piv], aug[rank]
        pv = aug[rank][col]
        for j in range(col, N+1):
            aug[rank][j] /= pv
        for r in range(M):
            if r != rank and aug[r][col] != 0:
                f = aug[r][col]
                for j in range(col, N+1):
                    aug[r][j] -= f * aug[rank][j]
        rank += 1
    # Check consistency
    for r in range(rank, M):
        if aug[r][N] != 0:
            return None
    x = [Fraction(0)] * N
    for i in range(rank):
        x[i] = aug[i][N]
    return x

def _try_rank(r, seed=42):
    """Try to find a valid rank-r decomposition."""
    if time.time() > DEADLINE - 10:
        return None
    
    rng = random.Random(seed)
    
    # Strategy: random sparse U, V with values in {-1, 0, 1}, solve for W
    # The system is linear in W entries.
    # Unknowns: W_flat[c * (p*n) + k*n + i] for c in range(r), i in range(n), k in range(p)
    # Total unknowns: r * p * n
    # Equations: for each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[c, k*n+i] = 1
    # Plus forbidden: sum_c U[i*m+j2,c]*V[j*p+k2,c]*W[c, k2*n+i2] = 0
    
    # Let's build the linear system
    num_unknowns = r * p * n
    equations = []
    rhs = []
    
    # For each (i,j,k) - must equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq = [0] * num_unknowns
                for c in range(r):
                    coeff = U_val = None  # placeholder
                # Actually we need U and V first
                # Let me restructure: fix U,V, then build system for W
                pass
    
    # Better approach: iterate over random U,V, build system, solve
    for attempt in range(2000):
        if time.time() > DEADLINE - 10:
            return None
        
        # Generate sparse random U and V
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        
        # Make U and V sparse with entries in {-1, 0, 1}
        density = max(0.1, 1.0 - 0.05 * attempt)
        for a in range(n*m):
            for c in range(r):
                if rng.random() < density:
                    U[a][c] = rng.choice([-1, 1])
        for b in range(m*p):
            for c in range(r):
                if rng.random() < density:
                    V[b][c] = rng.choice([-1, 1])
        
        # Build linear system for W
        # Unknown: W[c, idx] where idx = k*n+i, c in [0,r), total r*p*n unknowns
        # Equation for (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[c, k*n+i] = 1
        # This is linear in W!
        
        A_mat = []
        b_vec = []
        
        # Equations for (i,j,k) -> 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = [0] * num_unknowns
                    for c in range(r):
                        coeff = U[i*m+j][c] * V[j*p+k][c]
                        if coeff != 0:
                            idx = c * (p*n) + k*n + i
                            row[idx] = coeff
                    A_mat.append(row)
                    b_vec.append(1)
        
        # Add some forbidden equations (subset for speed)
        num_forbidden = min(50, (n*m*p)*(n*m*p - 1) // 10)
        for _ in range(num_forbidden):
            i = rng.randint(0, n-1)
            j = rng.randint(0, m-1)
            k = rng.randint(0, p-1)
            i2 = rng.randint(0, n-1)
            j2 = rng.randint(0, m-1)
            k2 = rng.randint(0, p-1)
            if i2 == i and j2 == j and k2 == k:
                continue
            row = [0] * num_unknowns
            for c in range(r):
                coeff = U[i*m+j2][c] * V[j*p+k2][c]
                if coeff != 0:
                    idx = c * (p*n) + k2*n + i2
                    row[idx] = coeff
            A_mat.append(row)
            b_vec.append(0)
        
        # Solve the system
        # Convert to use float for speed in solving, then verify
        try:
            import numpy as np
            A_np = np.array(A_mat, dtype=float)
            b_np = np.array(b_vec, dtype=float)
            
            # Use least squares if overdetermined
            if len(A_mat) >= num_unknowns:
                sol, residuals, rank_sol, sv = np.linalg.lstsq(A_np, b_np, rcond=None)
                if len(residuals) > 0 and residuals[0] > 1e-6:
                    continue
            else:
                # Underdetermined - use minimum norm solution
                sol = np.linalg.lstsq(A_np, b_np, rcond=None)[0]
            
            # Convert solution to W
            W = [[0]*(r) for _ in range(p*n)]
            for c in range(r):
                for idx in range(p*n):
                    W[idx][c] = sol[c*(p*n) + idx]
            
            # Quantize to rationals (try small denominators)
            W_q = []
            for row in W:
                W_q.append([])
                for val in row:
                    # Try to find a rational with small denominator
                    best = 0
                    best_err = float('inf')
                    for den in [1, 2, 3, 4, 5, 6, 7, 8]:
                        num = round(val * den)
                        err = abs(val - num/den)
                        if err < best_err:
                            best_err = err
                            best = num/den
                    W_q[-1].append(best)
            
            # Verify
            if _verify_fast(U, V, W_q):
                if _verify(U, V, W_q):
                    return {"U": U, "V": V, "W": W_q}
        except:
            continue
    
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 x 4x7 matrix multiplication."""
    start_time = time.time()
    
    # Start with schoolbook (rank 112)
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
    best = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = max(16, 28, 28) = 28
    # Also: n*m*p / max(n,m,p) = 112/7 = 16
    # Known: 4x4x4 has rank at most 49 (Strassen^2), possibly 23
    # For 4x4x7, let's try ranks from 28 upward
    
    for r in range(28, min(60, best_rank)):
        if time.time() > DEADLINE - 15:
            break
        
        result = _try_rank(r, seed=42 + r)
        if result is not None:
            best = result
            best_rank = r
            # Continue searching for even lower rank
            continue
        else:
            # Try a few more seeds for this rank
            for seed_offset in range(1, 5):
                if time.time() > DEADLINE - 15:
                    break
                result = _try_rank(r, seed=42 + r + seed_offset * 1000)
                if result is not None:
                    best = result
                    best_rank = r
                    break
    
    return best
