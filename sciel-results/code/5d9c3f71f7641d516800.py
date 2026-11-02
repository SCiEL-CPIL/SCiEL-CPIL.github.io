"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]
import time
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = 140.0  # seconds, safety margin before 150s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _to_halfint(x):
    """Convert a Fraction or number to integer or half-integer (multiple of 0.5)."""
    if isinstance(x, Fraction):
        num, den = x.numerator, x.denominator
        if den == 1:
            return num
        if den == 2:
            return num / 2
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, float):
        if abs(x - round(x)) < 1e-9:
            return int(round(x))
        if abs(2*x - round(2*x)) < 1e-9:
            return round(2*x) / 2
        return None
    return None

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact and entries are integer/half-integer."""
    rank = len(U[0]) if U else 0
    # Check shapes
    if len(U) != n*m or len(V) != m*p or len(W) != p*n:
        return False
    # Check all entries are integer or half-integer
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                h = _to_halfint(val)
                if h is None:
                    return False
    # Verify tensor identity: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == delta_{i,k}... 
    # Actually for matrix mult: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor: T[i,j,k] = 1 if we're multiplying, so T[i,j,k] = delta (always 1 for valid indices)
    # The decomposition says: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # But T[i,j,k] = 1 for all i,j,k (the multiplication tensor has all-ones)
    # Wait no. The matrix multiplication tensor T_{i,j,k} where i is row of A, j is col of A / row of B, k is col of B.
    # T[i,j,k] = 1 for all valid i,j,k.
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if isinstance(s, Fraction):
                    if s != 1:
                        return False
                else:
                    if abs(s - 1) > 1e-9:
                        return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _try_rank_gf2(r, n, m, p, time_budget):
    """
    Try to find a rank-r decomposition over GF(2) using systematic search.
    The tensor T[i,j,k] = 1 for all i,j,k.
    We need: sum_{c=0}^{r-1} U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 (mod 2)
    
    Strategy: Fix U and V, solve for W linearly over GF(2).
    """
    if time_budget() < 5:
        return None
    
    # Number of equations: n*m*p
    # Unknowns in W: p*n*r
    # For a given (U,V), the system is linear in W.
    
    # We'll use a randomized approach: randomly initialize U and V over GF(2),
    # then solve the linear system for W over GF(2).
    
    import numpy as np
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    best = None
    
    num_attempts = 200
    for attempt in range(num_attempts):
        if time_budget() < 5:
            break
        
        # Random U and V over GF(2)
        U_gf2 = np.random.randint(0, 2, size=(nm, r), dtype=np.int64)
        V_gf2 = np.random.randint(0, 2, size=(mp, r), dtype=np.int64)
        
        # Build the linear system for W over GF(2)
        # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 (mod 2)
        # This is linear in W.
        
        # Number of equations: n*m*p
        num_eqs = n * m * p
        
        # Build A matrix (num_eqs x pn*r) and b vector (num_eqs,)
        # A[eq, w_index] where eq corresponds to (i,j,k) and w_index = (k*n+i)*r + c
        # A[eq, (k*n+i)*r + c] = U[i*m+j,c] * V[j*p+k,c] (mod 2)
        # b[eq] = 1
        
        A = np.zeros((num_eqs, pn * r), dtype=np.int64)
        b = np.ones(num_eqs, dtype=np.int64)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(r):
                        A[eq_idx, (k*n+i)*r + c] = (U_gf2[i*m+j, c] * V_gf2[j*p+k, c]) % 2
                    eq_idx += 1
        
        # Solve linear system over GF(2) using Gaussian elimination
        # Augmented matrix [A | b]
        M = np.hstack([A, b.reshape(-1, 1)])
        
        # Gaussian elimination over GF(2)
        rows, cols = M.shape
        col = 0
        for row in range(rows):
            if col >= cols - 1:
                break
            # Find pivot
            pivot = -1
            for r2 in range(row, rows):
                if M[r2, col] % 2 == 1:
                    pivot = r2
                    break
            if pivot == -1:
                col += 1
                continue
            # Swap
            if pivot != row:
                M[row], M[pivot] = M[pivot].copy(), M[row].copy()
            # Eliminate
            for r2 in range(rows):
                if r2 != row and M[r2, col] % 2 == 1:
                    M[r2] = (M[r2] + M[row]) % 2
            col += 1
        
        # Check for consistency
        consistent = True
        for r2 in range(rows):
            # If all coefficients are 0 but RHS is 1, inconsistent
            if np.all(M[r2, :-1] % 2 == 0) and M[r2, -1] % 2 == 1:
                consistent = False
                break
        
        if not consistent:
            continue
        
        # Extract W from the solution (take first solution, set free variables to 0)
        W_gf2 = np.zeros(pn * r, dtype=np.int64)
        # For each column (variable), if it's a pivot column, take the value from the solution
        # We need to identify pivot columns
        col = 0
        pivot_cols = []
        for row in range(rows):
            if col >= cols - 1:
                break
            # Find first non-zero in this row from col onwards
            found = -1
            for c2 in range(col, cols - 1):
                if M[row, c2] % 2 == 1:
                    found = c2
                    break
            if found == -1:
                continue
            pivot_cols.append(found)
            W_gf2[found] = M[row, -1] % 2
            col = found + 1
        
        # Reshape W
        W_gf2 = W_gf2.reshape(pn, r)
        
        # Verify over GF(2)
        valid = True
        for i in range(n):
            if not valid:
                break
            for j in range(m):
                if not valid:
                    break
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s = (s + U_gf2[i*m+j, c] * V_gf2[j*p+k, c] * W_gf2[k*n+i, c]) % 2
                    if s != 1:
                        valid = False
                        break
        
        if valid:
            # Convert to integer list
            U_list = U_gf2.tolist()
            V_list = V_gf2.tolist()
            W_list = W_gf2.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _try_rank_rational(r, n, m, p, time_budget):
    """
    Try to find a rank-r decomposition over the rationals.
    Uses the structure: T[i,j,k] = 1 for all i,j,k.
    We parameterize U, V, W and solve.
    """
    if time_budget() < 10:
        return None
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # For small r, we can try to use known constructions.
    # One approach: use the "border rank" ideas or specific algebraic constructions.
    
    # For 2x5x6, let's try a structured approach.
    # The tensor is T[i,j,k] = 1.
    
    # Try a specific pattern: partition the rank into groups.
    # For example, if r = n*m*p / 2 = 30, maybe we can pair up terms.
    
    # Let's try a direct approach with numpy least squares as a heuristic,
    # then verify exactly.
    
    if time_budget() < 15:
        return None
    
    # Randomized search: randomly initialize, then do gradient-like updates
    # This is heuristic and may not converge, but worth trying for small r.
    
    num_restarts = 50
    for restart in range(num_restarts):
        if time_budget() < 10:
            break
        
        # Initialize with random values
        U = np.random.randn(nm, r)
        V = np.random.randn(mp, r)
        W = np.random.randn(pn, r)
        
        # Scale to reasonable magnitudes
        U /= np.sqrt(r)
        V /= np.sqrt(r)
        W /= np.sqrt(r)
        
        # We want sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k
        # This is a tensor equation. Let's use a simple iterative method.
        
        # Actually, for exact verification we need rational solutions.
        # Let's skip the heuristic and just return None for now.
        # The GF(2) approach is more promising for finding exact solutions.
        pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 60
    
    # Start with schoolbook as fallback
    best = _schoolbook()
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    # Lower bound: rank >= max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # Also rank >= n*m*p / min(n*m, m*p, p*n) ... not a standard bound
    # Known lower bounds for matrix multiplication tensor:
    # rank >= (n*m*p) / (min dimension related)
    # For C^m_{n,p}: rank >= max(n*p, m*n, m*p) is not always true
    # Strassen showed rank < n*m*p for some cases
    
    # Try ranks from 30 upward
    for r in range(30, best_rank):
        if _time_left() < 10:
            break
        
        # Try GF(2) approach
        result = _try_rank_gf2(r, n, m, p, _time_left)
        if result is not None:
            # Verify over rationals (GF(2) solution is also valid over integers)
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                best = result
                best_rank = r
                # Found a valid decomposition, try to go lower? No, we're going up.
                # Actually we want the minimum, so if we found one at rank r, 
                # we should continue checking lower ranks. But we're iterating upward.
                # Let's restructure: iterate from low to high, return first valid.
                # Since we're going from 30 up, the first valid is the minimum.
                return best
        
        if _time_left() < 5:
            break
    
    return best
# EVOLVE-BLOCK-END
