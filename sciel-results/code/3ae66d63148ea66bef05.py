"""Matrix multiplication 3x3 by 3x7 — rank-minimal decomposition search."""
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

DEADLINE = time.time() + 110  # safety margin before 150s wall-clock kill

def _to_matrix(A):
    """Convert list-of-lists to numpy matrix of Fractions."""
    return np.array(A, dtype=object)

def _verify(U, V, W, n=3, m=3, p=7):
    """Check that the factorization exactly decomposes the matrix multiplication tensor."""
    U = _to_matrix(U)
    V = _to_matrix(V)
    W = _to_matrix(W)
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Must equal 1 iff (i,j,k) corresponds to a valid multiplication term, i.e., always 1 for the tensor.
    # Actually the tensor T_{i,j,k} for matmul is: T[i,j,k] = 1 for all i,j,k (it's the identity-like tensor)
    # Wait — the matrix multiplication tensor T_{i,j,k} has T[i,j,k] = 1 for all (i,j,k) in range.
    # The decomposition: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k.
    # Let's verify.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if s != 1:
                    return False
    return True

def _verify_fast(U, V, W, n=3, m=3, p=7):
    """Fast verification using numpy with object dtype."""
    U = _to_matrix(U)
    V = _to_matrix(V)
    W = _to_matrix(W)
    rank = U.shape[1]
    # Build the tensor via Einstein summation
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Reshape for batch: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # For each (i,j,k): U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # We can compute all at once by iterating over r.
    T = np.zeros((n, m, p), dtype=object)
    for r in range(rank):
        # Outer product of U[:,r], V[:,r], W[:,r] with appropriate indexing
        for i in range(n):
            for j in range(m):
                T[i, j, :] += U[i*m+j, r] * V[j*p:(j+1)*p, r] * W[0:p*n:p*n, r]  # W[k*n+i, r] for k=0..p-1
    # Actually let me redo this more carefully
    T = np.zeros((n, m, p), dtype=object)
    for r in range(rank):
        u_col = U[:, r]  # shape (n*m,)
        v_col = V[:, r]  # shape (m*p,)
        w_col = W[:, r]  # shape (p*n,)
        for i in range(n):
            for j in range(m):
                u_val = u_col[i*m + j]
                for k in range(p):
                    v_val = v_col[j*p + k]
                    w_val = w_col[k*n + i]
                    T[i, j, k] += u_val * v_val * w_val
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if T[i, j, k] != 1:
                    return False
    return True

def _schoolbook(n=3, m=3, p=7):
    """Schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _random_search(rank, n=3, m=3, p=7, max_iters=5000, seed=42):
    """
    Random search for a rank-r decomposition.
    We set up a linear system: for each (i,j,k), sum_{r=0}^{rank-1} U[i,j,r]*V[j,k,r]*W[k,i,r] = 1.
    This is a system of n*m*p equations in rank*(n*m + m*p + p*n) unknowns (with the constraint that entries are from {0, 1/2, 1} or small rationals).
    
    Alternative: fix U and V randomly, solve for W? No, it's trilinear.
    
    Better approach: use the fact that if we fix U and V, then for each (i,j,k) we need
    sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1.
    This is linear in W! For each (k,i) pair, W[k,i,r] appears in the equation for (i,j,k) with coefficient U[i,j,r]*V[j,k,r].
    
    So: fix U and V, then for each (k,i), the equations over j give:
    sum_j [sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]] ... no wait, each (i,j,k) gives one equation.
    
    Let me think differently. Fix U and V. Then the system is linear in W.
    Unknowns: W[k*n+i, r] for k in 0..p-1, i in 0..n-1, r in 0..rank-1. Total: p*n*rank.
    Equations: n*m*p, one for each (i,j,k):
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    This is a linear system. If rank is small enough, we might be able to solve it.
    But the number of unknowns is p*n*rank and equations is n*m*p. For n=3,m=3,p=7: unknowns = 21*rank, equations = 63.
    If rank >= 3, unknowns >= 63, so we have a square or overdetermined system.
    
    Strategy: pick random U, V with entries in {0, 1}, then solve the linear system for W.
    Check if W has nice (small rational) entries.
    """
    rng = np.random.RandomState(seed)
    for it in range(max_iters):
        if time.time() > DEADLINE:
            return None
        # Random U with entries in {0, 1}
        U = [[int(rng.randint(0, 2)) for _ in range(rank)] for _ in range(n*m)]
        V = [[int(rng.randint(0, 2)) for _ in range(rank)] for _ in range(m*p)]
        
        # Build linear system for W
        # Unknowns indexed by (k, i, r) -> index k*n*rank + i*rank + r
        num_unknowns = p * n * rank
        # Equations: for each (i, j, k)
        A = []
        b = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = [0] * num_unknowns
                    for r in range(rank):
                        coeff = U[i*m+j][r] * V[j*p+k][r]
                        if coeff != 0:
                            idx = k * n * rank + i * rank + r
                            row[idx] = coeff
                    A.append(row)
                    b.append(1)
        
        A = np.array(A, dtype=object)
        b = np.array(b, dtype=object)
        
        # Try to solve using Gaussian elimination over Fractions
        # This is expensive; only do it if the system looks promising
        # Quick check: number of non-zero rows
        try:
            # Simple rank check: if A has more independent rows than unknowns, no solution
            # Use numpy float for quick rank estimate
            A_float = np.array(A, dtype=float)
            rank_A = np.linalg.matrix_rank(A_float)
            if rank_A > num_unknowns:
                continue
            if rank_A < n*m*p:
                # Underdetermined or rank-deficient; might have solution
                # Try solving with least squares or check consistency
                pass
            
            # Solve using fraction-based Gaussian elimination
            # This is the expensive part
            sol = _solve_linear_system(A, b)
            if sol is None:
                continue
            
            # Extract W
            W = [[0] * rank for _ in range(p * n)]
            for k in range(p):
                for i in range(n):
                    for r in range(rank):
                        W[k*n+i][r] = sol[k * n * rank + i * rank + r]
            
            # Check if W entries are "nice" (small numerators/denominators)
            max_denom = 1
            for k in range(p):
                for i in range(n):
                    for r in range(rank):
                        val = W[k*n+i][r]
                        if val != 0:
                            d = val.denominator
                            if d > 4:
                                max_denom = max(max_denom, d)
            if max_denom > 2:
                continue
            
            # Verify
            if _verify(U, V, W):
                return U, V, W
                
        except Exception:
            continue
    return None

def _solve_linear_system(A, b):
    """Gaussian elimination over Fractions. Returns solution or None if no unique solution."""
    n = len(b)
    m = len(A[0]) if n > 0 else 0
    # Augmented matrix
    aug = []
    for i in range(n):
        row = list(A[i]) + [b[i]]
        aug.append([Fraction(x) if not isinstance(x, Fraction) else x for x in row])
    
    # Forward elimination
    pivot = 0
    for col in range(m):
        if time.time() > DEADLINE:
            return None
        # Find pivot row
        pivot_row = None
        for row in range(pivot, n):
            if aug[row][col] != 0:
                pivot_row = row
                break
        if pivot_row is None:
            continue
        # Swap
        aug[pivot], aug[pivot_row] = aug[pivot_row], aug[pivot]
        # Eliminate
        pivot_val = aug[pivot][col]
        for row in range(n):
            if row != pivot:
                factor = aug[row][col] / pivot_val
                if factor != 0:
                    for j in range(col, m + 1):
                        aug[row][j] -= factor * aug[pivot][j]
        pivot += 1
        if pivot >= n:
            break
    
    # Check for inconsistency
    for i in range(n):
        row_sum = sum(aug[i][j] for j in range(m))
        if all(aug[i][j] == 0 for j in range(m)) and aug[i][m] != 0:
            return None
    
    # Back-substitution (for unique solution)
    # Actually, if underdetermined, we need a particular solution
    # For now, just check if we can find ANY solution
    # Use free variables set to 0
    sol = [Fraction(0)] * m
    # Identify pivot columns
    pivot_cols = []
    for col in range(m):
        for row in range(n):
            if aug[row][col] != 0:
                # Check if this row has col as its leftmost non-zero
                is_leftmost = all(aug[row][c] == 0 for c in range(col))
                if is_leftmost:
                    pivot_cols.append(col)
                    break
    
    # Back-substitute
    for col in reversed(pivot_cols):
        # Find the row where this col is the pivot
        for row in range(n):
            if aug[row][col] != 0 and all(aug[row][c] == 0 for c in range(col)):
                s = aug[row][m]
                for c in range(col + 1, m):
                    s -= aug[row][c] * sol[c]
                sol[col] = s / aug[row][col]
                break
    return sol

def _structured_search(rank, n=3, m=3, p=7, max_iters=2000, seed=0):
    """
    Structured search: exploit the block structure.
    For 3x3 by 3x7, we can think of it as multiplying a 3x3 matrix by a 3x7 matrix.
    We can decompose the 7 columns into blocks: 3+3+1.
    The 3x3 x 3x3 part has known rank <= 23 (Strassen-like).
    The 3x3 x 3x1 part has rank <= 9 (schoolbook, but maybe less).
    Total: 23 + 9 = 32, or we can share some factors.
    
    Alternatively, try rank values and see if we can find a solution.
    """
    # For now, try random search at various ranks
    for r in range(rank, 62):
        if time.time() > DEADLINE:
            break
        result = _random_search(r, n, m, p, max_iters=200, seed=seed + r)
        if result is not None:
            return result, r
    return None, None

def _try_known_rank23_m333():
    """
    Attempt to construct a rank-23 decomposition for 3x3 x 3x3.
    This is the Strassen-like result. The exact coefficients are complex.
    We'll return None if we can't verify it.
    """
    # The rank-23 algorithm for 3x3 matrix multiplication uses 23 multiplications.
    # Without the explicit coefficients, we can't reliably implement this.
    # Return None to indicate we don't have it.
    return None

def construct():
    """Construct a rank-minimal decomposition for 3x3 by 3x7 matrix multiplication."""
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best = (U_school, V_school, W_school)
    best_rank = n * m * p  # 63
    
    # Try to find better decompositions
    # First, try the structured approach: 3x3 x 3x7 = (3x3 x 3x3) + (3x3 x 3x1) + (3x3 x 3x3)
    # But we need a unified factorization, not block diagonal.
    
    # Try random search for various ranks
    for rank in range(30, best_rank):
        if time.time() > DEADLINE - 10:
            break
        result = _random_search(rank, n, m, p, max_iters=100, seed=rank * 1000)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W):
                best = (U, V, W)
                best_rank = rank
                break  # Found the minimum, stop
    
    # Also try a few more specific ranks if time permits
    if time.time() < DEADLINE - 20 and best_rank > 40:
        for rank in range(35, 50):
            if time.time() > DEADLINE - 10:
                break
            if rank >= best_rank:
                continue
            result = _random_search(rank, n, m, p, max_iters=50, seed=rank * 7777 + 123)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W):
                    best = (U, V, W)
                    best_rank = rank
    
    U, V, W = best
    return {"U": U, "V": V, "W": W}
