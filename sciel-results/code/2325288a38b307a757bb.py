"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def _snap_to_half_int(val, tol=1e-6):
    """Snap a float to nearest half-integer if close."""
    doubled = round(val * 2)
    if abs(val * 2 - doubled) < tol:
        return Fraction(doubled, 2)
    return None

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition. U: (n*m, r), V: (m*p, r), W: (p*n, r).
    Check that for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    All entries must be half-integers."""
    r = len(U[0]) if U else 0
    # First check all entries are half-integers
    for mat, name in [(U, 'U'), (V, 'V'), (W, 'W')]:
        for row in mat:
            for val in row:
                d = val * 2
                if abs(d - round(d)) > 1e-9:
                    return False
    # Verify trilinear form
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _try_random_decomposition(n, m, p, rank, max_iters, seed):
    """Try to find a rank decomposition via random search with refinement."""
    rng = random.Random(seed)
    U_size = n * m
    V_size = m * p
    W_size = p * n
    
    best = None
    for it in range(max_iters):
        # Random initialization with small integers
        U = [[Fraction(rng.randint(-2, 2), 2) for _ in range(rank)] for _ in range(U_size)]
        V = [[Fraction(rng.randint(-2, 2), 2) for _ in range(rank)] for _ in range(V_size)]
        W = [[Fraction(rng.randint(-2, 2), 2) for _ in range(rank)] for _ in range(W_size)]
        
        if _verify(U, V, W, n, m, p):
            return U, V, W
    
    return None

def _try_structured_search(n, m, p, rank, max_iters, seed):
    """Structured search: fix U and V, solve for W."""
    rng = random.Random(seed)
    U_size = n * m
    V_size = m * p
    W_size = p * n
    
    for it in range(max_iters):
        # Generate random U, V with half-integer entries
        U = [[Fraction(rng.randint(-3, 3), 2) for _ in range(rank)] for _ in range(U_size)]
        V = [[Fraction(rng.randint(-3, 3), 2) for _ in range(rank)] for _ in range(V_size)]
        
        # For each (i,k) pair in W, we need:
        # sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
        # This is a system of linear equations in W entries.
        # Variables: W[k*n+i, c] for all k,i,c -> W_size * rank variables
        # Equations: n*m*p = 24 equations
        
        # Build linear system: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # This is linear in W.
        num_eqs = n * m * p
        num_vars = W_size * rank
        
        # A is num_eqs x num_vars
        A = [[Fraction(0)] * num_vars for _ in range(num_eqs)]
        b = [Fraction(1)] * num_eqs
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(rank):
                        var_idx = (k * n + i) * rank + c
                        A[eq_idx][var_idx] = U[i * m + j][c] * V[j * p + k][c]
                    eq_idx += 1
        
        # Solve using Gaussian elimination with fractions
        # A is 24 x (8*rank)
        # For rank=21: 24 x 168 system - too large for exact Gaussian elimination quickly
        # For rank=22: 24 x 176
        # This might be slow. Let's try a different approach.
        
        # Actually, let's just try to solve and check if solution has half-integer entries
        if num_vars > num_eqs:
            # Underdetermined - pick a particular solution and check
            # Use simple method: set all but num_eqs variables to 0, solve square system
            # Pick first num_eqs variables
            if num_vars >= num_eqs:
                A_sq = [[A[eq][var] for var in range(num_eqs)] for eq in range(num_eqs)]
                # Gaussian elimination
                M = [row[:] for row in A_sq]
                bb = b[:]
                try:
                    for col in range(num_eqs):
                        # Find pivot
                        pivot = -1
                        for row in range(col, num_eqs):
                            if M[row][col] != 0:
                                pivot = row
                                break
                        if pivot == -1:
                            break
                        if pivot != col:
                            M[col], M[pivot] = M[pivot], M[col]
                            bb[col], bb[pivot] = bb[pivot], bb[col]
                        # Eliminate
                        pv = M[col][col]
                        for row in range(num_eqs):
                            if row != col:
                                factor = M[row][col] / pv
                                for c2 in range(col, num_eqs):
                                    M[row][c2] -= factor * M[col][c2]
                                bb[row] -= factor * bb[col]
                    # Back substitute
                    x = [Fraction(0)] * num_eqs
                    for row in range(num_eqs - 1, -1, -1):
                        if M[row][row] == 0:
                            break
                        s = bb[row]
                        for c2 in range(row + 1, num_eqs):
                            s -= M[row][c2] * x[c2]
                        x[row] = s / M[row][row]
                    # Check if all x are half-integers
                    valid = True
                    for val in x:
                        if val * 2 != round(float(val * 2)):
                            valid = False
                            break
                    if valid:
                        # Construct W
                        W = [[Fraction(0)] * rank for _ in range(W_size)]
                        for var_idx in range(num_eqs):
                            k_i = var_idx // rank
                            c = var_idx % rank
                            W[k_i][c] = x[var_idx]
                        if _verify(U, V, W, n, m, p):
                            return U, V, W
                except Exception:
                    continue
    return None

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    n, m, p = DIMENSIONS
    U_size = n * m  # 6
    V_size = m * p  # 12
    W_size = p * n  # 8
    
    # Helper to convert matrices of Fractions to floats for JSON serialization
    def to_floats(mat):
        return [[float(v) for v in row] for row in mat]

    # Try ranks from 21 upward
    # Note: The search for rank < 24 is computationally expensive and unlikely to succeed
    # quickly with simple random/structured search. We will attempt a very short search
    # but rely on the schoolbook fallback for stability.
    
    # Short attempt at rank 21-23
    for rank in [21, 22, 23]:
        if time.time() > deadline - 5: # Keep 5s for fallback construction and return
            break
        
        # Try random search with limited iterations
        max_iters = 100
        result = _try_random_decomposition(n, m, p, rank, max_iters, seed=42)
        if result is not None:
            U, V, W = result
            return {"U": to_floats(U), "V": to_floats(V), "W": to_floats(W)}
    
    # Fallback: schoolbook (Rank 24)
    rank = n * m * p
    U = [[0.0] * rank for _ in range(U_size)]
    V = [[0.0] * rank for _ in range(V_size)]
    W = [[0.0] * rank for _ in range(W_size)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
