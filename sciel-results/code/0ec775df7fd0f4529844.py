"""Matrix multiplication 4x5x6 - rank minimization via structured search."""
import time
import itertools
from fractions import Fraction
import numpy as np

DIMENSIONS = [4, 5, 6]

def _build_schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition as a safety fallback."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m) x rank
    V: (m*p) x rank
    W: (p*n) x rank
    The tensor T[i,j,k] = delta (i.e., (A*B)[i,k] = sum_j A[i,j]*B[j,k])
    must equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    rank = len(U[0])
    # Build the tensor from the decomposition
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Expected: T[i,j,k] = 1 if (i,j,k) is a valid triple (always is), 0 otherwise
    # Actually the matrix multiplication tensor is T[i,j,k] = 1 for all i,j,k
    # because (AB)[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor is all-ones on the valid indices.
    
    # Check a sample of entries to verify
    # For exact verification, check all entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1
                actual = 0
                for c in range(rank):
                    u_val = U[i * m + j][c]
                    v_val = V[j * p + k][c]
                    w_val = W[k * n + i][c]
                    if u_val and v_val and w_val:
                        actual += u_val * v_val * w_val
                if actual != expected:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p):
    """Faster verification using numpy."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # For each (i,j,k), check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_arr[i * m + j]
                v_row = V_arr[j * p + k]
                w_row = W_arr[k * n + i]
                val = np.dot(u_row * v_row, w_row)
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _try_rank_with_skeleton(n, m, p, rank, deadline):
    """
    Try to find a decomposition of given rank using a sparse skeleton approach.
    Fix U and V to have specific sparse patterns, then solve for W.
    """
    # The tensor equation: for all (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # If we fix U and V, this is linear in W.
    # For each (i,j,k), we need: sum_c (U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1
    # This is a system of n*m*p equations in p*n*rank unknowns (W entries).
    
    # Strategy: use a "diagonal" or structured pattern for U and V.
    # A common approach: make U and V have a specific block structure.
    
    # Let's try a simple approach: make each column of U and V have a specific sparsity.
    # For rank r, we can try to make U[i*m+j, c] non-zero only for specific (i,j) per c,
    # and similarly for V.
    
    # Actually, let's try the approach where we set up the linear system and solve it.
    # For a given pattern of non-zeros in U and V, we can solve for W exactly.
    
    # Simple pattern: for each column c, U has non-zeros in a subset of rows, V has non-zeros
    # in a subset of rows. The product U[i*m+j,c]*V[j*p+k,c] determines the coefficient
    # for W[k*n+i,c] in the equation for (i,j,k).
    
    # Let's try: U[i*m+j, c] = 1 if (i,j) is in some set S_c, else 0
    #            V[j*p+k, c] = 1 if (j,k) is in some set T_c, else 0
    # Then U[i*m+j,c]*V[j*p+k,c] = 1 iff (i,j) in S_c AND (j,k) in T_c
    # The equation for (i,j,k): sum_{c: (i,j) in S_c and (j,k) in T_c} W[k*n+i,c] = 1
    
    # This is a system of n*m*p equations in sum_c |S_c|*|T_c| ... no, in p*n*rank unknowns.
    
    # Let's just set up the full linear system for W given fixed U, V.
    # Unknowns: W[k*n+i, c] for k in range(p), i in range(n), c in range(rank)
    # Total unknowns: p * n * rank
    # Equations: n * m * p, one for each (i,j,k)
    
    # For a given (i,j,k), the equation is:
    # sum_{c=0}^{rank-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is linear in W. We can set up the system and solve.
    
    # The challenge is choosing U and V. Let's try random sparse patterns.
    
    import random
    random.seed(42)
    
    # Try a few random sparse patterns for U and V
    max_attempts = 50
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Create sparse U and V
        # Each row of U has a few non-zeros, each row of V has a few non-zeros
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        
        # For each column c, assign it to a random (i,j) for U and random (j,k) for V
        # This ensures each column contributes to exactly one (i,j,k) triple
        # But we need the sum over c to give 1 for each (i,j,k)
        
        # Alternative: make U and V such that for each (i,j,k), exactly one c has
        # U[i*m+j,c] * V[j*p+k,c] != 0, and set W[k*n+i,c] appropriately.
        # This is the schoolbook approach with rank = n*m*p.
        
        # For lower rank, we need to share columns across multiple (i,j,k) triples.
        
        # Let's try a different approach: use the known structure of matrix multiplication.
        # The tensor T[i,j,k] = delta (all ones) can be decomposed using the fact that
        # it's a "product" tensor.
        
        # Actually, let's just try to solve the linear system for W given random U, V.
        
        # Generate random U and V with small integer entries
        for i in range(n * m):
            for c in range(rank):
                if random.random() < 0.3:  # 30% non-zero
                    U[i][c] = random.choice([-1, 1])
                else:
                    U[i][c] = 0
        
        for j in range(m * p):
            for c in range(rank):
                if random.random() < 0.3:
                    V[j][c] = random.choice([-1, 1])
                else:
                    V[j][c] = 0
        
        # Set up the linear system for W
        # Unknowns: W[k*n+i, c] for k in range(p), i in range(n), c in range(rank)
        # Index: idx = (k * n + i) * rank + c
        num_unknowns = p * n * rank
        num_equations = n * m * p
        
        # Build the system A * x = b
        # x is a vector of length num_unknowns
        # For each equation (i,j,k):
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        
        # This is a sparse system. Let's use numpy to solve it.
        A = np.zeros((num_equations, num_unknowns), dtype=np.float64)
        b = np.ones(num_equations, dtype=np.float64)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(rank):
                        coeff = U[i * m + j][c] * V[j * p + k][c]
                        if coeff != 0:
                            # W index: (k*n + i) * rank + c
                            w_idx = (k * n + i) * rank + c
                            A[eq_idx, w_idx] += coeff
                    eq_idx += 1
        
        # Solve A * x = b
        try:
            # Use least squares since the system may be underdetermined
            x, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
        
        # Check if the solution is exact
        residual = np.linalg.norm(A @ x - b)
        if residual > 1e-6:
            continue
        
        # Check if x has small integer or half-integer values
        # If not, skip this attempt
        W_flat = x.reshape(p * n, rank)
        
        # Check if W entries are close to integers or half-integers
        W_rounded = np.round(W_flat * 2) / 2
        if np.max(np.abs(W_flat - W_rounded)) > 1e-4:
            continue
        
        # Verify the solution
        W = W_rounded.tolist()
        if _verify_fast(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s deadline, 10s safety margin
    
    # Start with the schoolbook decomposition as a fallback
    best = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find a lower rank decomposition
    # Known bounds: for 4x5x6, the rank is at most 4*5*6 = 120
    # Lower bounds: max(n*m, m*p, p*n) = max(20, 30, 24) = 30
    # So we can try ranks from 30 to 119
    
    # Try a few specific ranks
    for rank in range(30, min(best_rank, 120)):
        if time.time() > deadline:
            break
        
        result = _try_rank_with_skeleton(n, m, p, rank, deadline)
        if result is not None:
            best = result
            best_rank = rank
            break
    
    return best
