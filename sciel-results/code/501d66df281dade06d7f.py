import numpy as np
import scipy.linalg
from itertools import product

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We'll search for rank r decompositions where r < 18
    # Target: find rank-minimal exact tensor decomposition
    # U: (n*m, rank) = (6, r)
    # V: (m*p, rank) = (9, r)  
    # W: (p*n, rank) = (6, r)
    
    # The tensor T[i,j,k; j',k'; i',k''] = 1 if j==j' and i==i' and k==k'
    # Decomposition: sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k''*n+i', c] == T
    
    # Strategy: Work over GF(3) and use structured search
    # For each rank r from 12 to 17, try to find a valid decomposition
    
    for rank in range(12, 18):
        if _try_find_decomposition(n, m, p, rank):
            return _find_decomposition(n, m, p, rank)
    
    # Fallback to schoolbook (rank = 18)
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
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

def _try_find_decomposition(n, m, p, rank):
    """
    Try to find a rank-r decomposition over GF(3).
    Returns True if found, False otherwise.
    """
    # We'll use a greedy/structured approach:
    # For matrix multiplication, we can try decompositions where each term
    # corresponds to a partial multiplication pattern
    
    # Key insight: The tensor is the "identity" tensor for matrix multiplication.
    # We can try to find low-rank approximations by considering structured patterns.
    
    # For 2x3 * 3x3, the minimal rank is known to be at most 15 (from literature),
    # but let's search computationally.
    
    # Use random search over GF(3) with verification
    rng = np.random.default_rng(42)
    
    for attempt in range(500):
        # Generate random U and V over GF(3)
        U = rng.integers(0, 3, size=(n * m, rank)).astype(np.int64)
        V = rng.integers(0, 3, size=(m * p, rank)).astype(np.int64)
        
        # For each output element (i', j', k'), we need:
        # sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k''*n+i', c] == delta
        # where delta = 1 if (j==j' and i==i' and k==k'') else 0
        
        # This is a linear system for W given U and V.
        # Let's build the linear system over GF(3).
        
        # Number of equations: (n*m) * (m*p) * (p*n) = 6 * 9 * 6 = 324
        # Number of unknowns: (p*n) * rank = 6 * rank
        
        num_eqs = (n * m) * (m * p) * (p * n)
        num_unk = (p * n) * rank
        
        A = np.zeros((num_eqs, num_unk), dtype=np.int64)
        b = np.zeros(num_eqs, dtype=np.int64)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for jp in range(m * p):
                    j_prime = jp // p
                    k = jp % p
                    for kp in range(p * n):
                        k_prime = kp // n
                        i_prime = kp % n
                        
                        # Target value
                        if j == j_prime and i == i_prime and k == k_prime:
                            b[eq_idx] = 1
                        
                        # Coefficients for W[k_prime*n + i_prime, c]
                        for c in range(rank):
                            coeff = (U[i * m + j, c] * V[j_prime * p + k, c]) % 3
                            if coeff != 0:
                                w_idx = (k_prime * n + i_prime) * rank + c
                                A[eq_idx, w_idx] = (A[eq_idx, w_idx] + coeff) % 3
                        
                        eq_idx += 1
        
        # Solve over GF(3)
        W_flat = _solve_gf3(A, b)
        if W_flat is None:
            continue
        
        # Verify the solution
        W = W_flat.reshape((p * n, rank))
        if _verify_decomposition(n, m, p, rank, U, V, W):
            # Convert to Python lists with integer/half-integer values
            # Since we're working over GF(3), values are 0, 1, 2
            # We can map to integers: 0->0, 1->1, 2->-1 (or keep as 0,1,2)
            # The task allows integer/half-integer, so integers are fine
            U_out = U.tolist()
            V_out = V.tolist()
            W_out = W.tolist()
            return True
        
    return False

def _find_decomposition(n, m, p, rank):
    """
    Find and return a rank-r decomposition.
    Assumes _try_find_decomposition already verified it exists.
    """
    rng = np.random.default_rng(42)
    
    for attempt in range(500):
        U = rng.integers(0, 3, size=(n * m, rank)).astype(np.int64)
        V = rng.integers(0, 3, size=(m * p, rank)).astype(np.int64)
        
        num_eqs = (n * m) * (m * p) * (p * n)
        num_unk = (p * n) * rank
        
        A = np.zeros((num_eqs, num_unk), dtype=np.int64)
        b = np.zeros(num_eqs, dtype=np.int64)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for jp in range(m * p):
                    j_prime = jp // p
                    k = jp % p
                    for kp in range(p * n):
                        k_prime = kp // n
                        i_prime = kp % n
                        
                        if j == j_prime and i == i_prime and k == k_prime:
                            b[eq_idx] = 1
                        
                        for c in range(rank):
                            coeff = (U[i * m + j, c] * V[j_prime * p + k, c]) % 3
                            if coeff != 0:
                                w_idx = (k_prime * n + i_prime) * rank + c
                                A[eq_idx, w_idx] = (A[eq_idx, w_idx] + coeff) % 3
                        
                        eq_idx += 1
        
        W_flat = _solve_gf3(A, b)
        if W_flat is not None:
            W = W_flat.reshape((p * n, rank))
            if _verify_decomposition(n, m, p, rank, U, V, W):
                U_out = U.tolist()
                V_out = V.tolist()
                W_out = W.tolist()
                return {"U": U_out, "V": V_out, "W": W_out}
    
    # Should not reach here if _try_find_decomposition returned True
    return _schoolbook(n, m, p)

def _solve_gf3(A, b):
    """
    Solve A*x = b over GF(3).
    Returns solution vector or None if no unique solution.
    """
    n_rows, n_cols = A.shape
    # Augmented matrix
    M = np.column_stack((A, b)).astype(np.int64) % 3
    
    # Gaussian elimination over GF(3)
    pivot_cols = []
    for col in range(n_cols):
        # Find pivot
        pivot_row = None
        for row in range(col, n_rows):
            if M[row, col] != 0:
                pivot_row = row
                break
        if pivot_row is None:
            continue
        
        # Swap rows
        if pivot_row != col:
            M[[col, pivot_row]] = M[[pivot_row, col]]
        
        # Normalize pivot row
        inv = _gf3_inverse(M[col, col])
        M[col] = (M[col] * inv) % 3
        
        # Eliminate other rows
        for row in range(n_rows):
            if row != col and M[row, col] != 0:
                factor = M[row, col]
                M[row] = (M[row] - factor * M[col]) % 3
        
        pivot_cols.append(col)
    
    # Check for consistency
    for row in range(n_rows):
        # Check if all coefficients are 0 but b is non-zero
        coeff_sum = sum(M[row, :n_cols]) % 3
        if coeff_sum == 0 and M[row, n_cols] != 0:
            return None
    
    # Extract solution
    x = np.zeros(n_cols, dtype=np.int64)
    for i, col in enumerate(pivot_cols):
        x[col] = M[i, n_cols]
    
    return x

def _gf3_inverse(a):
    """Inverse in GF(3). 0 has no inverse, 1->1, 2->2 (since 2*2=4=1 mod 3)"""
    a = a % 3
    if a == 0:
        return 0
    elif a == 1:
        return 1
    elif a == 2:
        return 2

def _verify_decomposition(n, m, p, rank, U, V, W):
    """
    Verify that the decomposition exactly satisfies the tensor equation over GF(3).
    """
    for i in range(n):
        for j in range(m):
            for j_prime in range(m):
                for k in range(p):
                    for k_prime in range(p):
                        for i_prime in range(n):
                            lhs = 0
                            for c in range(rank):
                                lhs = (lhs + U[i * m + j, c] * V[j_prime * p + k, c] * W[k_prime * n + i_prime, c]) % 3
                            rhs = 1 if (j == j_prime and i == i_prime and k == k_prime) else 0
                            if lhs != rhs:
                                return False
    return True
# EVOLVE-BLOCK-END
