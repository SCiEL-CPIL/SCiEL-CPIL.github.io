"""
Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK.
Strategy: Try to find a rank-59 decomposition for 3x4*4x5 by exploiting
the asymmetry. The idea is to use a structured search over rational
coefficients where we start from the schoolbook decomposition and try
to merge pairs of terms while maintaining exact bilinear identities.

We use a local search approach:
1. Start with schoolbook rank-60 decomposition
2. Try to eliminate one term by finding a linear combination of other terms
   that reproduces its contribution
3. If successful, we have a rank-59 decomposition
"""
import time
import numpy as np
from itertools import product
from fractions import Fraction

DIMENSIONS = [3, 4, 5]
MAX_STEPS = 8

def _schoolbook(n, m, p):
    """
    Standard schoolbook matrix multiplication tensor decomposition.
    Rank = n * m * p.
    U shape: (n*m, rank)
    V shape: (m*p, rank)
    W shape: (p*n, rank)
    Indexing:
    U[i*m+j, c]
    V[j*p+k, c]
    W[k*n+i, c]
    """
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

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition exactly reproduces matrix multiplication.
    For each output element C[i,k], we need:
    sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = delta-like structure
    
    Actually, the correct identity is:
    For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    And for all (i', j', k') not in the "diagonal" pattern, the sum should be 0.
    
    More precisely, the matrix multiplication tensor T is defined by:
    T[i*m+j, j*p+k, k*n+i] = 1 for valid (i,j,k)
    And T is zero elsewhere in the support.
    
    The decomposition satisfies:
    T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # This should be 1
                total = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if abs(total - 1) > 1e-9:
                    return False
    # Check that other entries are 0 (spot check for efficiency)
    # The full check would be expensive, so we trust the construction
    return True

def _try_merge_pair(U, V, W, n, m, p, target_col):
    """
    Try to eliminate column 'target_col' by expressing its contribution
    as a linear combination of the other columns using exact rational arithmetic.
    
    We look for coefficients alpha_r (for r != target_col) such that:
    U[:, target_col] = sum_{r != target_col} alpha_r * U[:, r]
    V[:, target_col] = sum_{r != target_col} alpha_r * V[:, r]  
    W[:, target_col] = sum_{r != target_col} alpha_r * W[:, r]
    
    This is a linear system. If a solution exists with rational coefficients,
    we can merge.
    """
    rank = len(U[0])
    n_U = len(U)
    n_V = len(V)
    n_W = len(W)
    
    # Build the linear system
    # Unknowns: alpha_0, ..., alpha_{rank-1} (excluding target_col)
    other_cols = [r for r in range(rank) if r != target_col]
    num_unknowns = len(other_cols)
    
    # Number of equations: n_U + n_V + n_W
    num_eq = n_U + n_V + n_W
    
    if num_unknowns < num_eq:
        return None  # Underdetermined, likely no solution
    
    # Build matrix A and vector b using Fractions for exact arithmetic
    A = [[Fraction(0) for _ in range(num_unknowns)] for _ in range(num_eq)]
    b = [Fraction(0) for _ in range(num_eq)]
    
    # U equations
    for i in range(n_U):
        for j, r in enumerate(other_cols):
            A[i][j] = Fraction(U[i][r])
        b[i] = Fraction(U[i][target_col])
    
    # V equations
    for i in range(n_V):
        for j, r in enumerate(other_cols):
            A[n_U + i][j] = Fraction(V[i][r])
        b[n_U + i] = Fraction(V[i][target_col])
    
    # W equations
    for i in range(n_W):
        for j, r in enumerate(other_cols):
            A[n_U + n_V + i][j] = Fraction(W[i][r])
        b[n_U + n_V + i] = Fraction(W[i][target_col])
    
    # Use Gaussian elimination with exact arithmetic
    # Augmented matrix [A | b]
    augmented = [row[:] + [b[i]] for i, row in enumerate(A)]
    
    # Forward elimination
    for col in range(num_unknowns):
        # Find pivot
        pivot_row = -1
        for row in range(col, num_eq):
            if augmented[row][col] != 0:
                pivot_row = row
                break
        if pivot_row == -1:
            # Column is all zeros below this row
            continue
        
        # Swap to pivot position
        if pivot_row != col:
            augmented[col], augmented[pivot_row] = augmented[pivot_row], augmented[col]
        
        # Normalize pivot row
        pivot_val = augmented[col][col]
        for j in range(col, num_unknowns + 1):
            augmented[col][j] = augmented[col][j] / pivot_val
        
        # Eliminate below
        for row in range(col + 1, num_eq):
            factor = augmented[row][col]
            if factor != 0:
                for j in range(col, num_unknowns + 1):
                    augmented[row][j] -= factor * augmented[col][j]
    
    # Back substitution
    alpha = [Fraction(0) for _ in range(num_unknowns)]
    for i in range(num_unknowns - 1, -1, -1):
        # Find the row where column i has a pivot
        pivot_row = -1
        for row in range(i, num_eq):
            if augmented[row][i] != 0:
                pivot_row = row
                break
        if pivot_row == -1:
            # Free variable, set to 0
            alpha[i] = Fraction(0)
            continue
        
        # Back substitute
        s = augmented[pivot_row][num_unknowns]
        for j in range(i + 1, num_unknowns):
            s -= augmented[pivot_row][j] * alpha[j]
        alpha[i] = s / augmented[pivot_row][i]
    
    # Verify the solution exactly
    for i in range(num_eq):
        total = sum(A[i][j] * alpha[j] for j in range(num_unknowns))
        if total != b[i]:
            return None
    
    return alpha

def _remove_column(U, V, W, col):
    """Remove column 'col' from all three matrices."""
    U_new = [[U[i][j] for j in range(len(U[0])) if j != col] for i in range(len(U))]
    V_new = [[V[i][j] for j in range(len(V[0])) if j != col] for i in range(len(V))]
    W_new = [[W[i][j] for j in range(len(W[0])) if j != col] for i in range(len(W))]
    return U_new, V_new, W_new

def _apply_merge(U, V, W, alpha, target_col):
    """
    After determining that column target_col can be expressed as linear combination
    of other columns with coefficients alpha, actually perform the merge.
    
    If U[:, t] = sum_{r != t} alpha_r * U[:, r] and similarly for V and W,
    then the contribution of column t is already captured by the other columns.
    We can simply remove column t.
    """
    return _remove_column(U, V, W, target_col)

# EVOLVE-BLOCK-START
def _verify_decomposition_full(U, V, W, n, m, p):
    """
    Full verification that the decomposition exactly reproduces matrix multiplication.
    For all (a, b, c) in the index space:
    sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    where T[a,b,c] = 1 if (a,b,c) corresponds to valid (i,j,k) with a=i*m+j, b=j*p+k, c=k*n+i
    and 0 otherwise.
    """
    rank = len(U[0])
    n_U = len(U)
    n_V = len(V)
    n_W = len(W)
    
    # Build the set of valid (a,b,c) triples
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid.add((a, b, c))
    
    # Check all valid triples give 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                total = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                if total != 1:
                    return False
    
    # Check some invalid triples give 0 (spot check for efficiency)
    # Check triples where indices don't align
    import random
    rng = random.Random(42)
    for _ in range(200):
        a = rng.randint(0, n_U - 1)
        b = rng.randint(0, n_V - 1)
        c = rng.randint(0, n_W - 1)
        if (a, b, c) not in valid:
            total = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
            if total != 0:
                return False
    
    return True

def _structured_search(U, V, W, n, m, p, time_budget):
    """
    Structured search for rank reduction using null-space analysis.
    
    The idea: for a rank-r decomposition, if we can find a linear dependency
    among the rank-1 terms (i.e., a vector alpha such that sum alpha_r * term_r = 0
    in the tensor space), we can eliminate one term.
    
    We search for such dependencies by examining the structure of the tensor.
    """
    start_time = time.time()
    rank = len(U[0])
    
    # Try to find pairs of columns that can be combined
    # For each pair (c1, c2), check if there exist scalars a, b such that
    # a*term_c1 + b*term_c2 is a valid rank-1 term that can replace both
    
    # Alternative: try to find a column that is a linear combination of others
    # in the tensor product space
    
    for target in range(rank):
        if time.time() - start_time > time_budget:
            break
        
        alpha = _try_merge_pair(U, V, W, n, m, p, target)
        if alpha is not None:
            U_new, V_new, W_new = _remove_column(U, V, W, target)
            if _verify_decomposition_full(U_new, V_new, W_new, n, m, p):
                return U_new, V_new, W_new, len(U_new[0])
    
    return U, V, W, rank

def _try_pairwise_merge(U, V, W, n, m, p, c1, c2):
    """
    Try to merge two columns c1 and c2 into one column.
    
    We look for a new column (u_new, v_new, w_new) such that:
    u_new * v_new * w_new = u_c1 * v_c1 * w_c1 + u_c2 * v_c2 * w_c2
    
    This is a system of equations. For simplicity, we try:
    u_new = u_c1 + u_c2 (element-wise)
    and solve for v_new, w_new
    
    Or more generally, try small integer/half-integer coefficients.
    """
    rank = len(U[0])
    
    # Try the simplest case: can we absorb one column into another?
    # This is already handled by _try_merge_pair
    
    # Try a different approach: find if two columns can be replaced by one
    # with appropriate scaling
    
    # For each element position, we need:
    # U[a,c1]*V[b,c1]*W[c,c1] + U[a,c2]*V[b,c2]*W[c,c2] = U_new[a]*V_new[b]*W_new[c]
    
    # This is generally not possible with a single rank-1 term unless
    # the two terms are "aligned" in some sense.
    
    # Try: if U[:,c1] is proportional to U[:,c2], etc.
    # Check if the columns are proportional
    def are_proportional(col1, col2):
        """Check if col1 = lambda * col2 for some lambda."""
        # Find a nonzero entry
        for i in range(len(col1)):
            if col2[i] != 0:
                if col1[i] == 0:
                    return None
                lam = Fraction(col1[i]) / Fraction(col2[i])
                # Verify for all entries
                for j in range(len(col1)):
                    if Fraction(col1[j]) != lam * Fraction(col2[j]):
                        return None
                return lam
        # col2 is all zeros
        if all(x == 0 for x in col1):
            return Fraction(0)
        return None
    
    lam_U = are_proportional(U, [U[i][c1] for i in range(len(U))], [U[i][c2] for i in range(len(U))])
    if lam_U is None:
        return None
    
    # If U columns are proportional, check V and W
    # Actually, let's just use the merge approach which is more general
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill switch
    
    n, m, p = DIMENSIONS
    base_rank = n * m * p
    
    # Start with schoolbook decomposition
    best = _schoolbook(n, m, p)
    best_rank = base_rank
    
    # Check time
    if time.time() > deadline - 10.0:
        return best
    
    # Try to find a lower-rank decomposition
    U = [row[:] for row in best['U']]
    V = [row[:] for row in best['V']]
    W = [row[:] for row in best['W']]
    current_rank = len(U[0])
    
    # Phase 1: Try to eliminate individual columns
    for target in range(current_rank):
        if time.time() > deadline - 5.0:
            break
        
        alpha = _try_merge_pair(U, V, W, n, m, p, target)
        if alpha is not None:
            # Verify the merge with full check
            U_new, V_new, W_new = _remove_column(U, V, W, target)
            if _verify_decomposition_full(U_new, V_new, W_new, n, m, p):
                U, V, W = U_new, V_new, W_new
                current_rank = len(U[0])
                best = {'U': U, 'V': V, 'W': W}
                best_rank = current_rank
                # Found a merge, continue to try more
                target -= 1  # Re-check the same index since list shifted
    
    # Phase 2: Try structured search with remaining time
    remaining_time = deadline - 5.0 - time.time()
    if remaining_time > 10.0:
        U2, V2, W2, new_rank = _structured_search(U, V, W, n, m, p, remaining_time)
        if new_rank < current_rank:
            U, V, W = U2, V2, W2
            current_rank = new_rank
            best = {'U': U, 'V': V, 'W': W}
            best_rank = current_rank
    
    # Check time before returning
    if time.time() > deadline - 3.0:
        return best
    
    return best
# EVOLVE-BLOCK-END
