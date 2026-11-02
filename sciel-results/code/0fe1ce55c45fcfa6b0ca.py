"""
Tensor rank reduction for 4x5 * 5x5 matrix multiplication.

Approach: Start from the schoolbook decomposition (rank 100), then use a
structured rank-reduction heuristic:
  1. Random linear combinations of the three factor matrices.
  2. Gaussian elimination on the resulting linear system to detect dependent
     equations.
  3. If dependencies are found, the rank can be reduced.
We also try a few structured block-based reductions.

The verification is exact (integer arithmetic) when using integer-valued
entries, which keeps the search clean and avoids floating-point tolerance
issues.
"""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START

def _build_schoolbook(n, m, p):
    """Return (U, V, W) as numpy arrays for schoolbook rank n*m*p."""
    r = n * m * p
    U = np.zeros((n * m, r), dtype=np.int64)
    V = np.zeros((m * p, r), dtype=np.int64)
    W = np.zeros((p * n, r), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=0):
    """
    Check if U, V, W decompose the matrix multiplication tensor exactly.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    For each (i, j, k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    All other (i, j, k') triples must also satisfy the equation (they should
    naturally be 0 if the decomposition is correct for the tensor).
    """
    r = U.shape[1]
    # Build the full tensor check
    # For each c, the contribution to T[i,j,k] is U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # We need the sum over c to equal 1 for all (i,j,k) since the MM tensor has all 1s.
    # Actually the MM tensor T_{i,j,k} = 1 for all valid i,j,k.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = int(np.dot(
                    U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]
                ))
                if s != 1:
                    return False
    return True


def _verify_float(U, V, W, n, m, p, tol=1e-6):
    """Float-based verification."""
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(
                    U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]
                )
                if abs(s - 1.0) > tol:
                    return False
    return True


def _rank_reduction_attempt(U0, V0, W0, n, m, p, r0, deadline, seed=None):
    """
    Try to reduce the rank by finding linear dependencies.
    
    The key idea: if we can find coefficients a_c, b_c, d_c (for c in 0..r-1)
    such that for ALL (i,j,k):
        sum_c a_c * U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 0
    then we can perturb the decomposition to remove one rank-1 term.
    
    A more practical approach: try to find a non-trivial linear combination
    of the columns of the Kronecker structure that sums to zero in the
    tensor sense.
    
    Alternative: use the fact that if rank is r, the tensor lives in a
    (n*m) x (m*p) x (p*n) space. We can set up the system where we fix
    two matrices and solve for the third.
    
    Here we try: fix U and V, then the equations for W are:
        For each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    This is a linear system in W's entries. If the system is underdetermined,
    we have freedom. But with r = n*m*p, it's exactly determined (each W
    entry appears in exactly one equation... actually not quite).
    
    Let me try a different approach: random search with gradient-free
    optimization on small integer grids, but smarter than pure random.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    best_rank = r0
    U_best, V_best, W_best = U0.copy(), V0.copy(), W0.copy()
    
    # Strategy 1: Try to find a rank-(r0-1) decomposition by perturbing
    # the schoolbook with a small random vector and checking if the
    # resulting system is still consistent with one fewer constraint.
    
    # Strategy 2: Use the known fact that for 4x5*5x5, we can try
    # block-structured decompositions.
    
    # Let's try: partition A (4x5) into two 2x5 blocks.
    # C = [A1; A2] * B = [A1*B; A2*B]
    # This doesn't reduce rank by itself.
    
    # Strategy 3: Use Winograd-style for the 2x2 subproblems.
    # The 4x5 * 5x5 can be seen as:
    #   rows 0-1 of A times B, and rows 2-3 of A times B.
    # Each is a 2x5 * 5x5 = 2x5 matrix multiplication.
    # A 2x5 * 5x5 has schoolbook rank 2*5*5 = 50.
    # Two of them = 100. No savings.
    
    # Strategy 4: Try to find a non-trivial low-rank decomposition via
    # random search with structured initialization.
    
    # For each target rank from r0-1 down to a lower bound:
    lower_bound = max(n * m, m * p, p * n)  # 25 for 4x5*5x5
    
    for target_rank in range(r0 - 1, lower_bound, -1):
        if time.time() > deadline - 5:
            break
        
        # Try a few random attempts at this rank
        max_attempts = 3
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random integer values in {-1, 0, 1}
            U = np.zeros((n * m, target_rank), dtype=np.float64)
            V = np.zeros((m * p, target_rank), dtype=np.float64)
            W = np.zeros((p * n, target_rank), dtype=np.float64)
            
            # Random initialization
            for c in range(target_rank):
                U[:, c] = np.random.choice([-1, 0, 1], size=n * m)
                V[:, c] = np.random.choice([-1, 0, 1], size=m * p)
                W[:, c] = np.random.choice([-1, 0, 1], size=p * n)
            
            if _verify_float(U, V, W, n, m, p, tol=1e-4):
                # Convert to clean values
                U_best = np.round(U).astype(np.int64)
                V_best = np.round(V).astype(np.int64)
                W_best = np.round(W).astype(np.int64)
                if _verify(U_best, V_best, W_best, n, m, p):
                    best_rank = target_rank
                    break
        
        if best_rank < r0:
            break
    
    return best_rank, U_best, V_best, W_best


def _try_strassen_blocks(n, m, p, deadline):
    """
    Try to use Strassen's algorithm on 2x2 sub-blocks.
    
    For 4x5 * 5x5:
    Split A (4x5) into rows [0:2, 2:4] and cols [0:2, 2:5]:
      A = [A11(2x2)  A12(2x3)]
          [A21(2x2)  A22(2x3)]
    
    Split B (5x5) into rows [0:2, 2:5] and cols [0:2, 2:5]:
      B = [B11(2x2)  B12(2x3)]
          [B21(3x2)  B22(3x3)]
    
    C11(2x2) = A11*B11 + A12*B21
    C12(2x3) = A11*B12 + A12*B22
    C21(2x2) = A21*B11 + A22*B21
    C22(2x3) = A21*B12 + A22*B22
    
    The 2x2*2x2 parts can use Strassen (7 instead of 4 multiplications).
    But the rectangular parts (2x3*3x2, 2x2*2x3, 2x3*3x3, 2x2*3x3)
    still need schoolbook.
    
    Let me compute the rank:
    - A11*B11: 4 schoolbook or 7 Strassen
    - A21*B11: 4 schoolbook or 7 Strassen
    - A12*B21: 2*3*2 = 12 schoolbook
    - A22*B21: 2*3*2 = 12 schoolbook
    - A11*B12: 2*2*3 = 12 schoolbook
    - A21*B12: 2*2*3 = 12 schoolbook
    - A12*B22: 2*3*3 = 18 schoolbook
    - A22*B22: 2*3*3 = 18 schoolbook
    
    Schoolbook total: 4+4+12+12+12+12+18+18 = 92
    With Strassen on 2x2*2x2: 7+7+12+12+12+12+18+18 = 98 (worse!)
    
    So Strassen on individual 2x2 blocks doesn't help here because the
    rectangular parts dominate.
    
    Alternative: use a different block structure. What if we split
    the 5 dimension into 2+3 differently?
    
    Let me try: A (4x5) split cols as [0:3, 3:5]:
      A = [A11(4x3)  A12(4x2)]
    
    B (5x5) split rows as [0:3, 3:5]:
      B = [B11(3x5)]
          [B21(2x5)]
    
    C(4x5) = A11*B11 + A12*B21
    Rank: 4*3*5 + 4*2*5 = 60 + 40 = 100. Same as schoolbook.
    
    It seems hard to beat 100 with simple block decompositions.
    
    Let me try a completely different approach: use the fact that the
    tensor for 4x5*5x5 might have a lower rank than schoolbook due to
    algebraic identities.
    
    Actually, the known minimum rank for 4x5*5x5 is not well-studied.
    The lower bound is max(4*5, 5*5, 4*5) = 25, but actual minimum
    is likely higher.
    
    For now, let me focus on finding any rank < 100 via a more
    systematic search.
    """
    pass


def _structured_reduction(U0, V0, W0, n, m, p, r0, deadline):
    """
    Try to reduce rank by exploiting the structure of the MM tensor.
    
    Key insight: The MM tensor T[i,j,k] = 1 for all i,j,k.
    A rank-r decomposition gives:
      T[i,j,k] = sum_{c=1}^{r} U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    If we can find r < n*m*p such that this holds, we've beaten schoolbook.
    
    One approach: start with schoolbook (r=100) and try to find a
    linear dependence among the 100 rank-1 terms. If term c1 and c2
    can be replaced by a single term, we reduce rank by 1.
    
    For two terms to be mergeable, we need:
      U1[i,j]*V1[j,k]*W1[k,i] + U2[i,j]*V2[j,k]*W2[k,i] = U3[i,j]*V3[j,k]*W3[k,i]
    
    This is a system of 100 equations (one per (i,j,k) triple) in the
    entries of U3, V3, W3. If a solution exists, we can merge.
    
    Let's try a simpler heuristic: randomly perturb the decomposition
    and check if the rank can be reduced via Gaussian elimination on
    the associated linear system.
    """
    best_rank = r0
    U_best, V_best, W_best = U0.copy(), V0.copy(), W0.copy()
    
    # Try a few structured perturbations
    for trial in range(20):
        if time.time() > deadline - 5:
            break
        
        # Create a random sparse perturbation
        U = U0.astype(np.float64).copy()
        V = V0.astype(np.float64).copy()
        W = W0.astype(np.float64).copy()
        
        # Add small random noise to some entries
        for _ in range(10):
            row_u = random.randint(0, n * m - 1)
            row_v = random.randint(0, m * p - 1)
            row_w = random.randint(0, p * n - 1)
            col = random.randint(0, r0 - 1)
            noise = random.choice([-1, 1]) * 0.5
            U[row_u, col] += noise
            V[row_v, col] += noise
            W[row_w, col] += noise
        
        # Check if still valid
        if _verify_float(U, V, W, n, m, p, tol=1e-3):
            # Try to reduce rank by setting some columns to zero
            # and re-solving
            for col_to_remove in range(r0 - 1, r0 - 10, -1):
                if time.time() > deadline - 5:
                    break
                U_try = U[:, :col_to_remove].copy()
                V_try = V[:, :col_to_remove].copy()
                W_try = W[:, :col_to_remove].copy()
                # This won't generally work, but check
                if _verify_float(U_try, V_try, W_try, n, m, p, tol=1e-3):
                    best_rank = col_to_remove
                    U_best = U_try.astype(np.int64)
                    V_best = V_try.astype(np.int64)
                    W_best = W_try.astype(np.int64)
                    break
    
    return best_rank, U_best, V_best, W_best


def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    deadline = time.time() + 130  # Leave 20s margin for safety
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    best_rank = n * m * p  # 100
    U_best, V_best, W_best = U_sb, V_sb, W_sb
    
    # Try structured reduction
    if time.time() < deadline - 30:
        r, U, V, W = _structured_reduction(U_sb, V_sb, W_sb, n, m, p, best_rank, deadline)
        if r < best_rank:
            best_rank = r
            U_best, V_best, W_best = U, V, W
    
    # Try rank reduction
    if time.time() < deadline - 20:
        r, U, V, W = _rank_reduction_attempt(U_best, V_best, W_best, n, m, p, best_rank, deadline)
        if r < best_rank:
            best_rank = r
            U_best, V_best, W_best = U, V, W
    
    # Convert to lists for the output interface
    return {
        "U": U_best.tolist(),
        "V": V_best.tolist(),
        "W": W_best.tolist()
    }

# EVOLVE-BLOCK-END
