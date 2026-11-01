"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

import time
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
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
    Verify that the decomposition exactly reproduces matrix multiplication.
    The tensor T[i,j,k] = 1 if and only if (i,j,k) is a valid index triple.
    We check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True


def _verify_fast(U, V, W, n, m, p):
    """Faster verification using numpy."""
    rank = len(U[0])
    Uf = np.array(U, dtype=np.float64)
    Vf = np.array(V, dtype=np.float64)
    Wf = np.array(W, dtype=np.float64)
    # For each (i,j,k), compute sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # Build the full tensor check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                s = np.dot(Uf[u_idx], Vf[v_idx] * Wf[w_idx])
                if abs(s - 1) > 1e-9:
                    return False
    return True


def _try_rational_polish(U, V, W, n, m, p, max_iter=50):
    """
    Try to polish a floating-point decomposition into exact rational entries.
    Returns None if polishing fails.
    """
    rank = len(U[0])
    # For each column, we have three vectors u_c, v_c, w_c.
    # The constraint is: for all (i,j,k): sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1
    # This is a system of n*m*p equations in rank*3 variables (with the constraint that
    # each column is a rank-1 slice of the tensor).
    # 
    # A simpler approach: try to find integer/rational entries by rounding and verifying.
    # First, check if entries are already close to integers or half-integers.
    
    # Try rounding to nearest 0.5
    U_rounded = [[round(x * 2) / 2 for x in row] for row in U]
    V_rounded = [[round(x * 2) / 2 for x in row] for row in V]
    W_rounded = [[round(x * 2) / 2 for x in row] for row in W]
    
    if _verify_fast(U_rounded, V_rounded, W_rounded, n, m, p):
        return {"U": U_rounded, "V": V_rounded, "W": W_rounded}
    
    # Try rounding to nearest integer
    U_rounded = [[round(x) for x in row] for row in U]
    V_rounded = [[round(x) for x in row] for row in V]
    W_rounded = [[round(x) for x in row] for row in W]
    
    if _verify_fast(U_rounded, V_rounded, W_rounded, n, m, p):
        return {"U": U_rounded, "V": V_rounded, "W": W_rounded}
    
    return None


def _random_decomposition(rank, n, m, p, seed=None):
    """
    Generate a random decomposition and try to project it onto the valid manifold.
    This is a heuristic approach.
    """
    if seed is not None:
        np.random.seed(seed)
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Random initial matrices
    U = np.random.randn(nm, rank)
    V = np.random.randn(mp, rank)
    W = np.random.randn(pn, rank)
    
    # The constraint is that the tensor T = sum_c u_c ⊗ v_c ⊗ w_c must equal the 
    # matrix multiplication tensor, which has T[i,j,k] = 1 for all valid (i,j,k).
    # This is a very specific constraint. Random matrices won't satisfy it.
    # 
    # Instead, let's use a different approach: start from schoolbook and try to 
    # merge columns or find linear dependencies.
    
    return None


def _attempt_rank_reduction(n, m, p, target_rank, deadline):
    """
    Attempt to find a decomposition of rank target_rank.
    Uses a structured approach based on known algorithms for small matrix multiplication.
    
    For 2x3 * 3x6, the schoolbook rank is 36.
    We know from Strassen-type algorithms that lower ranks are possible.
    
    One approach: use the fact that we can share computations.
    For example, if we group columns of the first matrix and rows of the second,
    we can sometimes reduce the rank.
    
    Let's try a systematic search over structured decompositions.
    """
    start_time = time.time()
    
    # For 2x3 * 3x6, let's think about the structure.
    # A = [[a11, a12, a13],
    #      [a21, a22, a23]]
    # B = [[b11, b12, b13, b14, b15, b16],
    #      [b21, b22, b23, b24, b25, b26],
    #      [b31, b32, b33, b34, b35, b36]]
    # C = A * B, a 2x6 matrix.
    # C[0,0] = a11*b11 + a12*b21 + a13*b31
    # C[0,1] = a11*b12 + a12*b22 + a13*b32
    # ...
    # C[1,0] = a21*b11 + a22*b21 + a23*b31
    # ...
    
    # The tensor has 2*3*6 = 36 nonzero entries, all equal to 1.
    # We need to express this as a sum of rank-1 tensors.
    
    # Let's try a structured decomposition based on grouping.
    # Idea: group the 3 columns of A and 3 rows of B into blocks.
    
    # Actually, let's try a direct algebraic approach.
    # We want to find rank r such that there exist matrices U (6xr), V (18xr), W (12xr)
    # with the property that for all i in {0,1}, j in {0,1,2}, k in {0,1,2,3,4,5}:
    #   sum_c U[i*3+j, c] * V[j*6+k, c] * W[k*2+i, c] = 1
    
    # This is a system of 36 equations.
    # Let's try to find a solution with small integer entries.
    
    # Approach: use the fact that the tensor is "simple" (all entries are 1).
    # Try decompositions where U, V, W have entries in {-1, 0, 1}.
    
    # For rank 36, it's trivial (schoolbook).
    # For rank < 36, we need to find linear dependencies.
    
    # Let's try a specific structured approach:
    # Partition the 36 terms into groups that can be combined.
    
    # One known technique: if we have terms like a_ij * b_jk, we can sometimes
    # combine them using identities like (a+b)(c+d) = ac + ad + bc + bd.
    
    # For 2x3 * 3x6, let's try to use a "tensor train" or "CP" decomposition approach.
    
    # Actually, let's just try to solve the system of equations directly for small ranks.
    # This is a polynomial system, which is hard in general, but for small dimensions
    # and structured solutions, it might work.
    
    # Let's try a greedy approach: start with schoolbook and try to merge pairs of columns.
    
    # Schoolbook: column c corresponds to (i,j,k) with U[i*3+j,c]=1, V[j*6+k,c]=1, W[k*2+i,c]=1.
    # To merge two columns c1 and c2, we need to find a single column that represents both.
    # This is generally not possible unless the two terms are related.
    
    # Let's try a different approach: use known results from tensor rank theory.
    # For matrix multiplication [n,m,p], the tensor rank is known to be at most n*m*p,
    # and for some cases, lower bounds are known.
    
    # For [2,3,6], let's check if there's a known better decomposition.
    # The tensor rank of matrix multiplication [m,n,p] is related to the bilinear complexity.
    # For [2,3,6], this is a specific case.
    
    # Let's try a computational approach: use a SAT/SMT-style search or a numerical
    # optimization to find a low-rank decomposition.
    
    # Given the time constraint, let's try a few structured decompositions.
    
    # Approach 1: Try rank 36 (schoolbook) - always works.
    # Approach 2: Try to find a rank-35 decomposition by checking if one column can be
    #              expressed as a linear combination of others.
    
    # For the schoolbook decomposition, each column is linearly independent in the
    # tensor product space, so rank 36 is minimal for this specific form.
    # But there might be a different decomposition with lower rank.
    
    # Let's try a numerical approach: use numpy to find a low-rank approximation
    # of the tensor and then verify if it's exact.
    
    # Flatten the tensor T of shape (2,3,6) into a matrix and find its rank.
    T = np.ones((n, m, p))
    
    # Reshape to (n, m*p) - this gives the "fiber" matrix for mode-1
    T1 = T.reshape(n, m * p)
    # Reshape to (m, n*p) - mode-2
    T2 = T.reshape(m, n * p)
    # Reshape to (p, n*m) - mode-3
    T3 = T.reshape(p, n * m)
    
    # The tensor rank is at least the max of the ranks of these fiber matrices.
    r1 = np.linalg.matrix_rank(T1)
    r2 = np.linalg.matrix_rank(T2)
    r3 = np.linalg.matrix_rank(T3)
    lower_bound = max(r1, r2, r3)
    
    # For our tensor (all ones), the fiber matrices have rank 1 (since all rows are the same).
    # So the lower bound is 1, which is very weak.
    
    # The actual tensor rank is harder to determine. Let's try a direct search.
    
    # Given the complexity, let's just return the schoolbook decomposition for now,
    # but with a time check.
    
    if time.time() > deadline:
        return None
    
    # Try a structured decomposition based on grouping the 3 middle indices.
    # Idea: use the identity (x+y+z)(u+v+w) = xu + xv + xw + yu + yv + yz + zu + zv + zw
    # This doesn't directly help reduce the number of terms.
    
    # Let's try a different grouping:
    # For the 2x3 part, use a 2x3 matrix multiplication.
    # For the 3x6 part, use a 3x6 matrix multiplication.
    # The total is 2x3 * 3x6 = 2x6.
    
    # We can think of this as: first multiply 2x3 by 3x3 (giving 2x3), then 2x3 by 3x3 (giving 2x6).
    # But this doesn't reduce the rank.
    
    # Let's just return the schoolbook decomposition.
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with schoolbook decomposition (always valid)
    result = _schoolbook(n, m, p)
    
    # Try to find a better decomposition
    # For now, we'll just return the schoolbook decomposition.
    # A more advanced approach would search for lower-rank decompositions.
    
    # Check if we have time for a search
    if time.time() < deadline:
        # Try a few random perturbations of the schoolbook decomposition
        # and see if we can find a valid lower-rank decomposition.
        pass
    
    return result

# EVOLVE-BLOCK-END
