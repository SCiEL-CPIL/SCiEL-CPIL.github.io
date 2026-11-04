"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, product
from fractions import Fraction

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify_fast(U, V, W, n, m, p, T=None):
    """Fast verification using numpy."""
    if T is None:
        T = _build_tensor(n, m, p)
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Compute U @ V^T element-wise with W
    # T_decomp[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
    # Efficient: for each c, outer product
    T_decomp = np.zeros_like(T)
    for c in range(rank):
        T_decomp += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
    return np.allclose(T_decomp, T, atol=1e-9)

def _construct_schoolbook():
    n, m, p = DIMENSIONS
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

def _try_rank_reduction_by_merging(U, V, W, n, m, p, T, max_pairs=5000):
    """
    Try to merge pairs of rank-1 terms. If we have two terms (u1,v1,w1) and (u2,v2,w2),
    we can replace them with one term if there exist scalars a,b such that:
    a*u1 + b*u2 = u_new, a*v1 + b*v2 = v_new, a*w1 + b*w2 = w_new
    and the tensor contribution is preserved. This is only possible in special cases.
    
    More practically: try removing one term and see if the remaining terms can be adjusted.
    For a true rank reduction, we need to find a set of rank-1 tensors that sum to T
    with fewer terms.
    
    A more systematic approach: for each pair of columns in U,V,W, check if their
    span contains a rank-1 decomposition that can replace both.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to find linear dependencies among the rank-1 terms
    # For each pair (c1, c2), check if we can eliminate one by adjusting the other
    # This is a heuristic; true rank minimization is NP-hard
    
    # Alternative: use the fact that for matrix multiplication, there are known
    # lower bounds and constructions. For 3x5*5x7, the schoolbook is 105.
    # Known results: M(3,5,7) <= M(3,4,7)+M(0,1,7) etc.
    
    # Let's try a direct algebraic approach:
    # The tensor T has entries T[i,j,k] = 1 iff i//m == j//p and j%p == k%p and k%n == i%m
    # Actually: T[i*m+j, j*p+k, k*n+i] = 1
    
    # Key insight: we can use the structure of the indices.
    # For fixed j, the terms involving j are: for each i in 0..n-1 and k in 0..p-1:
    # T[i*m+j, j*p+k, k*n+i] = 1
    # This is an n*p "slice" for each j.
    
    # The schoolbook uses n*m*p = 105 terms.
    # Can we do better? The tensor rank of matrix multiplication M_{n,m,p} is known
    # to be at most n*m*p, and for some cases less.
    
    # For small cases, let's try to find a decomposition with rank < 105.
    # One approach: use the "Laderman" or "Strassen" style for rectangular matrices.
    
    # Actually, let's try a different structural approach:
    # View the multiplication as: C[i,k] = sum_j A[i,j] * B[j,k]
    # We can group by i and k. For each (i,k), we need sum_j A[i,j]*B[j,k].
    
    # The tensor is: T = sum_{i,j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    
    # Rank reduction idea: if we can find a linear combination of terms that
    # cancels out, we can reduce the rank.
    
    # For now, return the schoolbook as the best we can do without more search.
    return best_rank, best_U, best_V, best_W

def _search_low_rank(n, m, p, T, max_rank=None, timeout=100):
    """
    Search for a low-rank decomposition using a structured approach.
    
    Key insight: the matrix multiplication tensor for n×m × m×p has a natural
    decomposition. We can try to exploit the fact that the "middle" index j
    appears in both the first and second modes.
    
    Strategy: Try rank values from known lower bounds upward, and for each rank,
    try to find a valid decomposition using a greedy/linear-algebra approach.
    
    For 3×5 × 5×7, the schoolbook rank is 105.
    Lower bounds: max(n*m, m*p, p*n) = max(15, 35, 21) = 35.
    Upper bounds: n*m*p = 105.
    
    Let's try to find a decomposition with rank significantly less than 105.
    
    One known approach: use the "rectangular Strassen" algorithm.
    For 3×5 × 5×7, we can pad to 4×8 × 8×8 and use Strassen recursively.
    But that gives rank 7 * (rank of 2×4 × 4×4) which may not help.
    
    Alternative: use the fact that 3×5 × 5×7 can be decomposed using
    the "Bini" algorithm or other known fast matrix multiplication schemes.
    
    For now, let's try a direct search: for rank r from 35 to 104, try to
    find a valid decomposition. This is computationally expensive, so we'll
    use a heuristic.
    
    Actually, let's try a specific construction:
    
    The multiplication C = A * B where A is 3×5 and B is 5×7.
    C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    
    We can write this as a sum of rank-1 tensors. The schoolbook uses 15*7 = 105 terms.
    
    Can we do better? Consider that for each fixed j, we have 3*7 = 21 terms.
    If we could "share" some of these terms across different j values, we could reduce the rank.
    
    One approach: use the fact that the tensor has a lot of structure. The entries
    T[i*m+j, j*p+k, k*n+i] = 1 form a pattern that can be exploited.
    
    Let's try a different approach: use the "tensor train" or "matrix product state"
    representation to find a low-rank decomposition.
    
    For now, let's just return the schoolbook and rely on the evaluator to measure.
    """
    start = time.time()
    deadline = start + timeout
    
    # Try a few specific constructions
    
    # Construction 1: Schoolbook (rank 105)
    result = _construct_schoolbook()
    best_rank = 105
    best_result = result
    
    # Try to verify and see if we can find something better
    if _verify_fast(result["U"], result["V"], result["W"], n, m, p, T):
        best_rank = 105
        best_result = result
    
    # Try a "block" approach: split the 5 middle indices into groups
    # and use Strassen-like combinations
    
    # For 3×5 × 5×7, split the 5 middle indices as 3+2
    # A = [A1 | A2] where A1 is 3×3 and A2 is 3×2
    # B = [B1; B2] where B1 is 3×7 and B2 is 2×7
    # C = A1*B1 + A2*B2
    
    # A1*B1: 3×3 × 3×7, schoolbook rank = 63
    # A2*B2: 3×2 × 2×7, schoolbook rank = 42
    # Total: 105 (same as before)
    
    # Use Strassen on 3×3 × 3×3 part? But B1 is 3×7, not 3×3.
    # Pad B1 to 3×8? That doesn't help.
    
    # Let's try: 3×5 × 5×7 = (3×3 + 3×2) × (3×7 + 2×7)
    # = 3×3*3×7 + 3×2*2×7
    # For 3×3*3×7: use Strassen on the 3×3 part? 
    # 3×3 × 3×7: split 3×3 into 2×2 + 2×1, and 3×7 into 2×7 + 1×7
    # This gets complicated.
    
    # For now, return the best we have
    return best_rank, best_result

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    # Start with schoolbook
    result = _construct_schoolbook()
    
    if not _verify_fast(result["U"], result["V"], result["W"], n, m, p, T):
        # Should not happen, but handle it
        return result
    
    # Try to find a better decomposition
    best_rank = len(result["U"][0])
    best_result = result
    
    # Try a structured search for rank reduction
    # For 3×5 × 5×7, let's try to use the known fast matrix multiplication
    # algorithms for rectangular matrices.
    
    # One known result: M(3,5,7) <= M(3,4,7) + M(0,1,7)
    # M(3,4,7) can be computed with rank <= 3*4*7 = 84 using schoolbook
    # But we can do better with Strassen-like methods.
    
    # Actually, let's try a direct approach: use the "Laderman" algorithm
    # for 4×4 × 4×4 (rank 18) and adapt it.
    
    # For now, return the schoolbook as the best verified result
    return best_result
# EVOLVE-BLOCK-END
